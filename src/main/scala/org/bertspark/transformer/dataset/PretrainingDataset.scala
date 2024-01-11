/**
 * Copyright 2022,2023 Patrick R. Nicolas. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package org.bertspark.transformer.dataset

import ai.djl.engine.EngineException
import ai.djl.ndarray._
import ai.djl.ndarray.types.Shape
import ai.djl.training.dataset.Batch
import ai.djl.translate.Batchifier
import ai.djl.util.Progress
import org.apache.spark.sql._
import org.bertspark.nlp.token._
import org.bertspark.util.io.SingleS3Dataset
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.nlp.medical.noContextualDocumentEncoding
import org.bertspark.nlp.trainingset.{ContextualDocument, SentencePair}
import org.bertspark.transformer.dataset.PretrainingDataset.logger
import org.bertspark.util.rdbms.PredictionsTbl
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.{DjlDataset, RuntimeSystemMonitor}
import org.slf4j._



/**
 * Definition of of a BERT dataset
 * {{{
 *   This class encapsulate Spark dataset.
 *   It is used for both pre-training (mask tokens/indices + masked labels) and classification
 *   (mask tokens/indices + labels)
 * }}}
 * @param trainingSet Dataset of type org.apache.spark.sql.Dataset of elements of type DATA_TYPE
 * @param extractor Convert type of dataset into a generic Contextual Document
 * @param datasetConfig Configuration used to extract the data set
 * @tparam T Type of elements in the data set
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] class PretrainingDataset[T] protected (
  trainingSet: Dataset[T],
  extractor: T => ContextualDocument,
  datasetConfig: TDatasetConfig
)(implicit sparkSession: SparkSession, encoder: Encoder[T]) extends DjlDataset {

  protected[this] var documentCorpus: DocumentCorpus = null
  private var preProcessedTokenizer: TokenizerPreProcessor = _

  @inline
  final def getTrainingSet: Dataset[T] = trainingSet

  /**
   * The preparation consists of building a vocabulary using pre-defined words as potential constraint
   * @param progress Execution/training
   */
  override def prepare(progress: Progress): Unit = {
    import PretrainingDataset._

    preProcessedTokenizer = TokenizerPreProcessor(datasetConfig.getPreProcessedTokenizerType)
    logDebug(logger, "Bert pretraining dataset is ready!!")
  }

  @inline
  final def size: Long = trainingSet.count()


  /**
   * Retrieve a iterable collection of batches
   * @param ndManager Reference to the current NDArray manager
   * @return Java Iterable collection of batches
   */
  override def getData(ndManager: NDManager): java.lang.Iterable[Batch] = {
    logDebug(logger, s"Next epoch started .... for manager: ${ndManager.getName}")

    // If epoch sampling is used....
    if(documentCorpus == null)
      setDocumentCorpus(trainingSet.sample(1.0 / 8))

    // Finally generate the iterator for the data loader
    // We re-assign the iterator reference to force GC()
    new DataLoaderIterator(ndManager, datasetConfig, documentCorpus.getSentencePairs, documentCorpus.docIds)
  }

  def getShape: Shape = new Shape(datasetConfig.getBatchSize, datasetConfig.getMaxSeqLength)

  private def setDocumentCorpus(inputDS: Dataset[T]): Unit =
    documentCorpus = DocumentCorpus(inputDS, extractor, preProcessedTokenizer, datasetConfig.getSentencesBuilder)


  /**
   * Iterator for the data loader from a training corpus
   * @param ndManager Reference to the current ND Manager
   * @param datasetConfig Configuration for the data set:
   *                           batchSize: Int,
   *                           maxSeqLength: Int,
   *                           maxMasking: Int,
   *                           minTermFrequency: Int,
   *                           sentencesBuilderType: String
   *                           preProcessedTokenizerType: String,
   *                           preTrainingMode: Boolean
   * @param sampledSentencesPairs: Array of sentence pairs
   */
  final private class DataLoaderIterator(
    ndManager: NDManager,
    datasetConfig: TDatasetConfig,
    sampledSentencesPairs: Array[SentencePair],
    docIds: Seq[String]
  ) extends java.lang.Iterable[Batch] with java.util.Iterator[Batch] with RuntimeSystemMonitor with LatencyLog {

    var count: Int = 0
    override protected[this] val maxValue: Int = docIds.size* mlopsConfiguration.preTrainConfig.epochs

    // From configuration BERTDatasetConfig
    val batchSize: Int = datasetConfig.getBatchSize
    val maxSeqLength: Int = datasetConfig.getMaxSeqLength
    val maxMasking: Int = datasetConfig.getMaxMasking
    log(logger, 1, "Pretrain Iterator Init")

    private[this] var idx = batchSize

    private[this] val maskedInstances = {
      val sentencePairs: Array[SentencePair] =
        if(mlopsConfiguration.isSingleSegmentDocument)
          sampledSentencesPairs
        else {
          // If there is more than one segment then swap sentences on each other pair
          // then shuffle then sentence pairs
          val tempSentencePairs = SentencePair.swapEveryOther(sampledSentencesPairs)
          scala.util.Random.shuffle(tempSentencePairs).toArray
        }
      sentencePairs.map(TMaskedInstance(_, maxSeqLength, maxMasking, 0.2F))
    }
    override def iterator(): java.util.Iterator[Batch] = this

    override def hasNext: Boolean = idx < maskedInstances.size

    /**
     * Extract the next batch from the training data. This method behaves similar to __getItem__ in
     * PyTorch
     * @return Next batch
     */
    override def next: Batch = {
      import org.bertspark.implicits._

      val subManager = ndManager.newSubManager()
      count += 1
      log(logger, collectionInterval = 4, marker = "Pre-train Iterator")

      val maskInstancesBatch = maskedInstances.slice(idx - batchSize, idx)
      val ids = docIds.slice(idx - batchSize, idx)
      idx += 1
      try {
        val bertPretrainingBatch = TPretrainingBatch(maskInstancesBatch)
        val ndFeatures = bertPretrainingBatch.getFeatures(subManager)
        val ndLabels = bertPretrainingBatch.getLabels(subManager)
        // Record latency data
        log(logger)

        // This batch contains the list of document ids associated with input and labels
        val batch = new Batch(
            subManager,
            ndFeatures,
            ndLabels,
            maskInstancesBatch.size,
            Batchifier.STACK,
            Batchifier.STACK,
            idx,
            maskInstancesBatch.size,
            ids)
        batch
      }
      catch {
        case e: EngineException =>
          org.bertspark.error[EngineException, Batch](msg = "Engine exception", e)
        case e: IllegalStateException =>
          org.bertspark.error[IllegalStateException, Batch](msg = "Undefined state", e)
        case e: Exception =>
          org.bertspark.error[Exception, Batch](msg = "Undefined exception", e)
      }
    }
  }
}


/**
  * Singleton for constructor
  */
private[bertspark] final object PretrainingDataset {
  final private val logger: Logger = LoggerFactory.getLogger("PretrainingDataset")

  final val noNDList = new NDList()
  final def nullBatch(ndManager: NDManager) =
    new Batch(
      ndManager,
      noNDList,
      noNDList,
    0,
    null,
    null,
    0L,
    0L)

  /**
   *  Constructor of pre-training data set provided as a parameterized type Spark data set
   * @param trainingSet parameterized type Spark data set
   * @param extractor Conversion of parameterized type to contextual document
   * @param bertDatasetConfig Configuration for the data set
   * @param sparkSession Implicit reference to the current Spark context
   * @param encoder Implicit encoder for parameterized type of data set elements
   * @tparam T Parameterized type of data set elements
   * @return Instance of Pre-training data set
   */
  def apply[T](
    trainingSet: Dataset[T],
    extractor: T => ContextualDocument,
    bertDatasetConfig: TDatasetConfig
  )(implicit sparkSession: SparkSession, encoder: Encoder[T]): PretrainingDataset[T] =
    new PretrainingDataset[T](trainingSet, extractor, bertDatasetConfig)

  /**
   * Constructor of pre-training data set supplied as a data set of contextual document
   * @param trainingSet Training set provided as a Spark data set
   * @param datasetConfig Configuration for the data set
   * @param sparkSession Implicit reference to the current Spark context
   * @param encoder Implicit encoder for Contextual document
   * @return  Instance of Pre-training data set
   */
  def apply(
    trainingSet: Dataset[ContextualDocument],
    datasetConfig: TDatasetConfig
  )(implicit sparkSession: SparkSession): PretrainingDataset[ContextualDocument] = {
    import sparkSession.implicits._
    apply[ContextualDocument](trainingSet, noContextualDocumentEncoding, datasetConfig)
  }


  /**
   * Constructor for which the dataset is loaded from storage
   * @param s3Dataset Descriptor for the storage
   * @param bertDatasetConfig Configuration for the data set
   * @param sparkSession Implicit reference to the current Spark context
   * @param encoder Implicit encoder
   * @tparam T Type of elements in the Spark datasets
   * @return Instance of Pre-training data set
   */
  def apply[T](
    s3Dataset: SingleS3Dataset[T],
    bertDatasetConfig: TDatasetConfig
  )(implicit sparkSession: SparkSession, encoder: Encoder[T]): PretrainingDataset[T] = {
    new PretrainingDataset[T](s3Dataset.inputDataset, s3Dataset.extractCtxDocument, bertDatasetConfig)
  }


  /**
   * Constructor specific to Contextual document using S3 as input
   * @param s3Dataset Descriptor for the storage
   * @param bertDatasetConfig Configuration for the data set
   * @param sparkSession Implicit reference to the current Spark context
   * @return Instance of Pre-training data set
   */
  def apply(
    s3Dataset: SingleS3Dataset[ContextualDocument],
    bertDatasetConfig: TDatasetConfig
  )(implicit sparkSession: SparkSession): PretrainingDataset[ContextualDocument] = {
    import sparkSession.implicits._
    apply[ContextualDocument](s3Dataset, bertDatasetConfig)
  }


  /**
   * Constructor specific to Contextual document using RDBMS prediction records as input
   * @param predictionTbl Prediction table
   * @param bertDatasetConfig Configuration for the data set
   * @param sparkSession Implicit reference to the current Spark context
   * @return Instance of Pre-training data set
   */
  def apply(
    predictionTbl: PredictionsTbl,
    condition: String,
    bertDatasetConfig: TDatasetConfig
  )(implicit sparkSession: SparkSession): PretrainingDataset[ContextualDocument] = {
    import sparkSession.implicits._
    import org.bertspark.nlp.trainingset.implicits._
    val contextualDocuments: Seq[ContextualDocument] = (predictionTbl, condition)
    apply(contextualDocuments.toDS(), bertDatasetConfig)
  }
}
