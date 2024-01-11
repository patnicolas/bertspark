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
import ai.djl.training.dataset.Batch
import ai.djl.translate.Batchifier
import ai.djl.util.Progress
import java.util.concurrent.atomic.AtomicInteger
import org.apache.spark.sql._
import org.bertspark.util.io.SingleS3Dataset
import org.bertspark.nlp.medical.noContextualDocumentEncoding
import org.bertspark.nlp.token.{DocumentCorpus, TokenizerPreProcessor}
import org.bertspark.transformer.dataset.FeaturesDataset.{logger, DataLoaderIterator}
import org.bertspark.transformer.dataset.TFeaturesInstance.SegmentTokens
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.nlp.trainingset.ContextualDocument
import org.bertspark.{DjlDataset, RuntimeSystemMonitor}
import org.slf4j._


/**
 * Definition of of a BERT dataset
 * {{{
 *   This class encapsulate Spark dataset.
 *   It is used for both pre-training (mask tokens/indices + masked labels) and classification
 *   (mask tokens/indices + labels)
 * }}}
 *
 * @param trainingSet Dataset of type org.apache.spark.sql.Dataset of elements of type DATA_TYPE
 * @param extractor Convert type of dataset into a generic Contextual Document
 * @param datasetConfig Configuration used to extract the data set
 * @tparam T Type of elements in the data set
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class FeaturesDataset[T] protected (
  trainingSet: Dataset[T],
  extractor: T => ContextualDocument,
  datasetConfig: TDatasetConfig
)(implicit sparkSession: SparkSession, encoder: Encoder[T]) extends DjlDataset  {


  protected[this] var documentCorpus: Option[DocumentCorpus] = _


  /**
   * The preparation consists of building a vocabulary using pre-defined words as potential constraint
   * @param progress Execution/training
   */
  override def prepare(progress: Progress): Unit = {
    val preProcessedTokenizer = TokenizerPreProcessor(datasetConfig.getPreProcessedTokenizerType)

    documentCorpus = Some(
      DocumentCorpus(
        trainingSet,
        extractor,
        preProcessedTokenizer,
        datasetConfig.getSentencesBuilder
      )
    )
    logDebug(logger,  "Bert pretraining dataset is ready!!")
  }
  /**
   * Retrieve a iterable collection of batches
   * @param ndManager Reference to the current NDArray manager
   * @return Iterable collection of batches
   */
  override def getData(ndManager: NDManager): java.lang.Iterable[Batch] = {
    val batchDocSegmentTokens: Array[Array[SegmentTokens]] = documentCorpus
        .map(_.getSegments)
        .getOrElse({
          logger.warn("Could not find segment tokens for features dataset")
          Array.empty[Array[SegmentTokens]]
        })

      new DataLoaderIterator(
        ndManager,
        batchDocSegmentTokens,
        documentCorpus.map(_.docIds).getOrElse(Array.empty[String]),
        documentCorpus.map(_.getLabelIndices).getOrElse(Array.empty[Long]),
        datasetConfig.getBatchSize,
        datasetConfig.getMaxSeqLength,
        datasetConfig.getMaxMasking
      )
  }
}



private[bertspark] final object FeaturesDataset {
  final private val logger: Logger = LoggerFactory.getLogger("TFeaturesDataset")

  def apply[T](
    trainingSet: Dataset[T],
    extractor: T => ContextualDocument,
    bertDatasetConfig: TDatasetConfig
  )(implicit sparkSession: SparkSession, encoder: Encoder[T]): FeaturesDataset[T] =
    new FeaturesDataset[T](trainingSet, extractor, bertDatasetConfig)


  def apply[T](
    s3Dataset: SingleS3Dataset[T],
    bertDatasetConfig: TDatasetConfig
  )(implicit sparkSession: SparkSession, encoder: Encoder[T]): FeaturesDataset[T] = {
    val inputDS: Dataset[T] = s3Dataset.inputDataset
    apply(inputDS, s3Dataset.extractCtxDocument, bertDatasetConfig)
  }

  def apply(
    trainingSet: Dataset[ContextualDocument],
    bertDatasetConfig: TDatasetConfig
  )(implicit sparkSession: SparkSession, encoder: Encoder[ContextualDocument]): FeaturesDataset[ContextualDocument] =
    apply[ContextualDocument](trainingSet, noContextualDocumentEncoding, bertDatasetConfig)


  /**
   * Iterator for the data loader from a training corpus
   * @param ndManager Explicit reference to the current ND Manager
   * @param docSegments Batch of array of segments associated with a document
   * @param docIds Batch of document identifiers
   * @param labelIndices: Indices of the labels for pre-training
   * @param batchSize Size of a batch
   * @param maxSeqLength Maximum number of tokens per document or sentence
   * @param maxMasking Maximum number of masked tokens
   */
  final class DataLoaderIterator(
    ndManager: NDManager,
    docSegments: Array[Array[SegmentTokens]],
    docIds: Array[String],
    labelIndices: Array[Long],
    batchSize: Int,
    maxSeqLength: Int,
    maxMasking: Int) extends java.lang.Iterable[Batch] with java.util.Iterator[Batch] with RuntimeSystemMonitor {
    require(docSegments.nonEmpty, "Epoch batch iterator has undefined document segments tokens")
    require(docIds.nonEmpty, "Epoch batch iterator has undefined document ids")
    require(docSegments.size == docIds.size,
      s"Epoch batch iterator num doc segments ${docSegments.size} should be = num of documents ${docIds.size}")

    private[this] val start = System.currentTimeMillis()
    private[this] var idx = batchSize

    private[this] val rawInstances: Array[TFeaturesInstance] = docSegments.map(
      docSegment => TFeaturesInstance(docSegment, maxSeqLength)
    )
    logDebug(logger, "TFeaturesDataset.DataLoaderIterator initialized!")

    override def iterator(): java.util.Iterator[Batch] = this

    override def hasNext: Boolean = idx - batchSize < rawInstances.size

    /**
     * Extract the next batch from the training data. This method behaves similar to __getItem__ in
     * PyTorch
     * @return Next batch
     */
    override def next: Batch = {
      import org.bertspark.implicits._

      // Gets the next batch of data
      val rawInstancesBatch = rawInstances.slice(idx-batchSize, idx)
      val ids = docIds.slice(idx-batchSize, idx)
      idx += 1

      try {
        val subManager = ndManager.newSubManager()

        // Instantiation to generate the 3 input embeddings (tokens, type id and mask position)
        val bertFeaturesBatch = TFeaturesBatch(rawInstancesBatch)
        val ndFeatureBatch: NDList = bertFeaturesBatch.getFeatures(subManager)
        val shapes = ndFeatureBatch.getShapes()
        log

        new Batch(
          subManager,
          ndFeatureBatch,
          new NDList(),
          shapes.head.size().toInt,
          Batchifier.fromString("none"),
          Batchifier.fromString("none"),
          idx,
          shapes.head.size(),
          ids
        )
      }
      catch {
        case e: EngineException =>
          //  subManager.close()
          org.bertspark.error[EngineException, Batch](msg = "Engine exception", e)
        case e: IllegalStateException =>
          //  subManager.close()
          org.bertspark.error[IllegalStateException, Batch](msg = "Undefined state", e)
        case e: Exception =>
          //  subManager.close()
          org.bertspark.error[Exception, Batch](msg = "Undefined exception", e)
      }
    }

    private def log: Unit = {
      import DataLoaderIterator._
      logDebug(
        logger, {
          val recordCnt = recordCount.addAndGet(batchSize)
          val duration = (System.currentTimeMillis() - start)*0.001
          s"Transformer processed ${idx-batchSize} batches,  $recordCnt records in $duration sec with ave ${duration/recordCnt}"
        }
      )
    }
  }


  final object DataLoaderIterator {
    private val recordCount = new AtomicInteger(0)
  }
}
