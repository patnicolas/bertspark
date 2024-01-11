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
package org.bertspark.classifier.dataset

import ai.djl.engine.EngineException
import ai.djl.ndarray._
import ai.djl.training.dataset.Batch
import ai.djl.training.util.ProgressBar
import ai.djl.translate.Batchifier
import ai.djl.util.Progress
import java.util.concurrent.atomic.AtomicInteger
import org.apache.spark.sql.Dataset
import org.bertspark._
import org.bertspark.util.NDUtil
import org.bertspark.nlp.trainingset.LabelEmbedding
import org.bertspark.classifier.dataset.ClassifierDataset.DataLoaderIterator
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.slf4j._


/**
 * Define the data set for the BERT classifier as type
 * @param labeledDataDS Labeled training data set loaded from S3
 * @param labelIndexMap Label to index map
 * @param subModelName Name of sub model
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] class ClassifierDataset(
  labeledDataDS: Dataset[LabelEmbedding],
  subModelName: String,
  labelIndexMap: Map[String, Int]
) extends DjlDataset {
  import org.bertspark.config.MlopsConfiguration._

  final def size: Long = labeledDataDS.count()

  // @todo check the labeled data DS
  override def prepare(progress: Progress): Unit = {}

  def check: Unit =
    mlopsConfiguration.check(!labeledDataDS.isEmpty, "ClassifierDataset instance is incorrect")

  override def getData(ndManager: NDManager): java.lang.Iterable[Batch] = {
    val batchSize = mlopsConfiguration.executorConfig.batchSize
    new DataLoaderIterator(ndManager, labeledDataDS.collect, batchSize, subModelName, labelIndexMap)
  }
}


/**
  * Singleton for data loader iterator
  */
private[bertspark] final object ClassifierDataset {
  final private val logger: Logger = LoggerFactory.getLogger("ClassifierDataset")

  implicit def djlDataset2BertClassifierDataset(djlDataset: DjlDataset): ClassifierDataset = {
    // Set up the DJl dataset used for training
    val bertTrainingDataset = convertType[ClassifierDataset](djlDataset)
    bertTrainingDataset.prepare(new ProgressBar())
    bertTrainingDataset
  }

    /**
   * Data iterator for the training of the classifier
   * @param ndManager Reference to the system ND manager
   * @param labeledData Labeled data from data set loaded from S3
   * @param batchSize Size of the batch
   * @param labelIndexMap Map {label -> label index}
   * @param subModelName Name of sub model
   */
  final class DataLoaderIterator(
    ndManager: NDManager,
    labeledData: Array[LabelEmbedding],
    batchSize: Int,
    subModelName: String,
    labelIndexMap: Map[String, Int]
  ) extends java.lang.Iterable[Batch] with java.util.Iterator[Batch] with RuntimeSystemMonitor {
    require(labeledData.nonEmpty, s"Data loader iterator for $subModelName in classification has undefined input")
    require(labelIndexMap.nonEmpty, s"Label index map for $subModelName in classification is undefined")

    private[this] var idx = batchSize
    private[this] val start = System.currentTimeMillis()
    private[this] val recordCount = new AtomicInteger((0))
    override def iterator(): java.util.Iterator[Batch] = this

    override def hasNext: Boolean = idx < labeledData.size

    /**
     * Process the next batch (batch size is specified in the configuration file)
     * @return Batch instance
     */
    override def next: Batch = {
      val subManager = ndManager.newSubManager()
      val numClasses = labelIndexMap.size
      val metricsSummary = allMetrics("Classifier Iterator")
      if(metricsSummary.nonEmpty)
        logDebug(logger, s"${recordCount.addAndGet(batchSize)}: $metricsSummary")
        
      try {
        val labeledDataBatch = labeledData.slice(idx - batchSize, idx)
        val (ndCLSEmbeddingBatch, ndLabelBatch) = {
          // Case there is only one class (Oracle), match all CLS embedding to the single label
          if(numClasses == 1) {
            val ndCLSEmbeddings = labeledDataBatch.map {
              case (contextualDocEmbedding, _) => subManager.create(contextualDocEmbedding)
            }
            val labelArray = Array.fill(1)(1.0F)
            val ndLabel = subManager.create(labelArray)
            val ndLabels = Array.fill(batchSize)(ndLabel)
            buildBatches(ndCLSEmbeddings, ndLabels)
          }
            // Case of multiple classes
          else {
            val (ndCLSEmbeddings, ndLabels): (Array[NDArray], Array[NDArray]) = labeledDataBatch
                .filter {
                  case (_, label) =>
                    val labelIndex = labelIndexMap.getOrElse(label, -1)
                    if (labelIndex < 0 && labelIndex >= numClasses) {
                      val violationMessage = s"should be < $numClasses classes"
                      logger.error(s"Label index ${labelIndex} $violationMessage for sub model $subModelName")
                      false
                    }
                    else
                      true
                }
                .map {
                  case (contextualDocEmbedding, label) =>
                    val labelIndex = labelIndexMap.getOrElse(label.replace(",", " "), {
                      throw new DataBatchException(s"Label $label not found in label index map")
                    })
                    val ndEmbedding = subManager.create(contextualDocEmbedding)

                    // @todo Needs to be a single value
                    val labelArray = Array.fill(numClasses)(0.0F)
                    labelArray(labelIndex) = 1.0F
                    val ndLabel = subManager.create(labelArray)
                    (ndEmbedding, ndLabel)
                }.unzip

            // We need to abort this sub-model in case of issue with the batch
            if (ndLabels.isEmpty)
              throw new DataBatchException(s"Labels set for $subModelName is empty")

            // Extract the batch for embedding and labels
            buildBatches(ndCLSEmbeddings, ndLabels)
          }
        }
        idx += 1
        recordLatency
        subManager.attachAll(ndCLSEmbeddingBatch)
        subManager.attachAll(ndLabelBatch)

        new Batch(
            subManager,
            ndCLSEmbeddingBatch,
            ndLabelBatch,
            ndCLSEmbeddingBatch.size,
            Batchifier.STACK,
            Batchifier.STACK,
            idx,
            ndCLSEmbeddingBatch.size
        )
      }
      catch {
        case e: EngineException =>
          org.bertspark.error[EngineException, Batch]("Engine exception", e)
        case e: IllegalStateException =>
          org.bertspark.error[IllegalStateException, Batch]("Undefined state", e)
        case e: ArrayIndexOutOfBoundsException =>
          org.bertspark.error[ArrayIndexOutOfBoundsException, Batch]("index out of range", e)
        case e: Exception =>
          org.bertspark.error[Exception, Batch]("Undefined exception", e)
      }
    }

      // --------------------  Supporting methods ------------------------

    private def buildBatches(ndCLSEmbeddings: Array[NDArray], ndLabels: Array[NDArray]): (NDList, NDList) = {
      val inputs = ndLabels.map(new NDList(_))
      val batchedNDList = NDUtil.batchify(inputs)
      (new NDList(ndCLSEmbeddings: _*), batchedNDList)
    }


    private def recordLatency: Unit = {
      val recordCnt = recordCount.addAndGet(batchSize)
      if(recordCnt % 100*batchSize == 0) {
        val duration = System.currentTimeMillis() - start
        logDebug(
          logger,
          s"$subModelName: Processed $idx batches, $recordCnt docs ${duration*0.001} sec ave ${duration / recordCnt} msec/rec")
      }
    }
  }
}
