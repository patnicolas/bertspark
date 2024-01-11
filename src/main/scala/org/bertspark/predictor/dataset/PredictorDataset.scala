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
package org.bertspark.predictor.dataset

import ai.djl.engine.EngineException
import ai.djl.ndarray._
import ai.djl.training.dataset.Batch
import ai.djl.translate.Batchifier
import ai.djl.util.Progress
import java.util.concurrent.atomic.AtomicInteger
import org.apache.spark.sql.Dataset
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.nlp.trainingset.KeyedValues
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.predictor.dataset.PredictorDataset.DataLoaderIterator
import org.bertspark.DjlDataset
import org.slf4j._


/**
 * Wrapper for the data set used in the prediction
 * @param inputDataDS Input data set
 *
 * @author Patrick Nicolas
 * @version 0.3
 */
private[bertspark] final class PredictorDataset(
  inputDataDS: Dataset[KeyedValues]
) extends DjlDataset {

  override def prepare(progress: Progress): Unit = {}

  @inline
  final def getInputData: Dataset[KeyedValues] = inputDataDS

  def check: Unit = mlopsConfiguration.check(!inputDataDS.isEmpty, "PredictorDataset instance is incorrect")

  override def getData(ndManager: NDManager): java.lang.Iterable[Batch] =
    new DataLoaderIterator(ndManager, inputDataDS.collect)
}


/**
  * Singleton that wrapps the data loader
  */
private[bertspark] final object PredictorDataset {
  final private val logger: Logger = LoggerFactory.getLogger("PredictorDataset")

  /**
   * Data iterator for the training of the classifier
   * @param ndManager Reference to the system ND manager
   * @param docIdEmbeddingBatch Labeled data from data set loaded from S3
   * @param batchSize Size of the batch
   * @param subModelName Name of sub model
   */
  final class DataLoaderIterator(
    ndManager: NDManager,
    docIdEmbeddingBatch: Array[KeyedValues]
  ) extends java.lang.Iterable[Batch] with java.util.Iterator[Batch] {
    require(docIdEmbeddingBatch.nonEmpty, "Data loader iterator has undefined input")

    val batchSize = mlopsConfiguration.executorConfig.batchSize
    private[this] var idx = batchSize
    private[this] val start = System.currentTimeMillis()

    override def iterator(): java.util.Iterator[Batch] = this

    override def hasNext: Boolean = idx < docIdEmbeddingBatch.size

    /**
     * Process the next batch (batch size is specified in the configuration file)
     * @return Batch instance
     */
    override def next: Batch = {
      import org.bertspark.implicits._
      val subManager = ndManager.newSubManager()

      try {
        val docIdEmbeddingSlice = docIdEmbeddingBatch.slice(idx - batchSize, idx)
        val ndEmbeddingBatch = docIdEmbeddingSlice.map(docIdEmbedding => subManager.create(docIdEmbedding._2))
        val docIds = docIdEmbeddingSlice.map(_._1).toList
        idx += 1
        recordLatency

        new Batch(
          subManager,
          new NDList(ndEmbeddingBatch: _*),
          new NDList(),
          ndEmbeddingBatch.size,
          Batchifier.fromString("none"),
          Batchifier.fromString("none"),
          idx,
          ndEmbeddingBatch.size,
          docIds
        )
      }
      catch {
        case e: EngineException =>
          org.bertspark.error[EngineException, Batch](msg = "Engine exception", e)
        case e: IllegalStateException =>
          org.bertspark.error[IllegalStateException, Batch](msg = "Undefined state", e)
        case e: ArrayIndexOutOfBoundsException =>
          org.bertspark.error[ArrayIndexOutOfBoundsException, Batch](msg = "index out of range", e)
        case e: Exception =>
          org.bertspark.error[Exception, Batch](msg = "Undefined exception", e)
      }
    }

    private def recordLatency: Unit = {
      import DataLoaderIterator._

      val recordCnt = recordCount.addAndGet(batchSize)
      if (recordCnt % 100 * batchSize == 0) {
        val duration = System.currentTimeMillis() - start
        logDebug(
          logger,
          msg = s"Processed $idx batches, $recordCnt docs ${duration * 0.001} sec ave ${duration / recordCnt} msec/rec")
      }
    }
  }
  final object DataLoaderIterator {
    private val recordCount = new AtomicInteger(0)
  }
}

