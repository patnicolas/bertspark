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
 */
package org.bertspark.transformer.block

import ai.djl.engine.EngineException
import ai.djl.ndarray._
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.dl
import org.bertspark.transformer.config.BERTConfig
import org.bertspark.util.NDUtil
import org.slf4j._


/**
 * Block that define the inference for the pre-training model
 * {{{
 *
 *
 * }}}
 * @param bertConfig Configuration for the Pretraining BERT model
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] class BERTFeaturizerBlock protected (
  bertConfig: BERTConfig
) extends CustomPretrainingBlock(bertConfig, dl.gelulbl) {
  import BERTFeaturizerBlock._

  private[this] val segmentEmbeddingAggregation: SegmentEmbeddingAggregation =
    if(mlopsConfiguration.preTrainConfig.isEmbeddingConcatenate) new SegmentEmbeddingConcatenation
    else new SegmentEmbeddingSummation

  private[this] val segmentEmbeddingSelection: SegmentEmbeddingVec =
    if (mlopsConfiguration.isClsEmbedding) new CLSSegmentEmbedding
    else new PooledOutputSegmentEmbedding


  /**
   * Forward execution for the entire pre-training block
   * {{{
   *   The output are the segments embeddings for each document in the batch.
   *   The embedding can be extracted directly from the CLS token or pooled across the embeddings of all tokens
   *
   * The dimension for the segment input embeddings
   *       batch_size x num_segments_in_doc x number_embeddings x maxSegmentsLength
   *  i.e       4             2                     3                 128
   * }}}
   *
   * @param parameterStore Model parameters
   * @param segmentInputEmbeddingsBatch Input (batch) of tokensIs, typeIs, sequenceMask and masked indices
   * @param training Flag that specify training
   * @param params Extra parameters for training (usually null)
   * @return NDList of sentence predictor and masked token predictor if successful, an empty NDList otherwise
   */
  override protected def forwardInternal(
    parameterStore: ParameterStore,
    segmentInputEmbeddingsBatch: NDList,
    training : Boolean,
    params: PairList[String, java.lang.Object]): NDList =
    try {
      val ndChildManager = NDManager.subManagerOf(segmentInputEmbeddingsBatch)
      segmentInputEmbeddingsBatch.tempAttach(ndChildManager)

      // Retrieve the segment embeddings associated with the documents in this batch
      val segmentBatchNdEmbeddings = getSegmentEmbeddingsBatch(segmentInputEmbeddingsBatch, ndChildManager)

      // Implement the aggregation of the embeddings all the segments for each document in the batch
      val batchClsPredictions = segmentBatchNdEmbeddings.map(
        segmentNdEmbeddings => {
          // Compute the CLS (output) embedding for each of the segment of the document
          val segmentClsEmbeddings = segmentNdEmbeddings.map(
            forwardSegment(ndChildManager, parameterStore, _, training)
          )
          // Apply the aggregation of embeddings for each segments of the document
          segmentEmbeddingAggregation(segmentClsEmbeddings)
        }
      )
      // Return a batch of document embeddings... as either concatenation or polling/sum of
      // the CLS prediction/embedding from each document segment
      val result = NDUtil.batchify(batchClsPredictions.toArray)
      ndChildManager.ret(result)
    }
    catch {
      case e: EngineException =>
        org.bertspark.errorMsg[EngineException](msg = s"Engine failed for ${bertConfig.toString}", e, logger)
        new NDList()
      case e: ArrayIndexOutOfBoundsException =>
        org.bertspark.errorMsg[ArrayIndexOutOfBoundsException](msg = s"Indexing failed for  ${bertConfig.toString}", e, logger)
        new NDList()
      case e: Exception =>
        org.bertspark.errorMsg[Exception](msg = s"Undefined forward failure ${bertConfig.toString}", e, logger)
        new NDList()
    }


  // ---------------------  Supporting methods -----------------------------------

  protected def forwardSegment(
    ndChildManager: NDManager,
    parameterStore: ParameterStore,
    segmentNdEmbeddings: NDList,
    training: Boolean): NDList = {
    ndChildManager.attachAll(segmentNdEmbeddings)
    // Invoke the stack of transformer neural block
    val ndBertResult = thisTransformerBlock.forward(parameterStore, segmentNdEmbeddings, training)

    // Extract the embedding sequence and the pooled output
    // then apply the appropriate extraction of embedding given the configuration
    val predictionNDList = segmentEmbeddingSelection(ndBertResult)
    ndChildManager.ret(predictionNDList)
  }
}


/**
  * {{{
  * Singleton for
  * - Constructor variants
  * - Extracting the segment embeddings for each document and for a batch of documents
  * }}}
 */
private[bertspark] object BERTFeaturizerBlock {
  final private val logger: Logger = LoggerFactory.getLogger("BERTFeaturizerBlock")

  def apply(bertConfig: BERTConfig): BERTFeaturizerBlock = new BERTFeaturizerBlock(bertConfig)

  def apply(): BERTFeaturizerBlock = apply(BERTConfig())


  def getSegmentEmbeddingsBatch(ndBatch: NDList, ndManager: NDManager): Seq[Array[NDList]] = {
    val batchShapes: Array[Shape] = ndBatch.getShapes()
    val batchSize = batchShapes(0).get(0).toInt
    (0 until batchSize).map(index => getSegmentEmbeddings(ndBatch.head.get(index), ndManager))
  }

  def getSegmentEmbeddings(ndArrayBatch: NDArray, ndManager: NDManager): Array[NDList] = {
    // Get the number of segments per documents
    val numSegments = ndArrayBatch.getShape().get(0).toInt
    // Retrieve the embeddings associated with this batch of documents
    val segmentInputEmbeddings = (0 until numSegments).map(ndArrayBatch.get(_))

    // For each segment input embeddings retrieve the tokenIds, type ids and input mask
    segmentInputEmbeddings.map(
      segmentEmbedding => {
        val ndTokenIds = segmentEmbedding.get(0).expandDims(0)
        val typeIds = segmentEmbedding.get(1).expandDims(0)
        val inputMask = segmentEmbedding.get(2).expandDims(0)
        val ndSegmentEmbedding = new NDList(ndTokenIds, typeIds, inputMask)
        ndManager.attachAll(ndSegmentEmbedding)
        ndSegmentEmbedding
      }
    ).toArray
  }
}