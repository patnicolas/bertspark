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
package org.bertspark.transformer.block

import ai.djl.engine.EngineException
import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.nn.Block
import ai.djl.nn.transformer._
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import org.bertspark._
import org.bertspark.dl.block.BaseNetBlock
import org.bertspark.transformer.block.CustomPretrainingBlock.logger
import org.bertspark.transformer.config.{BERTConfig, BERTPretrainingBlocks}
import org.slf4j._

/**
 * Wrapper for BERT pre-training block. It can invoke either te pre-training model
 * of DJL or the custom implementation.
 * It override the default pre-training block defined in DJL
 *
 * @param bertConfig Configuration for the Pretraining BERT model
 * @param mlmActivation Activation function for the Masked Language Model
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] class CustomPretrainingBlock protected (
  bertConfig: BERTConfig,
  mlmActivation: String) extends BaseNetBlock {

  protected[this] val bertPreTrainingBlock: BERTPretrainingBlocks = {
    val confBlock = bertConfig(mlmActivation)

    // Register Bert block
    addChildBlock(s"BERT-${confBlock.getBertBlockName}", confBlock.getBertBlock)
    sequentialBlock.add(confBlock.getBertBlock)
    // Register Masked Language Model block
    addChildBlock(s"BERT-${confBlock.getMlmBlockName}", confBlock.getMlmBlock)
    sequentialBlock.add(confBlock.getMlmBlock)
    // Register Next Sentence predictor block
    addChildBlock(s"BERT-${confBlock.getNspBlockName}", confBlock.getNspBlock)
    sequentialBlock.add(confBlock.getNspBlock)
    confBlock
  }

  protected[this] lazy val thisTransformerBlock: BertBlock = bertPreTrainingBlock.getBertBlock
  private[this] lazy val thisMlmBlock: BertMaskedLanguageModelBlock = bertPreTrainingBlock
    .getMlmBlock
  private[this] lazy val thisNspBlock: BertNextSentenceBlock = bertPreTrainingBlock.getNspBlock


  final def getBertBlock: BertBlock = bertPreTrainingBlock.getBertBlock

  final def getBertPreTrainingBlock: Block = sequentialBlock


  /**
   * This method delegates processing to the block that actually implements the recursive
   * initialization of child block
   * @param ndManager Reference to the ND array manager
   * @param dataType data type (Default Float 32)
   * @param shapes Shape for the 4 embedding (batch size x embedding size)
   */
  override def initializeChildBlocks(
    ndManager: NDManager,
    dataType: DataType,
    shapes: Shape*): Unit = try {
    import org.bertspark.implicits._

    inputNames = Array[String]("tokenIds", "typeIds", "sequenceMasks", "maskedIndices")

    // Initialize the bert transformer encoder block
    thisTransformerBlock.initialize(ndManager, dataType, shapes: _*)

    val bertOutputShapes = thisTransformerBlock.getOutputShapes(shapes.toArray)
    val embeddedSequence = bertOutputShapes.head
    val pooledOutput = bertOutputShapes(1)
    val maskedIndices = shapes(2)
    // The embedding table has vocabulary_size x embedding_size dimension
    val embeddingTableShape = new Shape(
      thisTransformerBlock.getTokenDictionarySize,
      thisTransformerBlock.getEmbeddingSize
    )
    // Initialize the block for the Masked Language Model
    thisMlmBlock.initialize(ndManager, dataType, embeddedSequence, embeddingTableShape, maskedIndices)
    // Initialize the block for the Next Sentence Predictor
    thisNspBlock.initialize(ndManager, dataType, pooledOutput)
  } catch {
    case e: IllegalStateException =>
      logger.error(s"Failed initializing child blocks ${e.getMessage}")
    case e: Exception =>
      logger.error(s"Failed initializing child blocks ${e.getMessage}")
  }

  /**
   * Forward execution for the entire pre-training block
   * {{{
   *   The output are
   *   - embeddingTable Embedding for each token
   *   - Probability of predicting the next sentence correctly (NSP model)
   *   - Log probabilities of predicting the correct masked token (MLM model)
   * }}}
   * @param parameterStore Model parameters
   * @param inputNDList Input (batch) of tokensIs, typeIs, sequenceMask and masked indices
   * @param training Flag that specify training
   * @param params Extra parameters for training (usually null)
   * @return NDList of sentence predictor and masked token predictor if successful, an empty NDList otherwise
   */
  override protected def forwardInternal(
    parameterStore: ParameterStore,
    inputNDList: NDList,
    training : Boolean,
    params: PairList[String, java.lang.Object]): NDList = {
    import CustomPretrainingBlock._

    // Dimension batch_size x max_sentence_size
    val tokenIds = inputNDList.get(0)
    val typeIds = inputNDList.get(1)
    val inputMasks = inputNDList.get(2)
    // Dimension batch_size x num_masked_token
    val maskedIndices = inputNDList.get(3)

    try {
      val ndChildManager = NDManager.subManagerOf(tokenIds)
      ndChildManager.tempAttachAll(inputNDList)

      // Step 1: Process the transformer block for Bert
      val bertBlockNDInput = new NDList(tokenIds, typeIds, inputMasks)
      val ndBertResult = thisTransformerBlock.forward(parameterStore, bertBlockNDInput, training)

      // Step 2 Process the Next Sentence Predictor block
      // Embedding sequence dimensions are batch_size x max_sentence_size x embedding_size
      val embeddedSequence = ndBertResult.get(0)
      val pooledOutput = ndBertResult.get(1)

      // Need to un-squeeze for batch size =1,   (embedding_vector) => (1, embedding_vector)
      val unSqueezePooledOutput =
        if(pooledOutput.getShape.dimension() == 1) {
          logger.debug("Expand pooled output")
          val expanded = pooledOutput.expandDims(0)
          ndChildManager.tempAttachAll(expanded)
          expanded
        }
        else
          pooledOutput

      // We compute the NSP probabilities in case there are more than one single sentences
      val logNSPProbabilities: NDArray =
         thisNspBlock.forward(parameterStore, new NDList(unSqueezePooledOutput), training).singletonOrThrow

        // Step 3: Process the Masked Language Model block
        // Embedding table dimension are vocabulary_size x Embeddings size
      val embeddingTable = thisTransformerBlock
            .getTokenEmbedding
            .getValue(parameterStore, embeddedSequence.getDevice, training)

        // Dimension:  (batch_size x maskSize) x Vocabulary_size
      val logMLMProbabilities: NDArray = thisMlmBlock
            .forward(
              parameterStore,
              new NDList(embeddedSequence, maskedIndices, embeddingTable),
              training
            ).singletonOrThrow

        // Finally build the output
      val ndOutput = new NDList(logNSPProbabilities, logMLMProbabilities)
      ndChildManager.ret(ndOutput)
    }
    catch {
      case e: EngineException =>
        org.bertspark.errorMsg[EngineException](msg = s"Engine failed for ${bertConfig.toString}", e, logger)
        new NDList()
      case e: ArrayIndexOutOfBoundsException =>
        org.bertspark.errorMsg[ArrayIndexOutOfBoundsException](msg = s"Indexing failed for ${bertConfig.toString}", e, logger)
        new NDList()
      case e: UnsupportedOperationException =>
        org.bertspark.errorMsg[UnsupportedOperationException](msg = s"Unsupported operations for ${bertConfig.toString}", e, logger)
        new NDList()
      case e: Exception =>
        org.bertspark.errorMsg[Exception](msg = s"Undefined forward failure ${bertConfig.toString}", e, logger)
        new NDList()
    }
  }


  override def getOutputShapes(shapes: Array[Shape]): Array[Shape] = {
    require(shapes.length == 4, s"Output shapes for Pre training block ${shapes.length} should be" +
      s" 4")

    val batchSize = shapes.head.get(0)
    val maskedIndexCount = shapes(3).get(1)
    Array[Shape](
      new Shape(batchSize, 2),
      new Shape(batchSize, maskedIndexCount, thisTransformerBlock.getTokenDictionarySize)
    )
  }
}


/**
 * Singleton for constructors
 */
private[bertspark] final object CustomPretrainingBlock {
  final private val logger: Logger = LoggerFactory.getLogger("BERTPreTrainingBlock")

  def apply(bertConfig: BERTConfig, mlmActivation: String): CustomPretrainingBlock =
    new CustomPretrainingBlock(bertConfig, mlmActivation)

  def apply(bertConfig: BERTConfig): CustomPretrainingBlock = new CustomPretrainingBlock(bertConfig, dl.gelulbl)
}