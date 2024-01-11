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
package org.bertspark.transformer.config

import ai.djl.ndarray.NDArray
import ai.djl.nn.Block
import ai.djl.nn.transformer._
import org.bertspark.config.MlopsConfiguration
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.dl
import org.bertspark.dl.config._
import org.bertspark.nlp.vocabulary
import org.bertspark.transformer.config.BERTConfig.getBertConfig
import org.slf4j.{Logger, LoggerFactory}

/**
 * Define the component blocks for BERT training
 * @param bertNamedBlock Pair (Bert block name, bert block)
 * @param mlmNamedBlock Pair (NLM block name, NLM block)
 * @param nspNamedBlock  Pair (NSP block, NSP block)
 * @author Patrick Nicolas
 * @version 0.1
 */
final class BERTPretrainingBlocks(
  bertNamedBlock: (String, BertBlock),
  mlmNamedBlock: (String, BertMaskedLanguageModelBlock),
  nspNamedBlock: (String, BertNextSentenceBlock)) {
  @inline
  def getBertBlockName: String = bertNamedBlock._1

  @inline
  def getBertBlock: BertBlock = bertNamedBlock._2

  @inline
  def getMlmBlockName: String = mlmNamedBlock._1

  @inline
  def getMlmBlock: BertMaskedLanguageModelBlock = mlmNamedBlock._2

  @inline
  def getNspBlockName: String = nspNamedBlock._1

  @inline
  def getNspBlock: BertNextSentenceBlock = nspNamedBlock._2
}


/**
 * Generic configuration for BERT pre-training
 * @param blockType Type of neural block
 * @param bertModelType Type of BERT model (Micro, base....)
 * @param vocabularySize Number of entries in the vocabulary
 * @author Patrick Nicolas
 * @version 0.1
 */
case class BERTConfig(
  override val blockType: String,
  bertModelType: String,
  vocabularySize: Long) extends BlockConfig {

  /**
    * Generate a pre-training block given the model and block types
    * @return Block associated with transformer pre-training
    */
  override def apply(): Block = {
    import BERTConfig._

    val builder = getBertConfig(bertModelType)
    blockType match {
      case dl.bertLbl =>
        builder.setTokenDictionarySize(Math.toIntExact(vocabularySize)).build
      case dl.pretrainedBertLbl =>
        new BertPretrainingBlock(builder.setTokenDictionarySize(Math.toIntExact(vocabularySize)))
      case _ =>
        throw new UnsupportedOperationException(s"Bert block type $blockType is not supported")
    }
  }

  /**
    * {{{
    * Generate the 3 key component blocks
    *  - MLM block
    *  - BERT block
    *  - NSP block
    * }}}
    *
    * @param activationType Activation associated with the Masked Language Model
    * @return Group of 3 components blocks
    */
  def apply(activationType: String): BERTPretrainingBlocks = {
    val bertBlock = getBertConfig(bertModelType)
        .setTokenDictionarySize(Math.toIntExact(vocabularySize))
        .build
    val activationFunc: java.util.function.Function[NDArray, NDArray] = ActivationConfig.getNDActivationFunc(activationType)
    val bertMLMBlock = new BertMaskedLanguageModelBlock(bertBlock, activationFunc)
    val bertNSPBlock = new BertNextSentenceBlock
    new BERTPretrainingBlocks(
      ("transformer", bertBlock),
      ("mlm", bertMLMBlock),
      ("nsp", bertNSPBlock)
    )
  }
}


/**
  * Default constructors
  */
private[bertspark] object BERTConfig {
  private val logger: Logger = LoggerFactory.getLogger("BERTConfig")
  import org.bertspark.Labels._

  def apply(): BERTConfig =
    BERTConfig("BERT", mlopsConfiguration.preTrainConfig.transformer, MlopsConfiguration.vocabulary.size)


  def getBertConfig(bertModelType: String): BertBlock.Builder = bertModelType match {
    case `nanoBertLbl` =>
      // 4 encoders, 4 attention heads, embedding size: 256, dimension 256x4
      BertBlock.builder().nano()
    case `microBertLbl`=>
      // 12 encoders,8 attention heads, embedding size: 512, dimension 512x4
      BertBlock.builder().micro()
    case `baseBertLbl` =>
      // 12 encoders,12 attention heads, embedding size: 768, dimension 768x4
      BertBlock.builder().base()
    case `largeBertLbl` =>
      // 24 encoders,16 attention heads, embedding size: 1024, dimension 1024x4
      BertBlock.builder().large()
    case _ =>
      logger.warn(s"Bert block model type $bertModelType is not supported used Bert base")
      BertBlock.builder()
  }

  private final val bertNanoMinSeqLength = 128
  private final val bertMicroMinSeqLength = 128
  private final val bertBaseMinSeqLength = 256
  private final val bertLargeMinSeqLength = 512
  private final val bertXLargeMinSeqLength = 1024

  private final val bertNanoEmbeddingsSize = 256
  private final val bertMicroEmbeddingsSize = 512
  private final val bertBaseEmbeddingsSize = 768
  private final val bertLargeEmbeddingsSize = 1024
  private final val bertXLargeEmbeddingsSize = 1024

  final def getMinSeqLength(bertModelType: String): Int = bertModelType match {
    case `nanoBertLbl` => bertNanoMinSeqLength
    case `microBertLbl` => bertMicroMinSeqLength
    case `baseBertLbl` => bertBaseMinSeqLength
    case `largeBertLbl` => bertLargeMinSeqLength
    case `xLargeBertLbl` => bertXLargeMinSeqLength
    case _ =>
      logger.warn(s"Bert block model type $bertModelType is not supported used Bert base")
      bertBaseMinSeqLength
  }

  final def getEmbeddingsSize(bertModelType: String): Int = bertModelType match {
    case `nanoBertLbl` => bertNanoEmbeddingsSize
    case `microBertLbl` => bertMicroEmbeddingsSize
    case `baseBertLbl` => bertBaseEmbeddingsSize
    case `largeBertLbl` => bertLargeEmbeddingsSize
    case `xLargeBertLbl` => bertXLargeEmbeddingsSize
    case _ =>
      logger.warn(s"Bert block model type $bertModelType is not supported used Bert base")
      bertBaseEmbeddingsSize
  }
}

