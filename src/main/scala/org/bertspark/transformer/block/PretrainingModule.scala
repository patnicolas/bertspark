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

import ai.djl.nn.{AbstractBlock, Block, Parameter}
import ai.djl.nn.transformer.BertPretrainingBlock
import ai.djl.training.initializer.TruncatedNormalInitializer
import org.bertspark.config.MlopsConfiguration.{mlopsConfiguration, vocabulary}
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.transformer.config.BERTConfig
import org.slf4j.{Logger, LoggerFactory}

/**
 * {{{
 * Adapter or interface to the various pre-training neural bloc
 * - Default DJL pre-build pre-training block
 * - Customized pre-training block
 * }}}
 * @param bertPreTrainingBlock Pretraining block
 *
 * @author Patrick Nicolas
 * @version 0.5
 */
private[transformer] final class PretrainingModule private (bertPreTrainingBlock: AbstractBlock) {
  bertPreTrainingBlock.setInitializer(new TruncatedNormalInitializer(0.02F), Parameter.Type.WEIGHT)

  def getPretrainingBlock: Block = bertPreTrainingBlock
}


/**
 * Singleton for specialized constructor
 */
private[transformer] object PretrainingModule {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[PretrainingModule])
  /**
   * Constructor for the customized Pre-training block of type 'CustomPretrainingBlock'
   * @param bertConfig Configuration for the custom pre-training block
   * @return Instance of the Pretraining module
   */
  def apply(bertConfig: BERTConfig): PretrainingModule = {
    logDebug(logger, s"Custom pre-trained module ${mlopsConfiguration.preTrainConfig.transformer}")
    new PretrainingModule(CustomPretrainingBlock(bertConfig))
  }

  /**
   * Constructor for the default Pre-training block (DJL)
   * @return Instance of the Pretraining module
   */
  def apply(): PretrainingModule = {
    logDebug(logger, s"Default pre-trained module ${mlopsConfiguration.preTrainConfig.transformer}")

    val bertModel = BERTConfig.getBertConfig(mlopsConfiguration.preTrainConfig.transformer)
    val pretrainingBertBlock = new BertPretrainingBlock(bertModel.setTokenDictionarySize(Math.toIntExact(vocabulary.size)))
    new PretrainingModule(pretrainingBertBlock)
  }
}
