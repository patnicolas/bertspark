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
package org.bertspark.dl.block

import org.bertspark.dl.config.{ActivationConfig, BatchNormConfig, DeConvLayerConfig}


/**
 * Deconvolutional block that contains, in the following order
 * {{{
 *     De-convolutional layer
 *     Optional batch normalization
 *     Activation function
 * Contrary to convolutional network, deconvolutional do not have pooling
 * }}}
 *
 * @param deConvLayerConfig Configuration for the de-convolutional layer
 * @param batchNormConfig Optional Batch normalization configuration
 * @param activationConfig  Configuration for the Activation function
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class DeConvBlock protected (
  deConvLayerConfig: DeConvLayerConfig,
  batchNormConfig: Option[BatchNormConfig],
  activationConfig: ActivationConfig) extends BaseNetBlock {

  sequentialBlock.add(deConvLayerConfig())
  batchNormConfig.foreach(config =>  sequentialBlock.add(config()))
}


/**
 * Singleton for constructors
 */
private[bertspark] final object DeConvBlock {
  def apply(
    deConvLayerConfig: DeConvLayerConfig,
    batchNormConfig: BatchNormConfig,
    activationConfig: ActivationConfig): DeConvBlock =
    new DeConvBlock(deConvLayerConfig, Some(batchNormConfig), activationConfig)

  def apply(
    deConvLayerConfig: DeConvLayerConfig,
    activationConfig: ActivationConfig): DeConvBlock = new DeConvBlock(deConvLayerConfig, None, activationConfig)
}
