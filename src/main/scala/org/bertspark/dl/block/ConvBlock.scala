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

import org.bertspark.dl.config.{ActivationConfig, BatchNormConfig, ConvLayerConfig, PoolingConfig}


/**
 * Generic convolutional block composed of
 * {{{
 *   - Convolutional layer
 *   - Optional batch normalization
 *   - Activation function
 *   - Pooling function
 * }}}
 *
 * @param convLayerConfig Configuration for the convolutional layer
 * @param batchNormConfig Configuration for optional Batch normalization
 * @param activationConfig Configuration for the Activation function
 * @param poolingConfig Configuration for the Pooling function
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] class ConvBlock protected (
  convLayerConfig: ConvLayerConfig,
  batchNormConfig: Option[BatchNormConfig],
  activationConfig: ActivationConfig,
  poolingConfig: PoolingConfig) extends BaseNetBlock {

  sequentialBlock.add(convLayerConfig())
  batchNormConfig.foreach(config => sequentialBlock.add(config()))
  sequentialBlock.add(activationConfig())
  sequentialBlock.add(poolingConfig())

  def invert: BaseNetBlock =
    throw new UnsupportedOperationException("Inversion is not available for Convolutional block")
}


private[bertspark] final object ConvBlock {
  def apply(
    convLayerConfig: ConvLayerConfig,
    batchNormConfig: BatchNormConfig,
    activationConfig: ActivationConfig,
    poolingConfig: PoolingConfig): ConvBlock =
    new ConvBlock(convLayerConfig, Some(batchNormConfig), activationConfig, poolingConfig)

  def apply(
    convLayerConfig: ConvLayerConfig,
    activationConfig: ActivationConfig,
    poolingConfig: PoolingConfig): ConvBlock =
    new ConvBlock(convLayerConfig, None, activationConfig, poolingConfig)
}
