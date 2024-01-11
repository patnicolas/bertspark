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
package org.bertspark.dl.config

import ai.djl.ndarray.NDArray
import ai.djl.nn._
import org.bertspark.dl


/**
 * Define the configuration for the activation neural block
 * @param blockType Type of activation
 * @author Patrick Nicolas
 * @version 0.1
 */
case class ActivationConfig(override val blockType: String) extends BlockConfig {

  override def apply(): Block = blockType match {
    case dl.reluLbl => Activation.reluBlock
    case dl.gelulbl => Activation.geluBlock
    case dl.leakyReluLbl => Activation.leakyReluBlock(0.04F)
    case dl.eluLbl => Activation.eluBlock(0.05F)
    case dl.sigmoidLbl => Activation.sigmoidBlock
    case dl.tanhLbl => Activation.tanhBlock
    case dl.softSignLbl => Activation.softSignBlock
    case dl.softPlusLbl => Activation.softPlusBlock
    case dl.preluLbl => Activation.preluBlock
    case dl.seluLbl => Activation.seluBlock
    case _ =>
      throw new UnsupportedOperationException(s"Activation type $blockType is not supported")
  }

  override def toString: String = s"activation_$getId"
}


private[bertspark] final object ActivationConfig {

  /**
   * Retrieve the activation function on NDArray data associated with a given activation type
   * @param activationType Type of activation function
   * @return NDArray -> NDArray Java function
   */
  def getNDActivationFunc(activationType: String): java.util.function.Function[NDArray, NDArray] = activationType match {
    case dl.reluLbl => Activation.relu
    case dl.gelulbl => Activation.gelu
    case dl.sigmoidLbl => Activation.sigmoid
    case dl.tanhLbl => Activation.tanh
    case dl.leakyReluLbl => (ndArray: NDArray) => Activation.leakyRelu(ndArray, 0.02F)
    case dl.softSignLbl => Activation.softSign
    case dl.softPlusLbl => Activation.softPlus
    case dl.seluLbl => Activation.selu
    case _ =>
      throw new UnsupportedOperationException(s"Activation function $activationType is not supported")
  }
}