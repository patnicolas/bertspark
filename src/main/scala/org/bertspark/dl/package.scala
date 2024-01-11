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
package org.bertspark


/**
 * Constants/values shared across deep learning models
 * {{{
 *   Transformers have unique characteristics and are treated as separate family of networks within their own package
 * }}}
 * @author Patrick Nicolas
 * @version 0.1
 */
package object dl {

  final val batchFlattenLbl = "batchFlatten"
  final val linearLbl = "linear"
  final val conv2dLbl = "conv2d"
  final val conv1dLbl = "cond1d"
  final val conv3dLbl = "conv3d"
  final val deConv2dLbl = "deConv2d"
  final val deConv1dLbl = "deConv1d"
  final val deConv3dLbl = "deConv3d"
  final val bertLbl = "bert"
  final val pretrainedBertLbl = "pretrainedLbl"

  final val avgPool1dLbl = "avgPool1d"
  final val avgPool2dLbl = "avgPool2d"
  final val avgPool3dLbl = "avgPool3d"
  final val maxPool1dLbl = "maxPool1d"
  final val maxPool2dLbl = "maxPool2d"
  final val maxPool3dLbl = "maxPool3d"
  final val batchNormLbl = "batchNorm"

  final val reluLbl = "relu"
  final val gelulbl = "gelu"
  final val leakyReluLbl = "leakyRelu"
  final val eluLbl = "elu"
  final val sigmoidLbl = "sigmoid"
  final val tanhLbl = "tanh"
  final val softSignLbl = "softSign"
  final val softPlusLbl = "softPlus"
  final val preluLbl = "prelu"
  final val seluLbl = "selu"

  final val sgdLbl = "sgd"
  final val adamLbl = "adam"

  final val trainLabel = "train_epoch"
  final val validLabel = "valid_epoch"
}
