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

import ai.djl.ndarray.types.Shape
import ai.djl.nn.pooling.Pool
import ai.djl.nn.Block
import org.bertspark.dl



case class PoolingConfig(
  override val blockType: String,
  kernelShape: Shape,
  strideShape: Shape,
  paddingShape: Shape) extends BlockConfig {

  @throws(clazz = classOf[UnsupportedOperationException])
  override def apply(): Block = blockType match {
    case dl.avgPool1dLbl => Pool.avgPool1dBlock(kernelShape, strideShape, paddingShape)
    case dl.avgPool2dLbl => Pool.avgPool2dBlock(kernelShape, strideShape, paddingShape)
    case dl.avgPool3dLbl => Pool.avgPool3dBlock(kernelShape, strideShape, paddingShape)
    case dl.maxPool1dLbl => Pool.maxPool1dBlock(kernelShape, strideShape, paddingShape)
    case dl.maxPool2dLbl => Pool.maxPool2dBlock(kernelShape, strideShape, paddingShape)
    case dl.maxPool3dLbl => Pool.maxPool3dBlock(kernelShape, strideShape, paddingShape)
    case _ => throw new UnsupportedOperationException(s"CNN pooling model $blockType is not supported")
  }

  override def toString: String =
    s"conv_${getId}_(Kernel: $kernelShape, Stride: $strideShape, Padding: $paddingShape)"
}

final object PoolingConfig {
  def apply(
    blockType: String,
    kernelShape: Shape,
    strideShape: Shape
  ): PoolingConfig = PoolingConfig(blockType, kernelShape, strideShape, new Shape(-1, -1))

  def apply(
    blockType: String,
    kernelShape: Shape
  ): PoolingConfig = PoolingConfig(blockType, kernelShape, new Shape(-1, -1), new Shape(-1, -1))
}
