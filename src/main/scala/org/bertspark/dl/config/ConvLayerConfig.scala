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
import ai.djl.nn.Block
import ai.djl.nn.convolutional._
import org.bertspark.dl
import org.bertspark.dl.block.isValidShape


/**
 *
 * @param blockType
 * @param kernelShape
 * @param strideShape
 * @param paddingShape
 * @param numFilters
 * @param bias
 */
private[bertspark] case class ConvLayerConfig(
  override val blockType: String,
  kernelShape: Shape,
  strideShape: Shape,
  paddingShape: Shape,
  numFilters: Int,
  bias: Boolean) extends BlockConfig {

  @throws(clazz = classOf[UnsupportedOperationException])
  override def apply(): Block = blockType match {
    case dl.conv1dLbl => build1d
    case dl.conv2dLbl => build2d
    case _ => throw new UnsupportedOperationException(s"CNN model $blockType is not supported")
  }

  private def build2d: Block = {

    var _builder = Conv2d.builder.setKernelShape(kernelShape)
    if (isValidShape(strideShape))
      _builder = _builder.optStride(strideShape)
    if (isValidShape(paddingShape))
      _builder = _builder.optPadding(paddingShape)
    if (numFilters > 0)
      _builder = _builder.setFilters(numFilters)

    _builder.optBias(bias).build
  }

  private def build1d: Block = {
    var _builder = Conv1d.builder.setKernelShape(kernelShape)
    if (isValidShape(strideShape))
      _builder = _builder.optStride(strideShape)
    if (isValidShape(paddingShape))
      _builder = _builder.optPadding(paddingShape)
    if (numFilters > 0)
      _builder = _builder.setFilters(numFilters)
    _builder.optBias(bias).build
  }

  override def toString: String =
    s"conv_${getId}_(Kernel: $kernelShape, Stride: $strideShape, Padding: $paddingShape, $numFilters $bias)"
}

private[bertspark] final object ConvLayerConfig {
  def apply(
    blockType: String,
    kernelShape: Shape,
    strideShape: Shape,
    numFilters: Int,
    bias: Boolean): ConvLayerConfig =
    ConvLayerConfig(blockType, kernelShape, strideShape, new Shape(-1, -1), numFilters, bias)

  def apply(
    blockType: String,
    kernelShape: Shape,
    numFilters: Int,
    bias: Boolean): ConvLayerConfig =
    ConvLayerConfig(blockType, kernelShape, new Shape(-1, -1), new Shape(-1, -1), numFilters, bias)

}
