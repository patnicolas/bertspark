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

import ai.djl.ndarray._
import ai.djl.ndarray.types.DataType
import org.bertspark.dl.config.RBMConfig


private[bertspark] final class VariationalBlock(
  fcLayer: RBMConfig,
  meanLayer: RBMConfig,
  logVarianceLayer: RBMConfig,
  samplerLayer: RBMConfig) extends BaseNetBlock {

  sequentialBlock.add(fcLayer())
  sequentialBlock.add(meanLayer())
  sequentialBlock.add(logVarianceLayer())
  sequentialBlock.add(samplerLayer())


  def parameterize(mean: NDArray, logVariance: NDArray): NDArray = {
    val ndManager = NDManager.newBaseManager
    val std = logVariance.mul(0.5F).exp()
    val stdDev = ndManager.randomNormal(0.0F, 1.0F, mean.getShape, DataType.FLOAT32)
    stdDev.muli(std).addi(mean)
  }

}
