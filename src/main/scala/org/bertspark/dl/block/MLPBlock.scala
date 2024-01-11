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

import org.bertspark.dl.{batchFlattenLbl, linearLbl}
import org.bertspark.dl.config.RBMConfig


/**
 * Basic Multi-layer perceptron Network with a single hidden layer. This is a special case (shallow network) o
 * the stacked Restricted Boltzmann machine
 *
 * @param inputSize Size of input
 * @param hiddenLayerConf Number of hidden units
 * @param numOutputUnits  Number of output neurons
 * @see org.mlops.nn.blocks.RBMBlock
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class MLPBlock(
  inputSize: Int,
  hiddenLayerConf: BaseHiddenLayer,
  numOutputUnits: Int) extends RBMBlock(hiddenLayerConf) {

  sequentialBlock.add(RBMConfig(batchFlattenLbl,inputSize)())
  sequentialBlock.add(RBMConfig(linearLbl,numOutputUnits)())

}
