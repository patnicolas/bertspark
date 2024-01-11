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
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import org.bertspark.dl.config.{ActivationConfig, RBMConfig}
import org.bertspark.dl.linearLbl


/**
 * Generic Restricted Boltzman machine as an extension of
 * @param hiddenLayersConf Pair (number of units, activation_label)
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] class RBMBlock protected (hiddenLayersConf: BaseHiddenLayer) extends BaseNetBlock {

  sequentialBlock.add(RBMConfig(linearLbl, hiddenLayersConf._1)())
  sequentialBlock.add(ActivationConfig(hiddenLayersConf._2)())

  /**
   * Forward function for this Restricted Boltzmann machine
   * @param parameterStore Parameters list
   * @param inputNDList Input ND list (observations)
   * @param training Flag to specify if this is training environment
   * @param params Tuple parameters [Parameter_name, Parameter_value]
   * @return NDList value
   */
  override protected def forwardInternal(
    parameterStore: ParameterStore,
    inputNDList: NDList,
    training : Boolean,
    params: PairList[String, java.lang.Object]): NDList = {
    sequentialBlock.forward(parameterStore, inputNDList, training)
  }

  override def toString: String = ""
}



private[bertspark] final object RBMBlock {
  final val maxNumUnits = 1048576

  def apply(hiddenLayersConf: BaseHiddenLayer): RBMBlock =
    new RBMBlock(hiddenLayersConf)
}
