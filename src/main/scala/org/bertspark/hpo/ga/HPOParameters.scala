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
package org.bertspark.hpo.ga

import org.bertspark.config.{ClassifyConfig, DynamicConfiguration, MlopsConfiguration, PreTrainConfig}
import org.bertspark.hpo.ga.GASearch.HPOBaseParam


/**
 * Wrapper for the HPO parameters defined as list of HPO base parameter with type String, Array[INt],...
 * @param hpoParams List of HPO parameters
 * @author Patrick Nicolas
 * @version 0.4
 */
private[bertspark] final class HPOParameters(hpoParams: List[HPOBaseParam]) {
  import MlopsConfiguration._

  final def chromosomeRepresentation: java.util.List[HPOBaseParam] = {
    import org.bertspark.implicits._
    val chromosomes: java.util.List[HPOBaseParam] = hpoParams
    chromosomes
  }

  def updateMlopsConfiguration: Unit = {
    val newPreTrainConfig = mlopsConfiguration.preTrainConfig.copy(
      predictor = hpoParams(0).getValue.asInstanceOf[String],
      clsAggregation = hpoParams(1).getValue.asInstanceOf[String]
    )

    val newBaseLr: Float = hpoParams(5).getValue.asInstanceOf[Float]
    val newOptimizer = mlopsConfiguration.classifyConfig.optimizer.copy(baseLr = newBaseLr)

    val newClassifyConfig = mlopsConfiguration.classifyConfig.copy(
      dlLayout = hpoParams(2).getValue.asInstanceOf[Array[Int]],
      maxNumRecordsPerLabel = hpoParams(3).getValue.asInstanceOf[Int],
      optimizer = newOptimizer
    )

    DynamicConfiguration.apply[PreTrainConfig](newPreTrainConfig)
    DynamicConfiguration.apply[ClassifyConfig](newClassifyConfig)
  }

  override def toString: String = hpoParams.map(hpoParam => s"${hpoParam.label}: ${hpoParam.getValue}").mkString("\n")
}
