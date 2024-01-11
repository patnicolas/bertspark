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
package org.bertspark.config

import java.util.concurrent.atomic.AtomicBoolean
import MlopsConfiguration.mlopsConfiguration
import org.bertspark.delay
import org.slf4j.{Logger, LoggerFactory}



private[bertspark] final object DynamicConfiguration {
  final private val logger: Logger = LoggerFactory.getLogger("DynamicConfiguration")

  private val locked = new AtomicBoolean(false)

  /**
   * Update safely the current MLOPS configuration for Hyper parameters optimization
   * @param config Configuration element
   */
  @throws(clazz = classOf[UnsupportedOperationException])
  def apply[T](config: T): Unit = {
   // while (locked.get())
   //   delay(20L)

  //  locked.set(true)
    config match {
      case classifyConf: ClassifyConfig =>
        mlopsConfiguration = mlopsConfiguration.copy(classifyConfig = classifyConf)
      case preTrainConf: PreTrainConfig =>
        mlopsConfiguration = mlopsConfiguration.copy(preTrainConfig = preTrainConf)
      case preProcessConf: PreProcessConfig =>
        mlopsConfiguration = mlopsConfiguration.copy(preProcessConfig = preProcessConf)
      case _ =>
        throw new UnsupportedOperationException("Update configuration is not supported")
    }
  //  locked.set(false)
  }

  def apply(_runId: String, _modelId: String): Unit = {
 //   while (locked.get())
  //    delay(20L)

    // Grab the lock
  //  locked.set(true)
    logger.info(s"Update configuration for ${_runId} and ${_modelId}")
    val _classifyConfig = mlopsConfiguration.classifyConfig.copy(modelId = _modelId)
    mlopsConfiguration = mlopsConfiguration.copy(runId = _runId, classifyConfig = _classifyConfig)
    // Release the lock
 //   locked.set(false)
  }

  /**
   * Incremental either the runId (model identifier for the transformer) or modelId (model
   * identifier for the classifier)
   * @param isPreTraining Flag to specify this is a pre-training (Transformer) case
   */
  def ++(isPreTraining: Boolean): Unit = {
    while (locked.get())
      delay(20L)

    // Grab the lock
    locked.set(true)
    // Update configuration of the pre-training model
    if(isPreTraining) {
      val runIdCount = mlopsConfiguration.runId.toInt + 1
      logger.info(s"Increment configuration for pretraining ${runIdCount}")
      mlopsConfiguration = mlopsConfiguration.copy(runId = runIdCount.toString)
    }
      // Update configuration of the classifier
    else {
      val modelIdCount = mlopsConfiguration.classifyConfig.modelId.toInt + 1
      logger.info(s"Increment configuration for classifier $modelIdCount")
      val updateClassifyConfig = mlopsConfiguration.classifyConfig.copy(modelId = modelIdCount.toString)
      mlopsConfiguration = mlopsConfiguration.copy(classifyConfig = updateClassifyConfig)
    }
    // Release the lock
    locked.set(false)
  }

}