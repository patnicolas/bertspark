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
package org.bertspark.classifier.training

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.HasConvergedException
import org.bertspark.classifier.training.ClassifierConvergence.logger
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.HashMap


/**
  * Implement the various algorithm for the convergence of classifier training
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] trait ClassifierConvergence {

  /**
    * Test the convergence of the algorithm
    * {{{
    *  Compute the ratio of loss function as
    *    2*x(n)/[x(n-1) + x(n-2)]  if n > 2
    *    x(n)/x(n-1)  if n = 2
    *    1000F otherwise
    *  }}}
    * @return true if converged, false otherwise
    */
  def hasConverged(metricAccumulator: HashMap[String, List[Float]], epochNo: Int, modelName: String): Unit = {
    val convergenceRatio = mlopsConfiguration.classifyConfig.optimizer.convergenceLossRatio

    val converged =
      if(convergenceRatio > 0.0F)
        metricAccumulator.get("train_epoch_loss").map(
          losses =>
            if(losses.size > 2) {
              val averageRatio = 2.0F*losses.head/(losses(1) + losses(2))
              losses.head <= losses(1) && losses(1) <= losses(2) && averageRatio > convergenceRatio
            } else
              false
        ).getOrElse(false)
      else
        false

    // Throw an exception if training has converged..
    if(converged) {
      throw new HasConvergedException("")
    }
  }
}


private[bertspark] final object ClassifierConvergence {
  final private val logger: Logger = LoggerFactory.getLogger("ClassifierConvergence")
}
