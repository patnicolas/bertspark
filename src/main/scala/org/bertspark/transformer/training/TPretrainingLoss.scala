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
package org.bertspark.transformer.training

import ai.djl.ndarray._
import ai.djl.nn.transformer.BertPretrainingLoss
import java.util.concurrent.atomic.AtomicInteger
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.RuntimeSystemMonitor
import org.bertspark.config.MlopsConfiguration
import org.bertspark.util.TProgress
import org.slf4j._

/**
 * Override default Bert pre-training loss to add memory monitoring and debugging statements
 * @author Patrick Nicolas
 * @version 0.3
 */
private[bertspark] final class TPretrainingLoss()
    extends BertPretrainingLoss
        with RuntimeSystemMonitor
        with TProgress[Int, AtomicInteger] {
  import TPretrainingLoss._, MlopsConfiguration.mlopsConfiguration

  // @todo fix the max value by computing the number of records per selected subModel samples
  protected[this] val maxValue: Int = mlopsConfiguration.preTrainConfig.maxNumRecords*mlopsConfiguration.preTrainConfig.epochs
  protected[this] val progress = (cnt: AtomicInteger) => (cnt.get()*100.0/maxValue).floor.toInt

  override def evaluate(labels: NDList, predictions: NDList): NDArray = {
    logDebug(logger, allMetrics(descriptor = "Prior evaluation"))
    show(counter, descriptor = "Transformer pre-training loss")
    super.evaluate(labels, predictions)
  }
}

private[bertspark] final object TPretrainingLoss {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[TPretrainingLoss])
}
