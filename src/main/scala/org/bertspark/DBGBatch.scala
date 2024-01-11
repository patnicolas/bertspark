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

import ai.djl.ndarray._
import ai.djl.training.dataset.Batch
import ai.djl.translate.Batchifier
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.slf4j._


private[bertspark] final class DBGBatch(
  manager: NDManager,
  data: NDList,
  labels: NDList,
  size: Int,
  dataBatchifier: Batchifier,
  labelBatchifier: Batchifier,
  progress: Long,
  progressTotal: Long,
  indices: java.util.List[String] = null) extends Batch(manager: NDManager,
  data,
  labels,
  size,
  dataBatchifier,
  labelBatchifier,
  progress,
  progressTotal,
  indices) with RuntimeSystemMonitor {
  import DBGBatch._

  logDebug(logger, s"Create Batch: ${allMetrics("DbBatch")}")

  override def close(): Unit = {
    super.close()
    logDebug(logger, s"Close batch: ${allMetrics("DbBatch")}")
  }

}


private[bertspark] final object DBGBatch {
  final private val logger: Logger = LoggerFactory.getLogger("DBGBatch")
}