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
package org.bertspark.classifier.block

import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.nn.SequentialBlock
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import org.bertspark.classifier.block.DbgSequentialBlock.logger
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.slf4j._


/**
 * Override the original Sequential Block for debugging purpose
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class DbgSequentialBlock extends SequentialBlock {

  override def forwardInternal(
    parameterStore: ParameterStore,
    inputs: NDList,
    training: Boolean,
    params: PairList[String, AnyRef]): NDList = {
    import org.bertspark.implicits._

    var current = inputs
    children.values().foreach(
      block => current = block.forward(parameterStore, current, training)
    )
    current
  }

  override def initializeChildBlocks(manager: NDManager, dataType: DataType, inputShapes: Shape*): Unit = {
    import org.bertspark.implicits._

    var shapes: Array[Shape] = inputShapes.toArray
    var index = 0
    getChildren.values().foreach(
      block => {
        logDebug(logger,  s"Block #${index} input shapes: ${shapes.mkString(" ")}")
        val n = shapes
        block.initialize(manager, dataType, shapes:_*)
        shapes = block.getOutputShapes(shapes)
        logDebug(logger,  s"Block #${index} output shapes: ${shapes.mkString(" ")}")
        index += 1
      }
    )
  }
}


private[bertspark] final object DbgSequentialBlock {
  final private val logger: Logger = LoggerFactory.getLogger("DbgSequentialBlock")
}

