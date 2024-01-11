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

import ai.djl.nn._
import ai.djl.nn.core.Linear
import org.bertspark.dl


case class RBMConfig(override val blockType: String, numUnits: Int) extends BlockConfig {
  import RBMConfig._

  require(numUnits >= 0 && numUnits < maxNumUnits, s"Number of units $numUnits is out of range [1, $maxNumUnits]")

  @throws(clazz = classOf[UnsupportedOperationException])
  override def apply(): Block = blockType match {
    case dl.linearLbl => Linear.builder.setUnits(numUnits).build
    case dl.batchFlattenLbl => if (numUnits > 0) Blocks.batchFlattenBlock(numUnits) else Blocks
        .batchFlattenBlock()
    case _ => throw new UnsupportedOperationException(s"Single unit $blockType is not supported")
  }

  override def toString: String = s"${getId}_$numUnits"
}


private[bertspark] final object RBMConfig {
  final private val maxNumUnits = 16384
}

