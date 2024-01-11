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
package org.bertspark.dl.model

import ai.djl.nn.Block
import org.bertspark.dl.block.RBMBlock


/**
 * Simple auto-encoder model as a mirroring of a feed forward model.
 * @param inputBlock Input block (flatten)
 * @param rbmBlocks Sequence of restricted Boltzmann blocks
 * @param outputBlock Output block
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class AutoEncoderModel private (
  inputBlock: Block,
  rbmBlocks: Seq[RBMBlock],
  outputBlock: Block
) extends FFNNModel(inputBlock, rbmBlocks, outputBlock) {

  rbmBlocks.reverse.foreach(sequentialBlock.add(_))
  sequentialBlock.add(inputBlock)
}


private[bertspark] final object AutoEncoderModel {
  def apply(
    inputBlock: Block,
    rbmBlocks: Seq[RBMBlock],
    outputBlock: Block
  ): AutoEncoderModel = new AutoEncoderModel(inputBlock, rbmBlocks, outputBlock)


  def apply(ffnnModel: FFNNModel): AutoEncoderModel =
    new AutoEncoderModel(ffnnModel.getInputBlock, ffnnModel.getRbmBlocks, ffnnModel.getOutputBlock)
}
