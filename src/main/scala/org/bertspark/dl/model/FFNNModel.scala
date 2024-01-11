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

import ai.djl.ndarray._
import ai.djl.nn.Block
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import org.bertspark.dl.{batchFlattenLbl, linearLbl}
import org.bertspark.dl.block._
import org.bertspark.dl.config.RBMConfig
import org.bertspark.DjlDataset

/**
 * Feed forward neural network built from restricted Boltzmann layers
 * @param inputBlock Input block (flatten)
 * @param rbmBlocks Sequence of restricted Boltzmann blocks
 * @param outputBlock Output block
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] class FFNNModel protected (
  inputBlock: Block,
  rbmBlocks: Seq[RBMBlock],
  outputBlock: Block) extends BaseNetBlock() {

  sequentialBlock.add(inputBlock)
  rbmBlocks.foreach(sequentialBlock.add(_))
  sequentialBlock.add(outputBlock)


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
    params: PairList[String, java.lang.Object]): NDList =
    sequentialBlock.forward(parameterStore, inputNDList, training)

  def predict(trainingDataset: DjlDataset): Unit = ???

  @inline
  final def getInputBlock: Block = inputBlock

  @inline
  final def getRbmBlocks: Seq[RBMBlock] = rbmBlocks

  @inline
  final def getOutputBlock: Block = outputBlock

  override def toString: String = "" // builder.toString
}


/**
 * Constructors for Feed Forward Neural Network model
 */
private[bertspark] final object FFNNModel {
  final val maxNumUnits = 1048576

  def apply(
    inputBlock: Block,
    rbmBlocks: Seq[RBMBlock],
    outputBlock: Block): FFNNModel = new FFNNModel(inputBlock, rbmBlocks, outputBlock)

  def apply(
    inputSize: Int,
    hiddenLayersConf: Seq[BaseHiddenLayer],
    numOutputUnits: Int): FFNNModel = {
    check(inputSize, hiddenLayersConf.map(_._1), numOutputUnits)

    new FFNNModel(
      RBMConfig(batchFlattenLbl,inputSize)(),
      hiddenLayersConf.map(RBMBlock(_)),
      RBMConfig(linearLbl,numOutputUnits)())
  }


  def apply(
    hiddenLayersConf: Seq[BaseHiddenLayer],
    numOutputUnits: Int): FFNNModel =
    apply(0, hiddenLayersConf, numOutputUnits)

  @throws(clazz = classOf[IllegalArgumentException])
  final private def check(inputSize: Int, numHiddenUnits: Seq[Int], numOutputUnits: Int): Unit = {
    require(numHiddenUnits.nonEmpty, "No hidden layer is defined for this RBMBlock")

    require((inputSize >= 0 && inputSize < maxNumUnits) || inputSize == -1,
      s"Number of inputs $inputSize should be [1, $maxNumUnits]")
    numHiddenUnits.foreach(numHiddenUnits =>
      require(numHiddenUnits > 0 && numHiddenUnits < maxNumUnits,
        s"Number of hidden units $numHiddenUnits should be [1, $maxNumUnits]"))
    require(numOutputUnits > 0 && numOutputUnits < maxNumUnits,
      s"Number of hidden units $numOutputUnits should be [1, $maxNumUnits]")
  }
}


