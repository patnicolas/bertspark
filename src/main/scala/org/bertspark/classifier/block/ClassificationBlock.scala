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
import ai.djl.nn._
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.Dropout
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import org.bertspark.config.MlopsConfiguration
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.dl.block.BaseNetBlock
import org.bertspark.classifier.block.ClassificationBlock.logger
import org.bertspark.classifier.config.MlopsClassifierConfig
import org.bertspark.config.MlopsConfiguration.DebugLog.logInfo
import org.slf4j._


/**
 * Wrapper for the BERT decoder which consist of Fully connected neural network
 * {{{
 *  The input to the fully connected classifier has to match to the output of the
 *  aggregated prediction for the document (segment concatenation in case of concatenate mode)
 * }}}
 * @param numClasses Number of classes or labels
 * @todo This Decoder should be configurable
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class ClassificationBlock(numClasses: Long) extends BaseNetBlock {
  require(numClasses > 0  && numClasses <= 262144,
    s"Number of classes for classification blocks $numClasses should be [1, 262144]")

  val modelConfigs = MlopsClassifierConfig.getClassifierBlocks(Array[Int](numClasses.toInt))
  modelConfigs.foreach{ case (id, block) => add(id, block) }


  /**
   * This method delegates processing to the block that actually implements the recursive
   * initialization of child block
   * @param ndManager Reference to the ND array manager
   * @param dataType data type (Default Float 32)
   * @param shapes Shape for the 4 embedding (batch size x embedding size)
   */
  override def initializeChildBlocks(ndManager: NDManager, dataType: DataType, shapes: Shape*): Unit =
    super.initializeChildBlocks(ndManager, dataType, shapes:_*)


  override protected def forwardInternal(
    parameterStore: ParameterStore,
    inputNDList: NDList,
    training : Boolean,
    params: PairList[String, java.lang.Object]): NDList =
    sequentialBlock.forward(parameterStore, inputNDList, training,  params)


  override protected def forwardInternal(
    parameterStore: ParameterStore,
    inputNDList: NDList,
    labelNDList : NDList,
    params: PairList[String, java.lang.Object]): NDList =
    sequentialBlock.forward(parameterStore, inputNDList, labelNDList,  params)


  @inline
  final def getNumClasses: Long = numClasses


  private def add(name: String, block: Block): Unit = {
    sequentialBlock.add(block)
    addChildBlock(name, block)
  }

  override def toString: String = modelConfigs.map(_._1).mkString(" ")
}


/**
 * Singleton for classification
 */
private[bertspark] final object ClassificationBlock {
  final private val logger: Logger = LoggerFactory.getLogger("ClassificationBlock")


  final def concatenateNDArray(ndArrays: Array[NDArray]): NDArray =
    (1 until ndArrays.size).foldLeft(ndArrays.head) (
      (ndArray, index) => {
        ndArray.concat(ndArrays(index))
      }
    )

  val batchifierLambda: java.util.function.Function[NDList, NDList] = {
    (ndList: NDList) =>
      val initShapes = ndList.getShapes()
      val concatNDArray = (1 until ndList.size).foldLeft(ndList.get(0)) (
        (ndArray, index) => ndArray.concat(ndList.get(index))
      )

      val reshapedNDArray = concatNDArray.reshape(new Shape(ndList.size(), initShapes.head.size()))
      new NDList(reshapedNDArray)
  }
}

