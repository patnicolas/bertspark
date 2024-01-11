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
package org.bertspark.dl.block

import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.training.ParameterStore
import ai.djl.training.initializer._
import ai.djl.util._
import org.bertspark.classifier.block.DbgSequentialBlock
import org.slf4j._
import scala.annotation.varargs
import scala.collection.mutable.ListBuffer


/**
 * High level Block that adds child block to the this parent. The key functionality is to add
 * {{{
 *   In case the debug level == 'debug' the instrumental version of Sequential block,
 *   DbgSequentialBlock is used
 * }}}
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] abstract class BaseNetBlock extends AbstractBlock {

  protected[this] lazy val sequentialBlock: SequentialBlock = {
    import org.bertspark.config.MlopsConfiguration._
    if(mlopsConfiguration.isLogLevelDebug)  new DbgSequentialBlock else new SequentialBlock
  }

  override protected def forwardInternal(
    parameterStore: ParameterStore,
    inputNDList: NDList,
    training : Boolean,
    params: PairList[String, java.lang.Object]): NDList =
    sequentialBlock.forward(parameterStore, inputNDList, training)


  override def getOutputShapes(shapes: Array[Shape]): Array[Shape] = {
    val blockIterator = sequentialBlock.getChildren.values.iterator()

    @annotation.tailrec
    def _getOutputShapes(curShapes: Array[Shape]): Array[Shape] =
      if(!blockIterator.hasNext)
        curShapes
      else {
        val block = blockIterator.next
        val newShapes = block.getOutputShapes(curShapes)
        _getOutputShapes(newShapes)
      }
    _getOutputShapes(shapes)
  }


  /**
   * Initialize the base block with a shape current NDManager and initializer
   * @param ndManager Reference to the current NDManager
   * @param shape Shape
   * @param initializer Initializer
   */
  def initialize(ndManager: NDManager, shape: Shape, initializer: Initializer = new NormalInitializer()): Unit = {
    setInitializer(initializer, Parameter.Type.WEIGHT)
    initialize(ndManager, DataType.FLOAT32, shape)
  }

  /**
   * Initialization of the child blocks. This implementation uses the Scala tail recursion
   * @param ndManager Reference to the current NDManager
   * @param dataType Type of data (i.e. Float32, ....)
   * @param shapes Set of shapes for the underlying tensor
   */
  override def initializeChildBlocks(ndManager: NDManager, dataType: DataType, shapes: Shape*): Unit = {
    val childValues = sequentialBlock.getChildren.values
    val blockIterator = childValues.iterator()

    @annotation.tailrec
    def _initializeChildBlocks(ndManager: NDManager, dataType: DataType, curShapes: Shape*): Unit =
      if(blockIterator.hasNext) {
        val shapesArray: Array[Shape] = curShapes.toArray
        val block = blockIterator.next
        init(block, ndManager, dataType, shapesArray.toIndexedSeq: _*)
        val nextShapes = block.getOutputShapes(shapesArray)
        _initializeChildBlocks(ndManager, dataType, nextShapes.toIndexedSeq :_*)
      }
    _initializeChildBlocks(ndManager, dataType, shapes.toIndexedSeq :_*)
  }


  def showShapes(initialShape: Shape): Unit =
    println(collectShapes(
      initialShape,
      (pair: Pair[String, Block]) => s"${pair.getKey}: ${pair.getValue}").mkString("\n"))


  override def toString: String = collectShapes(
    sequentialBlock.getInputShapes()(0),
    (pair: Pair[String, Block]) => s"${pair.getKey}: ${pair.getValue}").mkString("\n")

      // ------------------  Supporting methods --------------------------------

  private def collectShapes(initialShape:Shape, f: Pair[String, Block] => String): List[String] = {
    var curShape = initialShape
    (0 until sequentialBlock.getChildren.size).foldLeft(ListBuffer[String]())(
      (buf, index) => {
        val block = sequentialBlock.getChildren.get(index)
        val newShape = block.getValue.getOutputShapes(Array[Shape](curShape))
        curShape = newShape(0)
        buf += f(block)
      }
    ).toList
  }

  @varargs
  private def init(block: Block, ndManager: NDManager, dataType: DataType, curShapes: Shape*): Unit = {
    block.initialize(ndManager, dataType, curShapes :_ *)
  }
}


private[bertspark] final object BaseNetBlock {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[BaseNetBlock])
}


