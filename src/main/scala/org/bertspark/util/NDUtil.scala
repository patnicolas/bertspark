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
package org.bertspark.util

import ai.djl.ndarray.{BaseNDManager, NDArray, NDArrays, NDList, NDManager}
import ai.djl.ndarray.types.DataType
import ai.djl.training.dataset.Batch
import ai.djl.translate.StackBatchifier
import org.bertspark.config.MlopsConfiguration
import org.bertspark.implicits
import org.bertspark.config.MlopsConfiguration.DebugLog.logTrace
import org.slf4j.{Logger, LoggerFactory}
import scala.annotation.tailrec
import scala.collection.concurrent.TrieMap
import scala.collection.mutable.ListBuffer


/**
 * Singleton for the Context Bound between NDArray and NDList and Scala type
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final object NDUtil {
  import org.bertspark.config.MlopsConfiguration._
  final val logger: Logger = LoggerFactory.getLogger("NDUtil")



  /**
   * Singleton to manage the status of ND manager (memory block
   */
  final object NDManagerStatus {
    private lazy val ndManagerMap = TrieMap[NDManager, Boolean]()
    private lazy val ndRootList = ListBuffer[String]()

    def +=(newNDManager: NDManager): Unit =
      if(MlopsConfiguration.DebugLog.isTraceLevel)
        if(ndManagerMap.contains(newNDManager))
          logger.warn(s"NDManagerStatus: newNDManager ${newNDManager.hashCode()} already created")
        else {
          ndManagerMap.put(newNDManager, false)
          logTrace(logger, s"NDManagerStatus: newNDManager ${newNDManager.hashCode()} is created => ${toString}")
        }


    def close(ndManager: NDManager, msg: String): Unit =
      if(MlopsConfiguration.DebugLog.isTraceLevel)
        if(ndManager.isOpen) {
          atClose(ndManager: NDManager)
          logTrace(logger, s"NDManagerStatus: Close ${ndManager.hashCode()} => ${toString}")
          ndManager.close()
        } else {
          logger.warn(s"NDManagerStatus: ${ndManager.hashCode()} $msg")
        }
      else
        if(ndManager.isOpen)
          ndManager.close()


    def close(batch: Batch, msg: String): Unit =
      if(MlopsConfiguration.DebugLog.isTraceLevel)
        if(batch.getManager().isOpen) {
          atClose(batch.getManager())
          logTrace(logger, s"NDManagerStatus: Close batch with ${batch.getManager().hashCode()} => ${toString}")
          batch.close()
        }
        else
          logger.warn(s"NDManagerStatus: batch ${batch.getManager().hashCode()} $msg")
      else
        if(batch.getManager().isOpen)
          batch.getManager().close()

    private def atClose(ndManager: NDManager): Unit = {
      if(!ndManagerMap.contains(ndManager))
        logger.error(s"NDManagerStatus: Cannot close un registered ND manager ${ndManager.hashCode()}")
      else if(ndManagerMap.get(ndManager).get)
        logger.error(s"NDManagerStatus: Cannot close an already closed ND manager")
      else {
        ndManagerMap.put(ndManager, true)
        logTrace(logger, s"NDManagerStatus: ${ndManager.hashCode()} status update => ${toString}")
      }
    }

    override def toString: String =
      s"${ndManagerMap.map{ case (ndManager, isClosed) => s"${ndManager.hashCode()}:$isClosed"}.mkString(" ")}"

    def debugDump(ndManager: NDManager, descriptor: String, level: Int = 3): Unit =
      if(MlopsConfiguration.mlopsConfiguration.isLogTrace) {
        val ndRoot = getRootManager(ndManager)

        if (ndRoot.isInstanceOf[BaseNDManager]) {
          logTrace(logger, s"$descriptor:\n${ndRoot.getName()}")
          ndRoot.asInstanceOf[BaseNDManager].debugDump(level)
        }
        else
          logger.error("Could not dump NDManager")
      }


    @tailrec
    private def getRootManager(ndManager: NDManager): NDManager = {
      val ndParent = ndManager.getParentManager()
      if(ndParent == null)
        ndManager
      else
        getRootManager(ndParent)
    }
  }

  @throws(clazz = classOf[UnsupportedOperationException])
  def display(ndArray: NDArray): String = {
    val shape = ndArray.getShape
    shape.dimension() match {
      case 3 => display((shape.get(0).toInt, shape.get(1).toInt, shape.get(2).toInt), ndArray)
      case 2 => display((shape.get(0).toInt, shape.get(1).toInt), ndArray)
      case 1 => display(shape.get(0).toInt, ndArray)
      case 0 => ndArray.toFloatArray.head.toString
      case _ => throw new UnsupportedOperationException(s"Display of tensor of ${shape.dimension()} not supported")
    }
  }

  def display(shape: (Int, Int), ndArray: NDArray): String = {
    val content = (0 until shape._1).map(
      index1 => {
        (0 until shape._2).map(
          index2 =>  "%.4f".format(ndArray.get(index1, index2).toFloatArray.head)
        ).mkString(" ")
      }
    ).mkString("\n")
    s"Shape: (${shape._1}, ${shape._2}}\n$content)"
  }


  def display(shape: Int, ndArray: NDArray): String = {
    val content = ndArray.toFloatArray.map("%.4f".format(_)).mkString(" ")
    s"Shape: $shape\n$content"
  }

  def display(shape: (Int, Int, Int), ndArray: NDArray): String = {
    val content = (0 until shape._1).map(
      index1 => {
        (0 until shape._2).map(
          index2 => {
            (0 until shape._3).map(
              index3 => "%.4f".format(ndArray.get(index1, index2, index3).toFloatArray.head)
            ).mkString("  ")
          }
        ).mkString("\n")
      }
    ).mkString("\n\n")
    s"Shape: (${shape._1}, ${shape._2}, ${shape._3}\n$content"
  }


  def str(ndArray: NDArray): String = {
    ndArray.getShape.dimension match {
      case 1 => ndArray.toFloatArray.mkString(", ")
      case 2 => str2NDArray(ndArray)
      case 3 =>
        val values = ndArray.toFloatArray
        val numMatrix = ndArray.size(0).toInt
        val numRows = ndArray.size(1)
        val numCols = ndArray.size(2)
        val sizeMatrix: Int = (numRows*numCols).toInt
        val internal = (0 until numMatrix).map(matrixIndex => {
          val startIndex: Int = matrixIndex*sizeMatrix
          val endIndex = startIndex + sizeMatrix
          str2Float(values.slice(startIndex, endIndex), numCols)
        }).mkString(",\n")
        s"[${internal}]"
      case _ =>
        throw new UnsupportedOperationException(s"Shape dimension not supported")
    }
  }

  private def str2NDArray(ndArray: NDArray): String = {
    val values = ndArray.toFloatArray
    val numCols = ndArray.size(1)
    str2Float(values, numCols)
  }

  private def str2Float(values: Array[Float], numCols: Long): String = {
    values.indices.foldLeft(new StringBuilder("["))(
      (sb, index) => {
        if(index > 0) {
          if((index % numCols) == 0x00)
            sb.append("\n")
          else
            sb.append(", ")
        }
        sb.append(values(index))
      }
    ).append("]").toString
  }

  @throws(clazz = classOf[UnsupportedOperationException])
  def str(ndList: NDList): String = {
    import implicits._
    ndList.subList(0, ndList.size).map(ndArray => ndArray.getDataType() match {
      case DataType.INT32 => ndArray.toIntArray.mkString(" ")
      case DataType.FLOAT32 => ndArray.toFloatArray.mkString(" ")
      case DataType.STRING => ndArray.toStringArray.mkString(" ")
      case _ =>
        throw new UnsupportedOperationException(s"Type for NDList ${ndArray.getDataType()} is not recognized")
    }).mkString("\n")
  }

  final object IntNDArray {
    implicit def fromVec(ndManager: NDManager, values: Array[Int]): NDArray = ndManager.create(values)
    implicit def toVec(ndArray: NDArray): Array[Int] = {
      val values = ndArray.toIntArray
      ndArray.close()
      values
    }

    implicit def fromMatrix(ndManager: NDManager, values: Array[Array[Int]]): NDArray = ndManager.create(values)
    implicit def toMatrix(ndArray: NDArray): Array[Array[Int]] = {
      val shapes = ndArray.getShape.getShape
      val values = partition(ndArray.toIntArray, shapes(1).toInt, List[Array[Int]]())
      ndArray.close()
      values
    }

    @annotation.tailrec
    private def partition(input: Array[Int], size: Int, xs: List[Array[Int]]): Array[Array[Int]] = {
      val splitted = input.splitAt(size)
      if(splitted._2.isEmpty)
        (splitted._1 :: xs).reverse.toArray
      else
        partition(splitted._2, size, splitted._1 ::xs)
    }
  }

  final object FloatNDArray {
    implicit def fromVec(ndManager: NDManager, values: Array[Float]): NDArray = ndManager.create(values)
    implicit def toVec(ndArray: NDArray): Array[Float] = {
      val values = ndArray.toFloatArray
      ndArray.close()
      values
    }

    implicit def fromMatrix(ndManager: NDManager, values: Array[Array[Float]]): NDArray = ndManager.create(values)
    implicit def toMatrix(ndArray: NDArray): Array[Array[Float]] = {
      val shapes = ndArray.getShape.getShape
      val values = partition(ndArray.toFloatArray, shapes(1).toInt, List[Array[Float]]())
      ndArray.close()
      values
    }

    @annotation.tailrec
    private def partition(input: Array[Float], size: Int, xs: List[Array[Float]]): Array[Array[Float]] = {
      val splitted = input.splitAt(size)
      if(splitted._2.isEmpty)
        (splitted._1 :: xs).reverse.toArray
      else
        partition(splitted._2, size, splitted._1 ::xs)
    }
  }


  final object IntNDList {
    implicit def fromVec(ndManager: NDManager, values: Array[Array[Int]]): NDList = {
      val ndArrays = values.map(ndManager.create(_))
      new NDList(ndArrays:_*)
    }

    implicit def toVec(ndList: NDList): Array[Array[Int]] =
      (0 until ndList.size()).map(ndList.get(_).toIntArray).toArray

    implicit def fromMatrix(ndManager: NDManager, values: Array[Array[Array[Int]]]): NDList= {
      val ndArrays = values.map(ndManager.create(_))
      new NDList(ndArrays:_*)

    }
    implicit def toMatrix(ndList: NDList): Array[Array[Array[Int]]] = {
      val values = (0 until ndList.size()).map(
        index => {
          val shape = ndList.get(index).getShape
          partition(ndList.get(index).toIntArray, shape.get(1).toInt, List[Array[Int]]())
        }
      ).toArray
      ndList.close()
      values
    }

    @annotation.tailrec
    private def partition(input: Array[Int], size: Int, xs: List[Array[Int]]): Array[Array[Int]] = {
      val splitted = input.splitAt(size)
      if(splitted._2.isEmpty)
        (splitted._1 :: xs).reverse.toArray
      else
        partition(splitted._2, size, splitted._1 ::xs)
    }
  }



  final object FloatNDList {
    implicit def fromVec(ndManager: NDManager, values: Array[Array[Float]]): NDList = {
      val ndArrays = values.map(ndManager.create(_))
      new NDList(ndArrays:_*)
    }

    implicit def toVec(ndList: NDList): Array[Array[Float]] =
      (0 until ndList.size()).map(ndList.get(_).toFloatArray).toArray

    implicit def fromMatrix(ndManager: NDManager, values: Array[Array[Array[Float]]]): NDList= {
      val ndArrays = values.map(ndManager.create(_))
      new NDList(ndArrays:_*)

    }
    implicit def toMatrix(ndList: NDList): Array[Array[Array[Float]]] = {
      val values = (0 until ndList.size()).map(
        index => {
          val shape = ndList.get(index).getShape
          partition(ndList.get(index).toFloatArray, shape.get(1).toInt, List[Array[Float]]())
        }
      ).toArray
      ndList.close()
      values
    }

    @annotation.tailrec
    private def partition(input: Array[Float], size: Int, xs: List[Array[Float]]): Array[Array[Float]] = {
      val splitted = input.splitAt(size)
      if(splitted._2.isEmpty)
        (splitted._1 :: xs).reverse.toArray
      else
        partition(splitted._2, size, splitted._1 ::xs)
    }
  }


  def batchify(inputs: Array[NDList]): NDList = {
    val stackBatchifier = new StackBatchifier()
    val outputNdList = stackBatchifier.batchify(inputs)
    outputNdList
  }


  /**
   * Concatenate a sequence of NDList along the first axis
   * @param inputs Array of NDList
   * @return NDList of concatenated (and flattend) arguments
   */
  def concat(inputs: Array[NDList]): NDList = concat(inputs, 0)


  /**
   * Concatenate a sequence of NDList along a given axis
   * @param inputs Array of NDList
   * @param axis Axis to concatenate against
   * @return NDList of concatenated arguments
   */
  def concat(inputs: Array[NDList], axis: Int): NDList = {
    if(inputs.size > 1) {
      val ndArrays = inputs.map(_.get(0))
      new NDList(concat(ndArrays))
    }
    else
      inputs.head
  }

  def concat(inputs: Array[NDArray]): NDArray =
    if(inputs.size > 1)
      inputs.drop(1).foldLeft(inputs.head) (
        (ndAcc, input) =>
          ndAcc.concat(input)
      )
    else
      inputs.head


  def add(inputs: Array[NDArray]): NDArray =
    if(inputs.size > 1)
      inputs.drop(1).foldLeft(inputs.head) (
        (ndAve, input) => ndAve.add(input)
      )
    else
      inputs.head

  def add(inputs: Array[NDList]): NDList =
    if(inputs.size > 1) {
      val ndArrays = inputs.map(_.get(0))
      new NDList( add(ndArrays))
    } else
      inputs.head

  def debug(desc: String, ndList: NDList): Unit = {
    import org.bertspark.implicits._

    if(ndList.size() > 0) {
      val dataType =  ndList.head.getDataType
      val xs =
        if(dataType.isFloating) ndList.subList(0, ndList.size).map(_.toFloatArray)
        else ndList.subList(0, ndList.size).map(_.toIntArray)

      val dataStr = xs.map(_.mkString(" ")).mkString("\n\n")
      println(s"--------------- $desc ---------------\n$dataStr")
    }
    else
      println("Empty data")
  }


  def cosine(x: NDArray, y: NDArray): Double = {
    val z = NDArrays.dot(x, y)
    val xNorm = x.norm()
    val yNorm = y.norm()
    val prod = z.div((xNorm.mul(yNorm))).toFloatArray
    prod(0)
  }

  /**
   * Compute the similarity between two tensor of type NDArray
   * @param x First tensor
   * @param y Second tensor
   * @param method Method to compute the similarity.. 'euclidean', 'jaccard' or 'cosine'
   * @return Similarity value
   * @throws UnsupportedOperationException if similarity method is not supported
   */
  @throws(clazz = classOf[UnsupportedOperationException])
  @throws(clazz = classOf[IllegalStateException])
  def computeSimilarity(
    x: NDArray,
    y: NDArray,
    method: String,
    size: Long = mlopsConfiguration.getEmbeddingsSize): Double = method match {
    case "cosine" => cosineSimilarity(x, y, size)
    case "euclidean" => euclideanSimilarity(x, y, size)
    case "jaccard" => jacardSimilarity(x, y, size)
    case _ =>
      throw new UnsupportedOperationException(s"Similarity method, $method is not supported")
  }

  /**
   * Compute the similarity between two tensor of type NDArray
   * @param x First tensor
   * @param y Second tensor
   * @param method Method to compute the similarity.. 'euclidean', 'jaccard' or 'cosine'
   * @return Similarity value
   * @throws UnsupportedOperationException if similarity method is not supported
   */
  @throws(clazz = classOf[UnsupportedOperationException])
  @throws(clazz = classOf[IllegalStateException])
  def computeBatchSimilarity(
    x: Array[NDArray],
    y: Array[NDArray],
    method: String,
    size: Long = mlopsConfiguration.getEmbeddingsSize): Double = {
    require(x.size == y.size, s"Size of batch ${x.size} should be = to ${y.size}")
    x.indices.map(index => computeSimilarity(x(index), y(index),method, size)).sum/x.size
  }




  private def jacardSimilarity(x: NDArray, y: NDArray, size: Long): Double =
    if(checkShape(x, y, size)){
      val diff = x.sub(y).toFloatArray
      diff.filter(Math.abs(_) < 0.01).size.toDouble/diff.size
    }
    else
      throw new IllegalStateException(
        s"""Similarity failed for ${x.getShape().toString} and ${y.getShape().toString}
           |with ${mlopsConfiguration.getEmbeddingsSize} embeddings""".stripMargin
            .replace("\n", " ")
      )

  private def cosineSimilarity(x: NDArray, y: NDArray, size: Long): Double =
    if(x.shapeEquals(y))
      0.5*(cosine(x, y) + 1.0)
    else
      throw new IllegalStateException(
        s"""Similarity failed for ${x.getShape().toString} and ${y.getShape().toString}
           |with ${mlopsConfiguration.getEmbeddingsSize} embeddings""".stripMargin
            .replace("\n", " ")
      )

  private def checkShape(x: NDArray, y: NDArray, size: Long): Boolean =
    x.shapeEquals(y) && x.size() == size

  private def euclideanSimilarity(x: NDArray, y: NDArray, size: Long): Double =
    if(checkShape(x, y, size))
      1.0 - x.sub(y).norm().div(4.0F*size).toFloatArray.head
    else
      throw new IllegalStateException(
        s"""Similarity failed for ${x.getShape().toString} and ${y.getShape().toString}
           |with ${mlopsConfiguration.getEmbeddingsSize} embeddings""".stripMargin
            .replace("\n", " ")
      )
}
