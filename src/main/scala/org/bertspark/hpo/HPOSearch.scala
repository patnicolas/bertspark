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
package org.bertspark.hpo

import org.slf4j._
import scala.collection.mutable.{HashSet, ListBuffer}


/**
 *  Generic signature for the HPO search strategy
 *  {{{
 *    paramValues  List of parameters values
 *    nextParamIndex Index of the next parameter
 *    nextParamValue Value of the next parameter
 *  }}}
 *
 *  @uthor Patrick Nicolas
 *  @version 0.4
 */
trait HPOSearch {
  protected[this] var nextParamIndex = 0
  protected[this] var nextParamValue = 0
  protected[this] val paramValues: Array[Array[Int]]

  final def getNextParamsIndex: Int = nextParamIndex
  final def getNextParamIndex: Int = nextParamValue
  final def getNextParamValue: Int = paramValues(nextParamIndex)(nextParamValue)
  def apply(): Boolean
}


/**
 * Wrapper for the random search
 * @param paramValueIndices
 */
final class HPORandomSearch(paramValueIndices: Array[Array[Int]])  {
  private[this] val rand = new scala.util.Random(9013L)
  private[this] val paramsSizes = paramValueIndices.map(_.size)
  private[this] val visited = HashSet[Int]()

  private def encode(nextParamIndex: Int, nextParamValueIndex: Int): Int =
    (nextParamIndex<<4) + nextParamValueIndex

  private[this] val internalRepresentation: Array[Int] = {
    val totalSize = paramsSizes.sum
    Range(0, totalSize).toArray
  }

  def apply(): (Int, Int)  = {
    val randomIndex = rand.nextInt(internalRepresentation.size)
    decode(randomIndex)
  }


  def decode(selectedIndex: Int): (Int, Int) = {
    var cumulative = 0
    var nextParamIndex = 0
    var nextParamValueIndex = -1
    do {
      cumulative += paramsSizes(nextParamIndex)
      if(cumulative <= selectedIndex) {
        nextParamIndex += 1
      }
      else {
        val relativeIndex = cumulative - paramsSizes(nextParamIndex)
        nextParamValueIndex = selectedIndex - relativeIndex
      }
    } while(nextParamValueIndex < 0)

    // Test if the parameters have been already generated (visited)
    // If so select the first value for any random parameter..
    val (finalNextParamIndex, finalNextParamValueIndex) =
      if(visited.contains(encode(nextParamIndex, nextParamValueIndex)))
        (rand.nextInt(paramValueIndices.size), 0)
      else
       (nextParamIndex, nextParamValueIndex)

    // Update the list of parameters configuration already visited.
    visited.add(encode(finalNextParamIndex, finalNextParamValueIndex))
    (finalNextParamIndex, finalNextParamValueIndex)
  }
}

final object HPORandomSearch {
  final private val logger: Logger = LoggerFactory.getLogger("HPORandomSearch")

  def getRepresentation(paramValueIndices: Array[Array[Int]]): Array[Int] = {
    val paramsSizes = paramValueIndices.map(_.size)
    val totalSize = paramsSizes.sum
    Range(0, totalSize).toArray
  }

  def getRepresentation(classifierTrainHPOConfig: ClassifierTrainHPOConfig): Array[Int] =
    getRepresentation(classifierTrainHPOConfig.getParams)
}


/**
 *
 * @param paramValues
 */
final class HPOGridSearch(override val paramValues: Array[Array[Int]]) extends HPOSearch {

  override def apply(): Boolean = {
    val nextParams: Array[Int] = paramValues(nextParamIndex)
    if (nextParamValue < nextParams.size-1) {
      nextParamValue += 1
      true
    } else if(nextParamIndex < paramValues.size -1){
      nextParamIndex += 1
      nextParamValue = 0
      true
    }
    else
      false
  }
}

final object HPOGridSearch {


  def changeOneParam(
    rawIndices: Array[Int],
    paramIndex: Int,
    collector: ListBuffer[Array[Int]]): Unit = {

    if(paramIndex >=  0) {
      val numParamValues = rawIndices(paramIndex)
      val prefix = collector.last.take(paramIndex)

      if (paramIndex == rawIndices.size - 1)
        (0 until numParamValues).foldLeft(collector)(
          (xs, index) => {
            val newIndices = prefix ++ Array[Int](index)
            collector += newIndices
          }
        )
      changeOneParam(rawIndices, paramIndex - 1, collector)
    }
  }

}