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

import java.util.concurrent.atomic.AtomicInteger
import org.apache.spark.sql.SparkSession
import org.bertspark.classifier.training.TClassifier
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.delay
import org.bertspark.hpo.ClassifierTrainHPOConfig.logger
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.{ClassifyConfig, DynamicConfiguration, PreTrainConfig}
import org.bertspark.hpo.ga.GASearch.{HPOBaseParam, HPOParamArrayInt, HPOParamFloat, HPOParamInt, HPOParamString}
import org.bertspark.hpo.ga.{HPOChromosome, HPOParameters}
import org.slf4j.{Logger, LoggerFactory}


/**
 * {{{
 * The hyper-parameters are
 *  - Learning rate
 *  - FFNN layers layout
 *  - Maximum number of records per sub-model
 *  - Convergence ratio
 *  - Document embedding mode
 *  - Segment embedding mode
 * }}}
 *
 * @param predictors List of predictors
 * @param clsAggregations List of aggregation methods
 * @param targetSampleSizes Size of target samples
 * @param convergenceLossRatios Various convergence rate for loss function
 * @param learningRates Various learning rate
 * @param numSubModels Number of sub-models
 * @param hpoSearchStrategy Search strategy: grid, random,...
 *
 * @author Patrick Nicolas
 * @version 0.4
 */
final class ClassifierTrainHPOConfig(
  predictors: Array[String],
  clsAggregations: Array[String],
  dlLayout: Array[Array[Int]],
  targetSampleSizes: Array[Int],
  convergenceLossRatios: Array[Float],
  learningRates: Array[Float],
  numSubModels: Int,
  hpoSearchStrategy: String
)(implicit sparkSession: SparkSession) {
  import DynamicConfiguration._

  private[this] val mlopsClassification = TClassifier()
  private[this] val counter = new AtomicInteger(0)

  private[this] val params = Array[Array[Int]](
    (0 until predictors.size).toArray,
    (0 until clsAggregations.size).toArray,
    (0 until dlLayout.size).toArray,
    (0 until targetSampleSizes.size).toArray,
    (0 until convergenceLossRatios.size).toArray,
    (0 until learningRates.size).toArray
  )

  @inline
  final def getNumSubModels: Int = numSubModels

  /**
   * Select the appropriate search strategy  (Grid, Random,...)
   */
  private[this] val hpoSearch = new HPORandomSearch(params)


  final def getHPOParamsList: List[HPOBaseParam] =
    List[HPOBaseParam](
      new HPOParamString("predictor", predictors),
      new HPOParamString("clsAggregation", clsAggregations),
      new HPOParamArrayInt("dlLayout", dlLayout),
      new HPOParamInt("targetSampleSize", targetSampleSizes),
      new HPOParamFloat("convergenceLossRatio", convergenceLossRatios),
      new HPOParamFloat("baseLr",  learningRates)
    )

  final def getHPOParameters: HPOParameters = new HPOParameters(getHPOParamsList)


  final def getHPOChromosome: HPOChromosome = new HPOChromosome(numSubModels, getHPOParameters)

  @inline
  final def getParams: Array[Array[Int]] = params

  /**
   * Update the dynamic configuration
   * @param paramIndex Index of the parameter
   * @param paramValue Value index of the parameter
   * @param experimentCnt Experiment counter
   */
  def getNextClassifierConfiguration(paramIndex: Int, paramValue: Int, experimentCnt: Int): Unit =
    paramIndex match {
      case 0 =>
        logDebug(logger, s"$experimentCnt Classify new HPO param: predictor=${predictors(paramValue)}")
        apply[PreTrainConfig](mlopsConfiguration.preTrainConfig.copy(predictor = predictors(paramValue)))
      case 1 =>
        logDebug(logger, s"$experimentCnt Classify new HPO param: clsAggregation=${clsAggregations(paramValue)}")
        apply[PreTrainConfig](mlopsConfiguration.preTrainConfig.copy(clsAggregation = clsAggregations(paramValue)))
      case 2 =>
        logDebug(logger, s"$experimentCnt Classify new HPO param: dlLayout=${dlLayout(paramValue).mkString(",")}")
        apply[ClassifyConfig](mlopsConfiguration.classifyConfig.copy(dlLayout = dlLayout(paramValue)))
      case 3 =>
        logDebug(logger, s"$experimentCnt Classify new HPO param: targetSampleSize=${targetSampleSizes(paramValue)}")
        apply[ClassifyConfig](mlopsConfiguration.classifyConfig.copy(maxNumRecordsPerLabel = targetSampleSizes(paramValue)))
      case 4 =>
        val newConvergenceLossRatio = convergenceLossRatios(paramValue)
        logDebug(logger, s"$experimentCnt Classify new HPO param: convergenceLossRatio=$newConvergenceLossRatio")
        val newOptimizer = mlopsConfiguration.classifyConfig.optimizer.copy(convergenceLossRatio = newConvergenceLossRatio)
        apply[ClassifyConfig](mlopsConfiguration.classifyConfig.copy(optimizer = newOptimizer))
      case 5 =>
        val newLearningRate = learningRates(paramValue)
        logDebug(logger, s"$experimentCnt Classify new HPO param: learningRate=$newLearningRate")
        val newOptimizer = mlopsConfiguration.classifyConfig.optimizer.copy(baseLr = newLearningRate)
        apply[ClassifyConfig](mlopsConfiguration.classifyConfig.copy(optimizer = newOptimizer))
      case _ =>
        logger.error(s"Params index $paramIndex is out of range")
    }


  /**
   * Select the next set of parameters using the select search mode
   * @return true if a next set of parameters is available, false otherwise
   */
  def next: Float = {
    import ClassifierTrainHPOConfig._

    // If there is another set of parameters to evaluate....
    val (paramIndex, paramValueIndex) = hpoSearch()
    val experimentCount = counter.incrementAndGet()

    getNextClassifierConfiguration(paramIndex, paramValueIndex, experimentCount)
    logger.debug(s"Ready to train for $experimentCount experiment with $paramIndex and $paramValueIndex")
    val accuracy = mlopsClassification()
    delay(1000L)
    accuracy
  }
}



private[bertspark] final object ClassifierTrainHPOConfig {
  final private val logger: Logger = LoggerFactory.getLogger("ClassifierTrainHPOConfig")


  final val allSubModelNames = Set[String]("ALL")

  final object LearningRates {
    def apply(lr: Float): Int = (lr*1e+5F).floor.toInt
    def apply(lr: Int): Float = lr*1e-5F
  }

  final object ConvergenceRatios {
    def apply(convergenceRatio: Float): Int =
      (convergenceRatio*100).floor.toInt
    def apply(convergenceRatio: Int): Float =
      convergenceRatio*0.01F
  }
}
