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

import org.apache.spark.sql.SparkSession
import org.bertspark._
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.hpo.ClassifierTrainHPO.logger
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.{DynamicConfiguration, ExecutionMode}
import org.bertspark.util.io.LocalFileUtil
import org.slf4j.{Logger, LoggerFactory}


/**
 * Hyper-parameters tuning for training the classifier....
 * {{{
 * The hyper-parameters are
 *  - Num epoch
 *  - Learning rate
 *  - FFNN layers layout
 *  - Maximum number of records per sub-model
 *  - Convergence ratio
 *  - Document embedding mode
 *  - Segment embedding mode
 * }}}
 * @param classifierTrainHPOConfig Current configuration for training the classifier
 * @param numCycles Number of execution or search cycle
 *
 * @author Patrick Nicolas
 * @version 0.4
 */
private[bertspark] final class ClassifierTrainHPO private (
  classifierTrainHPOConfig: ClassifierTrainHPOConfig,
  numCycles: Int) {
  import org.bertspark.config.DynamicConfiguration._

  @throws(clazz = classOf[HPOException])
  def execute: Unit = {
    ExecutionMode.setHpo
    (0 until numCycles).foreach(
      index => {
        logDebug(
          logger,
          s"Cycle# $index for preTrainer id=${mlopsConfiguration.runId} and classifier id=${mlopsConfiguration.classifyConfig.modelId}"
        )

        val accuracy = classifierTrainHPOConfig.next
        if(accuracy < 0.0F)
          logger.error(s"*** Completed HPO runs with accuracy $accuracy")
        ++(false)
      }
    )
  }
}


/**
 * Singleton for constructors
 */
private[bertspark] final object ClassifierTrainHPO {
  final private val logger: Logger = LoggerFactory.getLogger("ClassifierTrainHPO")
  final private val hpoConfigFile = "conf/HPOConfig.json"

  case class ClassifierTrainConfigParams(
    runId: String,
    prefixModelId: Int,
    numCycles: Int,
    populationSize: Int,
    numSubModels: Int,
    predictors: Array[String],
    clsAggregations: Array[String],
    dlLayout: Array[Array[Int]],
    targetSampleSizes: Array[Int],
    convergenceLossRatios: Array[Float],
    learningRates: Array[Float],
    hpoSearchStrategy: String
  ) {
    def getClassifierTrainHPOConfig(implicit sparkSession: SparkSession): ClassifierTrainHPOConfig =
      new ClassifierTrainHPOConfig(
        predictors,
        clsAggregations,
        dlLayout,
        targetSampleSizes,
        convergenceLossRatios,
        learningRates,
        numSubModels,
        hpoSearchStrategy
      )
  }


  def apply(classifierTrainHPOConfig: ClassifierTrainHPOConfig,  numExecutions: Int): ClassifierTrainHPO =
    new ClassifierTrainHPO(classifierTrainHPOConfig, numExecutions)

  final val parameterNames = Set[String](
    "Sentences builder",
    "Predictor",
    "CLS aggregation",
    "FFNN layout",
    "Target sample size",
    "Num. epochs",
    "Convergence loss rate",
    "Base learning rate",
    "train_epoch_Accuracy",
    "train_epoch_loss",
    "validate_epoch_Accuracy",
    "validate_epoch_loss"
  )


  def apply(args: Seq[String])(implicit sparkSession: SparkSession): Option[ClassifierTrainHPO] = {
    require(args.size == 2, "Arguments for classifier training HTO should be 'classifyHPO configurationFile'")
    this.apply(args(1))
  }

  def apply(hpoConfigurationFile: String)(
    implicit sparkSession: SparkSession
  ): Option[ClassifierTrainHPO] = {
    loadClassifierHPOConfiguration(hpoConfigurationFile).map(
      classifierTrainParams => {
        DynamicConfiguration(classifierTrainParams.runId, classifierTrainParams.prefixModelId.toString)
        new ClassifierTrainHPO(classifierTrainParams.getClassifierTrainHPOConfig, classifierTrainParams.numCycles)
      }
    )
  }

  private def loadClassifierHPOConfiguration(fileName: String): Option[ClassifierTrainConfigParams] =
    LocalFileUtil.Load.local(fileName).map(
      LocalFileUtil.Json.mapper.readValue(_, classOf[ClassifierTrainConfigParams])
    )

  def loadClassifierHPOConfiguration: Option[ClassifierTrainConfigParams] = loadClassifierHPOConfiguration(hpoConfigFile)


  def loadHPOConfiguration(fileName: String)(implicit sparkSession: SparkSession): Option[ClassifierTrainHPOConfig] =
    LocalFileUtil.Load.local(fileName).map(
      LocalFileUtil.Json.mapper.readValue(_, classOf[ClassifierTrainConfigParams])
    ).map(_.getClassifierTrainHPOConfig)

  def loadHPOConfiguration(implicit sparkSession: SparkSession): Option[ClassifierTrainHPOConfig] =
    loadHPOConfiguration(hpoConfigFile)
}
