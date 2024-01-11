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
package org.bertspark.config

import org.bertspark.config.ExecutionMode.isPretraining
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration


/**
 * Singleton wrapper for all the local paths
 * @author Patrick Nicolas
 * @version 0.3
 */
private[bertspark] final object FsPathNames {
  // Internal relative values
  private lazy val target = mlopsConfiguration.target
  private lazy val preTrainConfig = mlopsConfiguration.preTrainConfig
  private lazy val preProcessConfig = mlopsConfiguration.preProcessConfig
  private def classifyConfig =  mlopsConfiguration.classifyConfig
  private def runId = mlopsConfiguration.runId

  // ---------------------  Static local paths ---------------------------------------
  lazy val distributionJsonPath: String = "output/distribution.json"

  def getDistributionCsvPath(key: String): String = s"output/distribution$key.csv"

  lazy val getVocabularyFreqFile: String = s"vocabulary${preProcessConfig.vocabularyType}/$target.freq"


  final def getVocabularyExtension: String = preProcessConfig.vocabularyType

  def getClassifyModelOutput: String = s"${classifyConfig.modelPrefix}-$runId"

  def getPreTrainModelOutput: String = s"${preTrainConfig.modelPrefix}-$runId"

  def getTrainModelOutput: String = s"${classifyConfig.modelPrefix}-$runId-${classifyConfig.modelId}"

  final def getModelName: String = if(isPretraining) getPreTrainModelOutput else getClassifyModelOutput

  /**
    * The unique transformer model is loaded from S3 ('preTrained') into a default 'xxxx-0000.params' model
    * @return Complete path in local file directory
    */
  final def getFsTransformerModelPath: String = {
    val modelFileId = s"${preTrainConfig.modelPrefix}-$runId"
    s"models/$target/$modelFileId"
  }

  final def getFsTransformerModelFile: String = {
    val modelFileId = s"${preTrainConfig.modelPrefix}-$runId"
    s"$getFsTransformerModelPath/${modelFileId}-0000.params"
  }

  final def getFsClassifierModelPath(subClassificationModelName: String): String = {
    val fsRootPath = s"models/$target/${classifyConfig.modelPrefix}-$runId"
    s"$fsRootPath/$subClassificationModelName"
  }

  final def getFsClassifierModelFile(subClassificationModelName: String): String =
    s"${getFsClassifierModelPath(subClassificationModelName)}/$subClassificationModelName-0000.params"
}

