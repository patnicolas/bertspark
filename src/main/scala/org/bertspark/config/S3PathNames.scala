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

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration


/**
 * Wrapper for all the function computing the path to various S3 folders
 *
 * @author Patrick Nicolas
 * @version 0.5
 */
final object S3PathNames {
  // Internal relative values
  private lazy val storageConfig = mlopsConfiguration.storageConfig
  private lazy val target = mlopsConfiguration.target
  private lazy val preTrainConfig = mlopsConfiguration.preTrainConfig
  private def runId = mlopsConfiguration.runId
  private lazy val preProcessConfig = mlopsConfiguration.preProcessConfig
  private def classifyConfig =  mlopsConfiguration.classifyConfig

  // -------------------------   Static AWS-S3 paths ------------------------------------------

  lazy val s3CodeDescriptorFile = s"${storageConfig.s3RootFolder}/$target/codeDescriptors.csv"

  lazy val s3TokensEvaluationPath = s"${storageConfig.s3RootFolder}/$target/tokens/${preTrainConfig.sentenceBuilder}"

  lazy val s3TransformerModelPath = s"${storageConfig.s3RootFolder}/$target/models/$runId"

  lazy val s3ClassifierModelPath = s"$s3TransformerModelPath/${classifyConfig.modelId}"

  lazy val s3FeedbacksPath: String = s"${storageConfig.s3FeedbackFolder}/$target"

  lazy val s3RequestsPath: String = s"${storageConfig.s3RequestFolder}/$target"

  lazy val s3SubModelsStructure: String = getS3SubModelsStructure(target, runId)

  lazy val s3SubModelTaxonomy: String =
    s"${storageConfig.s3RootFolder}/$target/models/$runId/${classifyConfig.modelId}/modelTaxonomy.csv"

  def getS3SubModelsStructure(newTarget: String, transformerModel: String): String =
    s"${storageConfig.s3RootFolder}/$newTarget/models/$transformerModel/subModels.csv"

  def getS3SubModelsStructure(
    newTarget: String,
    transformerModel: String,
    customer: String,
    minNumRecordsPerLabel: Int): String =
    if(customer.nonEmpty)
      s"${storageConfig.s3RootFolder}/$newTarget/models/$transformerModel/subModels-$minNumRecordsPerLabel-$customer.csv"
    else
      s"${storageConfig.s3RootFolder}/$newTarget/models/$transformerModel/subModels-$minNumRecordsPerLabel.csv"



  def s3RequestPath(newTarget: String): String = s"${storageConfig.s3RequestFolder}/$newTarget"


  // ------------------------------  Dynamic AWS-S3 paths -------------------------------

  def getS3SimilarityOutput: String = s"${storageConfig.s3RootFolder}/$target/similarity/$runId"

  final def getS3ComparePath(subModelName: String, epochNo: Int): String =
    s"${s3ClassifierTrainingComparePath}/${classifyConfig.modelId}/$subModelName/epoch-$epochNo"

  def getS3FolderCompareSource =
    s"${storageConfig.s3RootFolder}/$target/compare/$runId/${classifyConfig.modelId}"

  def getS3EvaluationPath: String = s"${storageConfig.s3RootFolder}/$target/evaluation/$runId"

  def getS3AccuracyReportPath: String =
    s"${storageConfig.s3RootFolder}/$target/accuracy/${runId}/${classifyConfig.modelId}"

  def getS3TransformerModelPath: String =
    s"${storageConfig.s3RootFolder}/$target/models/$runId/${preTrainConfig.modelPrefix}-0000.params"

  lazy val s3PredictedClaimPath: String =
    s"${storageConfig.s3RootFolder}/$target/claims/$runId/${classifyConfig.modelId}"


        // --------------------------
        // Classification model path
        // ----------------------------

  def getS3ClassifierPath(subClassificationModelName: String) = {
    val s3RootFolder = s"${storageConfig.s3RootFolder}/$target/models"
    s"$s3RootFolder/$runId/${classifyConfig.modelId}/$subClassificationModelName/$subClassificationModelName-0000.params"
  }

  def getS3ClassificationPath(subClassificationModelName: String, transformerModelName: String, count: String) = {
    val modelRelativePath = ExecutionMode.convertModelName(transformerModelName)
    s"$getS3TransformerModelPath/${classifyConfig.modelId}/$subClassificationModelName/$modelRelativePath-$count.params"
  }


        // ----------
        // Metrics
        // ----------

  lazy val s3PreTrainingMetricPath: String =
    s"${storageConfig.s3RootFolder}/$target/metrics/$runId/pretraining"

  lazy val s3ClassifierTrainingMetricPath: String =
    s"${storageConfig.s3RootFolder}/$target/metrics/$runId/${classifyConfig.modelId}/training"

  lazy val s3PredictionMetricPath: String =
    s"${storageConfig.s3RootFolder}/$target/metrics/$runId/${classifyConfig.modelId}/prediction"

  lazy val s3ClassifierTrainingComparePath: String = s"${storageConfig.s3RootFolder}/$target/compare/$runId"

  def getS3MetricsPath(modelDescriptor: String): String =
    if(ExecutionMode.isPretraining) s3PreTrainingMetricPath
    else if(ExecutionMode.isClassifier) s3ClassifierTrainingMetricPath
    else if(ExecutionMode.isHpo) s"${s3ClassifierTrainingMetricPath}-$modelDescriptor"
    else ""


        // ---------------------------
        // Classifier parameters
        // ---------------------------

  lazy val s3ClassifierClassesPerSubModelsPath: String =
    s"${storageConfig.s3RootFolder}/$target/models/$runId/${classifyConfig.modelId}/numClasses.csv"

  def getS3ClassifierEmbeddingsPath: String = s"${storageConfig.s3RootFolder}/$target/models/$runId/embeddings"

  lazy val s3LabelIndexMapPath: String = s"${storageConfig.s3RootFolder}/$target/models/$runId/labelIndexMap.csv"

  def s3EvaluationSetPath = s"evaluationSet/${mlopsConfiguration.target}/${mlopsConfiguration.classifyConfig.modelId}"

        // -----------------------------
        // Vocabulary S3 folders
        // --------------------------

  lazy val s3VocabularyPath: String =
    s"${storageConfig.s3RootFolder}/$target/vocabulary/${preProcessConfig.vocabularyType}"

  def getVocabularyS3Path(vocabularyType: String): String =
    s"${storageConfig.s3RootFolder}/$target/vocabulary/$vocabularyType"


        // -----------------------------
        // Training set S3 folders
        // -------------------------------

  /**
    * Get the classifier training set path
    * @return Path of the training data for the classifier
    */
  lazy val s3ModelTrainingPath: String =
    s"${storageConfig.s3RootFolder}/$target/training/${preProcessConfig.vocabularyType}"

  def getS3ModelTrainingPath(vocabularyType: String): String =
    s"${storageConfig.s3RootFolder}/$target/training/$vocabularyType"

  def getS3ModelTrainingFilterPath: String = s"${storageConfig.s3RootFolder}/$target/training/filter.csv"



        // -----------------------------
        // Contextual Document S3 folders
        // -------------------------------
  /**
   * Constructs the S3 path for the contextual document
   * @return S3 path for the contextual document
   */
  lazy val s3ContextualDocumentPath: String =
    s"${storageConfig.s3RootFolder}/$target/${storageConfig.s3ContextDocumentFolder}/${preProcessConfig.vocabularyType}"

  def getS3ContextualDocumentPath(vocabularyType: String): String =
    s"${storageConfig.s3RootFolder}/$target/${storageConfig.s3ContextDocumentFolder}/$vocabularyType"

  def getS3ContextualDocumentGroupPath(vocabularyType: String): String =
    s"${storageConfig.s3RootFolder}/$target/${storageConfig.s3ContextDocumentFolder}/cluster$vocabularyType"

  lazy val s3ContextualDocumentGroupPath: String = getS3ContextualDocumentGroupPath(preProcessConfig.vocabularyType)
}

