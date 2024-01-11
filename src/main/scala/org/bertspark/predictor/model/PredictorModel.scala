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
package org.bertspark.predictor.model

import ai.djl.inference.Predictor
import ai.djl.ndarray._
import ai.djl.repository.zoo._
import java.io.File
import org.apache.spark.sql._
import org.bertspark.classifier.block.ClassificationBlock
import org.bertspark.classifier.model.ClassifierModelLoader
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.nlp.trainingset.TokenizedTrainingSet
import org.bertspark.predictor._
import org.bertspark.transformer.representation.PretrainingInference
import org.bertspark.util.io.S3Util
import org.slf4j._


/**
  * Predictor for the BERT classification model associated to a sub-model
  * {{{
  *   There are two component for the classifier
  *   - Pre-training predictor which predict the embedding vector associated to [CLS] token
  *         which model is trained using BERTPreTrainingModel
  *   - Classification block (MLP, CNN,....) which is loaded from local file which model is
  *         trained with BERTClassificationModel
  *   The models are loaded from specific local folders
  * }}}
  * @param pretrainingPredictor BERT pre-training predictor
  * @param classificationBlock Neural block for classification
  * @param transformerModelName Name or id of the transformer model
  * @param classificationModel Name, id of the classifier
  * @param subModelName Name of this sub-model
  * @param sparkSession Implicit reference to the current Spark context
  *
  * @author Patrick Nicolas
  * @version 0.4
  */
private[bertspark] final class PredictorModel private (
  pretrainingPredictor: PretrainingInference,
  override val classificationBlock: ClassificationBlock,
  override val transformerModelName: String,
  override val classificationModel: String,
  override val subModelName: String,
)(implicit sparkSession: SparkSession) extends ClassifierModelLoader {
  import PredictorModel._

  private[this] val subModelPredictor: Option[Predictor[NDList, NDList]] = model.map(_.newPredictor())
  private lazy val predictionSource =
    subModelPredictor.map(new PredictionFromSource(pretrainingPredictor, _, subModelName))

  @inline
  final def getClassificationSubModel: Option[ZooModel[NDList, NDList]] = model

  /**
    * Predict the classes/labels for a batch of data
    * @param ndManager Sub manager for ND arrays
    * @param inputRequestDS Dataset (Tokenized index or prediction request) training data set
    * @return Pair {Document id -> Prediction }
    */
  def predict(
    ndManager: NDManager,
    inputRequestDS: Dataset[TokenizedTrainingSet]): List[DocIdPrediction] =
    predictionSource.map(_.predict(ndManager, inputRequestDS)).getOrElse({
      logger.error(s"Prediction source is undefined")
      List.empty[DocIdPrediction]
    })
}


/**
  * Singleton for various constructor and
  */
private[bertspark] final object PredictorModel {
  final private val logger: Logger = LoggerFactory.getLogger("PredictorModel")

  /**
    * Default constructor
    * @param pretrainedPredictor Pre-training predictor
    * @param classificationBlock Classification Neural block
    * @param preTrainingModel Name/Id of the pre-training/transformer model
    * @param classificationModel Name/Id of the classification model
    * @param subModelName Sub model name
    * @param sparkSession Implicit reference to the current Spark context
    * @return Instance of TPredictorModel
    */
  def apply(
    pretrainedPredictor: PretrainingInference,
    classificationBlock: ClassificationBlock,
    preTrainingModel: String,
    classificationModel: String,
    subModelName: String
  )(implicit sparkSession: SparkSession): PredictorModel =
    new PredictorModel(pretrainedPredictor, classificationBlock, preTrainingModel, classificationModel, subModelName)


  /**
    * Constructor using configuration file parameters for the preTraining model and classification model
    * @param pretrainedPredictor Pre-training predictor
    * @param classificationBlock Classification Neural block
    * @param subModelName Sub model name
    * @param sparkSession Implicit reference to the current Spark context
    * @return Instance of TPredictorModel
    */
  def apply(
    pretrainedPredictor: PretrainingInference,
    classificationBlock: ClassificationBlock,
    subModelName: String
  )(implicit sparkSession: SparkSession): PredictorModel =
    new PredictorModel(
      pretrainedPredictor,
      classificationBlock,
      mlopsConfiguration.runId,
      mlopsConfiguration.classifyConfig.modelId,
      subModelName)


  /**
    * Load model from S3 with relevant pre-training model (runId) and classification model (classifyConfig.modelI
    * then copy to the local directory if it does not exist
    * @param preTrainingModel Name or id of the pre-training model
    * @param classificationModel Name or is of the classification model
    * @param subModelName Name of sub-model
    * @return Path of the local model file name
    */
  def localModelURLFromS3(preTrainingModel: String, classificationModel: String, subModelName: String): String = {
    val s3RootFolder = s"${mlopsConfiguration.storageConfig.s3RootFolder}/${mlopsConfiguration.target}/models"
    val s3ModelUrl = s"$s3RootFolder/$preTrainingModel/$classificationModel/$subModelName"
    val pattern = "0000.params"
    val latestModelKey = S3Util.getMaxKey(
      mlopsConfiguration.storageConfig.s3Bucket,
      s3ModelUrl,
      pattern,
      (s: String) => s.substring(0, 4).toInt)

    val modelName = s"Trained-Bert-$preTrainingModel-${latestModelKey.substring(latestModelKey.length - pattern.length)}"
    val fsModelDir = s"./models/Trained-Bert-$preTrainingModel/${mlopsConfiguration.target}/$subModelName"
    val fsModelUrl = s"$fsModelDir/$modelName"

    val dir = new File(fsModelDir)
    if(!(dir.exists() && dir.isDirectory)) {
      dir.mkdirs()
      S3Util.s3ToFs(fsModelUrl, latestModelKey, false)
    }
    fsModelUrl
  }
}

