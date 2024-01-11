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
package org.bertspark.classifier.model

import ai.djl.Model
import java.nio.file.Paths
import org.bertspark.util.io._
import org.bertspark.classifier.training.ClassifierTrainingListener
import org.bertspark.config._
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames.getS3MetricsPath
import org.bertspark.modeling.TrainingContext
import org.slf4j._


/**
  * Saving method for classification model. This trait is to support the related Training listener
  * {{{
  * The following components are stored on S3
  *   - Model
  *   - Metrics
  *   - Configuration
  * }}}
  * @author Patrick Nicolas
  * @version 0.6
  */
trait ClassifierModelSaver extends ModelSaver {
self: ClassifierTrainingListener =>
  import ClassifierModelSaver._

  protected[this] val trainingContext: TrainingContext


  /**
    * Save model for a given epoch number and an existing list of metrics
    * @param model Reference to the current classification model, loaded in the trainer
    * @param epochNo Number of epochs
    * @param metrics List of metric names
    */
  override def save(model: Model, epochNo: Int, metrics: List[String]): Unit = try {
    saveMetrics(subModelName, epochNo, metrics)
    saveModel(model, subModelName)
    saveConfiguration
  } catch {
    case e: IllegalArgumentException => logger.error(s"Save classifier ${e.getMessage}")
  }

  // ------------------  Supporting methods ---------------------------

  private def saveModel(subModel: Model, subClassifierModelName: String): Unit = {
    import S3FsPathMapping._

    // DEBUG
    logDebug(logger, s"Save subModel: $subModel classifier sub model: $subClassifierModelName")
    // END DEBUG
    val subModelName = subClassifierModelName.replace(",", " ")
    val (s3ModelPath, fsModelDir, _) = paths(classifierModelLbl, subModelName)
    val modelOutputPath = Paths.get(fsModelDir)

    // By default DJL stores the data into a local file, which needs to be transferred to S3
    // in Trained-bert-runId local directory
    subModel.save(modelOutputPath, subModelName)
    val absolutePath = modelOutputPath.toAbsolutePath.toString

    // We need to pick up the last model in the local directory (on AWS) containing the classification model.
    val epochCount = getCurrentEpoch(Paths.get(absolutePath), subModelName)
    val fsClassificationPath = s"$absolutePath/${subModelName}-${epochCount}.params"

    // Transfer from local file to S3 for persistence
    S3Util.fsToS3(fsClassificationPath, mlopsConfiguration.storageConfig.s3Bucket, s3Path = s3ModelPath)
  }


  private def saveMetrics(subModelName: String, epochNo: Int, metricsData: List[String]): Unit = {
    trainingContext.savePredictionLabels(epochNo)

    // First display on the standard output
    val metricInfo = s"$subModelName,*\n${metricsData.mkString("\n")}"
    logDebug(logger, s"\n---------------------------\n$metricInfo\n----------------------------")

    val (s3Folder, content) = getS3PathAndContent(metricInfo)
    if (s3Folder.nonEmpty && content.nonEmpty) {
      // Save the metrics to S3
      try {
        S3Util.upload(mlopsConfiguration.storageConfig.s3Bucket, s3Folder, content)
        logDebug(logger, s"Metric saved on S3 in $s3Folder")
      }
      catch {
        case e: IllegalArgumentException => logger.error(s"Save metrics: ${e.getMessage}")
      }
    }
  }

  private def getS3PathAndContent(metricsInfo: String): (String, String) = {
    val s3Folder =
      if (subModelName.nonEmpty) s"${getS3MetricsPath(subModelName)}-$subModelName"
      else getS3MetricsPath(subModelName)

    val content = s"${mlopsConfiguration.getParameters.mkString("\n")}\n$metricsInfo"
    (s3Folder, content)
  }
}


private[bertspark] final object ClassifierModelSaver {
  final private val logger: Logger = LoggerFactory.getLogger("ClassifierSaver")

  def saveConfiguration: Unit = MlopsConfiguration.saveConfiguration(false)
}