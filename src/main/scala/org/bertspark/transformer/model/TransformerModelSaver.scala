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
package org.bertspark.transformer.model

import ai.djl.Model
import java.nio.file.Paths
import org.bertspark.config._
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.delay
import org.bertspark.transformer.training.TransformerTrainingListener
import org.bertspark.util.io._
import org.slf4j._


/**
  * Save the various component of the transformer model into S3
  * {{{
  *   - Model
  *   - Metrics
  *   - Configuration
  * }}}
  * @author Patrick Nicolas
  * @version 0.6
  */
trait TransformerModelSaver extends ModelSaver {
self: TransformerTrainingListener =>
  import TransformerModelSaver._

  /**
    * Save model for a given epoch number and an existing list of metrics
    * @param model Reference to the current classification model, loaded in the trainer
    * @param epochNo Number of epochs
    * @param metrics List of metric names
    */
  override def save(model: Model, epochNo: Int, metrics: List[String]): Unit = try {
    saveMetrics(model.getName(), metrics, subModelName)
    saveModel(model)
    saveConfiguration
  } catch {
    case e: IllegalArgumentException => logger.error(s"Save transformer: ${e.getMessage}")
  }


  // -------------------------------    Supporting methods --------------------

  private def saveModel(model: Model, subModel: String = ""): Unit = {
    import S3FsPathMapping._

    // Save the transformer model to the local file
    val (s3ModelPath, fsModelDir, _) = paths(transformerModelLbl, "")
    model.save(Paths.get(fsModelDir), model.getName())

    // Save the associated configuration to the local file
    LocalFileUtil.Save.local(fsFileName = s"$fsModelDir/mlopsConfiguration.txt", mlopsConfiguration.toString)
    delay(timeInMillis = 2000L)
    val epochCount = getCurrentEpoch(Paths.get(fsModelDir), model.getName)

    try {
      val localFile = s"${mlopsConfiguration.preTrainConfig.modelPrefix}-${mlopsConfiguration.runId}-$epochCount.params"
      S3Util.fsToS3(fsSrcFilename = s"$fsModelDir/$localFile", s3ModelPath)
    }
    catch {
      case e: IllegalStateException =>
        logger.error(s"Failed to transfer $fsModelDir to $s3ModelPath, ${e.getMessage}")
      case e: Exception =>
        logger.error(s"Failed to transfer $fsModelDir to $s3ModelPath, ${e.getMessage}")
    }
  }

  private def saveMetrics(modelName: String, metricsData: List[String], subModel: String = ""): Unit = {
    // First display on the standard output
    val metricInfo = s"$modelName,*\n${metricsData.mkString("\n")}"
    logDebug(logger,  msg = s"\n---------------------------\n$metricInfo\n----------------------------")

    val (s3Folder, content) = getS3PathAndContent(modelName, metricInfo)
    if(content.nonEmpty) {
      // Save the metrics to S3
      S3Util.upload(mlopsConfiguration.storageConfig.s3Bucket, s3Folder, content)
      logDebug(logger, msg = s"Metric saved on S3 in $s3Folder")
    }
  }

  private def getS3PathAndContent(modelName: String, metricInfo: String): (String, String) = {
    val s3Folder = S3PathNames.getS3MetricsPath(modelName)
    val content =
      if(s3Folder.nonEmpty) s"${mlopsConfiguration.runId}\n${mlopsConfiguration.toString}\n$metricInfo"
      else ""
    (s3Folder, content)
  }
}


private[bertspark] final object TransformerModelSaver {
  final private val logger: Logger = LoggerFactory.getLogger("TransformerSaver")

  def saveConfiguration: Unit = MlopsConfiguration.saveConfiguration(true)
}
