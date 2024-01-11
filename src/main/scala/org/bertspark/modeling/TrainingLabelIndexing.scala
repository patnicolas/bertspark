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
  */
package org.bertspark.modeling

import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.{MlopsConfiguration, S3PathNames}
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.bertspark.util.io.S3Util
import org.slf4j.{Logger, LoggerFactory}


/**
  * *
  */
private[bertspark] final object TrainingLabelIndexing {
  import MlopsConfiguration._, S3PathNames._

  final private val logger: Logger = LoggerFactory.getLogger("TrainingLabelIndexing")


  def save(
    groupedSubModelsTrainingSetDS: Dataset[SubModelsTrainingSet]
  )(implicit sparkSession: SparkSession):  Seq[(String, Int)]  = {
    import sparkSession.implicits._

    val collectedLabels = groupedSubModelsTrainingSetDS
        .flatMap(_.labelIndices.map(_._1.replaceAll(",", " ")))
        .distinct()
        .collect
        .sortWith(_ < _)
    logDebug(logger, msg = s"Collect labels: ${collectedLabels.mkString(" ")}")
    saveLabelIndices(collectedLabels)

  }

  private def saveLabelIndices(labels: Array[String]): Seq[(String, Int)] = try {
    if(labels.nonEmpty) {
      val labelIndices = labels.zipWithIndex
      val labelIndicesStr = labelIndices.map{ case (k, v) => s"$k,$v"}.mkString("\n")
      S3Util.upload(s3LabelIndexMapPath, labelIndicesStr)
      labelIndices
    }
    else {
      logger.error(s"Cannot save empty training label index")
      Seq.empty[(String, Int)]
    }
  }
  catch {
    case e: IllegalArgumentException =>
      logger.error(s"TrainingLabelIndexing.save ${e.getMessage}")
      Seq.empty[(String, Int)]
  }


  def load: Map[String, Int] =
    try {
      S3Util.download(
        mlopsConfiguration.storageConfig.s3Bucket,
        s3LabelIndexMapPath
      ).map(
        content => {
          val lines = content.split("\n")
          lines.map(
            line => {
              val ar = line.split(",")
              if (ar.size != 2)
                throw new IllegalArgumentException(s"Failed to load label index map for ${s3LabelIndexMapPath}")
              (ar.head, ar(1).toInt)
            }
          )
        }
      ).map(_.toMap).getOrElse(
        throw new IllegalArgumentException(s"Failed to extractlabel index map for ${s3LabelIndexMapPath}")
      )
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(e.getMessage)
        Map.empty[String, Int]

      case e: Exception =>
        logger.error(e.getMessage)
        Map.empty[String, Int]
    }
}
