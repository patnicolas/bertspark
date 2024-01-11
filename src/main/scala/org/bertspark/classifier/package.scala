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
package org.bertspark

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames
import org.bertspark.config.S3PathNames.s3SubModelsStructure
import org.bertspark.util.io.LocalFileUtil.CSV_SEPARATOR
import org.bertspark.util.io.S3Util
import org.slf4j.{Logger, LoggerFactory}


/**
  * {{{
  * Wrapper that specify the parameters for the classifier such as
  * - Number of classes/labels associated with a sub model
  * - Label oracle (One label for each sub model)
  * }}}
  * @author Patrick Nicolas
  * @version 0.6
  */
package object classifier {
  final private val logger: Logger = LoggerFactory.getLogger("classifier")

  /**
    * Ancillary parameters for the classifier
    */
  final object ClassifierParams {
    def save(subModelNumClasses: Seq[(String, Long)]): Unit = {
      val existingSubModelClasses =
        if(SubModelNumClasses.exists) SubModelNumClasses.load.toSeq
        else Seq.empty[(String, Long)]
      SubModelNumClasses.save(existingSubModelClasses ++ subModelNumClasses)
    }
  }

  /**
    * Keep track of number of labels or classes per sub-model. The number of classes is used
    * to configure the siae of the output layer (Softmax) of the neural classifier
    */
  final protected object SubModelNumClasses {
    import S3PathNames._
    private lazy val s3Path = s3ClassifierClassesPerSubModelsPath

    /**
      * Save the pair of subModelName - NumClasses on S3
      *
      * @param subModelNumClasses List of pair {subModelName, numClasses}
      */
    def save(subModelNumClasses: Seq[(String, Long)]): Unit = try {
      S3Util.upload(
        mlopsConfiguration.storageConfig.s3Bucket,
        s3Path,
        subModelNumClasses
            .filter(_._2 > 0)
            .map {
              case (subName, numClasses) =>
                s"${subName.replace(",", " ")},$numClasses"
            }.mkString("\n")
      )
    }
    catch {
      case e: IllegalStateException =>
        logger.error(s"SubModelNumClasses.save${e.getMessage}")
    }


    final def exists: Boolean = S3Util.exists(mlopsConfiguration.storageConfig.s3Bucket, s3Path)

    /**
      * Load the number of classes associated with
      *
      * @return Map {SubModel -> Number of classes
      */
    def load: Map[String, Long] = try {
      S3Util.download(
        mlopsConfiguration.storageConfig.s3Bucket,
        s3Path
      ).map(
        _.split("\n").map(
          line => {
            val ar = line.split(",")
            (ar.head, ar(1).toLong)
          }
        ).toMap
      ).getOrElse({
        logger.warn(s"Failed to load class distribution from S3 $s3Path")
        Map.empty[String, Long]
      })
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"SubModelNumClasses.load ${e.getMessage}")
        Map.empty[String, Long]
    }
  }
}
