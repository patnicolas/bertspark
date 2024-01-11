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

import org.bertspark.config.ExecutionMode
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.hpo.ClassifierTrainHPO.parameterNames
import org.bertspark.hpo.HPOAnalyzer.ClassifierHPOAnalyzer
import org.bertspark.nlp.tokenSeparator
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.slf4j.{Logger, LoggerFactory}


/**
 * Analysis of output of classifier for each experiment
 * @param isPretraining Flag to specify this is pre-training
 * @param runId Run ID (pre-training model)
 * @param modelIdPrefix Model previs
 * @param useValidationAccuracy
 *
 * @author Patrick Nicolas
 * @version 0.4
 */
private[bertspark] final class HPOAnalyzer(
  isPretraining: Boolean,
  runId: String,
  modelIdPrefix: String,
  useValidationAccuracy: Boolean) {
  private[this] val hpoAnalyzer = new ClassifierHPOAnalyzer(s"$runId-$modelIdPrefix")


  def analyze: Unit = {
    ExecutionMode.setHpo

    val s3Folder = s"${mlopsConfiguration.storageConfig.s3RootFolder}/${mlopsConfiguration.target}/metrics/$runId"
    val marker = if(isPretraining) "Pretrained" else s"Trained-Bert-$runId-$modelIdPrefix"
    val groupedS3Keys = S3Util
        .getS3Keys(mlopsConfiguration.storageConfig.s3Bucket, s3Folder)
        .filter(_.contains(marker))
        .groupBy(
          key => {
            val endIndexMarker = key.lastIndexOf("-")
            val groupedKeys = key.substring(0, endIndexMarker)
            groupedKeys
          }
        )
    val s3Keys = groupedS3Keys.map(_._2)
    hpoAnalyzer.execute(s3Keys, useValidationAccuracy)
  }
}


/**
 * Singleton for HPO analyzer customized for classifier and pre-training
 */
private[bertspark] final object HPOAnalyzer {
  final private val logger: Logger = LoggerFactory.getLogger("HPOAnalyzer")

  trait HPOBaseAnalyzer  {
    def execute(s3Keys: Iterable[Iterable[String]], useValidationAccuracy: Boolean): Unit
  }

  final class ClassifierHPOAnalyzer(id: String) extends HPOBaseAnalyzer {

    override def execute(groupedS3Keys: Iterable[Iterable[String]], useValidationAccuracy: Boolean): Unit = {
      val accuracyPerExperiment = groupedS3Keys.map(compileExperiment(_, useValidationAccuracy))
      val rankedExperiments = accuracyPerExperiment
        .toSeq
        .map{ case (acc, params) => (acc, params.dropRight(4))}
        .sortWith(_._1 > _._1)

      val rankedExperimentsStr = rankedExperiments.map(displayExperimentResult(_)).mkString("\n\n\n")
      LocalFileUtil.Save.local(s"output/results-$id", rankedExperimentsStr)
    }
  }

  private def displayExperimentResult(result: (Double, Array[String])) =
    s"Accuracy: ${result._1}\n${result._2.mkString("\n")}"

  private def compileExperiment(s3Keys: Iterable[String], useValidationAccuracy: Boolean): (Double, Array[String]) = {
    val keyValues = s3Keys.map(
      s3Key => {
        val content = S3Util
            .download(mlopsConfiguration.storageConfig.s3Bucket, s3Key)
            .getOrElse({
              logger.warn(s"$s3Key is not found in S3")
              ""
            })

        if (content.nonEmpty) {
          val parameterValuesStr = content.split("\n")
          val parameterValues =  parameterValuesStr.filter(line => parameterNames.exists(line.contains(_)))

          // We need to verify that we collect all the parameters
          if (parameterValues.size == parameterNames.size + 2) {
            val accuracyParamIndex = if(useValidationAccuracy) parameterValues.size - 2 else parameterValues.size - 4
            val epochAccuracyLine = parameterValues(accuracyParamIndex)
            val keyValuePair = epochAccuracyLine.split(",")
            val maxAccuracy = keyValuePair(1).split(tokenSeparator).map(_.toDouble).max
            (maxAccuracy, parameterValues)
          }
          else {
            logger.error(s"Num. parameters ${parameterValues.size} should be ${parameterNames.size + 2}")
            (-1.0, Array.empty[String])
          }
        }
        else
          (-1.0, Array.empty[String])
      }
    )   .filter(_._2.nonEmpty)
        .toSeq

    // Post process the array of configuration
    if(keyValues.nonEmpty) {
      val params = keyValues.head._2
      val averageAccuracy = keyValues.map(_._1).sum / keyValues.size
      val experimentId = {
        val endExperimentIdIndex = s3Keys.head.lastIndexOf("-")
        s3Keys.head.substring(0, endExperimentIdIndex)
      }
      (averageAccuracy, Array[String](experimentId) ++ params)
    }
    else
      (-1.0F, Array.empty[String])
  }


  private def getSubModelName(s3Key: String): String = {
    val subModelNameStartIndex = s3Key.lastIndexOf("-") + 1
    if(subModelNameStartIndex > 0) s3Key.substring(subModelNameStartIndex) else ""
  }
}