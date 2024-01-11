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
package org.bertspark.nlp.trainingset

import org.apache.spark.sql._
import org.bertspark.util.SparkUtil
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.config.MlopsConfiguration.DebugLog.{logDebug, logInfo}
import org.bertspark.config.{ExecutionMode, S3PathNames}
import org.bertspark.config.S3PathNames.s3SubModelsStructure
import org.bertspark.util.io.S3Util
import org.slf4j._

/**
 * {{{
 *   Sampler methods for Training set
 *   - downSampler sample the entire dataset of labeled claims to the minium number of occurrences
 *   - sampleByFrequency Sample by frequency
 * }}}
 *
 * @author Patrick Nicolas
 * @version 0.3
 */
private[bertspark] final object TrainingSetSampler {
  final private val logger: Logger = LoggerFactory.getLogger("TrainingSetSampler")

  /**
    * Select supported sub models to be trained, and evaluated
    * @return List of tuples (sub_model_name, num_labels, labels)
    */
  def selectSupportedSubModels: (Seq[(String, Int, String)], Seq[(String, Int, String)]) = {
    val (oracleSubModels, predictiveSubModels) = generate
    val totalNumSubModels = oracleSubModels.size + predictiveSubModels.size

    val oracleSubModelsRatio = oracleSubModels.size.toFloat / totalNumSubModels
    logDebug(logger, msg = s"Ratio Oracle to trained model $oracleSubModelsRatio")
    (oracleSubModels, predictiveSubModels)
  }


  /**
    * Extract tuple {sub-model, num_labels_in_sub_model, labels } from S3 list of trained model and
    * subModels
    * @return Tuples (sub-model, num_labels_in_sub_model, labels) [Oracles, Predictives]
    */
  private def generate: (Array[(String, Int, String)], Array[(String, Int, String)]) = try {
    if(ExecutionMode.isEvaluation) {
      val entries: Array[(String, Int, String)] = S3Util
          .download(mlopsConfiguration.storageConfig.s3Bucket, s3SubModelsStructure)
          .map(_.split("\n"))
          .getOrElse(Array.empty[String])
          .map {
            entry => {
              val fields = entry.split(",")
              (fields.head.trim, fields(1).toInt, fields(2).replace("  ", " "))
            }
          }

      if (entries.nonEmpty) {
        val (oracles, trained) = entries.partition(_._2 == 1)

        val trainedSubModelSet = S3Util.getS3Keys(
          mlopsConfiguration.storageConfig.s3Bucket,
          S3PathNames.s3ClassifierModelPath
        )
            .filter(_.endsWith(".params"))
            .map(
              path => {
                val endIndex = path.indexOf("-0000")
                val beginIndex = path.lastIndexOf("/")
                val subModel = path.substring(beginIndex + 1, endIndex)
                subModel
              }
            ).toSet

        val trainedSubModels = trained.filter(t => trainedSubModelSet.contains(t._1.trim))

        // We can evaluate the model with or without Oracle
        if (mlopsConfiguration.evaluationConfig.classifierOnly)
          (Array.empty[(String, Int, String)], trainedSubModels)
        else {
          (oracles, trainedSubModels)
        }
      }
      else {
        logger.error(s"Unable to load data from $s3SubModelsStructure")
        (Array.empty[(String, Int, String)], Array.empty[(String, Int, String)])
      }
    }
      // For training classifier ..
    else {
      val entries = S3Util
          .download(mlopsConfiguration.storageConfig.s3Bucket, s3SubModelsStructure)
          .map(_.split("\n"))
          .getOrElse(Array.empty[String])
          .map {
            entry => {
              val fields = entry.split(",")
              (fields.head.trim, fields(1).toInt, fields(2).replace("  ", " "))
            }
          }


      logDebug(logger, s"Loaded ${entries.length} sub-models from $s3SubModelsStructure")

      if (entries.nonEmpty) {
        val (oracle, toTrain) = entries.partition(_._2 == 1)
        logDebug(
          logger,
          msg = s"Loaded ${oracle.size} oracle and ${toTrain.size} to train sub models from $s3SubModelsStructure"
        )
        (oracle, toTrain)
      }
      else {
        logger.error(s"Unable to load data from $s3SubModelsStructure")
        (Array.empty[(String, Int, String)], Array.empty[(String, Int, String)])
      }

    }
  }
  catch {
    case e: IllegalArgumentException =>
      logger.error(s"Generate sub models taxonomy failed ${e.getMessage}")
      (Array.empty[(String, Int, String)], Array.empty[(String, Int, String)])
  }



  /**
   * Down sample the training data
   *
   * @param labeledClaimDS Data set of labeled claim
   *                     [LabeledClaim(id: String, emrCodes: Seq[MlEMRCodes], feedbackLineItems: Seq[FeedbackLineItem]]
   * @param minNumOccurrences Minimum number of occurrences to allow the records
   * @param sparkSession      Implicit reference to the current Spark context
   * @return Down sampled labeled training data
   */
  def downSampleClaim(
    labeledClaimDS: Dataset[LabeledClaim],
    minNumOccurrences: Int
  )(implicit sparkSession: SparkSession): Dataset[LabeledClaim] = {
    import sparkSession.implicits._

    labeledClaimDS.map(
      labeledClaim => (labeledClaim.getClaim, Seq[LabeledClaim](labeledClaim))
    ).groupByKey(_._1)
        .reduceGroups(
          (f: (String, Seq[LabeledClaim]), g: (String, Seq[LabeledClaim])) => (f._1, f._2 ++ g._2))
        .flatMap { case (_, (_, labeledTrainData)) => labeledTrainData.take(minNumOccurrences) }
  }


  /**
    * {{{
    *   Group a hierarchical labeled claim associated with a given emr,by the remaining claim
    *   We filter for eligible labels: Number of notes associated with a given note should be > minLabelFreq
    *   The number of notes per labels is capped to maxLabelFreq to avoid unbalanced training set
    * }}}
    *
    * @param trainingLabels Hierarchical labeled claim (id, emr, remainingClaim)
    * @return Sequence of hierarchical labeled claims group by their remaining codes
    */
  def filterPerNumNotesLabels(
    trainingLabels: List[TrainingLabel]
  ): Seq[List[TrainingLabel]] = {
    val groupTrainingLabels: Seq[List[TrainingLabel]] = trainingLabels.groupBy(_.claim).map(_._2).toSeq
    // We filter for number of occurrences be minNumOccurrences of notes per label/claims
    groupTrainingLabels.filter(_.size >= mlopsConfiguration.preProcessConfig.minLabelFreq)
  }

  def getNumNotesPerLabels(
    trainingLabels: List[TrainingLabel]
  ): Seq[(String, Int)] = {
    val claimDistribution = trainingLabels.groupBy(_.claim).map{
      case (claim, notes) => (claim, notes.size)
    }.toSeq.sortWith(_._2 > _._2)
    claimDistribution
  }


  /**
   * Sample per minimum frequency. Only the training records for which the number of claims is above the
   * threshold are considered for the classification
   *
   * @param labeledClaimDS Raw labeled claim dataset
   * @param minFrequency   Minimum number of claim occurrences (frequency) to quality the training record
   * @param sparkSession   Implicit reference to the current Spark context
   * @return Labeled claimed training data which meet the minimum frequency requirement.
   */
  def sampleByFrequency(
    labeledClaimDS: Dataset[LabeledRequest],
    minFrequency: Int)(implicit sparkSession: SparkSession): Dataset[LabeledRequest] = {
    import sparkSession.implicits._

    logInfo(logger, msg = s"Total number of labeled requests: ${labeledClaimDS.count}")
    val labeledCountDS = labeledClaimDS.map(labeledRequest => (labeledRequest.claimStr, 1))
    val aggregatedRDD = SparkUtil.groupBy[LabeledCount, String](
      (labeledCount: LabeledCount) => labeledCount._1,
      reducer,
      labeledCountDS)

    val rankedRDD = aggregatedRDD.sortBy[Int]((f: LabeledCount) => f._2, false)
    val filteredRDD = rankedRDD.filter(_._2 > minFrequency)
    val filteredSet = filteredRDD.map(_._1).collect.toSet

    val finalLabeledRequestDS = labeledClaimDS.filter(labeledRequest => filteredSet.contains(labeledRequest.claimStr))
    logInfo(logger, msg = s"Number of high frequency requests: ${finalLabeledRequestDS.count}")
    finalLabeledRequestDS
  }


  private val reducer = (labeledCount1: LabeledCount, labeledCount2: LabeledCount) =>
    (labeledCount1._1, labeledCount1._2 + labeledCount2._2)
}
