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
package org.bertspark.analytics

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Encoder, SparkSession}
import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalFeedback, InternalRequest, Modality}
import org.bertspark.util.io.S3Util
import org.bertspark.util.SparkUtil
import org.bertspark.config.MlopsConfiguration.DebugLog.logInfo
import org.slf4j.{Logger, LoggerFactory}
import scala.reflect.ClassTag


/**
 * Cleanser utility for raw requests (type MlPredictReq) and feedbacks (type MlFeedbackReq)
 * {{{
 *   cleanseFeedbackLineItem  Apply a repair/fix method for feedback line item
 *   removedDuplicates Remove duplicates keeping the most recent version
 * }}}
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final object Cleanser {
  import org.bertspark.config.MlopsConfiguration._
  final private val logger: Logger = LoggerFactory.getLogger("Cleanser")

  final private val modifiersCorrector = (input: InternalFeedback) => {
    val updatedLineItems = input.finalized.lineItems.map(
      lineItem => {
        if(lineItem.modifiers.nonEmpty) {
          val updatedModifiers = lineItem.modifiers.filter(_.nonEmpty)
          if (updatedModifiers.size < lineItem.modifiers.size) {
            logger.error(s"Found incorrect modifier for ${input.id} ${lineItem.modifiers.mkString(",")}")
            lineItem.copy(modifiers = updatedModifiers)
          }
          else
            lineItem
        }
        else
          lineItem
      }
    )
    val updatedFinalized = input.finalized.copy(lineItems = updatedLineItems)
    input.copy(finalized = updatedFinalized, autoCodable=false)
  }

  final val modalityCorrector = (input: InternalRequest) => {
    val _modality =
      if(input.context.EMRCpts.nonEmpty) Modality.getModalityFromCpt(input.context.EMRCpts.head.cpt)
      else Modality.unknownModality

    val _context = input.context.copy(modality = _modality)
    input.copy(context = _context)
  }

  /**
   * Correct the feedback line item for incorrect modifiers
   * @param s3FeedbackFolder S3 folder containing the feedback
   * @param sparkSession Implicit reference to the current Spark context
   */
  def cleanseFeedbackLineItem(s3FeedbackFolder: String)(implicit sparkSession: SparkSession): Unit =
    cleanseFeedback(s3FeedbackFolder, modifiersCorrector)

  /**
   * Remove duplicates from Feedback
   * @param sourceFeedbackDS Original Data set for Feedbacks
   * @param sparkSession Implicit reference to the current Spark context
   * @return Data set of feedbacks without duplicates.
   */
  def removedDuplicates(
    sourceFeedbackDS: Dataset[InternalFeedback]
  )(implicit sparkSession: SparkSession): Dataset[InternalFeedback] = {
    import sparkSession.implicits._
    removedDuplicates[InternalFeedback](sourceFeedbackDS, (feedback: List[InternalFeedback]) => feedback.head.id)
  }

  /**
   * Generic method to remove duplicate of type T
   * @param sourceDS Source/input data set of type T
   * @param key Function to extract key (identifier) the define the duplicates
   * @param sparkSession Implicit reference to the current Spark context
   * @param encoder Implicit encoder for type T
   * @tparam T Type of the element in the data set
   * @return Data set of type T without duplicates
   */
  def removedDuplicates[T](
    sourceDS: Dataset[T],
    key: List[T] => String
  )(implicit sparkSession: SparkSession,
    t: ClassTag[T],
    enc1: Encoder[T],
    enc2: Encoder[List[T]],
    enc3: Encoder[(String, List[T])]): Dataset[T] = {
    import sparkSession.implicits._
    val accDS = sourceDS.map(List[T](_))

    // We record the last feedback (which should be the audit)
    val groupedById: RDD[List[T]] = SparkUtil.groupBy[List[T], String](
      key,
      (_: List[T], feedback2: List[T]) => feedback2,
      accDS
    )
    groupedById.map(_.head).toDS()
  }

  def cleanseRequest(
    s3InputRequest: String,
    correct: InternalRequest => InternalRequest
  )(implicit sparkSession: SparkSession): Unit = {
    import sparkSession.implicits._

    val predictReqDS = try {
      S3Util.s3ToDataset[InternalRequest](
        mlopsConfiguration.storageConfig.s3Bucket,
        s3InputRequest,
        header = false,
        "json"
      )
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"cleanseRequest: ${e.getMessage}")
        sparkSession.emptyDataset[InternalRequest]
    }

    val predictReqDSSplit = predictReqDS.randomSplit(Array.fill(20)(0.05))
    var count = 0
    predictReqDSSplit.map(
      splitDS => {
        count += 1
        logInfo(logger, s"Split #$count")
        val cleansedSplitDS = splitDS.map(correct(_))

        S3Util.datasetToS3[InternalRequest](
          mlopsConfiguration.storageConfig.s3Bucket,
          cleansedSplitDS,
          s3OutputPath = s"temp/$s3InputRequest",
          header = false,
          fileFormat = "json",
          toAppend = true,
          numPartitions = 4
        )
      }
    )
  }

  private def cleanseFeedback(
    s3FeedbackFolder: String,
    correct: InternalFeedback => InternalFeedback
  )(implicit sparkSession: SparkSession): Unit = {
    import sparkSession.implicits._

    val feedbackDS = try {
      S3Util.s3ToDataset[InternalFeedback](
        mlopsConfiguration.storageConfig.s3Bucket,
        s3FeedbackFolder,
        false,
        "json"
      )
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"Cleanse feedback: ${e.getMessage}")
        sparkSession.emptyDataset[InternalFeedback]
    }

    logger.info(s"${feedbackDS.count} original feedbacks")
    val feedbackWithoutDuplicatesDS = removedDuplicates(feedbackDS)
    logger.info(s"${feedbackWithoutDuplicatesDS.count} unique feedbacks")

    val correctedDS = feedbackWithoutDuplicatesDS.map(correct(_))
    S3Util.datasetToS3[InternalFeedback](
      mlopsConfiguration.storageConfig.s3Bucket,
      correctedDS,
      s3OutputPath = s"$s3FeedbackFolder-corrected",
      header = false,
      fileFormat = "json",
      toAppend = false,
      numPartitions = 32
    )
  }
}
