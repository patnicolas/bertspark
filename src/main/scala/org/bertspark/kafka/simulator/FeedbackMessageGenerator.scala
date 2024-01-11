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
package org.bertspark.kafka.simulator

import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames
import org.bertspark.delay
import org.bertspark.kafka.prodcons.TypedKafkaProducer
import org.bertspark.kafka.serde.FeedbackSerDe.FeedbackMessage
import org.bertspark.kafka.simulator.MessageGenerator.logger
import org.bertspark.modeling.SubModelsTaxonomy.subModelTaxonomy
import org.bertspark.nlp.medical.MedicalCodingTypes.{FeedbackLineItem, InternalFeedback, MlClaimEntriesWithCodes}
import org.bertspark.predictor.TPredictor.PredictedClaim
import org.bertspark.util.io.S3Util
import org.bertspark.util.SparkUtil


/**
  * Versatile generator for request messages
  *
  * @param topic Producer topic - feedback topic
  * @param feedbacks Iterator for feedback requests
  * @param ingestIntervalMs Time out (sleep) in milliseconds, between batches
  * @param numRepeats  Number of batches of requests used for the generator
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] final class FeedbackMessageGenerator private (
  override protected val topic: String,
  feedbacks: Seq[InternalFeedback],
  ingestIntervalMs: Long = 0L,
  numRepeats: Int = 1)
    extends MessageGenerator[FeedbackMessage](
      new TypedKafkaProducer[FeedbackMessage](org.bertspark.kafka.serde.FeedbackSerDe.serializingClass, topic),
      feedbacks.map(FeedbackMessage(_)),
      ingestIntervalMs,
      numRepeats) {
  import MessageGenerator._

  def execute: Boolean = {
    logger.info(s"Starts generating of ${feedbacks.size} feedback requests")
    this.start()
    val expectedDuration = 2000000L
    delay(expectedDuration)
    true
  }
}


/**
  * Singleton for constructors
  */
private[bertspark] final object FeedbackMessageGenerator {

  /**
    * Constructor using S3 file a source
    * @param s3FeedbacksPath Path to S3 folder file
    * @param produceTopic Kafka topic for the producer
    * @param numRequests Maximum number of request per batch
    * @return Instance of FeedbackMessageGenerator
    */
  def apply(
    s3FeedbacksPath: String,
    produceTopic: String,
    numRequests: Int)(implicit sparkSession: SparkSession): FeedbackMessageGenerator = {
    apply(s3FeedbacksPath,  produceTopic, 1000L, numRequests, 1)
  }


  /**
    * Constructor using S3 file a source
    * @param s3FeedbacksPath Path to S3 folder file
    * @param produceTopic Kafka topic for the producer
    * @param ingestIntervalMs Time interval between batches
    * @param numRequests Maximum number of request per batch
    * @param numRepeats Number of batches
    * @return Instance of FeedbackMessageGenerator
    */
  def apply(
    s3FeedbacksPath: String,
    produceTopic: String,
    ingestIntervalMs: Long,
    numRequests: Int,
    numRepeats: Int)(implicit sparkSession: SparkSession): FeedbackMessageGenerator = {
    import org.bertspark.config.MlopsConfiguration._
    import sparkSession.implicits._
    import collection.JavaConverters._

    // Load feedback records
    val feedbacks: Seq[InternalFeedback] = try {
      S3Util.s3ToDataset[InternalFeedback](
        mlopsConfiguration.storageConfig.s3Bucket,
        s3FeedbacksPath,
        header = false,
        fileFormat = "json")
          .dropDuplicates("id")
          .limit(numRequests)
          .collect()
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"FeedbackMessageGenerator: ${e.getMessage}")
        Seq.empty[InternalFeedback]
    }

    logDebug(logger, s"Load ${feedbacks.size} feedback records")

    new FeedbackMessageGenerator(produceTopic, feedbacks, ingestIntervalMs, numRepeats)
  }



  /**
    * Constructor for testing Kafka feedback topic only
    * @param produceTopic Kafka topic for the producer
    * @param numRequests Maximum number of request per batch
    * @return Instance of FeedbackMessageGenerator
    */
  def apply(
    produceTopic: String,
    numRequests: Int
  )(implicit sparkSession: SparkSession): FeedbackMessageGenerator =
    new FeedbackMessageGenerator(produceTopic, Seq.empty[InternalFeedback], 1000L, 1)


  /**
    * {{{
    *  Constructor using S3 feedback and Predicted claim as source.
    *  Steps:
    *   - 1: Load predicted claim
    *   - 2 Feedbacks - Load all feedback without duplicates
    *   - 3 Feedbacks - Sample a sub set of feedback if the number of requests is limited
    *   - 4 Feedbacks - Filter by valid, trained sub-models if specified in the configuration file
    *   - 5 Join the predicted claims with internal feedbacks records
    *   - 6 Instantiate the generator
    * }}}
    *
    * @param s3PredictedClaimPath  S3 path for predicted claim (abridged Prediction response)
    * @param s3FeedbacksPath Path to S3 folder file
    * @param produceTopic Kafka topic for the producer
    * @param ingestIntervalMs Time interval between batches
    * @param numRequests Maximum number of request per batch
    * @return Instance of FeedbackMessageGenerator
    */
  def apply(
    s3PredictedClaimPath: String,
    s3FeedbacksPath: String,
    produceTopic: String,
    ingestIntervalMs: Long,
    numRequests: Int)(implicit sparkSession: SparkSession): FeedbackMessageGenerator = {
    import sparkSession.implicits._

    // Step 1: Load predicted claim
    val predictedClaimDS: Dataset[PredictedClaim] = try {
      S3Util.s3ToDataset[PredictedClaim](
        S3PathNames.s3PredictedClaimPath,
        header = false,
        fileFormat = "json"
      )
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"computeOverlapRate: ${e.getMessage}")
        sparkSession.emptyDataset[PredictedClaim]
    }
    logDebug(logger, s"Loaded ${predictedClaimDS.count()} predicted claims")

    // Step 2: Feedbacks - Load all feedback without duplicates
    val allInternalFeedbackDS = try {
      S3Util.s3ToDataset[InternalFeedback](
        mlopsConfiguration.storageConfig.s3Bucket,
        s3FeedbacksPath,
        header = false,
        fileFormat = "json"
      )   .dropDuplicates("id")
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"FeedbackMessageGenerator: ${e.getMessage}")
        sparkSession.emptyDataset[InternalFeedback]
    }


    // Step 3: Feedbacks - Sample a sub set of feedback if the number of requests is limited
    val sampledInternalFeedbackDS =
      if(numRequests > 0) allInternalFeedbackDS.limit(numRequests)
      else allInternalFeedbackDS

    // Step 4: Feedback - Filter by valid, trained sub-models if specified in the configuration file
    val subModelNames = subModelTaxonomy.getSupportedSubModelNameSet
    val internalFeedbackDS =    sampledInternalFeedbackDS.filter(feedback => subModelNames.contains(feedback.context.emrLabel))

    logDebug(logger, s"Loaded ${internalFeedbackDS.count()} feedbacks for trained sub models")

    // Step 5: Join the predicted claims with internal feedbacks records
    val feedbacks: Seq[InternalFeedback] = mergePredictedClaimFeedbacks(predictedClaimDS, internalFeedbackDS)

    // Step 6: Instantiate the generator
    new FeedbackMessageGenerator(produceTopic, feedbacks, ingestIntervalMs, numRepeats = 1)
  }


  private def mergePredictedClaimFeedbacks(
    predictedClaimDS: Dataset[PredictedClaim],
    internalFeedbackDS: Dataset[InternalFeedback])(implicit sparkSession: SparkSession): Seq[InternalFeedback] = {
    import sparkSession.implicits._

    val updatedInternalFeedbackDS = SparkUtil.sortingJoin[PredictedClaim, InternalFeedback](
      predictedClaimDS,
      "id",
      internalFeedbackDS,
      "id"
    ).map{
      case (predictedClaim, internalFeedback) =>
        // We need at least one line item with one medical code (CPT)
        if(predictedClaim.lineItems.isEmpty || predictedClaim.lineItems.head.cpt.isEmpty) {
          logger.error(s"Cannot find line item for ${predictedClaim.id}")
          InternalFeedback()
        }
        else {
          val feedbackLineItems = predictedClaim.lineItems.map(FeedbackLineItem(_))
          val correctedAutoCoded = MlClaimEntriesWithCodes(feedbackLineItems)
          internalFeedback.copy(autocoded = correctedAutoCoded)
        }
    }.filter(_.id.nonEmpty)

    updatedInternalFeedbackDS.show()
    logDebug(logger, s"Create generator for ${updatedInternalFeedbackDS.count()} labeled claims")
    updatedInternalFeedbackDS.collect()
  }
}
