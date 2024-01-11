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
package org.bertspark.predictor

import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.TEvaluator
import org.bertspark.analytics.MetricsCollector
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.medical.MedicalCodingTypes.{lineItemSeparator, FeedbackLineItem, InternalFeedback, InternalRequest, MlClaimEntriesWithCodes, MlLineItem, PRequest}
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.bertspark.util.io.S3Util
import org.bertspark.util.SparkUtil
import org.slf4j.{Logger, LoggerFactory}


/**
  * Evaluator using request, predictions and labels from S3 store
  * {{{
  * Evaluation modes:
  *    - load Load pre-evaluation data set
  *    - fromTraining  Evaluate classifier from data extracted from training set
  *    - fromRequest  Evaluate classifier from data extracted from raw requests
  *
  * Command line arguments:
  *   - evaluate s3 fromRequest numRequests  # Evaluate using random sample from requests and feedback
  *   - evaluate s3 fromTraining             # Evaluate using random sample from training set and feedbacks
  *   - evaluate Kafka numRandomRequests     # Evaluate using Kafka messages
  * }}}
  * @param args Command line arguments
  *
  * @author Patrick Nicolas
  * @version 0.7
  */
private[bertspark] final class S3Evaluator(args: Seq[String]) extends TEvaluator with MetricsCollector {
  require(
    args.size > 1,
    s"S3 Evaluator argument ${args.mkString(" ")} should be 'evaluate {load,fromTraining,fromRequest}"
  )
  protected[this] val lossName: String = "Evaluation loss"

  /**
    * {{{
    *  Execute the entire evaluation cycle
    *  1- Load the requests from S3
    *  2- Select a random number of requests
    *  3- Load the feedback or labels
    *  4- Join the random sample of requests with the feedbacks or labels
    *  5- Compute the metrics
    *  6- Save metrics into S3
    * }}}
    * @param sparkSession Implicit reference to the current Spark context
    */
  override def execute(implicit sparkSession: SparkSession): Unit = {
    val completeInternalFeedbacks = S3Evaluator.execute(args)
    updateMetrics(completeInternalFeedbacks)
    save
  }
}


/**
  * Singleton that update the Feedback with the prediction generated from a random
  * sample of requests
  */
private[bertspark] object S3Evaluator {
  final private val logger: Logger = LoggerFactory.getLogger("S3Evaluator")

  def apply(numRequests: Int): S3Evaluator = new S3Evaluator(
    Seq[String]("evaluate", "fromRequest" ,numRequests.toString)
  )

  case class PredictedClaimStr(id: String, lineItems: Seq[MlLineItem]) {
    override def toString: String = s"Id:$id, ${lineItems.map(_.toCodesSpace).mkString(lineItemSeparator)}"
  }


  // -----------------------   Supported methods -------------------------------

  private def execute(args: Seq[String])(implicit sparkSession: SparkSession): Seq[InternalFeedback] = {
    val evaluationSetManager = args(2) match {
      case "load" =>  new EvaluationFromLoadedFeedbacks
      case "fromTraining" => new EvaluationFromTrainingSet
      case "fromModels" => new EvaluationFromModels
      case "fromRequest" => new EvaluationFromRequests
      case _ =>
        throw new UnsupportedOperationException(s"${args(2)} operation for evaluation is not supported")

    }

    val numRandomRequests = args(3).toInt
    val sizeSplits = args(4).toInt
    val completeInternalFeedbackDS = evaluationSetManager(numRandomRequests, sizeSplits)
    val completeInternalFeedbacks = completeInternalFeedbackDS.collect()

    val nonEmptyCompleteInternalFeedbacks = completeInternalFeedbacks.filter(_.autocoded.lineItems.nonEmpty)
    logDebug(
      logger, {
      val totalNumberCompleteFeedbacks = completeInternalFeedbacks.length
      val numNonEmptyCompleteFeedbacks = nonEmptyCompleteInternalFeedbacks.length
      s"$numNonEmptyCompleteFeedbacks complete non empty feedbacks out of $totalNumberCompleteFeedbacks"
    })
    nonEmptyCompleteInternalFeedbacks
  }


      // ---------------------------

  trait EvaluationManager {

    def apply(numRequests: Int, sizeRequestsDSSplit: Int)(implicit sparkSession: SparkSession): Dataset[InternalFeedback] = {
      val (requestDS, feedbackDS) = extract(numRequests)
      annotateFeedback(requestDS, feedbackDS, sizeRequestsDSSplit)
    }

    protected def extract(numRequests: Int): (Dataset[InternalRequest], Dataset[InternalFeedback])

    private def annotateFeedback(
      internalRequestDS: Dataset[InternalRequest],
      feedbackDS: Dataset[InternalFeedback],
      sizeRequestsDSSplit: Int
    )(implicit sparkSession: SparkSession) = {
      import sparkSession.implicits._

      val requestDS = internalRequestDS.map(PRequest(_))
      val predictor = TPredictor()

      val rawNumRequests = requestDS.count()
      logger.info(s"Ready to evaluate $rawNumRequests requests")

      if(rawNumRequests > sizeRequestsDSSplit) {
        val numSplits = (rawNumRequests/sizeRequestsDSSplit.toDouble).ceil.toInt
        val requestDSSeq = requestDS.randomSplit(Array.fill(numSplits)(1.0/numSplits))
        val internalFeedbackDSSeq = requestDSSeq.map(consolidateFeedback(_, feedbackDS, predictor))
        internalFeedbackDSSeq.tail.foldLeft(internalFeedbackDSSeq.head)(
          (acc, feedbackDS) => acc.union(feedbackDS)
        )
      }
      else
        consolidateFeedback(requestDS, feedbackDS, predictor)
    }


    private def consolidateFeedback(
      requestDS: Dataset[PRequest],
      feedbackDS: Dataset[InternalFeedback],
      predictor: TPredictor)(implicit sparkSession: SparkSession): Dataset[InternalFeedback] = {
      import sparkSession.implicits._

      val selectedRequests = requestDS.collect
      val predictedClaims = predictor
          .apiProcess(selectedRequests.toSeq)
          .map(
            response => {
              if(response.claim.lineItems.isEmpty)
                logger.warn("response.claim.lineItems is empty")
              PredictedClaimStr(response.id, response.claim.lineItems)
            }
          )

      logDebug(logger, msg = s"Extracted ${predictedClaims.filter(_.lineItems.nonEmpty)} non empty predicted claims")

      // 4- Join the random sample of requests with the feedbacks or labels
      //    and add the prediction to the auto-coded field of the feedback
      SparkUtil.sortingJoin[PredictedClaimStr, InternalFeedback](
        predictedClaims.toDS(),
        tDSKey = "id",
        feedbackDS,
        uDSKey = "id"
      ).map {
        // Update the auto-coded field with the prediction
        case (predictedClaim, feedback) =>
          val feedbackLineItems = predictedClaim.lineItems.map(FeedbackLineItem(_))
          val predicted = MlClaimEntriesWithCodes(feedbackLineItems)
          feedback.copy(autocoded = predicted)
      }
    }
  }



  final class EvaluationFromLoadedFeedbacks(implicit sparkSession: SparkSession) extends EvaluationManager {

    def apply(): Dataset[InternalFeedback] = {
      import sparkSession.implicits._
      S3Util.s3ToDataset[InternalFeedback](s3InputFile = S3PathNames.s3EvaluationSetPath)
    }

    override protected def extract(numRequests: Int): (Dataset[InternalRequest], Dataset[InternalFeedback]) = ???
  }


  /**
    *
    * @param numRequests Number of requests used in the evaluation
    * @param sparkSession Implicit reference to the current Spark context
    */
  final class EvaluationFromRequests(implicit sparkSession: SparkSession) extends EvaluationManager {

    override protected def extract(numRequests: Int): (Dataset[InternalRequest], Dataset[InternalFeedback]) =  {
      import sparkSession.implicits._

      // 1- Load the requests from S3
      val internalRequestDS = try {
        S3Util.s3ToDataset[InternalRequest](
          mlopsConfiguration.storageConfig.s3Bucket,
          S3PathNames.s3RequestsPath,
          header = false,
          fileFormat = "json"
        ).dropDuplicates("id")
      }
      catch {
        case e: IllegalArgumentException =>
        logger.error(s"S3 evaluator request ${e.getMessage}")
        sparkSession.emptyDataset[InternalRequest]
      }

      // 3- Load the feedback or labels
      val feedbackDS: Dataset[InternalFeedback] = try {
        S3Util.s3ToDataset[InternalFeedback](S3PathNames.s3FeedbacksPath).dropDuplicates("id")
      }
      catch {
        case e: IllegalArgumentException =>
        logger.error(s"S3 evaluator feedback ${e.getMessage}")
        sparkSession.emptyDataset[InternalFeedback]
      }
      (internalRequestDS, feedbackDS)
    }
  }


  /**
    * Extract the pair of dataset (requests, filtered feedback) used in the evaluation of models...
    * @param sparkSession Implicit reference to the current Spark context
    */
  final class EvaluationFromModels(implicit sparkSession: SparkSession) extends EvaluationManager {
    import sparkSession.implicits._

    override protected def extract(numRequests: Int): (Dataset[InternalRequest], Dataset[InternalFeedback]) =  try {
      // 1- Load the sub-models which have been trained..
      val s3Folder = S3PathNames.s3ClassifierModelPath
      val subModelSet = S3Util.getS3Keys(mlopsConfiguration.storageConfig.s3Bucket, s3Folder).map(
        path => {
          val relativePath = path.substring(s3Folder.length+1)
          val separatorIndex = relativePath.indexOf("/")
          if(separatorIndex != -1) relativePath.substring(0, separatorIndex) else ""
        }
      ).filter(_.nonEmpty).toSet

      // 2- Load the requests from S3
      val internalRequestDS: Dataset[InternalRequest] = try {
        val rawInternalRequestDS = S3Util.s3ToDataset[InternalRequest](
          mlopsConfiguration.storageConfig.s3Bucket,
          S3PathNames.s3RequestsPath,
          header = false,
          fileFormat = "json"
        )   .dropDuplicates("id")

        val requestDS = rawInternalRequestDS.filter(request => subModelSet.contains(request.context.emrLabel))
        logDebug(
          logger,
          msg = s"Evaluation from models ${requestDS.count()} filtered from ${rawInternalRequestDS.count} loaded"
        )

        val totalNumRequests = requestDS.count()

        if(numRequests > 0) {
          val fraction = numRequests.toFloat / totalNumRequests
          logDebug(logger, msg = s"Evaluate $numRequests random requests out of $totalNumRequests")
          requestDS.sample(fraction)
        }
        else {
          logDebug(logger, msg = s"Evaluate all $totalNumRequests requests")
          requestDS
        }
      }
      catch {
        case e: IllegalArgumentException =>
          logger.error(s"Evaluation from models request ${e.getMessage}")
          sparkSession.emptyDataset[InternalRequest]
      }

      // 3- Load the feedback or labels
      val feedbackDS: Dataset[InternalFeedback] = try {
        logDebug(logger, msg = s"Evaluation from models loaded ${subModelSet.size} sub-models from ${s3Folder}")

        val rawFeedbackDS = S3Util.s3ToDataset[InternalFeedback](S3PathNames.s3FeedbacksPath).dropDuplicates("id")
        val filteredFeedbackDS = rawFeedbackDS.filter(feedback => subModelSet.contains(feedback.context.emrLabel))
        logDebug(
          logger,
          msg = s"Evaluation from models loaded ${filteredFeedbackDS.count} filtered feedbacks from ${rawFeedbackDS.count()}"
        )
        filteredFeedbackDS
      }
      catch {
        case e: IllegalArgumentException =>
          logger.error(s"Evaluation from models failed ${e.getMessage}")
          sparkSession.emptyDataset[InternalFeedback]
      }
      (internalRequestDS, feedbackDS)
    }
    catch {
      case e: IllegalStateException =>
        logger.error(s"Evaluation from sub-models failed to load requests or feedbacks: ${e.getMessage}")
        (sparkSession.emptyDataset[InternalRequest],   sparkSession.emptyDataset[InternalFeedback])
    }
  }


  /**
    * Generation set directly from the training set of type SubModelsTrainingSet in S3
    * @param sparkSession Implicit reference to the current Spark context
    */
  final class EvaluationFromTrainingSet(implicit sparkSession: SparkSession) extends EvaluationManager {
    import sparkSession.implicits._

    override protected def extract(numRequests: Int):  (Dataset[InternalRequest], Dataset[InternalFeedback]) = try {
      val trainingSetPath = S3PathNames.s3ModelTrainingPath
      val subModels = S3Util.s3ToDataset[SubModelsTrainingSet](s3InputFile = trainingSetPath)
      val docIdsDS = subModels.flatMap(_.labeledTrainingData.map(_.contextualDocument.id))
      val docIdsSet = docIdsDS.collect().toSet

      val internalRequestDS = S3Util.s3ToDataset[InternalRequest](S3PathNames.s3RequestsPath)
          .filter(req => docIdsSet.contains(req.id))
      logDebug(logger, msg = s"Evaluation from TrainingSet ${internalRequestDS.count()} requests")

      val feedbackDS = S3Util.s3ToDataset[InternalFeedback](S3PathNames.s3FeedbacksPath)
          .filter(req => docIdsSet.contains(req.id))
      logDebug(logger, msg = s"Evaluation from TrainingSet ${feedbackDS.count()} feedbacks")

      (internalRequestDS, feedbackDS)
    }
    catch {
      case e: IllegalStateException =>
        logger.error(s"Evaluation from training set failed to load requests or feedbacks: ${e.getMessage}")
        (sparkSession.emptyDataset[InternalRequest],  sparkSession.emptyDataset[InternalFeedback])
    }
  }
}