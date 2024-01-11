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
package org.bertspark.kafka

import org.apache.spark.sql.SparkSession
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.{EvaluationConfig, S3PathNames}
import org.bertspark.{delay, TEvaluator}
import org.bertspark.kafka.prodcons.{KafkaFeedbackProc, KafkaPredictionProc}
import org.bertspark.kafka.simulator.{FeedbackMessageGenerator, RequestMessageGenerator}
import org.slf4j.{Logger, LoggerFactory}
import org.bertspark.config.MlopsConfiguration._


/**
  * {{{
  * Generic workflow for generation and evaluation of predictions using Kafka message bus:
  * Workflow:
  *    1 Simulate request
  *    2 Consume request and predict claims
  *    3 Simulate feedback
  *    4 Consume feedback
  *    5 Compute and stored metrics
  * }}}
  *
  * @param args Command line argument for the evaluation using Kafka messaging queue
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] final class KafkaEvaluator(args: Seq[String]) extends TEvaluator {
  import KafkaEvaluator._
  val stage = args(2).toInt
  require((stage >= 0 && stage < 4) || (stage >= 10 && stage < 14), s"Stage $stage should be [0, 3] or [10, 13]")

  /**
    * Execute the entire evaluation cycle
    *
    * @param sparkSession Implicit reference to the current Spark context
    */
  override def execute(implicit sparkSession: SparkSession): Unit = {
    if (stage <= allEvalStage || stage == generateRequestStage) // Stage 0   Generate request message
      generateRequest(mlopsConfiguration.evaluationConfig)

    if (stage <= claimFeedbackMetricStage || stage == generatePredictionStage)
      generatePredictions(mlopsConfiguration.evaluationConfig)

    if (stage <= feedbackMetricStage || stage == generateFeedbackStage)
      generateFeedbacks(mlopsConfiguration.evaluationConfig)

    if (stage <= metricOnlyStage || stage == generateMetricsStage)
      generateMetrics(mlopsConfiguration.evaluationConfig)
  }


    // ---------------------  Supporting methods ----------------------

  private def generateRequest(evaluationConfig: EvaluationConfig)(implicit sparkSession: SparkSession): Unit = {
    RequestMessageGenerator(
      evaluationConfig.s3RequestPath,
      mlopsConfiguration.runtimeConfig.requestTopic,
      evaluationConfig.ingestIntervalMs,
      evaluationConfig.numRequestPerSubModel).proceed

    delay(10000L)
    logDebug(
      logger,
      msg = s"Generation of requests to ${mlopsConfiguration.runtimeConfig.requestTopic} from ${evaluationConfig.s3RequestPath} is done"
    )
  }

  private def generatePredictions(evaluationConfig: EvaluationConfig)(implicit sparkSession: SparkSession): Unit = {
    KafkaPredictionProc.executeBatch(
      mlopsConfiguration.runtimeConfig.requestTopic,
      mlopsConfiguration.runtimeConfig.responseTopic,
      evaluationConfig.numRequestPerSubModel)
    delay(10000L)
    logDebug(
      logger,
      msg = s"Generation of claims from ${evaluationConfig.s3RequestPath} to ${S3PathNames.s3PredictedClaimPath} is done"
    )
  }

  private def generateFeedbacks(evaluationConfig: EvaluationConfig)(implicit sparkSession: SparkSession): Unit = {
    FeedbackMessageGenerator(
      S3PathNames.s3PredictedClaimPath,
      evaluationConfig.s3FeedbackPath,
      mlopsConfiguration.runtimeConfig.feedbackTopic,
      evaluationConfig.ingestIntervalMs,
      -1).proceed
    delay(timeInMillis = 10000L)
    logDebug(
      logger,
      msg = s"Generation of feedbacks ${mlopsConfiguration.runtimeConfig.feedbackTopic} from ${evaluationConfig.s3RequestPath} is done"
    )
  }

  private def generateMetrics(evaluationConfig: EvaluationConfig)(implicit sparkSession: SparkSession): Unit = {
    KafkaFeedbackProc.executeBatch(
      mlopsConfiguration.runtimeConfig.feedbackTopic,
      mlopsConfiguration.runtimeConfig.ackTopic,
      evaluationConfig.numRequestPerSubModel)
    delay(timeInMillis = 2000L)
    logDebug(logger, msg = s"Evaluation from ${mlopsConfiguration.runtimeConfig.feedbackTopic} is done")
  }
}



private[bertspark] final object KafkaEvaluator {
  final private val logger: Logger = LoggerFactory.getLogger("KafkaEvaluator")

  final private val allEvalStage = 0
  final private val claimFeedbackMetricStage = 1
  final private val feedbackMetricStage = 2
  final private val metricOnlyStage = 3
  final private val generateRequestStage = 10
  final private val generatePredictionStage = 11
  final private val generateFeedbackStage = 12
  final private val generateMetricsStage = 13
}