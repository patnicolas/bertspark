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

import org.bertspark.delay
import org.bertspark.kafka.prodcons.{KafkaFeedbackProc, KafkaPredictionProc}
import org.bertspark.kafka.prodcons.KafkaPredictionProc.predictionPipeline
import org.slf4j.{Logger, LoggerFactory}


/**
  * Define a pipeline to generate request through Kafka topic
  * {{{
  *  Command line arguments:
  *  KafkaPipeline requestTopic responseTopic pollingInterval requestType
  *  Example
  *    'kafkaPipeline ml-request-mlops ml-response-mlops 4500 requests'  # for requests
  *    'kafkaPipeline ml-feedback-mlops ml-ack-mlops 4500 feedbacks'  # for feedbacks
  * }}}
  *
  * @author Patrick Nicolas
  * @version 0.5
  */
private[bertspark] final object KafkaPipeline {
  final private val logger: Logger = LoggerFactory.getLogger("KafkaPipeline")


  def apply(args: Seq[String]): Boolean = {
    require(args.size == 5,
      s"""Kafka pipeline command line arguments should be
         |'kafkaPipeline requestTopic responseTopic pollingInterval requestType' """.stripMargin)

    val requestTopic = args(1)
    val responseTopic = args(2)
    val pollingTimeIntervalMs = args(3).toInt
    val requestType = args(4)

    apply(RequestRunTimeConfig(requestTopic, responseTopic, pollingTimeIntervalMs), requestType)
  }

  /**
    * *
    * @param predictionRunTimeConfig
    * @param requestType
    * @return
    */
  def apply(predictionRunTimeConfig: RequestRunTimeConfig, requestType: String): Boolean = {
    val numProcessedResponses =
      requestType match {
        case "requests" =>
          KafkaPredictionProc.executeBatch(
            predictionRunTimeConfig.requestTopic,
            predictionRunTimeConfig.responseTopic,
            -1
          )
        case "feedbacks" =>
          KafkaFeedbackProc.executeBatch(
            predictionRunTimeConfig.requestTopic,
            predictionRunTimeConfig.responseTopic,
            -1
          )
        case _ =>
          throw new UnsupportedOperationException(s"Request type $requestType is not supported!")
        }

    logger.info(s"$numProcessedResponses predictions sent to ${predictionRunTimeConfig.responseTopic}")

    val delayMs = 40000L
    delay(delayMs)
    true
  }
}
