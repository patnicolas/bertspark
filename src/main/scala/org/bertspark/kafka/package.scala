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

/**
 * @note The default values should be actually defined in the configuration file ???
 * @author Patrick Nicolas
 * @version 0.6
 */
package object kafka {
  import org.bertspark.util.ResourceUtil.getFileFromResourceAsStream

  final val applicationPropertiesFile = "application.properties"

  final val testPredictRequestTopic = "testPredictRequest"
  final val testPredictResponseTopic = "testPredictResponse"
  final val testFeedbackRequestTopic = "testFeedbackRequest"
  final val testFeedbackResponseTopic = "testFeedbackResponse"
  final val saslJaasConfigLabel = "sasl.jaas.config"
  lazy val initialProperties = getFileFromResourceAsStream(applicationPropertiesFile, "=")

  trait RequestExecConfig {
    self =>
  }

  /**
    * Configuration for Kafka producer
    * @param requestTopic Topic for requests for prediction
    * @param responseTopic Topic for responses for prediction
    * @param pollingTimeIntervalMs Polling interval for Kafka consumer in millis
    */
  case class RequestRunTimeConfig(
    requestTopic: String,
    responseTopic: String,
    pollingTimeIntervalMs: Long) extends RequestExecConfig

  /**
    * Companion object to convert command line arguments into PredictionRunTimeConfig
    */
  final object RequestRunTimeConfig {

    @throws(clazz = classOf[IllegalArgumentException])
    def apply(args: Seq[String]): RequestRunTimeConfig = {
      require(args.size == 5,
        s"""Incorrect arguments:[${args.mkString(" ")}]
           |Should be [local kafkaProcCons requestTopic responseTopic pollingTimeIntervalMs""".stripMargin)
      val requestTopic = args(2)
      val responseTopic = args(3)
      val pollingTimeIntervalMs = args(4).toInt

      RequestRunTimeConfig(requestTopic, responseTopic, pollingTimeIntervalMs)
    }
  }


  /**
    * Configuration for Kafka producer
    * @param requestTopic Topic for requests for prediction
    * @param ingestIntervalMs Polling interval for Kafka consumer in millis
    * @param numRequests Max number of requests
    * @param numRepeats Number of repeats
    * @param simulatorType Type of request (requests, feedbacks)
    */
  case class RequestGenConfig(
    s3RequestFile: String,
    requestTopic: String,
    ingestIntervalMs: Long,
    numRequests: Int,
    numRepeats: Int,
    simulatorType: String) extends RequestExecConfig

  /**
    * Companion object to convert command line arguments into PredictionRunTimeConfig
    */
  final object RequestGenConfig {

    @throws(clazz = classOf[IllegalArgumentException])
    def apply(args: Seq[String]): RequestGenConfig = {
      require(args.size == 5,
        s"""Incorrect argument ${args.mkString(" ")}\nShould be
           |'kafkaSimulator requestFile requestTopic numRequests simulatorType')""".stripMargin)
      val s3RequestFile = args(1)
      val requestTopic = args(2)
      val sendingTimeIntervalMs = 100
      val numRequests = args(3).toInt
      val numRepeats = 2
      val simulatorType = args(4)

      RequestGenConfig(s3RequestFile, requestTopic, sendingTimeIntervalMs, numRequests, numRepeats, simulatorType)
    }
  }
}

