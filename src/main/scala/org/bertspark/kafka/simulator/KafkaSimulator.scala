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

import org.apache.spark.sql.SparkSession
import org.bertspark.config.S3PathNames
import org.bertspark.kafka.RequestGenConfig


/**
  * Generic generator for Kafka messages
  * {{{
  *  Commmand line
  *  kafkaSimulator s3RequestFile requestTopic ingestInterval numRequests numBatches $simulationType
  *     with simulationType {"requests", "feedbacks" }
  * }}}
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] final object KafkaSimulator {
  final val simulatedRequestLbl = "requests"
  final val simulatedFeedbackLbl = "feedbacks"
  final val simulatedJoinFeedbackLbl = "joinFeedbacks"

  /**
    * Product a prediction or feedback message to Kafka
    * @param genConf Request generator configuration
    * @return True if at least one message has been produced, false otherwise
    */
  def apply(genConf: RequestGenConfig)(implicit sparkSession: SparkSession): Unit = genConf.simulatorType match {
      case `simulatedRequestLbl` =>
        RequestMessageGenerator(
          genConf.s3RequestFile,
          genConf.requestTopic,
          genConf.ingestIntervalMs,
          genConf.numRequests,
          genConf.numRepeats).execute
      case `simulatedFeedbackLbl` =>
        FeedbackMessageGenerator(
          genConf.s3RequestFile,
          genConf.requestTopic,
          genConf.ingestIntervalMs,
          genConf.numRequests,
          genConf.numRepeats).execute
      case `simulatedJoinFeedbackLbl` =>
        FeedbackMessageGenerator(
          S3PathNames.s3PredictedClaimPath,
          genConf.s3RequestFile,
          genConf.requestTopic,
          genConf.ingestIntervalMs,
          genConf.numRequests).execute
      case _ => throw new UnsupportedOperationException(s"Simulator of ${genConf.simulatorType} is not supported ")
    }

  def apply(args: Seq[String])(implicit sparkSession: SparkSession): Unit = apply(RequestGenConfig(args))
}
