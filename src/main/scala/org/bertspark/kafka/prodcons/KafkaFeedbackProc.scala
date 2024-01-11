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
package org.bertspark.kafka.prodcons

import org.apache.kafka.common.errors.{SerializationException, TimeoutException}
import org.apache.kafka.common.KafkaException
import org.apache.spark.SparkException
import org.bertspark.analytics.MetricsCollector
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.kafka.prodcons.KafkaFeedbackProc.feedbackTransform
import org.bertspark.kafka.serde.AckSerDe.AckMessage
import org.bertspark.kafka.serde.{AckSerDe, FeedbackSerDe}
import org.bertspark.kafka.serde.FeedbackSerDe.FeedbackMessage
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalFeedback
import org.bertspark.predictor.TFeedback
import org.slf4j.{Logger, LoggerFactory}


/**
  * Kafka processor for feedback (requests) and Ackknowledgment (response)
  * @param consumer Consumer of feedback requests
  * @param producer Producer of Acknowledgement
  * @param transform Default pre-defined Transform for feedback
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] final class KafkaFeedbackProc(
  override val consumer: TypedKafkaConsumer[FeedbackMessage],
  override val producer: TypedKafkaProducer[AckMessage]) extends KafkaProc[FeedbackMessage, AckMessage]{
  override val transform: Seq[FeedbackMessage] => Seq[AckMessage] = feedbackTransform
}


/**
  * Singleton for the constructor and execution of batch of requests
  */
private[bertspark] final object KafkaFeedbackProc {
  final private val logger: Logger = LoggerFactory.getLogger("KafkaFeedbackProc")
  final private val maxNumCycles = 20
  lazy val tFeedback = new TFeedback

  final private def feedbackTransform: Seq[FeedbackMessage] => Seq[AckMessage] =
    (feedbackMessages: Seq[FeedbackMessage]) => {
      // Just return default Acknowledgment
      val feedbacks: Seq[InternalFeedback] = feedbackMessages.map(_.payload)
      tFeedback.update(feedbacks)
      Seq.fill(feedbackMessages.length)(AckMessage("OK"))
    }

  /**
    * Constructor of Kafka feedback proc using topic
    * @param consumeTopic topic for consumer of feedback requests
    * @param produceTopic topic for producing acknowledgment
    * @return Instance of Kafka feedback proc
    */
  def apply(consumeTopic: String, produceTopic: String): KafkaFeedbackProc = {
    // Build the Kafka consumer for prediction request
    val consumer = new TypedKafkaConsumer[FeedbackMessage](FeedbackSerDe.deserializingClass, consumeTopic)
    // Build the Kafka producer for prediction response
    val producer = new TypedKafkaProducer[AckMessage](AckSerDe.serializingClass, produceTopic)
    new KafkaFeedbackProc(consumer, producer)
  }


  /**
    * Execute a batch of messages consumed from Kafka queue if the execution Mode, executors.executionMode is set
    * to 'kafka-spark'
    * {{{
    * Exceptions to be handled:
    *   - SparkException
    *   - IllegalStateException
    *   - InterruptedException
    *   - SerializationException
    *   - TimeoutException
    *   - KafkaException
    *
    *  Data flow:
    *   Step 1: Consume a batch of Kafka message (prediction or feedback requests)
    *   Step 2: Apply the transformation (execute pipeline)
    *   Step 3: Produce Kafka message (prediction and feedback response)
    *   Step 4: Commit the offsets
    * }}}
    *
    * @param consumeTopic Topic for consuming messages from (requests)
    * @param produceTopic Topic to produce messages to (response)
    * @param maxNumResponses Minimum number of expected responses. (-1 for no limit)
    */
  def executeBatch(
    consumeTopic: String,
    produceTopic: String,
    maxNumResponses: Int = -1): Int = {

    logDebug(logger, msg = s"Execute feedbacks for consumer $consumeTopic and producer $produceTopic")
    val kafkaHandler = KafkaFeedbackProc(consumeTopic, produceTopic)
    var counter = 0
    var numCycles = 0

    while((maxNumResponses == -1 || counter < maxNumResponses) && numCycles < maxNumCycles) {
      numCycles += 1
      logDebug(logger, msg = s"Cycle #$numCycles")

      // Pool the request topic (has its own specific Kafka exception handler)
      val consumerRecords = kafkaHandler.consumer.receive
      if (consumerRecords.nonEmpty) {
        // Generate and apply transform to the batch
        val start = System.currentTimeMillis()
        logger.info(s"Start processing ${consumerRecords.size} feedbacks")

        try {   // Exception related to processing
          val acks = feedbackTransform(consumerRecords.map(_._2))
          val responses = acks.zip(consumerRecords).map{ case (ack, consumerRec) => (consumerRec._1, ack)}

          if(responses.nonEmpty) {
            counter += responses.size

            val duration = System.currentTimeMillis - start
            logger.info(s"Executed in $duration msecs, average: ${duration * 0.001 / responses.size} secs, total: $counter")
            kafkaHandler.producer.send(responses)
            // It is assumed nothing goes wrong after this point
            kafkaHandler.consumer.asyncCommit
          }
          else
            logger.warn("No acknowledgement is generated!")
        }
        catch {
          case e: SparkException =>
            logger.error(s"Spark failure: ${e.getMessage}")
          case e: IllegalStateException =>
            logger.error(s"Illegal state: ${e.getMessage}")
          case e: InterruptedException =>
            logger.error(s"Producer interrupted: ${e.getMessage}")
          case e: SerializationException =>
            logger.error(s"Producer failed serialization: ${e.getMessage}")
          case e: TimeoutException =>
            logger.error(s"Producer time out: ${e.getMessage}")
          case e: KafkaException =>
            logger.error(s"Producer Kafka error: ${e.getMessage}")
          case e: Exception =>
            logger.error(s"Undefined Kafka error: ${e.getMessage}")
        }
      }
    }
    tFeedback.save

    logger.warn(s"Exit Kafka prediction handler and close $consumeTopic after $counter messages")
    kafkaHandler.close
    counter
  }

}