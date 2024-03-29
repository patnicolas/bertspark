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

import java.io.IOException
import java.time.Duration
import java.util._
import org.apache.kafka.clients.consumer._
import org.apache.kafka.common.errors.WakeupException
import org.bertspark.kafka.TKafkaConfig.getParameterValue
import org.bertspark.kafka.{initialProperties, saslJaasConfigLabel}
import org.slf4j._


/**
 * Generic, typed Kafka consumer
 * The kafka consumer is responsible to keep the connectivity to Kafka service open
 * @param valueDeserializerClass Class used to deserialize the value retrieved from Kafka
 * @param pollTimeIntervalMs Timeout for polling in milliseconds
 * @tparam T Type of the value or payload
 * @see https://agilesde.atlassian.net/wiki/spaces/DS/pages/2419195989/Kafka+interfaces
 *
 * @author Patrick Nicolas
 * @version 0.5
 */
@throws(clazz = classOf[IllegalStateException])
private[bertspark] case class TypedKafkaConsumer[T](
  valueDeserializerClass: String,
  kafkaTopic: String) {
  import TypedKafkaConsumer._

  private[this] var isStarted = false
  private[this] val consumerProperties = getConsumerProperties(valueDeserializerClass)
  private[this] val kafkaConsumer =
    consumerProperties.map(new KafkaConsumer[String, T](_)).getOrElse(
      throw new IllegalStateException("Could not instantiate consumer")
    )
  kafkaConsumer.subscribe(Collections.singletonList(kafkaTopic))


  private[this] val pollTimeDurationMs: Duration = {
    val pollTimeInterval = consumerProperties.map(_.getProperty("poolTimeIntervalMs").toInt).getOrElse(5200)
    Duration.ofMillis(pollTimeInterval)
  }

  final def close: Unit = kafkaConsumer.close

  @inline
  def syncCommit: Unit = kafkaConsumer.commitSync()

  @inline
  def asyncCommit: Unit = kafkaConsumer.commitAsync()



  /**
   * Main loop to process event records. Topic are extracted dynamically
   * @param kafkaEventRecord Received a single Kafka event
   */
  def receive: Seq[(String, T)] =
      // Iterate through records assuming auto-commit of offset.
    try {
      import scala.collection.convert.ImplicitConversions.`iterable AsScalaIterable`
      // Simple health check to start
      check
      val consumerRecords: ConsumerRecords[String, T] = kafkaConsumer.poll(pollTimeDurationMs)
      val records = consumerRecords.records(kafkaTopic)
      // If records have been detected.....
      if(records.size > 0) {
        logger.info(s"Consume ${records.size} messages")
        records.map(record => (record.key, record.value)).toSeq
      }
      else
        Seq.empty[(String, T)]
    }
        // Manual commit of offsets...
    catch {
      case e: WakeupException =>
        logger.error(s"To wake up ${e.getMessage}")
        kafkaConsumer.wakeup
        Seq.empty[(String, T)]

      case e: UnsatisfiedLinkError =>
        logger.error(s"Unsatisfied link ${e.getMessage}")
        e.printStackTrace()
        Seq.empty[(String, T)]

      case e: IOException =>
        logger.error(s"Native file not found ${e.getMessage}")
        e.printStackTrace()
        Seq.empty[(String, T)]

      case e: Exception =>
        logger.error(s"Generic exception ${e.getMessage}")
        e.printStackTrace()
        Seq.empty[(String, T)]
    }

  private def check: Unit =
    if(!isStarted) {
      logger.info(s"\nReady to consume $kafkaTopic with pooling duration of ${pollTimeDurationMs.toString} msecs. .....")
      isStarted = true
    }
}


/**
 * Singleton to retrieve Consumer properties
 * @author Patrick Nicolas
 * @version 0.6
 */
private[bertspark] final object TypedKafkaConsumer {
  final val logger: Logger = LoggerFactory.getLogger("TypedKafkaConsumer")

  /**
   * Constructor for the consumer configuration
   * {{{
   *   BOOTSTRAP_SERVERS_CONFIG
   *   GROUP_ID_CONFIG
   *   ENABLE_AUTO_COMMIT_CONFIG
   *   AUTO_COMMIT_INTERVAL_MS_CONFIG
   *   SESSION_TIMEOUT_MS_CONFIG
   * }}}
   * @param valueDeserializerClass Class for the deserializer of the value
   * @return Properties
   * @review Should I load a template properties file?
   */
  final def getConsumerProperties(valueDeserializerClass: String): Option[Properties] =
    initialProperties.map(
      props => {
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, getParameterValue(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, ""))
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringDeserializer")
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, valueDeserializerClass)
        props.put("poolTimeIntervalMs", getParameterValue("poolTimeIntervalMs", "5200"))
        props.put(saslJaasConfigLabel, getParameterValue(saslJaasConfigLabel, ""))
        props.put(ConsumerConfig.GROUP_ID_CONFIG, getParameterValue(ConsumerConfig.GROUP_ID_CONFIG, "group-1"))
        props.put(ConsumerConfig.FETCH_MAX_BYTES_CONFIG, getParameterValue(ConsumerConfig.FETCH_MAX_BYTES_CONFIG, "52428800"))
        props.put(ConsumerConfig.MAX_POLL_RECORDS_CONFIG, getParameterValue(ConsumerConfig.MAX_POLL_RECORDS_CONFIG, "512"))
        props.put(ConsumerConfig.MAX_PARTITION_FETCH_BYTES_CONFIG, getParameterValue(ConsumerConfig.MAX_PARTITION_FETCH_BYTES_CONFIG, "1048576"))
        props
      }
    )
}

