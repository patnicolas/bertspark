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


import java.util.Properties
import org.apache.kafka.clients.producer._
import org.apache.kafka.common.errors.{InterruptException, SerializationException, TimeoutException}
import org.apache.kafka.common.KafkaException
import org.bertspark.kafka.{initialProperties, saslJaasConfigLabel}
import org.bertspark.kafka.TKafkaConfig.getParameterValue
import org.slf4j.{Logger, LoggerFactory}

/**
 * Generic wrapper for Typed Kafka producer
 * @param valueSerializerClass Serializer for the value sent by the producer
 * @param topic Topic to which produce Kafka responses
 * @keying Function to extract the key => tuple (key, entity)
 *
 * @author Patrick Nicolas
 * @version 0.5
 */
private[bertspark] final class TypedKafkaProducer[T](valueSerializerClass: String, topic: String)  {
  import TypedKafkaProducer._

  private[this] val producerProperties: Option[Properties] = getProducerProperties(valueSerializerClass)
  private[this] val kafkaProducer = {
    producerProperties.map(prop => {
      val kafkaProducer = new KafkaProducer[String, T](prop)
      kafkaProducer
    }).getOrElse(throw new IllegalStateException("Could not instantiate producer"))
  }

  /**
   * Send a wrapper to Kafka using 'round-robin' across partitions
   * @param event Wrapper to be produced to Kafka
   */
  def send(producingMessage: (String, T)): Unit = try {
    val (key, value) = producingMessage
    val producer = new ProducerRecord[String, T](topic, key, value)
    kafkaProducer.send(producer)
  }
  catch {
    case e: InterruptException =>
      logger.error(s"Producer interrupted: ${e.getMessage}")
    case e: SerializationException =>
      logger.error(s"Producer failed serialization: ${e.getMessage}")
    case e: TimeoutException =>
      logger.error(s"Producer time out: ${e.getMessage}")
    case e: KafkaException =>
      logger.error(s"Producer Kafka error: ${e.getMessage}")
  }

  def send(producingMessages: Seq[(String, T)]): Unit = producingMessages.foreach(send(_))

  final def close: Unit = kafkaProducer.close
}


/**
 * Singleton to retrieve Kafka producer properties
 * @todo Get Number of partitions from
 * @author Patrick Nicolas
 * @version 0.5
 */
private[bertspark] final object TypedKafkaProducer {
  final val logger: Logger = LoggerFactory.getLogger("TypedKafkaProducer")

  /**
   * Constructor for the producer configuration
   * @param valueDeserializerClass Class for the serializer of the value
   * @return Optional; Properties
   * @review Should I load a template properties file?
   */
  final def getProducerProperties(valueDeserializerClass: String): Option[Properties] =
    initialProperties.map(
      props => {
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer")
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, valueDeserializerClass)
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, getParameterValue(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, ""))
        props.put(ProducerConfig.LINGER_MS_CONFIG, getParameterValue(ProducerConfig.LINGER_MS_CONFIG, "60"))
        props.put(ProducerConfig.BATCH_SIZE_CONFIG, getParameterValue(ProducerConfig.BATCH_SIZE_CONFIG, "24576"))
        props.put(saslJaasConfigLabel, getParameterValue(saslJaasConfigLabel, ""))
        props.put(ProducerConfig.COMPRESSION_TYPE_CONFIG, getParameterValue(ProducerConfig.COMPRESSION_TYPE_CONFIG, "snappy"))
        props.put(ProducerConfig.ACKS_CONFIG, getParameterValue(ProducerConfig.ACKS_CONFIG, "all"))
        props
      }
    )
}
