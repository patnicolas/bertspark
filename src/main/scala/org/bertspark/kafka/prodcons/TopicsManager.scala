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
package org.bertspark.kafka.prodcons

import java.util.Properties
import org.apache.kafka.clients.admin.{AdminClient, ConsumerGroupListing, NewTopic, TopicListing}
import org.bertspark.kafka.prodcons.TypedKafkaConsumer.getConsumerProperties
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.convert.ImplicitConversions.`collection AsScalaIterable`


/**
 * Manager of Kafka topics given a properties
 * @param properties Properties defined from the consumer
 * @author Patrick Nicolas
 * @version 0.2
 */
private[bertspark] final class TopicsManager private (properties: Properties) {
  import TopicsManager._


  final def describeTopic(topic: String): String = {
    import org.bertspark.implicits._

    val adminClient = AdminClient.create(properties)
    val describeTopicsResult = adminClient.describeTopics(Seq[String](topic))
    describeTopicsResult.values.toString
  }

  final def describeFeatures: scala.collection.mutable.Map[String, String] = {
    import org.bertspark.implicits._

    val adminClient = AdminClient.create(properties)
    val featureResult = adminClient.describeFeatures
    featureResult.featureMetadata().get.supportedFeatures.map{
      case (feature, value) => (feature, value.toString)
    }
  }


  final def listConsumerGroupIds: Seq[String] = {
    import org.bertspark.implicits._

    val adminClient = AdminClient.create(properties)
    val consumerGroups = adminClient.listConsumerGroups.all.get
    val groups: Seq[ConsumerGroupListing] = consumerGroups
    groups.map(_.groupId)
  }



  /**
   * List the current topics associated with this consumer
   * @return List of topics for this consumer
   */
  final def listTopics: Iterable[String] = {
    val adminClient = AdminClient.create(properties)
    val listingsFuture = adminClient.listTopics.listings
    val listings = listingsFuture.get
    val topics = listings.map(_.name)
    adminClient.close
    topics
  }

  /**
   * Create a new topic, if it does not exist, yet
   * @param topic Name of the topic to create
   * @param numPartitions Number of partitions (default 6)
   * @param numReplications Number of replications (default 3)
   * @return Updated list of topics
   */
  final def createTopic(
    topic: String,
    numPartitions: Int = defaultNumPartitions,
    numReplications: Short = defaultNumReplications): Iterable[String] = {
    import org.bertspark.implicits._

    val topicsSeq = listTopics.toSeq
    // If the topic does not exist, create it
    if(!topicsSeq.contains(topic)) {
      val adminClient = AdminClient.create(properties)
      val newTopic = new NewTopic(topic, numPartitions, numReplications)

      val results = adminClient.createTopics(scala.collection.immutable.List[NewTopic](newTopic))
      if(results.values.nonEmpty) topicsSeq ++ Seq[String](topic) else topicsSeq
    }
    else {
      logger.warn(s"Topic $topic already exists")
      topicsSeq
    }
  }


  /**
   * Delete a topic from Kafka service
   * @param topic Topic to be removed
   * @return Updated list of topics
   */
  final def deleteTopic(topic: String): Iterable[String] = {
    import org.bertspark.implicits._

    val adminClient = AdminClient.create(properties)
    adminClient.deleteTopics(Seq[String](topic))

    val listingsFuture = adminClient.listTopics.listings
    val listings: Seq[TopicListing] = listingsFuture.get
    val topics = listings.map(_.name)
    adminClient.close
    topics
  }
}


/**
 * Singleton for constructors and default values
 */
private[bertspark] final object TopicsManager {
  final val logger: Logger = LoggerFactory.getLogger("TopicsManager")

  final private val defaultNumPartitions: Int = 2
  final private val defaultNumReplications: Short = 3

  def apply(properties: Properties): TopicsManager = new TopicsManager(properties)

  def apply(valueDeserializerClass: String): Option[TopicsManager] = {
    val properties = getConsumerProperties(valueDeserializerClass)
    properties.map(new TopicsManager(_))
  }
}

