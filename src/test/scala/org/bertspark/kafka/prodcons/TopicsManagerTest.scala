package org.bertspark.kafka.prodcons

import org.bertspark.delay
import org.bertspark.kafka.serde.{AckSerDe, FeedbackSerDe, RequestSerDe}
import org.scalatest.flatspec.AnyFlatSpec

private[prodcons] final class TopicsManagerTest extends AnyFlatSpec {

  ignore should "Succeed get topic description" in {
    val topic = "testPredictRequest"
    TopicsManager(RequestSerDe.deserializingClass).map(
      predictTopicManager => {
        println(predictTopicManager.describeTopic(topic))
      }
    )
  }

  ignore should "Succeed get configuration" in {
    TopicsManager(RequestSerDe.deserializingClass).map(
      predictTopicManager => {
        val keyValuePairs = predictTopicManager.describeFeatures
        println(keyValuePairs.mkString("\n"))
      }
    )
  }

  ignore should "Succeed retrieving list of group ids" in {
    TopicsManager(RequestSerDe.deserializingClass).map(
      predictTopicManager => {
        val groupIds = predictTopicManager.listConsumerGroupIds
        println(groupIds.mkString("\n"))
      }
    )
  }

  ignore should "Succeed listing topics" in {
    val predictTopicManager = TopicsManager(RequestSerDe.deserializingClass)
    val predictTopicsSeq = predictTopicManager.map(_.listTopics).getOrElse({
      println("Could not instantiate a topic manager")
      Iterable.empty[String]
    }).toSeq.sortWith(_ < _)
    println(s"List of ${predictTopicsSeq.size} current topics:\n${predictTopicsSeq.mkString("\n")}")
  }


  ignore should "Succeed create a topic" in {
    val newTopic = "ml-feedback-mlops"
    val topicManager = TopicsManager(FeedbackSerDe.deserializingClass)
    val topicsSeq = topicManager.map(_.createTopic(newTopic)).getOrElse({
      println(s"Could not instantiate a topic manager")
      Seq.empty[String]
    })
    println(topicsSeq.mkString(" "))

    assert(topicsSeq.nonEmpty)
    delay(4000L)
    val topicsSeq2 = topicManager.map(_.listTopics).getOrElse({
      println(s"Could not instantiate a topic manager")
      Seq.empty[String]
    })
    println(topicsSeq2.mkString("\n"))
  }


  it should "Succeed delete a topic" in {
    val topicManager = TopicsManager(FeedbackSerDe.deserializingClass)
    val newTopic = "ml-request-mlops"
    val topicsSeq3 = topicManager.map(_.deleteTopic(newTopic)).getOrElse({
      println(s"Could not instantiate a topic manager")
      Seq.empty[String]
    })
    println(topicsSeq3.mkString(" "))
  }


  ignore should "Succeed create and delete a topic" in {
    val newTopic = "new-topic"
    val topicManager = TopicsManager(RequestSerDe.deserializingClass)
    val topicsSeq = topicManager.map(_.createTopic(newTopic)).getOrElse({
      println(s"Could not instantiate a topic manager")
      Seq.empty[String]
    })

    println(topicsSeq.mkString(" "))
    assert(topicsSeq.nonEmpty == true)
    delay(4000L)
    val topicsSeq2 = topicManager.map(_.listTopics).getOrElse({
      println(s"Could not instantiate a topic manager")
      Seq.empty[String]
    })
    println(topicsSeq2.mkString(" "))

    val topicsSeq3 = topicManager.map(_.deleteTopic(newTopic)).getOrElse({
      println(s"Could not instantiate a topic manager")
      Seq.empty[String]
    })
    println(topicsSeq3.mkString(" "))
  }
}
