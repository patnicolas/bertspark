package org.bertspark.kafka.simulator

import org.bertspark.delay
import org.bertspark.kafka.serde.FeedbackSerDe.FeedbackMessage
import org.bertspark.nlp.medical.MedicalCodingTypes.{FeedbackLineItem, InternalContext, InternalFeedback}
import org.scalatest.flatspec.AnyFlatSpec

private[simulator] final class FeedbackMessageGeneratorTest extends AnyFlatSpec {

  it should "Succeed producing a feedback message request" in {
    import org.bertspark.implicits._

    val feedbackLineItem = FeedbackLineItem(0, "77123",  Seq[String]("26"), Seq[String]("Z12.31"), 0, "UN", 0.0)
    val feedback = InternalFeedback("id", true, InternalContext(), feedbackLineItem)
    val feedbackMessage = FeedbackMessage(feedback)

    val numRequests = 20
    val producerTopic = "ml-request-mlops"
    val feedbackMessageGenerator = FeedbackMessageGenerator(producerTopic, numRequests)
    (0 until 10).foreach(
      index => feedbackMessageGenerator.produce((index.toString, feedbackMessage))
    )
    delay(5000L)
  }

  it should "Succeed producing feedback message to Kafka" in {
    import org.bertspark.implicits._

    val numRequests = 20
    val producerTopic = "ml-request-mlops"
    val s3FeedbackPath = s"feedbacks/Cornerstone"
    val feedbackMessageGenerator = FeedbackMessageGenerator(s3FeedbackPath, producerTopic, numRequests)
    feedbackMessageGenerator.execute
  }
}
