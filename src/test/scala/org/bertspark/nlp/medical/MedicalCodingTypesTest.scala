package org.bertspark.nlp.medical

import org.bertspark.analytics.MetricsCollector
import org.bertspark.analytics.MetricsCollector.metrics
import org.bertspark.nlp.medical.MedicalCodingTypes.{lineItemSeparator, FeedbackLineItem, InternalFeedback, MlEMRCodes, MlLineItem}
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalFeedback.removeDuplicateLineItems
import org.scalatest.flatspec.AnyFlatSpec

private[medical] final class MedicalCodingTypesTest extends AnyFlatSpec {


  ignore should "Succeed stringize a line item" in {
    val lineItem = MlLineItem(0, "89913", Seq[String]("26", "RT"),  Seq[String]("R61.2", "J77.4"), 1, "UN", 0.0, "0")
    val lineItemSpace = lineItem.toCodesSpace
    val lineItemComma = lineItem.toCodesComma
    println(s"Line item Space: $lineItemSpace")
    println(s"Line item comma: $lineItemComma")

    val revertedLineItem = MlLineItem.getLineItem(lineItemSpace)
    println(s"Line item from space: ${revertedLineItem.toString}")

    val revertedLineItem2 = MlLineItem(lineItemComma)
    println(s"Line item from comma: ${revertedLineItem2.toString}")
  }

  it should "Succeed stringize a feedback line item" in {
    val lineItem = FeedbackLineItem(0, "89913", Seq[String]("26", "RT"),  Seq[String]("R61.2", "J77.4"), 1, "UN", 0.0)
    val lineItemSpace = lineItem.lineItemSpace
    val lineItemComma = lineItem.lineItemComma
    println(s"Line item Space: $lineItemSpace")
    println(s"Line item comma: $lineItemComma")

    val lineItem2 = FeedbackLineItem(0, "G3297", Seq[String]("26"),  Seq[String]("Z00.8"), 1, "UN", 0.0)
    val lineItemsStr = FeedbackLineItem.str(Seq[FeedbackLineItem](lineItem, lineItem2))
    println(s"Feedback line items: $lineItemsStr")

    val revertedLineItem = FeedbackLineItem(lineItemSpace)
    println(s"Reverted feedback line item: ${revertedLineItem.toString}")

    val revertedLineItems = FeedbackLineItem.toLineItems(lineItemsStr)
    println(s"Reverted line items:\n${revertedLineItems.mkString("\n")}")
  }


  ignore should "Succeed compare line items" in {
    val lineItem1 = MlLineItem(0, "89913", Seq[String]("26", "RT"),  Seq[String]("R61.2", "J77.4"), 1, "UN", 0.0, "0")
    val lineItem2 = MlLineItem(0, "89913", Seq[String]("26"),  Seq[String]("R61.2", "J77.4"), 1, "UN", 0.0, "0")
    val lineItem3 = MlLineItem(0, "89913", Seq[String]("26", "RT"),  Seq[String]("R61.2"), 1, "UN", 0.0, "0")
    val lineItem4 = MlLineItem(0, "89913", Seq.empty[String],  Seq[String]("R61.2"), 1, "UN", 0.0, "0")
    val lineItem5 = MlLineItem(0, "89913", Seq.empty[String],  Seq[String]("R61.2"), 1, "UN", 0.0, "0")
    val lineItem6 = MlLineItem(0, "89913", Seq.empty[String],  Seq[String]("R61.2", "J77.4"), 1, "UN", 0.0, "0")

    assert(lineItem1.isEqual(lineItem1))
    assert(!lineItem1.isEqual(lineItem2))
    assert(!lineItem1.isEqual(lineItem3))
    assert(lineItem4.isEqual(lineItem5))
    assert(lineItem4.isEqual(lineItem5))
    assert(!lineItem1.isEqual(lineItem6))
  }

  ignore should "Succeed extracting Feedback line items" in {
    val feedbackStr = "77891 26 GC M67.11 Z12.13"
    val lineItem = FeedbackLineItem(feedbackStr)
    println(lineItem.toString)
    val feedbackStr2 = "90122 I10.2"
    val lineItems = FeedbackLineItem.toLineItems(s"$feedbackStr$lineItemSeparator$feedbackStr2")
    println(lineItems.mkString("\n"))
  }


  ignore should "Succeed removing duplicates in feedback" in {
    val feedbackLineItem1 = FeedbackLineItem(0, "77123",  Seq[String]("26"), Seq[String]("Z12.31"), 0, "UN", 0.0)
    val feedbackLineItem2 = FeedbackLineItem(0, "77881",  Seq[String]("26", "LT"), Seq[String]("M78.919"), 0, "UN", 0.0)
    val feedbackLineItem3 = FeedbackLineItem(0, "77881",  Seq[String]("26", "LT"), Seq[String]("M78.919"), 0, "UN", 0.0)

    val feedbackLineItems = Seq[FeedbackLineItem](feedbackLineItem1, feedbackLineItem2, feedbackLineItem3)
    val mlEmrCodes =  Seq[MlEMRCodes](MlEMRCodes(0, "90012", Seq.empty[String], Seq.empty[String], 1, "UN"))

    val feedback: InternalFeedback = InternalFeedback.apply("id", mlEmrCodes, feedbackLineItems)
    val correctedFeedback = removeDuplicateLineItems(feedback)
    println(correctedFeedback.toString)
  }

  ignore should "Succeed not removing duplicates" in {
    val feedbackLineItem1 = FeedbackLineItem(0, "77123",  Seq[String]("26"), Seq[String]("Z12.31"), 0, "UN", 0.0)
    val feedbackLineItem2 = FeedbackLineItem(0, "77881",  Seq[String]("26", "LT"), Seq[String]("M78.919"), 0, "UN", 0.0)

    val feedbackLineItems = Seq[FeedbackLineItem](feedbackLineItem1, feedbackLineItem2)
    val mlEmrCodes =  Seq[MlEMRCodes](MlEMRCodes(0, "90012", Seq.empty[String], Seq.empty[String], 1, "UN"))

    val feedback: InternalFeedback = InternalFeedback.apply("id", mlEmrCodes, feedbackLineItems)
    val correctedFeedback = removeDuplicateLineItems(feedback)
    println(correctedFeedback.toString)
  }
}


