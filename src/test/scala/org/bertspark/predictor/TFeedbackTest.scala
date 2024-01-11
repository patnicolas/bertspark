package org.bertspark.predictor

import org.bertspark.nlp.medical.MedicalCodingTypes.{FeedbackLineItem, InternalContext, MlClaimEntriesWithCodes, MlEMRCodes, InternalFeedback}
import org.bertspark.predictor.TFeedbackTest.createFeedbacks
import org.scalatest.flatspec.AnyFlatSpec


private[predictor] final class TFeedbackTest extends AnyFlatSpec {


  it should "Succeed updating metrics for identical feedbacks" in {
    val tFeedback = new TFeedback
    tFeedback update createFeedbacks("same")

    val (strictMetrics, coreMetrics) = tFeedback.getMetrics
    assert(strictMetrics.accuracy == 1.0F)
    assert(strictMetrics.f1 == 1.0F)
    println(s"Strict metrics (identical):\n${strictMetrics.toString}")
    println(s"Core metrics (identical):\n${coreMetrics.toString}")
  }


  it should "Succeed updating metrics for different feedbacks" in {
    val tFeedback = new TFeedback
    tFeedback update createFeedbacks("different")

    val (strictMetrics, coreMetrics) = tFeedback.getMetrics
    assert(Math.abs(strictMetrics.accuracy - 2.0/3) < 0.0001)
    assert(Math.abs(strictMetrics.f1 - 0.8) < 0.0001)
    println(s"Strict metrics (different):\n${strictMetrics.toString}")
    println(s"Core metrics (different):\n${coreMetrics.toString}")
  }



  it should "Succeed updating metrics for core feedbacks" in {
    val tFeedback = new TFeedback
    tFeedback update createFeedbacks("core")

    val (strictMetrics, coreMetrics) = tFeedback.getMetrics
    assert(Math.abs(strictMetrics.accuracy - 0.5) < 0.0001)
    assert(Math.abs(strictMetrics.f1 - 2.0/3) < 0.0001)
    assert(Math.abs(coreMetrics.accuracy - 1.0) < 0.0001)
    assert(Math.abs(coreMetrics.f1 - 1.0) < 0.0001)
    println(s"Strict metrics (core):\n${strictMetrics.toString}")
    println(s"Core metrics (core):\n${coreMetrics.toString}")
  }


  it should "Succeed updating metrics for any feedbacks" in {
    val tFeedback = new TFeedback
    tFeedback update createFeedbacks("any")

    val (strictMetrics, coreMetrics) = tFeedback.getMetrics
    println(s"Strict metrics (any):\n${strictMetrics.toString}")
    println(s"Core metrics (any):\n${coreMetrics.toString}")
  }
}


private[predictor] final object TFeedbackTest {

  def createFeedbacks(testType: String): Seq[InternalFeedback] = {
    val emrCpts1 = Seq[MlEMRCodes](
      MlEMRCodes(0, "77889", Seq[String]("26", "LT"), Seq.empty[String])
    )
    val lineItems1 = Seq[FeedbackLineItem](
      FeedbackLineItem(0, "77889", Seq[String]("26", "LT"), Seq[String]("B89.11", "M78.12"), 0, "UN", 0.1),
      FeedbackLineItem(0, "G6738", Seq.empty[String], Seq[String]("I25.1"), 0, "UN", 0.1),
      FeedbackLineItem(0, "77000", Seq[String]("GC"), Seq[String]("Z12.31"), 0, "UN", 0.1),
    )
    val context1 = InternalContext("", 11, "M", "T", "Cu", "Cl", "M", "Pos", "Dos", emrCpts1, "", "", "", "", "", "")

    val feedback1 = InternalFeedback(
      "1",
      true,
      context1,
      MlClaimEntriesWithCodes(lineItems1),
      MlClaimEntriesWithCodes(lineItems1),
      MlClaimEntriesWithCodes()
    )

    val emrCpts2 = Seq[MlEMRCodes](
      MlEMRCodes(0, "66001", Seq[String]("GC"), Seq.empty[String])
    )
    val lineItems2 = Seq[FeedbackLineItem](
      FeedbackLineItem(0, "G6738", Seq.empty[String], Seq[String]("I25.1"), 0, "UN", 0.1),
      FeedbackLineItem(0, "66001", Seq[String]("GC"), Seq[String]("I10", "M78.99"), 0, "UN", 0.1)
    )
    val context2 = InternalContext("", 11, "M", "T", "Cu", "Cl", "M", "Pos", "Dos", emrCpts2, "", "", "", "", "", "")

    val feedback2 = InternalFeedback(
      "2",
      true,
      context2,
      MlClaimEntriesWithCodes(lineItems1),
      MlClaimEntriesWithCodes(lineItems2),
      MlClaimEntriesWithCodes()
    )

    val emrCpts3 = Seq[MlEMRCodes](
      MlEMRCodes(0, "77889", Seq[String]("26", "LT"), Seq.empty[String])
    )
    val lineItems3 = Seq[FeedbackLineItem](
      FeedbackLineItem(0, "77889", Seq[String]("26", "LT"), Seq[String]("B89.11", "Z12.31"), 0, "UN", 0.1),
      FeedbackLineItem(0, "G6738", Seq.empty[String], Seq[String]("I25.1"), 0, "UN", 0.1),
      FeedbackLineItem(0, "77000", Seq[String]("GC"), Seq[String]("Z12.31"), 0, "UN", 0.1),
    )
    val context3 = InternalContext("", 11, "M", "T", "Cu", "Cl", "M", "Pos", "Dos", emrCpts1, "", "", "", "", "", "")

    val feedback3 = InternalFeedback(
      "1",
      true,
      context1,
      MlClaimEntriesWithCodes(lineItems1),
      MlClaimEntriesWithCodes(lineItems3),
      MlClaimEntriesWithCodes()
    )

    testType match {
      case "same" => Seq[InternalFeedback](feedback1, feedback1, feedback1)
      case "different" => Seq[InternalFeedback](feedback1, feedback2, feedback1)
      case "core" => Seq[InternalFeedback](feedback1, feedback3)
      case _ => Seq[InternalFeedback](feedback1, feedback2, feedback3)
    }
  }
}
