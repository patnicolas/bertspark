package org.bertspark.analytics

import org.apache.spark.rdd.RDD
import org.bertspark.analytics.Cleanser.modalityCorrector
import org.bertspark.analytics.CleanserTest.{OldFeedback, OldRequest}
import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalContext, InternalFeedback, InternalRequest, MlClaimEntriesWithCodes, MlEMRCodes}
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.{ContextualDocument, SubModelsTrainingSet}
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.bertspark.util.SparkUtil
import org.scalatest.flatspec.AnyFlatSpec


private[analytics] final class CleanserTest extends AnyFlatSpec{


  it should "Succeed extracting CMBS sub models with label with high number of records" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val feedbackDS = S3Util.s3ToDataset[InternalFeedback](
      "feedbacksProd/CMBS",
      false,
      "json"
    ).dropDuplicates("id").persist()

    val listFeedbackDS = feedbackDS.map(List[InternalFeedback](_))

    val subModelsGroupedByAveRecordsPerLabel: RDD[(String, Float, Iterable[List[String]])] =
      SparkUtil.groupBy[List[InternalFeedback], String](
      (feedbacks: List[InternalFeedback]) => feedbacks.head.context.emrLabel,
      (f1: List[InternalFeedback], f2: List[InternalFeedback]) => f1 ::: f2,
      listFeedbackDS
    ).map(
      xs => {
        val labelCountMap: Iterable[(String, Int, List[String])] = xs.groupBy(_.toFinalizedSpace)
            .map{ case (label, records) => (label, records.size, records.map(_.id)) }

        val aveNumRecords: Float =
          if(labelCountMap.nonEmpty) labelCountMap.map(_._2).reduce(_ + _).toFloat/ labelCountMap.size
          else 0.0F

        val subModelName = xs.head.context.emrLabel
        val ids = labelCountMap.map(_._3)
        (subModelName, aveNumRecords, ids)
      }
    )

    val subModelsWithHighestNumRecords = subModelsGroupedByAveRecordsPerLabel.collect().sortWith(_._2 < _._2).take(4)
    val ids: Array[String] = subModelsWithHighestNumRecords.flatMap(_._3).flatten
    println(s"${ids.mkString(" ")}")

    val selectedRequestDS = S3Util.s3ToDataset[InternalRequest](
      "requestsProd/CMBS",
      false,
      "json"
    ).dropDuplicates("id").filter(req => ids.contains(req.id))

    val selectedFeedbackDS = feedbackDS.filter(feedback => ids.contains(feedback.id))

    S3Util.datasetToS3[InternalRequest](
      selectedRequestDS,
      "requestsProd/SELECT",
      false,
      "json",
      false,
      8
    )

    S3Util.datasetToS3[InternalFeedback](
      selectedFeedbackDS,
      "feedbacksProd/SELECT",
      false,
      "json",
      false,
      8
    )
  }

  ignore should "Succeed augmenting the dictionary" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    import org.bertspark.config.MlopsConfiguration._

    val tokensDS = S3Util.s3ToDataset[ContextualDocument](
      "mlops/ALL/contextDocument/AMA",
      false,
      "json"
    ).flatMap(_.text.split(tokenSeparator).filter(_.startsWith("##")))

    val existingVocabulary = S3Util.download(
      mlopsConfiguration.storageConfig.s3Bucket,
      "mlops/ALL/vocabulary/AMA"
    )
    val newVocabulary = existingVocabulary.map(_.split("\n")).map(lines => (lines ++ tokensDS.collect).sortWith(_ < _))
    newVocabulary.foreach(
      lines => {
        val content = lines.distinct.mkString("\n")
        S3Util.upload(
          "mlops/ALL/vocabulary/AMA3",
          content
        )
      }
    )
  }

  ignore should "Succeed join contextual document with specific dataset" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val contextualDocumentDS = S3Util.s3ToDataset[SubModelsTrainingSet](
      "mlops/CMBS/training/AMA",
      false,
      "json"
    ).flatMap(_.labeledTrainingData.map(_.contextualDocument.id))

    val internalRequestDS = S3Util.s3ToDataset[InternalRequest](
      "requestsProd/CMBS",
      false,
      "json"
    ).dropDuplicates("id").map(_.id)

    val contextualDocIds = contextualDocumentDS.collect()
    val internalRequestIds = internalRequestDS.collect().toSet
    val intersection = contextualDocIds.filter(internalRequestIds.contains(_))
    println(s"Overlap of ${intersection.size} records")
  }

  ignore should "Succeed creating a small customer request" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val requestDS = S3Util.s3ToDataset[InternalRequest](
      "requestsProd/CMBS",
      false,
      "json"
    ).dropDuplicates("id").filter(
      req => {
        val subModelSet = Set[String]("9310", "76642 LT", "76642 RT", "73562 LT")
        subModelSet.contains(req.context.emrLabel)
      }
    )

    val requestsSet = requestDS.map(_.id).collect.toSet
    println(s"${requestsSet.size} requests for SMALL")

    val feedbackDS = S3Util.s3ToDataset[InternalFeedback](
      "feedbacksProd/CMBS",
      false,
      "json"
    ).dropDuplicates("id")
        .filter(
          req => {
            val subModelSet = Set[String]("9310", "76642 LT", "76642 RT", "73562 LT")
            subModelSet.contains(req.context.emrLabel)
          }
        )
        .filter(feedback => requestsSet.contains(feedback.id))

    println(s"${feedbackDS.count} feedbacks for SMALL")

    S3Util.datasetToS3[InternalRequest](
      requestDS,
      "requestsProd/SMALL2",
      false,
      "json",
      false,
      4
    )

    S3Util.datasetToS3[InternalFeedback](
      feedbackDS,
      "feedbacksProd/SMALL2",
      false,
      "json",
      false,
      4
    )
  }

  ignore should "Succeed converting request type" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val inputDS = S3Util.s3ToDataset[OldRequest](
      "aideo-tech-autocoding-v1",
      "requests/40/7/cmb",
      false,
      "json"
    ).filter(_.context.EMRCpts.nonEmpty)
    inputDS.show

    val newRequestDS = inputDS.filter(_.notes.head.size > 1024).map(
      oldRequest => {
        import CleanserTest._
        val newRequests: InternalRequest = oldRequest
        newRequests
      }
    )

    S3Util.datasetToS3[InternalRequest](
      newRequestDS,
      "cmbs",
      false,
      "json",
      false,
      16
    )
  }

  ignore should "Succeed converting feedback type" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val inputDS = S3Util.s3ToDataset[OldFeedback](
      "aideo-tech-autocoding-v1",
      "feedbacks/40/7/cmb",
      false,
      "json"
    ).filter(_.context.EMRCpts.nonEmpty)
    inputDS.show

    val newFeedbackDS = inputDS.map(
      oldRequest => {
        import CleanserTest._
        val newRequests: InternalFeedback = oldRequest
        newRequests
      }
    )

    newFeedbackDS.show
    S3Util.datasetToS3[InternalFeedback](
      newFeedbackDS,
      "cmbsFeedback",
      false,
      "json",
      false,
      16
    )
  }

  ignore should "Succeeds cleansing the feedback entry" in {
    import org.bertspark.implicits._

    Cleanser.cleanseFeedbackLineItem("feedbacks/Cornerstone")
  }

  ignore should "Succeed cleansing requests" in {
    import org.bertspark.implicits._
    Cleanser.cleanseRequest("requests/Cornerstone", modalityCorrector)
  }
}

object CleanserTest {
  case class OldContext(
    claimType: String,
    sectionMode: String,
    sectionGroups: Seq[String],
    age: Int,
    gender: String,
    taxonomy: String,
    placeOfService: String,
    dateOfService: String,
    EMCode: String,
    EMRCpts: Seq[MlEMRCodes],
    providerId: String,
    patientId: String,
    sectionHeaders: Seq[String],
    planId: String
  )

  val customer = "CMBS"
  val client = "Any"

  implicit def old2NewContext(oldContext: OldContext): InternalContext = {
    val modality = modalityFromCPT.getOrElse(oldContext.EMRCpts.head.cpt, "unknown")
    InternalContext(
      oldContext.claimType,
      oldContext.age,
      oldContext.gender,
      oldContext.taxonomy,
      customer,
      client,
      modality,
      oldContext.placeOfService,
      oldContext.dateOfService,
      oldContext.EMRCpts,
      oldContext.providerId,
      oldContext.patientId,
      oldContext.planId,
      "",
      "",
      ""
    )
  }

  case class OldRequest(id: String, context: OldContext, notes: Seq[String])

  case class OldFeedback(
    id: String,
    autoCodable: Boolean,
    context: OldContext,
    autocoded: MlClaimEntriesWithCodes,
    finalized: MlClaimEntriesWithCodes,
    audited: MlClaimEntriesWithCodes
  )


  implicit def old2NewFeedback(oldFeedback: OldFeedback): InternalFeedback = {
    val newContext: InternalContext = oldFeedback.context
    InternalFeedback(
      oldFeedback.id,
      oldFeedback.autoCodable,
      newContext,
      oldFeedback.autocoded,
      oldFeedback.finalized,
      oldFeedback.audited
    )
  }

  implicit def old2NewRequest(oldRequest: OldRequest): InternalRequest = {
    val newContext: InternalContext = oldRequest.context
    InternalRequest(oldRequest.id, newContext, oldRequest.notes)
  }

  val modalityFromCPT = LocalFileUtil.Load.local(s"conf/cpt-modality.csv")
      .map(
        content => {
          val lines = content.split("\n")
          lines.map(
            line => {
              val ar = line.split(",")
              (ar.head.trim, ar(1).trim)
            }
          ).toMap
        }
      ).getOrElse(Map.empty[String, String])
}
