package org.bertspark.nlp.medical

import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalFeedback, InternalRequest}
import org.bertspark.util.io.S3Util
import org.scalatest.flatspec.AnyFlatSpec
import LegacyCodingTypes._


private[medical] final class LegacyCodingTypesTest extends AnyFlatSpec{

  ignore should "Succeed converting old request to new request" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val customer = "CMBS"
    val inputDS = S3Util.s3ToDataset[OldRequest](
      "aideo-tech-autocoding-v1",
      "requests/40/7/cmb",
      false,
      "json"
    ).filter(_.context.EMRCpts.nonEmpty)
    inputDS.show

    val newRequestDS = inputDS.filter(_.notes.head.size > 512).map(
      oldRequest => {
        val newRequests: InternalRequest = old2NewRequest(oldRequest, customer)
        newRequests
      }
    )

    S3Util.datasetToS3[InternalRequest](
      newRequestDS,
      s"requests/$customer",
      false,
      "json",
      false,
      32
    )
  }

  it should "Succeed converting old feedback to new feedback" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val customer = "CMBS"
    val inputDS = S3Util.s3ToDataset[OldFeedback](
      "aideo-tech-autocoding-v1",
      "feedbacks/40/7/cmb",
      false,
      "json"
    ).filter(_.context.EMRCpts.nonEmpty)
    inputDS.show

    val newFeedbackDS = inputDS.map(
      oldRequest => {
        val newRequests: InternalFeedback = old2NewFeedback(oldRequest, customer)
        newRequests
      }
    )

    S3Util.datasetToS3[InternalFeedback](
      newFeedbackDS,
      s"feedbacks/$customer",
      false,
      "json",
      false,
      32
    )
  }
}
