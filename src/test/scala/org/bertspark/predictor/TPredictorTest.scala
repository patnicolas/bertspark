package org.bertspark.predictor

import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalRequest, MlEMRCodes, PRequest}
import org.bertspark.util.io.S3Util
import org.scalatest.flatspec.AnyFlatSpec



private[predictor] final class TPredictorTest extends AnyFlatSpec {


  ignore should "Succeed filtering a small set of requests" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = s"requestsProd/Cornerstone"
    val requests = S3Util.s3ToDataset[InternalRequest](s3Folder, false, "json").limit(6).collect()

    val context1 = requests.head.context
    val newContext1 = context1.copy(
      EMRCpts = Seq[MlEMRCodes](MlEMRCodes(0, "76700", Seq[String]("26"), Seq.empty[String], 1, "UN"))
    )
    val request1 = requests.head.copy(context = newContext1)

    val context4 = requests(4).context
    val newContext4 = context4.copy(
      EMRCpts = Seq[MlEMRCodes](MlEMRCodes(0, "93971", Seq[String]("26", "LT"), Seq.empty[String], 1, "UN"))
    )
    val request4 = requests.head.copy(context = newContext4)

    val sampledRequests = Seq[InternalRequest](request1, request4) ++ requests
    val pResponses = TPredictor().apiProcess(sampledRequests.map(PRequest(_)))
    println(pResponses.mkString("\n"))
  }


  ignore should "Succeed filtering requests" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = s"requestsProd/Cornerstone"
    val ds = S3Util.s3ToDataset[InternalRequest](s3Folder, false, "json").limit(40000)
    val cnt = ds.count()
    println(s"CNT: $cnt")
    val pRequests: Seq[PRequest] = ds
        .filter(req => req.context.emrLabel == "71046 26" || req.context.emrLabel == "76700 26")
        .map(PRequest(_))
        .collect()

    val pResponses = TPredictor().apiProcess(pRequests)
    println(pResponses.mkString("\n"))
  }
}

