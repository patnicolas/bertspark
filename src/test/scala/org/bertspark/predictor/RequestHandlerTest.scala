package org.bertspark.predictor

import org.bertspark.nlp.medical.MedicalCodingTypes.{MlEMRCodes, Pipe, PRequest, PRequestContext}
import org.scalatest.flatspec.AnyFlatSpec


private[predictor] final class RequestHandlerTest extends AnyFlatSpec {
    import RequestHandlerTest._

  it should "Succeed to load classifier models" in {
    val requestHandler = TPredictor()
    println(requestHandler.toString)
  }


  it should "Succeed to classify requests" in {
    val requestHandler = TPredictor()
    val requests = getRequests
    requestHandler.apiProcess(requests)
  }
}


private[predictor] final object RequestHandlerTest {
  def getRequests: Seq[PRequest] = {
    // "73721 26 RT"
    val emrCodes1 = Seq[MlEMRCodes](MlEMRCodes(0, "73721", Seq[String]("26", "RT"), Seq.empty[String], 0, "UN"))
    val request1 =  PRequest("1",emrCodes1, "Hello")

    val emrCodes2= Seq[MlEMRCodes](MlEMRCodes(0, "80911", Seq[String]("26"), Seq.empty[String], 0, "UN"))
    val request2 =  PRequest("2",emrCodes2, "Hello2")
  // 93923 26
    val emrCodes3= Seq[MlEMRCodes](MlEMRCodes(0, "93923", Seq[String]("26"), Seq.empty[String], 0, "UN"))
    val request3 =  PRequest("3",emrCodes3, "Hello3")
    Seq[PRequest](request1, request2, request3)
}


}
