package org.bertspark.util.rdbms

import org.bertspark.nlp.medical.MedicalCodingTypes.MlLineItem
import org.bertspark.predictor.TPredictor.PredictedClaim
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.scalatest.flatspec.AnyFlatSpec

private[rdbms] final class PredictedClaimTableTest extends AnyFlatSpec {

  ignore should "Succeed create Prediction claim table" in {
    val postgreSql = PostgreSql()
    val table = PredictedClaimTbl(postgreSql)
    assert(table.isReady)
    table.close
  }

  it should "Succeed insert entry in prediction claim table" in {
    val id = "1"
    val age = 45
    val gender = "M"
    val customer = "PSI"
    val client = "SERL"
    val procedureCategory = "DIAGNOSTIC"
    val emr = "72040 26"
    val pos = "11"
    val dos = "2023-03-02"
    val note = """\\r\\n (\)================================================================================\\r\\n\\r\\n                              MARY LANNING 'HEALTHCARE\\r\\n\\r\\nPatient Name: Todd A Johnson\\r\\nBirth Date: 10/14/1960\\r\\nSSN: 506-72-2187\\r\\nReferring Doctor: ELLIOT, BRIAN K\\r\\nReading Doctor: Rodriguez, Paul \\r\\nVisit No.: 3100582096\\r\\nOrder No.: RD0680158\\r\\nExam Date: 06/03/2019 00:00:00\\r\\nExam: 72192 - CT Bony Pelvis\\r\\n================================================================================\\r\\n\\r\\nNo acute fracture. I agree with preliminary report.\\r\\nCT BONY PELVIS HISTORY: Fall, R/O occult frx. COMPARISON: None. TECHNIQUE: CT of the pelvis. FINDINGS: There is mild osteopenia. There is osteophytosis of the right SI joints and at L5-S1. Posterior osteophyte disc complex at L5-S1 results in mild spinal stenosis. There is no free air. No acute fracture.\\r\\n\\r\\nElectronically Signed By: Vanderlijn, Pieter, MD\\r\\nSigned Date: 06/04/2019 7:46 AM CD"""


    val autoCodeState = 5
    val lineItems = Seq[MlLineItem](
      MlLineItem(0, "72040", Seq[String]("26"), Seq[String]("M78.1"), 1, "UN", 0.4, "1"),
      MlLineItem(1, "73610", Seq[String]("26", "LT"), Seq[String]("R10.9"), 1, "UN", 0.4, "1")
    )
    val latency = 98L
    val predictionClaim1 = PredictedClaim(
      id, age, gender, customer, client, procedureCategory, emr, pos, dos, note.replaceAll("\\)", "").replaceAll("'", ""), autoCodeState, lineItems, latency)

    val postgreSql = PostgreSql()
    PredictedClaimTbl.insertPrediction(Seq[PredictedClaim](predictionClaim1, predictionClaim1), postgreSql)
    postgreSql.close
  }

  ignore should "Succeed inserting a prediction from JSON" in {
    val jsonInput =
      s"""
         |\r\n (\) ================================================================================\r\n\r\n                              THE CHILDRENS HOSPITAL OF AL\r\n\r\nPatient Name: CARLISLE RASHAAD\r\nReferring Doctor: HEAD, RONA \r\nReading Doctor: ORTIZ, CLARA L\r\nVisit No.: 12590031\r\nOrder No.: 3998650\r\nExam Date: 12/01/2022 00:00:00\r\nExam: 73130 - BRD - HAND, COMPLETE - LEFT\r\n================================================================================\r\n\r\n		\r\n\r\nPatient Name:  	CARLISLE, RASHAAD  	MR #:  	     1419972\r\n	2412 BROOKHAVEN AVE SW	Clinic/SVC	    EMR	\r\n	BIRMINGHAM, AL 35211	Adm #:  	    12590031\r\n	(205)203-3468	Discharge D/T:	    Dec  1 2022	\r\nDOB:  02/07/2010    Age:   12Y   M     	ER Patient   -		\r\n\r\nOrder Dr.:  STAFF PHYSICIAN ED	Attending Dr.:  EMILY SKOOG	Alternate Dr.:  STAFF PHYSICIAN ED\r\n\r\nFOR QUESTIONS, CONCERNS, OR FOLLOW-UP PLEASE CALL (205) 638-9730\r\n\r\n\r\nTech Inits:  ETAYLO                \r\nInterpretating Physician:  CLARA  LUCIA ORTIZ on Dec  1 2022  8:02A	\r\nSigned: CLARA LUCIA ORTIZ on Dec  1 2022  8:07A	\r\n	\r\nELECTRONICALLY SIGNED ___________________________________________________________________________________________________________\r\n*** Final Report ***\r\n\r\nSEPARATE RESULT?\r\nREADING DOCTOR:  \r\nREVIEWING DOCTORS(S):  \r\nREADING DATE:  \r\nTRANSCRIBED BY:  \r\nSPECIAL CASE:  \r\nRELEASE RESULTS: (Y)\r\nBEGIN INTERFACE RESULT:\r\nPROCEDURE:  BRD - HAND, COMPLETE - LEFT	ACCESSION#  3998650\r\n\r\nPROCEDURE DATE:  12/01/2022   07:45	CPT CODE:  73130\r\nREASON:	LEFT FINGER INJURY; LEFT HAND INJURY; \r\n\r\nSTART RESULTS:\r\nACCESSION:  (3998650)\r\nCRITICAL VALUE: \r\nTEACHING CASE:  \r\nACR CODES(S):\r\nCLINICAL INFORMATION: 12 years old Male Injury. Left hand injury.\r\n\r\nCOMPARISON: None.\r\n\r\nTECHNIQUE: Left hand three views 12/01/2022 7:45 AM\r\n\r\nFINDINGS/IMPRESSION: A Salter II fracture of the proximal phalanx of the left middle digit is seen. No significant angulation.\r\n\r\n\r\n\r\n\r\nEND VIEW:\r\nEND INTERFACE RESULT:\r\n\r\n"
         |""".stripMargin

    val correctedJsonInput = jsonInput.replaceAll("", "")

    val predictedClaim = LocalFileUtil.Json.mapper.readValue[PredictedClaim](correctedJsonInput, classOf[PredictedClaim])
    val postgreSql = PostgreSql()
    PredictedClaimTbl.insertPrediction(Seq[PredictedClaim](predictedClaim, predictedClaim), postgreSql)
    postgreSql.close
  }

}
