package org.bertspark.kafka.prodcons

import org.bertspark.kafka.prodcons.KafkaPredictionProcTest.generateRequestMessage
import org.bertspark.kafka.serde.RequestSerDe.RequestMessage
import org.bertspark.nlp.medical.MedicalCodingTypes.{MlEMRCodes, PRequest, PRequestContext}
import org.scalatest.flatspec.AnyFlatSpec

private[prodcons] final class KafkaPredictionProcTest extends AnyFlatSpec {

  it should "Succeed processing Kafka prediction" in {
    val requestMessages = generateRequestMessage
    val responsesMessages = KafkaPredictionProc.process(requestMessages)
    println(responsesMessages.mkString("\n"))
  }
}



private[prodcons] final object KafkaPredictionProcTest  {
/*
76506 26,44
G9637,497
20206,15
73552 77 26 LT GC,4
 */
  def generateRequestMessage: Seq[RequestMessage] = {
    val emrCodes1 = Seq[MlEMRCodes](
      MlEMRCodes(0, "76506", Seq[String]("26"), Seq.empty[String], 0, "UN")
    )
    val prContext1 = PRequestContext(emrCodes1)
    val note1 = "\\r\\n\\r\\nDOB: 04/07/1944\\r\\nM\\r\\nM2015201478\\r\\n2020-05-31\\r\\n2035\\r\\n54126\\r\\nRON\\r\\nPEARSON\\r\\n12528\\r\\nJACKIE\\r\\nLIVESAY\\r\\nI\\r\\nDIAG\\r\\nM20607299\\r\\nGLADYS\\r\\nG\\r\\nCOWAN\\r\\n1944-04-07\\r\\nF\\r\\nMEDICARE BLUECROSS ADVANTAGE\\r\\n10022\\r\\nPROCEDURE ORDERED: 03-XR-20-128542\\r\\n2552913365\\r\\nXR CHEST 1 VIEW PORTABLE\\r\\n71045\\r\\n2968540043\\r\\nADMITTING DX: FALL\\r\\nWORKING DX: FALL\\r\\nEXAM: PORTABLE UPRIGHT AP VIEW OF THE CHEST\\r\\nINDICATION: FALL, PAIN\\r\\nCOMPARISON: RADIOGRAPHS OF THE THORACIC SPINE 07/14/2016\\r\\n\\r\\nFINDINGS:\\r\\nTHE LUNGS ARE SYMMETRICALLY INFLATED AND WITHOUT FOCAL CONSOLIDATION.  NO LARGE PLEURAL EFFUSION OR VISIBLE PNEUMOTHORAX.  THE CARDIOMEDIASTINAL SILHOUETTE IS ENLARGED.  NO ACUTE OSSEOUS ABNORMALITY.\\r\\n\\r\\nIMPRESSION:\\r\\nNO ACUTE CARDIOPULMONARY ABNORMALITY.\\r\\n\\r\\nCARDIOMEGALY.\\r\\n\\r\\nTRANSCRIPTIONIST- RON PEARSON, PHYSICIAN - RADIOLOGIST\\r\\nREAD BY- RON PEARSON, PHYSICIAN - RADIOLOGIST\\r\\nREVIEWED AND E-SIGNED BY- RON PEARSON, PHYSICIAN - RADIOLOGIST\\r\\nRELEASED DATE TIME- 05/31/20 21:05\\r\\n------------------------------------------------------------------------------\\r\\n\\r\\n"
    val prRequest1 = PRequest("1", prContext1, Seq[String](note1))

    val emrCodes2 = Seq[MlEMRCodes](
      MlEMRCodes(0, "G9637", Seq.empty[String], Seq.empty[String], 0, "UN")
    )
    val prContext2 = PRequestContext(emrCodes2)
    val note2 = "\\r\\n\\r\\nUnit : 1\\r\\nExam Description :US BREAST LIMITED RIGHT\\r\\nCPT Code :76642\\r\\nPrimary Physician :Charles A Montgomery, MD\\r\\nDOB :  12/31/1946\\r\\nAttending Physician : \\r\\nR breast cyst\\r\\n\\r\\n\\r\\nDiagnostic bilateral mammogram and focused right breast ultrasound History: Follow-up right breast cyst and left lumpectomy, history of left breast cancer TISSUE DENSITY DENSITYCODE:  C heterogeneously dense (51-75%) FINDINGS Standard bilateral CC, MLO, and true lateral views were performed.  Focused right breast ultrasound and magnification views of the right breast were also performed. Prominent bilateral fibrocystic change again demonstrated, similar to the previous study.  There is a new grouping of microcalcifications in the upper inner left breast appear heterogeneous in similar to numerous additional scattered microcalcifications on magnification images.  Mild scarring in the lateral left breast. Focused right breast ultrasound was performed which demonstrated prominent scattered fibrocystic change in the upper outer, lower outer, and lower inner quadrants.  The largest cyst appears enlarged and is present in the lower outer quadrant measuring 4.7 x 2.9 cm in greatest dimensions.  A complex cystic area in the upper outer quadrant at the 11 o`clock position is not significantly changed measuring approximately 1.8 x 1.5 cm. No suspicious masses, areas of concerning architectural distortion, or suspiciously clustered microcalcifications are identified. Compared to prior exams, there has been no suspicious mammographic change.   Films were reviewed by R2 computer-assisted detection system. A negative/benign mammographic/ultrasonographic report should not delay or preclude biopsy in the setting of suspicious clinical findings. ADDITIONAL RECOMMENDATION If the patient has not had a recent physical examination of the breasts, it is suggested that an appointment be made with the clinician for physical exam correlation. ASSESSMENT Prominent bilateral fibrocystic change which is favored to be benign in etiology.  The new microcalcifications in the left breast are also favored to be benign in etiology. Recommendation: The patient should return in 6 months for diagnostic bilateral mammography with magnification views of the left breast and a repeat right breast ultrasound. BIRADS:  3 - Probable Benign Finding-Short Interval Follow-Up Suggested FOLLOWUP:  1003 Short Interval Follow-Up in 6 Months Report dictation location: GCHWMA1\\r\\n\\r\\n"

    val prRequest2 = PRequest("2", prContext2, Seq[String](note2))
    val requestMessage1 = RequestMessage(11111L, prRequest1)
    val requestMessage2 = RequestMessage(11112L, prRequest2)
    Seq[RequestMessage](requestMessage1, requestMessage2)
  }
}