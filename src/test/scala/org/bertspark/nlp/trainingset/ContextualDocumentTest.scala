package org.bertspark.nlp.trainingset

import org.bertspark.nlp.medical.MedicalCodingTypes._
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalContext.getEmrCodesComma
import org.bertspark.nlp.medical.NoteProcessors
import org.bertspark.nlp.medical.NoteProcessors.specialCharCleanserRegex
import org.bertspark.nlp.token.CtxTxtNSentencesBuilder
import org.bertspark.nlp.trainingset.ContextualDocument.ContextualDocumentBuilder
import org.bertspark.nlp.trainingset.ContextualDocument.ContextualDocumentBuilder.extractContextualDocument
import org.scalatest.flatspec.AnyFlatSpec


private[trainingset] final class ContextualDocumentTest extends AnyFlatSpec {

  it should "Succeed converting a set of records " in {
    val note = "\\r\\n\\r\\nDOB: 12/22/1996\\r\\nHistory Number:1784837\\r\\nExam: Fluoroscopically guided right knee arthrogram \\r\\nProcedure performed by Phillip Clarke, MD with assistance by resident Hunter Upton, M.D.\\r\\nINDICATION: 22 years old Female with  ACUTE RT KNEE PN SURG 12.28.2018 \\r\\nFluoro time: 0.3 min\\r\\nDAP: 22.1 uGy*m2\\r\\n\\r\\nThe patient was interviewed prior to exam. Patient was informed of procedural details, as well as purpose,  benefits, risks (to include contrast reaction and infection), and alternatives of procedure. The patient expressed understanding, signed consent was obtained. No significant contraindications to exam were noted. A formal time-out was performed and everyone agreed upon stated location and procedure.\\r\\n\\r\\nScout Image demonstrates no acute fracture. Anatomic alignment. Subtle irregularity of the patella inferior pole at site of full-thickness cartilage loss. Small joint effusion.\\r\\n\\r\\nProcedure:\\r\\nSterile barrier technique was employed. Local lidocaine anesthesia was injected subcutaneously.  Using fluoroscopic control,  a 25 gauge needle was placed into the right knee. Intra-articular location was confirmed by injection of contrast. No fluid was aspirated.  Subsequently, 16 cc of ProHance/saline mixture was injected.\\r\\n\\r\\nThere were no immediate complications. Patient was given post-arthrogram instructions and instruction sheet.\\r\\n\\r\\nIMPRESSION:\\r\\nRight knee arthrogram, preparatory for MR arthrogram.  Please see separate MR arthrogram report for diagnostic evaluation.\\r\\n\\r\\n-- \\r\\n \\r\\n \\r\\n  I have personally reviewed the image(s) and the resident interpretation and agree with the findings.\\r\\n\\r\\nPLAT-50-148\\r\\n\\r\\n\\r\\n.\\r\\nAuthenticated By: CLARK MD, PHILLIP D                         11/12/2019 16:37\\r\\nResident: Upton MD RES, Hunter B\\r\\n\\r\\n**FINAL REPORT**\\r\\n\\r\\n"
    val internalContext = InternalContext(
      "",
      12,
      "M",
      "",
      "Cornerstone",
      "c",
      "radiology",
      "11",
      "",
      Seq[MlEMRCodes](MlEMRCodes(0, "99214", Seq[String]("26", "LT"), Seq.empty[String], 1, "UN")),
      "",
      "",
      "",
      "",
      "",
      ""
    )
    /*
          val terms = NoteProcessors.cleanse(text, specialCharCleanserRegex)
      // Step 4: Replace the medical abbreviation with descriptors
     // val textWithReplacedAbbr = MedicalAbbreviations().build(terms).map(_.toLowerCase)
      terms.map(_.toLowerCase).mkString(" ")
     */
    val tokens = NoteProcessors.cleanse(note, specialCharCleanserRegex).map(_.toLowerCase)
    println(s"Num raw tokens: ${tokens.size}")
    val internalRequest1 = InternalRequest("id1", internalContext, Seq[String](note))
    val internalRequest2 = InternalRequest("id2", internalContext, Seq[String](note))

    import org.bertspark.implicits._
    import sparkSession.implicits._
    val ds = Seq[InternalRequest](internalRequest1, internalRequest2).toDS()
    val contextualDocumentBuilder = ContextualDocumentBuilder()
    contextualDocumentBuilder(ds, 5)
  }

  ignore should "Succeed display tokens per segments" in {
    import org.bertspark.implicits._
    val sampleSize = 200
    ContextualDocumentBuilder.evaluate(sampleSize)
  }

  it should "Succeed processing tokens for a given note" in {
    val note = "\r\n\r\nDOB: 03/11/1938\r\nHistory Number:875102\r\nEXAM:  CT of the head without IV contrast.\nINDICATION: fall with right sided weakness\nCOMPARISON: CT head December 31, 2018. \n\nOne or more of the following dose-reduction techniques was utilized: automated exposure control, iterative reconstruction, and/or manual adjustment of tube current and voltage for size.\n\nFindings:  \nMixed density left cerebral convexity hematoma, which measures 12 mm in thickness, and results in partial effacement of the left lateral ventricle and 5 mm of left to right midline shift.\nScattered periventricular subcortical white matter hyperdensities likely represent sequela of chronic microvascular ischemic changes.\nGray-white matter differentiation is grossly maintained. \nBasilar cisterns are patent.\n \nKnown large mass in the left upper neck measuring at least 3.7 x 2.6 cm, which was biopsied on December 14, 2018.\nBilateral intraocular lens replacement.\nThe paranasal sinuses and mastoid air cells are clear. \nNo acute osseous abnormalities.\n\nImpression:\n\nAcute on chronic left cerebral convexity subdural hematoma which results in 5 mm left right midline shift.\n\n\n--Communication\nThese findings were phoned to Dr. Barnett by Dr. Matthew Ramsey at 2/9/2019 6:26 PM EST \n\n\n\n\n-- \n \n \n  I have personally reviewed the image(s) and the resident interpretation and agree with the findings.\n\nAUR-HOME\n\r\n\r\n.\nAuthenticated By: BROWN MD, STEPHEN J                         02/09/2019 18:45\nResident: Ramsey MD RES, Matthew T\n\n**FINAL REPORT**\r\n"
    val context = InternalContext(
      "c0",
      54,
      "F",
      "889adf",
      "Cornerstone",
      "cl-4",
      "MRI",
      "22",
      "",
      Seq[MlEMRCodes](MlEMRCodes(0, "78123", Seq[String]("26"), Seq.empty[String], 4, "UN")),
      "p1",
      "pt-4",
      "",
      "",
      "",
      ""
    )
    val sentenceBuilder = new CtxTxtNSentencesBuilder
    val request = InternalRequest("990", context, Seq[String](note))
    val contextualDocument = extractContextualDocument(request)
    val tokens = sentenceBuilder(contextualDocument)
    println(tokens.mkString(" "))
  }

  it should "Succeed converting a labeled request" in{
    val contextualDocument1 = ContextualDocument("1")
    val emrCpts1 = Seq[MlEMRCodes](
      MlEMRCodes(0, "77889", Seq[String]("26", "LT"), Seq.empty[String])
    )
    val emrWithComma1 = getEmrCodesComma(emrCpts1)
    val lineItems1 = Seq[FeedbackLineItem](
      FeedbackLineItem(0, "77889", Seq[String]("26", "LT"), Seq[String]("B89.11", "M78.12"), 0, "UN", 0.1),
      FeedbackLineItem(0, "G6738", Seq.empty[String], Seq[String]("I25.1"), 0, "UN", 0.1),
      FeedbackLineItem(0, "77000", Seq[String]("GC"), Seq[String]("Z12.31"), 0, "UN", 0.1),
    )
    val labeledRequest1 = LabeledRequest(contextualDocument1, emrCpts1, lineItems1)
    val converted = TrainingLabel.mkStringLineItems(labeledRequest1, emrWithComma1)
    println(converted)
  }
}


