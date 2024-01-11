package org.bertspark.nlp.medical

import org.bertspark.nlp.medical.MedicalCodingTypes._
import org.bertspark.nlp.medical.ContextEncoder._
import org.scalatest.flatspec.AnyFlatSpec

private[bertspark] final class ContextEncoderTest extends AnyFlatSpec {

  it should "Embed numerical values" in {
    val embedded1 = encodeNumeric(0, 120, 10, 12)
    println(embedded1)
    val embedded2 = encodeNumeric(0, 120, 10, 87)
    println(embedded2)
  }

  it should "Embed context from a request" in {
    val context = InternalContext(
      "C1",
      78,
      "F",
      "radiology",
      "BH",
      "AAA",
      "ER",
      "22",
      "2022-08-01",
      Seq.empty[MlEMRCodes],
      "myProvider",
      "myPatient",
      "",
      "",
      "",
      ""
    )
    println(encodeContext(context).mkString(" "))
  }
}