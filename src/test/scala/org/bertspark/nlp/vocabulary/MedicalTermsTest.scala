package org.bertspark.nlp.vocabulary

import org.scalatest.flatspec.AnyFlatSpec

private[vocabulary] final class MedicalTermsTest extends AnyFlatSpec{

  ignore should "Succeed building a vocabulary component with medical terms" in {
    val output = MedicalTerms.buildFromCorpus()
    assert(output.size > 0)
    println(output.take(10).mkString(" "))
  }

  it should "Succeed augmenting the current TfIdf vocabulary" in {
    val globalTfIdfFile = "output/globalRelTf-CMBS.csv"
    val tfIdfThreshold = 0.9F
    val numWordPieceNotes = 32000
    val output = MedicalTerms.augmentTf(globalTfIdfFile, tfIdfThreshold, numWordPieceNotes)
    println(s"Final vocabulary size: ${output.size}")
  }
}
