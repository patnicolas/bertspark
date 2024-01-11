package org.bertspark.nlp.vocabulary

import org.bertspark.nlp.medical.NoteProcessors
import org.bertspark.nlp.medical.NoteProcessors.specialCharCleanserRegex
import org.bertspark.nlp.token.TokenizerPreProcessor.AbbreviationMap.abbreviationMap
import org.bertspark.nlp.tokenSeparator
import org.scalatest.flatspec.AnyFlatSpec


private[vocabulary] final class MedicalVocabularyBuilderTest extends AnyFlatSpec {

  ignore should "Succeed pre-processing notes to build vocabulary" in {
    import org.bertspark.implicits._

    val medicalCodingVocabulary = MedicalVocabularyBuilder()
    medicalCodingVocabulary.build
  }

  it should "Succeed cleansing abbreviations" in {
    val abbreviations = abbreviationMap.flatMap(_._2.split(tokenSeparator)).toSeq.distinct.mkString("\n")
    val cleansedAbbreviations = NoteProcessors.cleanse(abbreviations, specialCharCleanserRegex)
    println(cleansedAbbreviations.take(1024).sortWith(_ < _).mkString("\n"))
  }

  it should "Succeed cleansing code descriptions" in {
    val codeDescription = MedicalCodeDescriptors.getCptDescriptors.flatMap(_._2).distinct.mkString("\n")
    val cleansedCodeDescription = NoteProcessors.cleanse(codeDescription, specialCharCleanserRegex)
    println(s"\n---------------------------------------------------\n\n\n${cleansedCodeDescription.take(2048).sortWith(_ < _).mkString("\n")}")
  }
}
