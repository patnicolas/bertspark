package org.bertspark.nlp.vocabulary

import org.bertspark.config.MlopsConfiguration.vocabulary
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.nlp.medical.NoteProcessors._
import org.bertspark.nlp.medical.encodePredictReq
import org.bertspark.util.io.SingleS3Dataset
import org.scalatest.flatspec.AnyFlatSpec

private[nlp] final class MedicalVocabularyTest extends AnyFlatSpec {

  ignore should "Succeed Building 1 and 2-Grams" in {
    val tokens = Array[String]("hello", "world", "Was", "today", "just", "another", "day")
    val nGrams = MedicalVocabulary.buildNGrams(tokens, 2)
    println(nGrams.mkString("\n"))
  }

  ignore should "Succeed Building 1, 2 and 3-Grams" in {
    val tokens = Array[String]("hello", "world", "Was", "today", "just", "another", "day")
    val nGrams = MedicalVocabulary.buildNGrams(tokens, 3)
    println(nGrams.mkString("\n"))
  }


  it should "Succeed loading the vocabulary" in {
    val sz = vocabulary.size()
    val token4 = vocabulary.getToken(4)
    val token908 = vocabulary.getToken(908)
    val token346 = vocabulary.getToken(346)
  }
}
