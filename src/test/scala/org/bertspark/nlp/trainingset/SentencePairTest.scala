package org.bertspark.nlp.trainingset

import org.scalatest.flatspec.AnyFlatSpec


private[trainingset] final class SentencePairTest extends AnyFlatSpec {

  it should "Succeed apply xOver to a list of sentence pairs" in {
    val firstSentence = Array[String]( "hello",
      "not",
      "today",
      "review",
      "in",
      "August",
      "or",
      "September")
    val secondSentence = Array[String]( "This", "is", "not", "a", "great", "day")
    val thirdSentence = firstSentence.reverse
    val sentencePairs = Array[SentencePair](
      new SentencePair(firstSentence, thirdSentence),
      new SentencePair(secondSentence, thirdSentence),
      new SentencePair(firstSentence, secondSentence)
    )
    val updatedSentencePairs = SentencePair.swapEveryOther(sentencePairs)
    println(updatedSentencePairs.mkString(", "))
  }

  it should "Succeed truncate the sentence pair" in {
    val firstSentence = Array[String]( "hello",
      "not",
      "today",
      "review",
      "in",
      "August",
      "or",
      "September")
    val secondSentence = Array[String]( "This", "is", "not", "a", "great", "day")
    val sentencePair = new SentencePair(firstSentence, secondSentence)
    println(s"Before truncation: ${   sentencePair.toString}")
    println(s"Truncated to 7: ${   sentencePair.truncate(7).toString}")
  }
}
