package org.bertspark.transformer.dataset

import org.bertspark.nlp.trainingset.SentencePair
import org.bertspark.transformer.dataset.TMaskedInstance.{createLabeledTokens, createMaskedIndices, createInputMasks, createTokenTypeIds}
import org.bertspark.transformer.dataset.TMaskedInstanceTest.createBertMaskedInstance
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable.ListBuffer
import scala.util.Random

private[dataset] final class TMaskedInstanceTest extends AnyFlatSpec {

  it should "Succeed padding for value" in {
    val input = Array[Int](9, 4, 5, 1, 9)
    val output = input.padTo(10, 0)
    assert(output.size == 10)
    assert(output(2) == 5)
    assert(output(8) == 0)
    println(output.mkString(". "))
  }

  it should "Succeed aggregating labels as list items" in {
    val list1 = List[String]("bb1", "bb2")
    val list2 = List[String]("ddd1", "ddd2", "dd3")

    val xs = ("aaa" :: list1) ::: ("ccc" :: list2) :::  ("eee" :: List[String]())
    println(xs.mkString(", "))

    val l = new ListBuffer[String]()
    l.append("aaa")
    l.appendAll(list1)
    l.append("bbb")
    l.appendAll(list2)
    l.append("eee")
    println(l.mkString(", "))
  }

  it should "Succeed creating type ids" in {
    val labels = Array[String](
      clsLabel,
      "aicd",
      "not",
      "allopurinol",
      "review",
      "in",
      "angioedema",
      ".",
      sepLabel,
      "asphyxia",
      "is",
      "bladder",
      "in",
      "carcinomatous")
    val typeIds = createTokenTypeIds(labels)
    println(s"Type ids ${typeIds.mkString(", ")}")
  }

  it should "Succeed creating labels from sentences" in {
    val firstSentence = Array[String]( "hello",
      "aicd",
      "today",
      "review",
      "in",
      "allopurinol",
      "or",
      "angioedema")
    val secondSentence = Array[String]( "asphyxia", "is", "not", "allopurinol", "bladder", "carcinomatous")
    val sentencesPair = new SentencePair(firstSentence, secondSentence)

    val labels = createLabeledTokens(sentencesPair)
    println(s"Create labels ${labels.mkString(", ")}")
  }


  it should "Succeed create masked labels" in {
    val maskingIndices = Array[Int](2, 3, 5, 0, 1, 8)

    val labels = Array[String](
      "aicd",
      "not",
      "allopurinol",
      "review",
      mskLabel,
      "angioedema",
      "asphyxia",
      "is",
      "bladder",
      mskLabel,
      "carcinomatous")
    val maskedLabels = createInputMasks(maskingIndices, labels, new Random(42L))
    println(s"Masked labels: ${maskedLabels.mkString(", ")}")
  }


  it should "Succeed generating masked indices 1" in {
    val maxMasking = 6
    val maskingProb = 0.2F
    val labels = Array[String](
      "aicd",
      "not",
      "allopurinol",
      "review",
      mskLabel,
      "angioedema",
      "asphyxia",
      "is",
      "bladder",
      mskLabel,
      "carcinomatous")
    val maskedIndices = createMaskedIndices(maxMasking, maskingProb, labels)
    println(s"Masked indices for max 6 and prob 0.2:  ${maskedIndices.mkString(", ")}")
  }

  it should "Succeed generating masked indices 2" in {
    val maxMasking = 3
    val maskingProb = 0.35F
    val labels = Array[String](
      "aicd",
      "not",
      "allopurinol",
      "review",
      mskLabel,
      "angioedema",
      "asphyxia",
      "is",
      "bladder",
      mskLabel,
      "carcinomatous")
    val maskedIndices = createMaskedIndices(maxMasking, maskingProb, labels)
    println(s"Masked indices for max 3 and prob 0.35:  ${maskedIndices.mkString(", ")}")
  }

  it should "Succeed generating masked indices 3" in {
    val maxMasking = 6
    val maskingProb = 0.5F
    val labels = Array[String](
      "aicd",
      "not",
      "allopurinol",
      "review",
      mskLabel,
      "angioedema",
      "asphyxia",
      "is",
      "bladder",
      mskLabel,
      "carcinomatous")
    val maskedIndices = createMaskedIndices(maxMasking, maskingProb, labels)
    println(s"Masked indices for max 6 and prob 0.35:  ${maskedIndices.mkString(", ")}")
  }

  it should "Succeed generating type ids, labels, masked labels and masked indices - 1" in {
    val firstSentence = Array[String]( "hello",
      "aicd",
      "today",
      "review",
      "in",
      "allopurinol",
      "or",
      "angioedema")
    val secondSentence = Array[String]( "asphyxia", "is", "not", "allopurinol", "bladder", "carcinomatous")
    val bertMaskedInstance = createBertMaskedInstance(firstSentence, secondSentence, 6, 0.2F)
    println(s"max masking 6, prob 0.2F: ${bertMaskedInstance.toString}")
  }

  it should "Succeed generating type ids, labels, masked labels and masked indices - 2" in {
    val firstSentence = Array[String]( "hello",
      "aicd",
      "today",
      "review",
      "in",
      "allopurinol",
      "or",
      "angioedema")
    val secondSentence = Array[String]( "asphyxia", "is", "not", "allopurinol", "bladder", "carcinomatous")
    val bertMaskedInstance = createBertMaskedInstance(firstSentence, secondSentence, 6, 0.45F)
    println(s"max masking 6, prob 0.45F: ${bertMaskedInstance.toString}")
  }
}

private[dataset] final object TMaskedInstanceTest {

  def createBertMaskedInstance(
    firstSentence: Array[String],
    secondSentence: Array[String],
    maxMasking: Int,
    maskingProb: Float): TMaskedInstance =  {

    val sentencesPair = new SentencePair(firstSentence, secondSentence)
    TMaskedInstance(sentencesPair, 16, maxMasking, maskingProb)
  }
}
