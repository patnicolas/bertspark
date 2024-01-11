package org.bertspark.transformer.dataset

import org.bertspark.transformer.dataset.TFeatureInstanceTest.createSegmentTokens
import org.bertspark.transformer.dataset.TFeaturesInstance.SegmentTokens
import org.scalatest.flatspec.AnyFlatSpec


private[dataset] final class TFeatureInstanceTest extends AnyFlatSpec {

  it should "Succeed getting token ids" in {
    val bertFeatureInstance = TFeaturesInstance(createSegmentTokens, 32)
    println(s"Token ids:\n${bertFeatureInstance.getTokenIds.map(_.mkString(" ")).mkString("\n")}")
  }

  it should "Succeed getting type ids" in {
    val bertFeatureInstance = TFeaturesInstance(createSegmentTokens, 32)
    println(s"Type ids:\n${bertFeatureInstance.getTypeIds.map(_.mkString(" ")).mkString("\n")}")
  }

  it should "Succeed getting input mask" in {
    val bertFeatureInstance = TFeaturesInstance(createSegmentTokens, 32)
    println(s"Input mask:\n${bertFeatureInstance.getInputMasks.map(_.mkString(" ")).mkString("\n")}")
  }
}


private[dataset] final object TFeatureInstanceTest {

  private def createSegmentTokens: Array[SegmentTokens] = {
    val segment1 = Array[String](
      clsLabel,
      "aicd",
      "not",
      "allopurinol",
      "review",
      "in",
      "angioedema",
      ".")

    val segment2 = Array[String](
      clsLabel,
      "unwitnessed",
      "aaaaa",
      "shadowing",
      "palpation",
      "morphology",
      "discharged",
      "roof",
      "angioedema",
      "farner",
      "stasis",
      "ordered",
      "chiles",
      "BBBBB",
      "perivesical",
      "inframalleolar",
      "acutely",
      ".")
    Array[SegmentTokens](segment1, segment2)
  }
}
