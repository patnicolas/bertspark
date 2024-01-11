package org.bertspark.transformer.dataset

import ai.djl.ndarray._
import org.bertspark.transformer.dataset.TFeaturesInstance.SegmentTokens
import org.scalatest.flatspec.AnyFlatSpec


private[dataset] final class TFeaturesBatchTest extends AnyFlatSpec {
  import TFeaturesBatchTest._

  it should "Succeed retrieving features for features instances" in {
    val ndManager = NDManager.newBaseManager()
    val bertFeaturesBatch = TFeaturesBatch(createSegmentsBatch)
    val ndFeatures: NDList = bertFeaturesBatch.getFeatures(ndManager)
    println(s"Shape: ${ndFeatures.getShapes().mkString(" ")}")
    ndManager.close()
  }
}



final object TFeaturesBatchTest {

  def createSegmentsBatch: Array[TFeaturesInstance] = Array[TFeaturesInstance](
    TFeaturesInstance(createSegmentTokens1, 32),
    TFeaturesInstance(createSegmentTokens2, 32)
  )


  private def createSegmentTokens1: Array[SegmentTokens] = {
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

  private def createSegmentTokens2: Array[SegmentTokens] = {
    val segment1 = Array[String](
      clsLabel,
      "palpation",
      "not",
      "hall",
      "discharged",
      "why",
      "inframalleolar",
      "click",
      "lonnie",
      "blood",
      "confluence",
      "projecting",
      ".")

    val segment2 = Array[String](
      clsLabel,
      "provided",
      "aaaaa",
      "spondyloarthropathy",
      "provider",
      "morphology",
      "adenocystic",
      "roof",
      "axially",
      "cores",
      "reformed",
      "ordered",
      "angular",
      "chrisman",
      "toe")
    Array[SegmentTokens](segment1, segment2)
  }
}