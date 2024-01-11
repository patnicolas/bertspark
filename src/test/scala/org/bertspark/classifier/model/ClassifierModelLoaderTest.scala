package org.bertspark.classifier.model

import org.bertspark.classifier.block.ClassificationBlock
import org.scalatest.flatspec.AnyFlatSpec

private[classifier] final class ClassifierModelLoaderTest extends AnyFlatSpec {

  it should "Succeed loading classifier model" in {
    val numClasses = 7

    val loader = new ClassifierModelLoader {
      override val transformerModelName = "420"
      override val classificationModel = "C-420"
      override val subModelName = "73721 26 RT"
      override val classificationBlock: ClassificationBlock = new ClassificationBlock(numClasses)
    }
    val subModel = loader.model
    assert(subModel.isDefined == true)
    assert(loader.isReady == true)
  }
}
