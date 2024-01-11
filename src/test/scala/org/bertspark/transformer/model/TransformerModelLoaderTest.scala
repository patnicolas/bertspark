package org.bertspark.transformer.model

import org.bertspark.transformer.block.BERTFeaturizerBlock
import org.scalatest.flatspec.AnyFlatSpec

private[transformer] final class TransformerModelLoaderTest extends AnyFlatSpec {

  it should "Succeed loading a pre-trained model" in {
    val transformerLoader = new TransformerModelLoader {
      override protected[this] val preTrainingBlock: BERTFeaturizerBlock = BERTFeaturizerBlock()
    }
    assert(transformerLoader.model.isDefined)
  }

  ignore should "Failed loading an undefined pre-trained model" in {
    val transformerLoader = new TransformerModelLoader {
      override protected[this] val preTrainingBlock: BERTFeaturizerBlock = BERTFeaturizerBlock()
    }
    assert(transformerLoader.model.isDefined == false)
  }
}
