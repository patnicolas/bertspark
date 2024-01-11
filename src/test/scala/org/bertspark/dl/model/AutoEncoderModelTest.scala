package org.bertspark.dl.model

import ai.djl.ndarray.types.Shape
import org.bertspark.dl.block.BaseHiddenLayer
import org.scalatest.flatspec.AnyFlatSpec

private[dl] final class AutoEncoderModelTest extends AnyFlatSpec {

  it should "Succeed defining an Feed forward auto-encoder" in {
    val numFeatures = 16
    val hiddenLayerConfig = Seq[BaseHiddenLayer](
      (16, "relu"),
      (6, "relu")
    )
    val autoEncoderModel = AutoEncoderModel(FFNNModel(numFeatures, hiddenLayerConfig, 2))
    println(autoEncoderModel.showShapes(new Shape(numFeatures)))
  }
}
