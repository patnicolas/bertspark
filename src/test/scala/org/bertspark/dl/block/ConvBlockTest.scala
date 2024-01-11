package org.bertspark.dl.block

import ai.djl.ndarray.types.Shape
import org.bertspark.dl.config._
import org.bertspark.dl
import org.scalatest.flatspec.AnyFlatSpec



private[block] final class ConvBlockTest extends AnyFlatSpec {

  it should "Succeed inverting a Stacked RBM block" in {
    val convBlock = ConvBlock(
      ConvLayerConfig(dl.conv1dLbl, new Shape(1, 8), new Shape(1, 2), new Shape(1, 2), 64, false),
      BatchNormConfig(dl.batchNormLbl, 1, true, 0.001F, 0.98F, false),
      ActivationConfig(dl.reluLbl),
      PoolingConfig(dl.maxPool1dLbl, new Shape(1, 23), new Shape(1, 3), new Shape(1, 5))
    )
    println(convBlock.toString)
  }
}

