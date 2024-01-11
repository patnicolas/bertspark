package org.bertspark.dl.model

import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import org.scalatest.flatspec.AnyFlatSpec

private[model] final class NeuralModelTest extends AnyFlatSpec {

  it should "Succeed extracting name of loss" in {
    val loss = new SoftmaxCrossEntropyLoss()
    println(loss.getName)
  }
}
