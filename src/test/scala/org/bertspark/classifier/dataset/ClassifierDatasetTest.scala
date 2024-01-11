package org.bertspark.classifier.dataset

import ai.djl.ndarray._
import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import org.scalatest.flatspec.AnyFlatSpec

private[dataset] final class ClassifierDatasetTest extends AnyFlatSpec {

  ignore should "Succeed processing Softmax cross entropy loss" in {
    val ndManager = NDManager.newBaseManager()
    val ndData = ndManager.create(Array[Float](0.4F, 0.6F, 0.3F))
    val labelsArray = Array.fill(3)(0)
    labelsArray(1) = 1
    val ndLabel: NDArray = ndManager.create(labelsArray)    //.ones(new Shape(1))
    val softmaxCrossEntropyLoss = new SoftmaxCrossEntropyLoss("loss", 1, -(1), false, true)
    val ndLoss: NDArray = softmaxCrossEntropyLoss.evaluate(new NDList(ndLabel), new NDList(ndData))
    val lossType = ndLoss.getDataType
    println(s"loss Type: ${lossType}")
    println(ndLoss.toFloatArray().mkString(" "))
  }

  it should "Succeed processing Softmax cross entropy loss with batch" in {
    val ndManager = NDManager.newBaseManager()
    val ndData = ndManager.create(
      Array[Array[Float]](
        Array[Float](0.4F, 0.6F, 0.3F),
        Array[Float](0.2F, 0.0F, 0.9F),
        Array[Float](0.2F, 0.6F, 0.9F)
      )
    )
    val labelsArray = Array[Array[Float]](
      Array[Float](23.0F),
      Array[Float](0.0F),
      Array[Float](0.0F)
    )

    val ndLabel: NDArray = ndManager.create(labelsArray)    //.ones(new Shape(1))
    val softmaxCrossEntropyLoss = new SoftmaxCrossEntropyLoss("loss", 1, -(1), true, true)
    val ndLoss: NDArray = softmaxCrossEntropyLoss.evaluate(new NDList(ndLabel), new NDList(ndData))
    val lossType = ndLoss.getDataType
    println(s"loss Type: ${lossType}")
    println(ndLoss.toFloatArray().mkString(" "))
  }
}
