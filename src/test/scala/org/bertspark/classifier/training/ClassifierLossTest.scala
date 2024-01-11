package org.bertspark.classifier.training

import ai.djl.ndarray._
import org.scalatest.flatspec.AnyFlatSpec


private[classifier] final class ClassifierLossTest extends AnyFlatSpec{
  import ClassifierLossTest._

  it should "Succeed computing the loss for a given label and prediction" in {
    val ndManager = NDManager.newBaseManager()

    val loss = computeLoss(ndManager, Array[Float](0.0F, 0.4F, 0.6F, 0.5F, 0.7F))
    println(s"Loss: ${loss.mkString(" ")}")

    val loss2 = computeLoss(ndManager, Array[Float](0.0F, 0.0F, 0.0F, 0.0F, 1.0F))
    println(s"Loss2: ${loss2.mkString(" ")}")

    val loss3 = computeLoss(ndManager, Array[Float](1.0F, 0.0F, 0.0F, 0.0F, 0.0F))
    println(s"Loss3: ${loss3.mkString(" ")}")
    ndManager.close()
  }

  it should "Succeed computing the loss for a given label and prediction batch" in {
    val ndManager = NDManager.newBaseManager()

    val loss = computeBatchLoss(ndManager,
      Array[Array[Float]](
        Array[Float](0.0F, 0.4F, 0.6F, 0.5F, 0.7F),
        Array[Float](0.1F, 0.0F, 0.1F, 0.0F, 0.2F),
        Array[Float](0.9F, 0.1F, 0.0F, 0.1F, 0.0F)
      ))
    println(s"Batched loss: ${loss.mkString(" ")}")

    val loss2 = computeBatchLoss(ndManager,
      Array[Array[Float]](
        Array[Float](0.1F, 0.0F, 0.1F, 0.0F, 0.2F),
        Array[Float](0.1F, 0.0F, 0.1F, 0.0F, 0.2F),
        Array[Float](0.1F, 0.0F, 0.1F, 0.0F, 0.2F)
      ))
    println(s"Batched loss2: ${loss2.mkString(" ")}")

    val loss3 = computeBatchLoss(ndManager,
      Array[Array[Float]](
        Array[Float](0.0F, 0.0F, 0.0F, 0.0F, 1.0F),
        Array[Float](1.0F, 0.0F, 0.0F, 0.0F, 0.0F),
        Array[Float](0.0F, 1.0F, 0.0F, 0.0F, 0.0F)
      )
    )
    println(s"Batched loss3: ${loss3.mkString(" ")}")

    val loss4 = computeBatchLoss(ndManager,
      Array[Array[Float]](
        Array[Float](1.0F, 0.0F, 0.0F, 0.0F, 0.0F),
        Array[Float](0.0F, 0.0F, 0.0F, 0.0F, 0.0F),
        Array[Float](0.0F, 0.0F, 0.0F, 0.0F, 1.0F)
      )
    )
    println(s"Batched loss4: ${loss4.mkString(" ")}")
    ndManager.close()
  }
}

private[classifier] final object ClassifierLossTest {

  def computeLoss(ndManager: NDManager, prediction: Array[Float]): Array[Float] = {
    val labelIndexMap = Map.empty[Int, String]
    val lossName = "Loss"
    val sparseLabel = false
    val fromLogit = true
    val subModeName = "XXX"

    val bertSoftmaxCrossEntropyLoss = ClassifierLoss(
      labelIndexMap,
      lossName,
      sparseLabel,
      fromLogit,
      subModeName)

    val label: Array[Float] = Array[Float](0.0F, 0.0F, 0.0F, 0.0F, 1.0F)
    val ndPrediction: NDArray = ndManager.create(prediction)
    val ndLabel: NDArray = ndManager.create(label)
    val ndLoss = bertSoftmaxCrossEntropyLoss.evaluate(new NDList(ndLabel), new NDList(ndPrediction))
    ndLoss.toFloatArray
  }


  def computeBatchLoss(ndManager: NDManager, prediction: Array[Array[Float]]): Array[Float] = {
    val labelIndexMap = Map.empty[Int, String]
    val lossName = "Loss"
    val sparseLabel = false
    val fromLogit = true
    val subModeName = "XXX"

    val bertSoftmaxCrossEntropyLoss = ClassifierLoss(
      labelIndexMap,
      lossName,
      sparseLabel,
      fromLogit,
      subModeName)

    val label = Array[Array[Float]](
        Array[Float](0.0F, 0.0F, 0.0F, 0.0F, 1.0F),
        Array[Float](1.0F, 0.0F, 0.0F, 0.0F, 0.0F),
        Array[Float](0.0F, 1.0F, 0.0F, 0.0F, 0.0F)
      )

    assert(label.size == prediction.size && label.head.size == prediction.head.size)

    val ndPrediction: NDArray = ndManager.create(prediction)
    val ndLabel: NDArray = ndManager.create(label)
    val ndLoss = bertSoftmaxCrossEntropyLoss.evaluate(new NDList(ndLabel), new NDList(ndPrediction))
    ndLoss.toFloatArray
  }
}

