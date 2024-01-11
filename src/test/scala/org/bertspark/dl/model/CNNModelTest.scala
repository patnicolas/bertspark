package org.bertspark.dl.model

import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.ndarray.types.Shape
import ai.djl.ndarray.NDManager
import ai.djl.training.dataset.Dataset
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.loss.Loss
import org.bertspark.dl.block._
import org.bertspark.dl.config._
import org.bertspark.dl
import org.bertspark.modeling.TrainingContext
import org.scalatest.flatspec.AnyFlatSpec


private[dl] final class CNNModelTest extends AnyFlatSpec {

  ignore should "Succeed laying out a CNN" in {
    val convBlock1 = ConvBlock(
      ConvLayerConfig(dl.conv1dLbl, new Shape(5, 5), new Shape(2, 2), new Shape(-1, -1), 6, false),
      BatchNormConfig(dl.batchNormLbl, 1, true, 0.001F, 0.98F, false),
      ActivationConfig(dl.reluLbl),
      PoolingConfig(dl.maxPool1dLbl, new Shape(5, 5), new Shape(2, 2), new Shape(2, 2))
    )
    val convBlock2 = ConvBlock(
      ConvLayerConfig(dl.conv1dLbl, new Shape(5, 5), new Shape(-1, -1), new Shape(-1, -1), 16, false),
      ActivationConfig(dl.reluLbl),
      PoolingConfig(dl.maxPool1dLbl, new Shape(5, 55), new Shape(2, 2), new Shape(2, 2))
    )

    val numFeatures = 128
    val fnnModel = FFNNModel(128, Seq[BaseHiddenLayer]((32, "sigmoid"), (16, "sigmoid")), 16)

    val cnnNetwork = CNNModel(Seq[ConvBlock](convBlock1, convBlock2), fnnModel)
    println(cnnNetwork.toString)
  }



  it should "Succeed training CNN model of fashion data set" in {
    val ndManager = NDManager.newBaseManager

    val batchSize = 32
    val trainDataset = FashionMnist.builder
        .optManager(ndManager)
        .optUsage(Dataset.Usage.TRAIN)
        .setSampling(batchSize, true)
        .build
    val testDataset = FashionMnist.builder
        .optManager(ndManager)
        .optUsage(Dataset.Usage.TEST)
        .setSampling(batchSize, true)
        .build

    val convLayer1Conf = ConvLayerConfig(
      dl.conv2dLbl,
      new Shape(5, 5),
      new Shape(-1, -1),
      new Shape(2,2),
      6,
      false)
    val poolingLayer1Conf = PoolingConfig(
      dl.avgPool2dLbl,
      new Shape(5, 5),
      new Shape(2, 2),
      new Shape(2,2))
    val conv1Block = ConvBlock(convLayer1Conf, ActivationConfig(dl.sigmoidLbl), poolingLayer1Conf)

    val convLayer2Conf = ConvLayerConfig(
      dl.conv2dLbl,
      new Shape(5, 5),
      16,
      false)
    val poolingLayer2Conf = PoolingConfig(
      dl.avgPool2dLbl,
      new Shape(5, 5),
      new Shape(2, 2),
      new Shape(2, 2))
    val conv2Block = ConvBlock(convLayer2Conf, ActivationConfig(dl.sigmoidLbl), poolingLayer2Conf)

    val fullyConnected = FFNNModel(Seq[BaseHiddenLayer]((120, dl.sigmoidLbl), (84, dl.sigmoidLbl)), 10)
    val cnnModel = CNNModel(Seq[ConvBlock](conv1Block, conv2Block), fullyConnected)

    val x = ndManager.randomUniform(0.0F, 1.0F, new Shape(1, 1, 28, 28))
    println(s"CNN configuration:\n${cnnModel.toString}")

    val numEpochs = 8
    val trainContext = TrainingContext(
      Optimizers.sgd(0.8F, 0.0F),
      new NormalInitializer(),
      Loss.softmaxCrossEntropyLoss(),
      numEpochs,
      1,
      1,
      Seq[String]("loss"),
      "CNN"
    )
    cnnModel.train(trainContext, trainDataset, testDataset, "")
  }
}
