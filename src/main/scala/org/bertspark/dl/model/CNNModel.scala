/**
 * Copyright 2022,2023 Patrick R. Nicolas. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package org.bertspark.dl.model

import ai.djl.metric.Metrics
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Block
import ai.djl.training.EasyTrain
import ai.djl.training.util.ProgressBar
import ai.djl.Model
import org.bertspark.dl.block._
import org.bertspark.DjlDataset
import org.bertspark.modeling.{TrainingContext, TrainingSummary}

/**
 * Class to configure and train CnnModel
 * @param convBlocks Set of convolutional blocks
 * @param flattenBlock Flattening tensor block
 * @param rbmBlocks Stacked RBM blocks
 * @param outputBlock Output block
 * @param inputShape Shape of input date
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class CNNModel private (
  convBlocks: Seq[ConvBlock],
  flattenBlock: Block,
  rbmBlocks: Seq[RBMBlock],
  outputBlock: Block,
  inputShape: Option[Shape]) extends BaseNetBlock with NeuralModel[CNNModel] with TrainingSummary {
  require(convBlocks.nonEmpty, s"Cannot create a CNN network without convolutional blocks")
  require(inputShape.isDefined, "Input shape for CNN model is undefined")

  convBlocks.foreach(sequentialBlock.add(_))
  sequentialBlock.add(flattenBlock)
  rbmBlocks.foreach(sequentialBlock.add(_))
  sequentialBlock.add(outputBlock)


  /**
   * Training for a given training context, shape of input data and training, validation dataset
   * @param trainingContext Context/Configuration for the training
   * @param trainDatasetIter Training dataset
   * @param testDatasetIter Test dataset
   */
  override def train(
    trainingContext: TrainingContext,
    trainDatasetIter: DjlDataset,
    testDatasetIter: DjlDataset,
    subModelName: String): String = {

    trainDatasetIter.prepare(new ProgressBar)
    testDatasetIter.prepare(new ProgressBar)

    // Initialize the training configuration
    val model = Model.newInstance("cnn")
    model.setBlock(sequentialBlock)
    val trainer = model.newTrainer(trainingContext.getDefaultTrainingConfig)
    trainer.setMetrics(new Metrics)
    inputShape.foreach(trainer.initialize(_))
    inputShape.foreach(showShapes(_))

    // Train method
    val testDataset = testDatasetIter
    EasyTrain.fit(trainer, trainingContext.getNumEpochs, trainDatasetIter, testDataset)
    val res = trainer.getTrainingResult
    println(s"\nTraining Loss: ${res.getTrainLoss}")
    println(s"Validation Loss: ${res.getValidateLoss}")
    apply(trainer)
    textSummary(trainingContext)
    // Text summary
    println(toRawText)
    // Plot enabled/disabled in the training context
    trainer.getModel.getModelPath.toString
  }
}


private[bertspark] final object CNNModel {

  def apply(
    convBlocks: Seq[ConvBlock],
    flattenBlock: Block,
    rbmBlocks: Seq[RBMBlock],
    outputBlock: Block,
    inputShape: Shape): CNNModel = new CNNModel(convBlocks, flattenBlock, rbmBlocks, outputBlock, Some(inputShape))

  def apply(
    convBlocks: Seq[ConvBlock],
    flattenBlock: Block,
    rbmBlocks: Seq[RBMBlock],
    outputBlock: Block): CNNModel = new CNNModel(convBlocks, flattenBlock, rbmBlocks, outputBlock, None)


  def apply(convBlocks: Seq[ConvBlock], fullyConnected: FFNNModel, inputShape: Shape): CNNModel =
    new CNNModel(
      convBlocks,
      fullyConnected.getInputBlock,
      fullyConnected.getRbmBlocks,
      fullyConnected.getOutputBlock,
      Some(inputShape))

  def apply(convBlocks: Seq[ConvBlock], fullyConnected: FFNNModel): CNNModel =
    new CNNModel(
      convBlocks,
      fullyConnected.getInputBlock,
      fullyConnected.getRbmBlocks,
      fullyConnected.getOutputBlock,
      None)

}
