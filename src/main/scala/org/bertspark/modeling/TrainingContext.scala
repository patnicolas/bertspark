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
package org.bertspark.modeling

import ai.djl.engine.Engine
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener._
import ai.djl.training.loss._
import ai.djl.training.tracker._
import ai.djl.training.tracker.WarmUpTracker.Mode
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.optimizer._
import ai.djl.Device
import ai.djl.nn.transformer.BertPretrainingLoss
import ai.djl.training.initializer._
import java.util.concurrent.Executors
import org.bertspark.classifier.training._
import org.bertspark.config._
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.dl.model.NeuralModel.defaultOnSaveTrainingModel
import org.bertspark.transformer.training.TransformerTrainingListener
import org.slf4j._


/**
 * Training context used for both pre-training and classification models
 * @param optimizer Optimized used for pre-training or classification  [Shared across all models)
 * @param loss Loss function  - Not shared
 * @param numEpochs Number of epochs - shared
 * @param batchSize Size of the batch
 * @param numDevices Number of devices
 * @param evaluationMetrics Evaluation metrics - Not shared
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class TrainingContext private (
  optimizer: Optimizer,
  initializer: Initializer,
  loss: Loss,
  numEpochs: Int,
  batchSize: Int,
  numDevices: Int,
  evaluationMetrics: Seq[String],
  subModelName: String) {
  import TrainingContext._

  final def getInitializer: Initializer = initializer

  override def toString: String =
    s"""Optimizer      ${optimizer.toString}
       |Loss:          ${loss.toString}
       |Initializer    ${initializer.getClass.getName}
       |Num. epochs:   $numEpochs
       |Batch size:    $batchSize
       |Num. devices:  $numDevices
       |Eval. metrics: ${evaluationMetrics.mkString(" ")}
       |Sub model:     $subModelName
       |""".stripMargin

  val listener = new SaveModelTrainingListener("build/model")
  listener.setSaveModelCallback(defaultOnSaveTrainingModel)

  private[this] lazy val trainingConfig = {
    val devices = extractDevices
    new DefaultTrainingConfig(loss)
        .optOptimizer(optimizer)
        .optDevices(devices)
        .optExecutorService(executorService)
        .addEvaluator(new Accuracy())
        .addTrainingListeners(trainingListeners: _*)
        .addTrainingListeners(listener)
  }


  /**
   * retrieve the classifier loss option
   * @return Option of classifier loss if type is appropriate, None otherwise
   */
  final def getClassifierLoss: Option[ClassifierLoss] =
    if ((ExecutionMode.isClassifier || ExecutionMode.isHpo) && loss.isInstanceOf[ClassifierLoss])
      Some(loss.asInstanceOf[ClassifierLoss])
    else None

  def getTrainingListener: Option[BaseTrainingListener] =
    if(trainingListeners.last.isInstanceOf[BaseTrainingListener])
      Some(trainingListeners.last.asInstanceOf[BaseTrainingListener])
    else None

  /**
    * Retrieve the classifier training listener from the training context
    * @return Optional classifier training listener if correct type, None otherwise
    */
  def getClassifierTrainingListener: Option[ClassifierTrainingListener] =
    if(trainingListeners.last.isInstanceOf[ClassifierTrainingListener])
      Some(trainingListeners.last.asInstanceOf[ClassifierTrainingListener])
    else None

  /**
    * Retrieve the transformer pre-training listener from the training context
    * @return Optional transformer pre-training listener if correct type, None otherwise
    */
  def getTransformerTrainingListener: Option[TransformerTrainingListener] =
    if(trainingListeners.last.isInstanceOf[TransformerTrainingListener])
      Some(trainingListeners.last.asInstanceOf[TransformerTrainingListener])
    else None


  final def getEpochNo: Int = getTrainingListener.map(_.getEpochNo).getOrElse(-1)


  /**
   * Save comparison of predictions and labels
   */
  final def savePredictionLabels(epochNo: Int): Unit = getClassifierLoss.foreach(_.save(epochNo))


  final def getDefaultTrainingConfig: DefaultTrainingConfig = trainingConfig

  @inline
  final def getBatchSize: Int = batchSize

  @inline
  final def getNumEpochs: Int = numEpochs

  @inline
  final def getLossName: String = loss.getName

  @inline
  final def getEvaluationMetrics: Seq[String] = evaluationMetrics


  final def getDevices: Array[Device] = trainingConfig.getDevices


    // ----------------  Support methods -----------------------

  private def trainingListeners = Array[TrainingListener](
    new EpochTrainingListener(),
    new EvaluatorTrainingListener(),
    //  new DivergenceCheckTrainingListener(),
    new MemoryTrainingListener(),
    if(ExecutionMode.isPretraining) TransformerTrainingListener(this, subModelName)
    else ClassifierTrainingListener(this, subModelName)
  )

  // Device.of("gpu", 0)
  private def extractDevices: Array[Device] = mlopsConfiguration.executorConfig.dlDevice match {
    case "gpu" => Array.tabulate(numDevices)(Device.of("gpu", _))
    case "cpu" => Array[Device](Device.cpu())
    case "any" => Engine.getInstance.getDevices(numDevices)
  }
}


  /**
   * Custom constructors for the training context
   */
private[bertspark] final object TrainingContext {
    import MlopsConfiguration._
    final private val logger: Logger = LoggerFactory.getLogger("TrainingContext")


    private val executorService = Executors.newFixedThreadPool(mlopsConfiguration.executorConfig.numThreads)

    final def adamOptimizer =  {
      val optimizerConfig =
        if(ExecutionMode.isPretraining) mlopsConfiguration.preTrainConfig.optimizer
        else mlopsConfiguration.classifyConfig.optimizer

      val numSteps = optimizerConfig.numSteps
      val baseLearningRate = optimizerConfig.baseLr
      val epsilon = optimizerConfig.epsilon

      val mainTracker = PolynomialDecayTracker.builder
          .setBaseValue(baseLearningRate)
          .setEndLearningRate(baseLearningRate/numSteps)
          .setDecaySteps(numSteps*10)
          .optPower(1.0F)
          .build
      val learningRateTracker = WarmUpTracker.builder
          .optWarmUpBeginValue(0F)
          .optWarmUpSteps(numSteps)
          .optWarmUpMode(Mode.LINEAR)
          .setMainTracker(mainTracker)
          .build

      Adam.builder.optEpsilon(epsilon).optLearningRateTracker(learningRateTracker).build
    }

    def apply(
      optimizer: Optimizer,
      initializer: Initializer,
      loss: Loss,
      numEpochs: Int,
      batchSize: Int,
      numDevices: Int,
      evaluationMetrics: Seq[String],
      subModelName: String): TrainingContext =
      new TrainingContext(optimizer, initializer, loss, numEpochs, batchSize, numDevices, evaluationMetrics, subModelName)

    def apply(
      optimizer: Optimizer,
      initializer: Initializer,
      numEpochs: Int,
      batchSize: Int,
      loss: Loss,
      evaluationMetrics: Seq[String],
      subModelName: String): TrainingContext =
      apply(optimizer,initializer, loss, numEpochs, batchSize, 1, evaluationMetrics, subModelName)

    def apply(
      optimizer: Optimizer,
      initializer: Initializer,
      loss: Loss,
      batchSize: Int,
      numEpochs: Int,
      evaluationMetrics: Seq[String],
      subModelName: String): TrainingContext =
      apply(optimizer, initializer, loss, numEpochs, batchSize, 1, evaluationMetrics, subModelName)


    def apply(subModelName: String = ""): TrainingContext =
      if (subModelName.nonEmpty)
        apply(
          adamOptimizer,
          mlopsConfiguration.getInitializer,
          new SoftmaxCrossEntropyLoss(),
          mlopsConfiguration.executorConfig.batchSize,
          mlopsConfiguration.executorConfig.numDevices,
          mlopsConfiguration.classifyConfig.epochs,
          Seq[String]("Accuracy"),
          subModelName)
      else
        apply(
          adamOptimizer,
          new TruncatedNormalInitializer(0.15F),
          new BertPretrainingLoss(),
          mlopsConfiguration.executorConfig.batchSize,
          mlopsConfiguration.executorConfig.numDevices,
          mlopsConfiguration.preTrainConfig.epochs,
          Seq[String]("Accuracy"),
          subModelName)
  }

