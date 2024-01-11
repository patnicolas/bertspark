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

import ai.djl.Model
import ai.djl.training._
import ai.djl.training.initializer.Initializer
import ai.djl.training.loss.{Loss, SoftmaxCrossEntropyLoss}
import java.util.function.Consumer
import org.bertspark.dl.config.Optimizers
import org.bertspark.DjlDataset
import org.bertspark.classifier.training.ClassifierLoss
import org.bertspark.modeling.TrainingContext
import org.slf4j.{Logger, LoggerFactory}


/**
 * Generic Neural Network training
 * @tparam T Type of Neural Network
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] trait NeuralModel[T <: NeuralModel[T]] {

  def train(
    trainingCtx: TrainingContext,
    trainIter: DjlDataset,
    testIter: DjlDataset,
    subModelName: String = ""): String
}


/**
 * Singleton to build the training context
 */
final object NeuralModel {
  import org.bertspark.config.MlopsConfiguration._
  final private val logger: Logger = LoggerFactory.getLogger("NeuralModel")

  final val defaultOnSaveTrainingModel: Consumer[Trainer] =
    _defaultOnSaveModel(
      (res: TrainingResult) => res.getTrainEvaluation("Accuracy"),
      (res: TrainingResult) => res.getTrainLoss()
    )


  final val defaultOnSaveValidationModel: Consumer[Trainer] =
    _defaultOnSaveModel(
      (res: TrainingResult) => res.getValidateEvaluation("Accuracy"),
      (res: TrainingResult) => res.getValidateLoss()
    )


  private def _defaultOnSaveModel(
    accuracyFunc: TrainingResult => Float,
    lossFunc: TrainingResult => Float): Consumer[Trainer] = (trainer: Trainer) => {
    val result = trainer.getTrainingResult
    val model = trainer.getModel
    val accuracy = accuracyFunc(result)
    val loss = lossFunc(result)
    model.setProperty("Accuracy", "%.5f".format(accuracy))
    model.setProperty("Loss", "%.6f".format(loss))
  }

  /**
   * Get training context which depends on the loss function
   * @param indexLabelsMap Map of index to string for the labels
   * @param subModelName Name of sub-model
   * @return Training context
   */
  @throws(clazz = classOf[UnsupportedOperationException])
  def buildTrainingContext(indexLabelsMap: Map[Int, String], subModelName: String): TrainingContext = {
    val loss =
      if(mlopsConfiguration.classifyConfig.isMlopsLoss)
        ClassifierLoss(indexLabelsMap, "loss", false, true, subModelName)
      else
        new SoftmaxCrossEntropyLoss("loss", 1, -(1), false, true)
    val initializer: Initializer = mlopsConfiguration.getInitializer
    buildTrainingContext(initializer, loss, subModelName)
  }

  /**
   * Get training context which depends on the loss function
   * @param loss Loss function
   * @param subModelName Name of submodel
   * @return Training context
   */
  @throws(clazz = classOf[UnsupportedOperationException])
  def buildTrainingContext(initializer: Initializer, loss: Loss, subModelName: String): TrainingContext = {
    val evaluationMetrics = Seq[String]("Accuracy")
    val (optimizerConfig, numEpochs, batchSize) = mlopsConfiguration.getTrainingParams

    val decayStepsFactor = 100
    val optimizer = Optimizers.adam(optimizerConfig, decayStepsFactor)
    val numDevices = mlopsConfiguration.executorConfig.numDevices

    val lossName = loss.getClass.getSimpleName
    if(isLossValid(lossName))
      TrainingContext(
        optimizer,
        initializer,
        loss,
        numEpochs,
        batchSize,
        numDevices,
        evaluationMetrics,
        subModelName)
    else
      throw new UnsupportedOperationException(s"Loss type: $lossName is not supported")
  }

  private def isLossValid(lossName: String): Boolean =
    lossName == "BertPretrainingLoss" ||
        lossName == "BertMaskedLanguageModelLoss" ||
        lossName == "SoftmaxCrossEntropyLoss" ||
        lossName == "ClassifierLoss" ||
        lossName == "TPretrainingLoss"
}
