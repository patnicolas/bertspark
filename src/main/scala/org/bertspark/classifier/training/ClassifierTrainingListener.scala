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
package org.bertspark.classifier.training

import ai.djl.training.Trainer
import org.bertspark._
import org.bertspark.classifier.model.ClassifierModelSaver
import org.bertspark.classifier.training.ClassifierTrainingListener.{coreMetricsLbl, logger, strictMetricsLbl}
import org.bertspark.config.ExecutionMode
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.modeling.{BaseTrainingListener, TrainingContext}
import org.slf4j.{Logger, LoggerFactory}


/**
 *  Listener dedicated to training of classifier
 * @param trainingContext Training context
 * @param subModelName Description of the test
 * @see org.mlops.modeling.BaseTrainingListener
 *
 * @author Patrick Nicolas
 * @version 0.3
 */
private[bertspark] final class ClassifierTrainingListener private (
  override val trainingContext: TrainingContext,
  override val subModelName: String)
    extends BaseTrainingListener(trainingContext, subModelName)
        with ClassifierModelSaver
        with ClassifierConvergence {
  import org.bertspark.config.MlopsConfiguration._

  private[this] var counter = 0
  final val numEpochBetweenSaves = (mlopsConfiguration.classifyConfig.epochs/8.0).floor.toInt

  /**
   * Save the content of the accumulated metrics and models into local file and S3
   * @param trainer Reference to the current trainer
   * @param enabled Enabled saving the model for classification into local file and S3
   */
  override protected def record(trainer: Trainer, enabled: Boolean): Unit =
    // Apply only if this is a classifier or Hyper-parameter optimization
    if((ExecutionMode.isClassifier || ExecutionMode.isHpo) && (counter % numEpochBetweenSaves) == 0 ) {
      counter += 1
      val subModelPath = subModelName.replace(",", " ")
      collectQualityMetrics

      // Catch an exception if converged.
     val converged = try {
        hasConverged(metricAccumulator, epochNo.get(), subModelPath)
        false
      }
      catch {
        case e: HasConvergedException => true
      }

      val numEpochs = epochNo.get()
      // Record the metrics and sub-model parameters if converged or if max mum of epochs is reached
      if(converged || numEpochs >= trainingContext.getEpochNo) {
        val metricData = getMetrics(subModelPath)
        save(trainer.getModel, numEpochs, metricData)
      }

      // If converge re-throw the exception to exit the training cycle
      if(converged) {
        val convergenceMessage = s"Classifier for $subModelName after $numEpochs epochs"
        logDebug(logger, convergenceMessage)
        throw new HasConvergedException(convergenceMessage)
      }
    }

  private def collectQualityMetrics: Unit = trainingContext.getClassifierLoss.foreach(
    loss => {
      val (strictMetrics, coreMetrics) = loss.getMetrics
      val strictMap = strictMetrics.toMap(strictMetricsLbl)
      add(strictMap)
      val coreMap = coreMetrics.toMap(coreMetricsLbl)
      add(coreMap)
    }
  )
}


/**
  * Singleton for constructor
  */
private[bertspark] final object ClassifierTrainingListener {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[ClassifierTrainingListener])

  final val strictMetricsLbl = "strict"
  final val coreMetricsLbl = "core"

  /**
   * Default constructor
   * @param trainingContext Training context
   * @param descriptor Description of the test
   */
  def apply(
    trainingContext: TrainingContext,
    descriptor: String
  ): ClassifierTrainingListener = new ClassifierTrainingListener(trainingContext, descriptor)
}