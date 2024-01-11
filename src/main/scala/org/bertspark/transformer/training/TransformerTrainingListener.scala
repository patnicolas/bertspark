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
package org.bertspark.transformer.training

import ai.djl.training.Trainer
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.config.{ExecutionMode, S3PathNames}
import org.bertspark.modeling.{BaseTrainingListener, TrainingContext}
import org.bertspark.transformer.model.TransformerModelSaver
import org.slf4j.{Logger, LoggerFactory}


/**
 * Listener for the training of transformer
 * @param trainingContext Training context associated with this transformer
 * @param subModelName Name of the transformer model
 * @see  org.mlops.modeling.BaseTrainingListener
 *
 * @author Patrick Nicolas
 * @version 0.3
 */
private[bertspark] final class TransformerTrainingListener private (
  trainingContext: TrainingContext,
  override val subModelName: String)
    extends BaseTrainingListener(trainingContext, subModelName)
        with TransformerModelSaver {

  /**
   * Save the content of the accumulated metrics into a local file
   * @param trainer Reference to the current trainer
   * @param enabled Enabled saving the model for pre-training into local file and S3
   */
  override protected def record(trainer: Trainer, enabled: Boolean): Unit =
    if(ExecutionMode.isPretraining && enabled) {
      val metricData = getMetrics()
      save(trainer.getModel, getEpochNo, metricData)
    }
}


private[bertspark] final object TransformerTrainingListener {
  final private val logger: Logger = LoggerFactory.getLogger("TransformerTrainingListener")
  /**
   * Default constructor
   * @param trainingContext Training context
   * @param descriptor Description of the test
   */
  def apply(
    trainingContext: TrainingContext,
    descriptor: String
  ): TransformerTrainingListener = new TransformerTrainingListener(trainingContext, descriptor)
}