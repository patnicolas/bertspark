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
package org.bertspark.transformer

import ai.djl.ndarray.types.Shape
import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.training.Trainer
import java.io.File
import org.bertspark.getPretrainingModelPath
import org.bertspark.modeling.TrainingContext
import org.bertspark.nlp.trainingset.ContextualDocument
import org.bertspark.transformer.dataset.PretrainingDataset


package object model {

  final def isModelExist: Boolean = {
    val file = new File(getPretrainingModelPath)
    file.exists() && file.isDirectory
  }

  def initShape(dataset: PretrainingDataset[ContextualDocument], embeddingSize: Long): Shape = {
    val datasetShape: Shape = dataset.getShape
    val maxSeqLength = datasetShape.getShape()(1)
    new Shape(maxSeqLength, embeddingSize)
  }

  def getTrainer(model: Model, trainingCtx: TrainingContext, inputShape: Shape): Trainer = {
    val trainingConfig = trainingCtx.getDefaultTrainingConfig
    val trainer: Trainer = model.newTrainer(trainingConfig)
    trainer.setMetrics(new Metrics())
    trainer.initialize(inputShape, inputShape, inputShape, inputShape)
    trainer
  }
}
