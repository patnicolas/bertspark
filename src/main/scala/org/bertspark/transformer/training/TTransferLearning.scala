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


import ai.djl.Model
import ai.djl.nn.transformer.BertPretrainingLoss
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.loss.Loss
import org.bertspark.{implicits, DLException, RuntimeSystemMonitor}
import org.bertspark.config.ExecutionMode
import org.bertspark.config.MlopsConfiguration.DebugLog.logInfo
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.dl.model.NeuralModel
import org.bertspark.modeling.ModelExecution
import org.bertspark.transformer.block.BERTFeaturizerBlock
import org.bertspark.transformer.config.BERTConfig
import org.bertspark.transformer.model.{TransformerModelLoader, TTransferLearningModel}
import org.bertspark.transformer.training.TPretraining.BaseTrainingSet
import org.slf4j.{Logger, LoggerFactory}


/**
  *
  */
private[bertspark] final class TTransferLearning
    extends ModelExecution
        with TransformerModelLoader
        with RuntimeSystemMonitor{
  import org.bertspark.implicits._
  import TTransferLearning._

  ExecutionMode.setTransferLearning

  override val preTrainingBlock = BERTFeaturizerBlock()
  private[this] val bertPreTrainingDataset = BaseTrainingSet.initTrainingSet

  private[this] val transferLearningModel =  new TTransferLearningModel(
    BERTConfig.getEmbeddingsSize(mlopsConfiguration.preTrainConfig.transformer),
    model.get)

  /**
    * Train the pre-training model
    */
  override protected def train(): Float = {
    var outputModel: Model = null
    try {
      logInfo(logger,  msg = s"Training set is ready!!")
      val loss: Loss = new BertPretrainingLoss()
      val trainingContext = NeuralModel.buildTrainingContext(new NormalInitializer, loss, "Transfer learning")
      // Train the model
      transferLearningModel.train(trainingContext, bertPreTrainingDataset, testDataset = null)
      // Update the state of the training set
      1.0F
    }
    catch {
      case e: DLException =>
        org.bertspark.printStackTrace(e)
        logger.error(s"DL failure ${e.getMessage}")
        0.0F
      case e: Exception =>
        org.bertspark.printStackTrace(e)
        logger.error(s"Unknown exception ${e.getMessage}")
        0.0F
    }
    finally {
      import implicits._
      if(outputModel != null)
        outputModel.close()
      close
    }
  }
}


private[bertspark] final object TTransferLearning {
  final private val logger: Logger = LoggerFactory.getLogger("TTransferLearning")

}

