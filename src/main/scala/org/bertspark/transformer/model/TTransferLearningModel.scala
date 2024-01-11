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
package org.bertspark.transformer.model

import ai.djl.engine.EngineException
import ai.djl.ndarray.NDList
import ai.djl.nn.transformer.BertPretrainingBlock
import ai.djl.repository.zoo.ZooModel
import ai.djl.training._
import java.io.{File, IOException}
import org.apache.spark.sql.SparkSession
import org.bertspark._
import org.bertspark.config.MlopsConfiguration.DebugLog._
import org.bertspark.dl.model.NeuralModel
import org.bertspark.nlp.trainingset.ContextualDocument
import org.bertspark.transformer.config.BERTConfig
import org.bertspark.transformer.dataset.PretrainingDataset
import org.bertspark.config.MlopsConfiguration.{mlopsConfiguration, vocabulary}
import org.bertspark.modeling.TrainingContext
import org.slf4j.{Logger, LoggerFactory}


/**
  * Implement the Transfer learning model to upgrade the weights of the pre-trained model
  * @param embeddingSize Size of embedding
  * @param pretrainedTransformer Transformer model pre-trained
  * @param sparkSession Implicit reference to the current Spark context
  *
  * @author Patrick Nicolas
  * @version 0.7
  */
private[bertspark] final class TTransferLearningModel (
  embeddingSize: Long,
  pretrainedTransformer: ZooModel[NDList, NDList],
)(implicit sparkSession: SparkSession) extends NeuralModel[TTransferLearningModel] {
  import TTransferLearningModel._


  private[this] val pretrainingBertBlock = {
    val bertModel = BERTConfig.getBertConfig(mlopsConfiguration.preTrainConfig.transformer)
    new BertPretrainingBlock(bertModel.setTokenDictionarySize(Math.toIntExact(vocabulary.size)))
  }

  @throws(clazz = classOf[DLException])
  override def train(
    trainingCtx: TrainingContext,
    trainingDataset: DjlDataset,
    testDataset: DjlDataset,
    subModelName: String): String = try {

    logDebug(logger, msg = s"Training context devices: ${trainingCtx.getDevices.map(_.getDeviceType).mkString(" ")}")

    // Instantiate the model as an instance of BERT data set of medical notes
    val preTrainingDataset = convertType[PretrainingDataset[ContextualDocument]](trainingDataset)
    val trainer = createTrainer(trainingCtx, preTrainingDataset)
    logInfo(logger,  msg = "Trainer initialized")

    // Launch training without validation set
    val validateDataset = null
    EasyTrain.fit(trainer, trainingCtx.getNumEpochs, preTrainingDataset, validateDataset)
    logDebug(logger, msg = s"RSS memory history: ${RuntimeSystemMonitor.rssHistory.mkString("\n")}")

    val modelPath = trainer.getModel.getModelPath.toString
    trainer.getModel.close()
    modelPath
  }
  catch {
    case e: EngineException =>
      org.bertspark.error[EngineException, String](msg = "DL engine failed:", e)
    case e: IllegalArgumentException =>
      org.bertspark.error[IllegalArgumentException, String](msg = "Incorrect arguments:", e)
    case e: IllegalStateException =>
      org.bertspark.error[IllegalStateException, String](msg = "Undefined state:", e)
    case e: IndexOutOfBoundsException =>
      org.bertspark.error[IndexOutOfBoundsException, String](msg = "Index out of bounds:", e)
    case e: IOException =>
      org.bertspark.error[IOException, String](msg = "I/O failed:", e)
    case e: Exception =>
      org.bertspark.error[Exception, String](msg = "Undefined exception:", e)
    // Treats the out of memory failure uniquely
    case e: OutOfMemoryError =>
      org.bertspark.releaseReserve
      printStackTrace(e)
      throw e
  }

  private def createTrainer(
    trainingCtx: TrainingContext,
    preTrainingDataset: PretrainingDataset[ContextualDocument]): Trainer = {

    // Condition to either load or create a new model (if the model is undefined
    if(isModelExist)
      throw new IllegalStateException(s"Transformer model ${mlopsConfiguration.runId} already exists!")

    pretrainedTransformer.setBlock(pretrainingBertBlock)
    val inputShape = initShape(preTrainingDataset, embeddingSize)
    getTrainer(pretrainedTransformer, trainingCtx, inputShape)
  }
}


private[bertspark] final object TTransferLearningModel {
  final private val logger: Logger = LoggerFactory.getLogger("TTransferLearningModel")
}