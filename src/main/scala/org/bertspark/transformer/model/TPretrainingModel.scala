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

import ai.djl.training._
import ai.djl.Model
import ai.djl.engine.EngineException
import java.io._
import java.nio.file.Paths
import org.bertspark._
import org.bertspark.config.FsPathNames
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.dl.model.NeuralModel
import org.bertspark.transformer.block.{CustomPretrainingBlock, PretrainingModule}
import org.bertspark.transformer.config.BERTConfig
import org.bertspark.transformer.dataset.PretrainingDataset
import org.bertspark.config.MlopsConfiguration.DebugLog.{logDebug, logInfo}
import org.bertspark.modeling.{TrainingContext, TrainingSummary}
import org.bertspark.nlp.trainingset.ContextualDocument
import org.bertspark.transformer.model.TPretrainingModel.logger
import org.slf4j._


/**
 * Implementation of BERT pre-training model and related training algorithm
 * {{{
 *   The pre-training model is used for
 *   - Pre-training ('train')
 *   - Predicting [CLS] feature for fine-tuning or downstream applications
 *
 * The prediction of CLS tag is the input to the classifier..
 * }}}
 * @param pretrainingModule Pretraining block module
 * @param embeddingSize Size of embeddings
 * @param modelName Name of model for logging and storing purpose
 * @todo Replace  bertPreTrainingBlock: BERTPretrainingBlock by TPretrainingBlock as adapter
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class TPretrainingModel (
  pretrainingModule: PretrainingModule,
  embeddingSize: Long,
  modelName: String
) extends NeuralModel[TPretrainingModel] with TrainingSummary {

  /**
    * Train the Transformer model
    * @param trainingCtx Training context
    * @param trainingDataset DJL training set
    * @param testDataset DJL validation set
    * @param subModelName Ignore (just for polymorphic call)
    * @throws DLException if training fails
    * @return Path of the S3 folder the model is saved to
    */
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

  /**
   * Compute and record the system metrics (Memory, CPU, GPU....)
   * @param trainingCtx Training context
   * @param trainingDataset Training DJL data set
   * @return Last model
   */
  def evaluateSystemMetrics(
    trainingCtx: TrainingContext,
    trainingDataset: DjlDataset
  ): String = try {
    System.setProperty("collect-memory", "true")

    val preTrainingDataset = convertType[PretrainingDataset[ContextualDocument]](trainingDataset)
    val trainer = createTrainer(trainingCtx, preTrainingDataset)
    val validateDataset = null

    EasyTrain.fit(trainer, trainingCtx.getNumEpochs, preTrainingDataset, validateDataset)
    val modelPath = trainer.getModel.getModelPath.toString
    trainer.getModel.close()
    modelPath
  }
  catch {
    case e: Exception =>
      org.bertspark.error[Exception, String](msg = "Undefined exception:", e)
    // Treats the out of memory failure uniquely
    case e: OutOfMemoryError =>
      org.bertspark.releaseReserve
      printStackTrace(e)
      throw e
  }


  @inline
  final def getEmbeddingSize: Long = embeddingSize

  override def toString: String =
    s"Block: ${pretrainingModule.toString}\nEmbedding size: ${embeddingSize}\nModel: $modelName"



  // ----------------------------------  Supporting/helper methods ------------------------------

  private def createTrainer(
    trainingCtx: TrainingContext,
    preTrainingDataset: PretrainingDataset[ContextualDocument]): Trainer = {

    // Condition to either load or create a new model (if the model is undefined
    if(isModelExist)
      throw new IllegalStateException(s"Pre-training model ${mlopsConfiguration.runId} already exists!")

    val model = Model.newInstance(modelName)
    // DEBUG
    model.setBlock(pretrainingModule.getPretrainingBlock)
    // END DEBUG
    val inputShape = initShape(preTrainingDataset, embeddingSize)
    getTrainer(model, trainingCtx, inputShape)
  }
}



private[bertspark] final object TPretrainingModel {
  final private val logger: Logger = LoggerFactory.getLogger("TPretrainingModel")

  /**
   * Load model from local file defined by the modelPath and the model name
   * @param vocabularySize Size of vocabulary
   * @throws DLException If the model was not found or improperly formatted
   * @return model
   */
  def loadModel(vocabularySize: Long): Model = try {
    val modelName = FsPathNames.getModelName

    val bertConfig = BERTConfig(modelName, mlopsConfiguration.getTransformer, vocabularySize)
    val bertPreTrainingBlock = CustomPretrainingBlock(bertConfig)

    val model = Model.newInstance(modelName)
    model.setBlock(bertPreTrainingBlock)
    val modelPath = getPretrainingModelPath
    model.load(Paths.get(modelPath), modelName)
    model
  }
  catch {
    case e: FileNotFoundException =>
      error[FileNotFoundException, Model](msg = s"Could not find model for $getPretrainingModelPath/${FsPathNames.getModelName}", e)
    case e: IOException =>
      error[IOException, Model](msg = s"I/O Failure for $getPretrainingModelPath/${FsPathNames.getModelName}", e)
    case e: Exception =>
      error[Exception, Model](msg = s"Undefined failure loading $getPretrainingModelPath/${FsPathNames.getModelName}", e)
  }
}