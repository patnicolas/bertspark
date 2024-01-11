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
package org.bertspark.classifier.model

import ai.djl.Model
import ai.djl.engine.EngineException
import ai.djl.metric.Metrics
import ai.djl.ndarray._
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Parameter
import ai.djl.training._
import ai.djl.translate.TranslateException
import java.io.IOException
import org.apache.spark.sql._
import org.bertspark._
import org.bertspark.config.FsPathNames
import org.bertspark.nlp.trainingset._
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.dl.model.NeuralModel
import org.bertspark.nlp.medical.LabeledDjlDataset
import org.bertspark.transformer.block.CustomPretrainingBlock
import org.bertspark.transformer.model.TPretrainingModel
import org.bertspark.classifier.block.ClassificationBlock
import org.bertspark.classifier.training.TClassifier.predictionCount
import org.bertspark.config.MlopsConfiguration.DebugLog.{logDebug, logTrace}
import org.bertspark.modeling.{TrainingContext, TrainingSummary}
import org.slf4j._


/**
 * Define the BERT classifier model as
 * {{{
 *   - BERT pre-trained model encoder
 *   - Classifier/decoder which is Fully connected network
 *  The training data is of type (Prediction request, index of class or labels)
 *  The progress display relies on the TrainingSummary mixin
 * }}}
 * @param sparkSession Implicit reference to the current Spark context
 * @param encoder Encoder for the train data
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class TClassifierModel (
  implicit sparkSession: SparkSession, encoder: Encoder[TrainData]
) extends NeuralModel[TClassifierModel] with TrainingSummary {
  import TClassifierModel._


  /**
    * Train the classifier from Tokenizer indexed training set
    * {{{
    *   Structure of TokenizedIndexedTrainingSet:  [ContextualDocument, label, CLS embedding]
    * }}}
    * @param labeledDjlDataset Labeled DJL dataset
    * @param subModelName Identifier for the sub-model
    * @todo return (numClasses, Accuracy)
    * @return Number of classes associated with this model.
    */
  @throws(clazz = classOf[DLException])
  @throws(clazz = classOf[IllegalStateException])
  def train(labeledDjlDataset: LabeledDjlDataset, subModelName: String): (Long, Float) = {
    import org.bertspark.config.MlopsConfiguration._

    var classificationModel: Option[Model] = None
    var trainingContext: Option[TrainingContext] = None

    try {
      trainingContext = Some(NeuralModel.buildTrainingContext(labeledDjlDataset.getIndexLabelsMap, subModelName))
      val numClasses = labeledDjlDataset.getNumClasses

      // @todo compute the ratio data size / num classes
      logDebug(
        logger,
        s"Train classifier for $subModelName with $numClasses classes from ${mlopsConfiguration.target}"
      )

      (for {
        classificationTrainingDataset <- labeledDjlDataset.djlTrainingData
        classificationValidationDataset <- labeledDjlDataset.djlValidationData
      } yield {
        // Instantiate the model and initialize the trainer.
        classificationModel = Some(Model.newInstance(FsPathNames.getPreTrainModelOutput))
        val trainer = createTrainer(trainingContext.get, classificationModel.get, numClasses)

        // Launch training. Convergence throws an HasConvergedException
        try {
          EasyTrain.fit(
            trainer,
            trainingContext.get.getNumEpochs,
            classificationTrainingDataset,
            classificationValidationDataset
          )
          trainer.notifyListeners(_.onTrainingEnd(trainer))
        }
        catch {
          case e: HasConvergedException => logger.warn(e.getMessage)
        }

        val accuracyMetrics = trainer.getMetrics.getMetric("train_epoch_Accuracy")
        val accuracyMetric = accuracyMetrics.get(accuracyMetrics.size()-1)
        logDebug(logger, s"Classifier model $subModelName trained with an accuracy ${accuracyMetric.getValue.toFloat}")

        classificationModel.foreach(
          model =>
            if(model.getNDManager().isOpen) {
              logDebug(logger, "Closing the classification model..")
              model.close()
            }
            else
              logger.warn(s"Classifier model was already closed!")
        )
        (numClasses, accuracyMetric.getValue.toFloat)
      }).getOrElse(
        throw new IllegalStateException("Training or validation set are empty")
      )
    }
    catch {
      case e: EngineException =>
        error[EngineException, (Long, Float)]("Improper selection of DL framework:", e)
      case e: IllegalArgumentException =>
        error[IllegalArgumentException, (Long, Float)]("Incorrect arguments:", e)
      case e: IndexOutOfBoundsException =>
        error[IndexOutOfBoundsException, (Long, Float)]("Index out of bounds:", e)
      case e: IOException =>
        error[IOException, (Long, Float)]("I/O failed: ", e)
      case e: TranslateException =>
        error[TranslateException, (Long, Float)]("Translation failed:", e)
      case e: IllegalStateException =>
        error[IllegalStateException, (Long, Float)]("Illegal state:", e)
      case e: Exception =>
        error[Exception, (Long, Float)]("Undefined exception:", e)
    }
  }

  // ---------------------  Supporting methods --------------------------------

  private def createTrainer(
    trainingContext: TrainingContext,
    classificationModel: Model,
    numClasses: Long): Trainer = {

    val classificationBlock = new ClassificationBlock(numClasses)
    classificationBlock.setInitializer(trainingContext.getInitializer, Parameter.Type.WEIGHT)

    classificationModel.setBlock(classificationBlock)
    val trainer: Trainer = classificationModel.newTrainer(trainingContext.getDefaultTrainingConfig)
    trainer.setMetrics(new Metrics())

    val inputShape = new Shape(mlopsConfiguration.getPredictionOutputSize)
    trainer.initialize(inputShape)
    logDebug(logger,  s"Trainer initialized for $numClasses classes")
    trainer
  }

  private def log(
    logger: Logger,
    tokenizedIndexedDS: Dataset[TokenizedTrainingSet],
    subModelName: String,
  ): Unit = logTrace(
    logger,
    {
      val tokenizedInput = tokenizedIndexedDS.collect
      val cnt = predictionCount.addAndGet(tokenizedInput.size)
      val str1 = s"${mlopsConfiguration.target}/$subModelName with $tokenizedInput records total: ${cnt}"
      val str2 = tokenizedInput.map(
        tokenizedIndex => s"${tokenizedIndex.label} ${tokenizedIndex.contextualDocument.summary}"
      ).take(10).mkString("\n")
      s"Start training the classifier for $str1\n$str2"
    }
  )

  /**
   * Train a BERT classifier using a DJL training and validation set given a training context
   * {{{
   *  - The model is saved into local files
   *  - The display relies on the TrainingSummary mixin
   * }}}
   * @param trainingCtx Training context (Parameters)
   * @param trainingDataset DJL training set
   * @param validationDataset DJL validation set
   * @param subModelName Name of sub-model
   * @return Model
   */
  @throws(clazz = classOf[DLException])
  override def train(
    trainingCtx: TrainingContext,
    trainingDataset: DjlDataset,
    validationDataset: DjlDataset,
    subModelName: String): String = ???
}




private[bertspark] final object TClassifierModel {
  import TPretrainingModel._
  final private val logger: Logger = LoggerFactory.getLogger("TClassifierModel")


  @throws(clazz = classOf[IllegalStateException])
  def getTPreTrainedModel: Model = loadModel(vocabulary.size())

  @throws(clazz = classOf[IllegalStateException])
  def getTPreTrainedBlock: CustomPretrainingBlock = getTPreTrainedModel.getBlock.asInstanceOf[CustomPretrainingBlock]
}