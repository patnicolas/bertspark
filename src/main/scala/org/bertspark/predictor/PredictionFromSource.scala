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
package org.bertspark.predictor

import ai.djl.inference.Predictor
import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.training.dataset.Batch
import java.util.ArrayList
import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.modeling.TrainingLabelIndexing
import org.bertspark.nlp.trainingset.{KeyedValues, TokenizedTrainingSet, TrainingSet}
import org.bertspark.predictor.dataset.PredictorDataset
import org.bertspark.predictor.PredictionFromSource.{getSelectedClasses, logger}
import org.bertspark.predictor.model.DocIdPrediction
import org.bertspark.transformer.representation.PretrainingInference
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer


/**
  * {{{
  * Segregate the prediction workflow by source of data
  * - Dataset from S3  (PredictionFromS3)
  * - Dataset generated from Kafka stream  (PredictionFromKafka)
  *
  * Parameters:
  * scopedNdManager:  Sub scoped ND manager
  * classificationPredictor: Predictor for a given classifier sub-model
  * pretrainingInference Predictor for transformer
  * predictorDataset: Dataset input to the predictor
  * indexLabelMap: Indexed label (label -> index)
  * subClassifierModelName: Name of classifier sub model
  * tokenize: Convert Parameterized data set to TokenizedIndexedTraining set
  * }}}
  *
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] class PredictionFromSource(
  transformerModel: PretrainingInference,
  subClassifierModelPredictor: Predictor[NDList, NDList],
  subClassifierModelName: String) {

  /**
    * {{{
    * Execute the full prediction using
    * - Tokenizer
    * - Transformer model
    * - Neural Classifier
    * }}}
    *
    * @param ndManager NDManager for GPU
    * @param tokenizedIndexedDS Dataset of tokenized training set
    * @param sparkSession Implicit reference to the current Spark context
    * @return List of pairs (documentId: String, prediction: String)
    */
  def predict(
    ndManager: NDManager,
    tokenizedIndexedDS: Dataset[TokenizedTrainingSet]
  )(implicit sparkSession: SparkSession): List[DocIdPrediction] = {
    import sparkSession.implicits._

    logDebug(logger, msg = s"Predict ${tokenizedIndexedDS.count()} documents from source")

    // Produces the list for pair (documentId, CLSEmbedding)
    val clsPredictions: List[KeyedValues] = transformerModel.predict(
      ndManager,
      tokenizedIndexedDS.map(_.contextualDocument))
    logDebug(logger, msg = s"Got ${clsPredictions.size} CLS predictions")

   // transformerModel.close()

    // Build the Labeled DJL formatted data set
    val labelIndexMap = TrainingLabelIndexing.load
    val trainingSet = TrainingSet(tokenizedIndexedDS, labelIndexMap)

    // Retrieve the data set containing the input ot prediction
    val predictorDataset: PredictorDataset = trainingSet.toPredictionDataset(clsPredictions, subClassifierModelName)
    logDebug(logger, msg = s"Predictor dataset size: ${predictorDataset.getInputData.count()}")

    val keyedPredictions = predict(
      ndManager,
      subClassifierModelPredictor,
      predictorDataset,
      trainingSet.indexLabelMap)
    logDebug(logger, msg = s"Keyed predictions size: ${keyedPredictions.size}")

    val validKeyedPredictions = keyedPredictions.filter(_._2.nonEmpty)
    logDebug(logger, msg = s"Valid keyed predictions size: ${validKeyedPredictions.size}")

    // Complete the prediction with the sub model key
    validKeyedPredictions.map{ case (id, prediction) => (id, s"$subClassifierModelName:$prediction") }
  }

  /**
    * Generic prediction method give a ND manager, a classification predictor and a dataset
    * @param scopedNdManager Sub scoped ND manager
    * @param classificationPredictor Predictor for a given classifier sub-model
    * @param predictorDataset Dataset input to the predictor
    * @param indexLabelMap Indexed label (label -> index)
    * @return List of pair (DocumentId, prediction)
    */
  protected def predict(
    scopedNdManager: NDManager,
    classificationPredictor: Predictor[NDList, NDList],
    predictorDataset: PredictorDataset,
    indexLabelMap: Map[Int, String]): List[DocIdPrediction] = {
    import org.bertspark.implicits._

    val subNDManager = scopedNdManager.newSubManager()
    predictorDataset.prepare()

    // Execute prediction code
    val batchIterator: java.util.Iterator[Batch] = try {
      val batches = predictorDataset.getData(subNDManager)
      batches.iterator()
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"PredictionFromSource batch iterator failed ${e.getMessage}")
        val list: java.util.List[Batch] = new ArrayList[Batch]()
        list.iterator()
    }

    // Collect the predictions
    val predictions = ListBuffer[DocIdPrediction]()
    // Iterate across the various models
    while(batchIterator.hasNext) {
      val nextBatch: Batch = batchIterator.next()
      val nextData = nextBatch.getData
      val batchPrediction: NDList = classificationPredictor.predict(nextData)

      // Retrieve the first embedding in the batch
      val docIds = nextBatch.getIndices.map(_.asInstanceOf[String])
      (0 until batchPrediction.size()).foreach(
        index => {
          val shape = batchPrediction.get(index).getShape()
          val numLabels = shape.getShape()(1)
          // If there is more than one class, it needs to be predicted
          if(numLabels > 1) {
            logDebug(logger, msg = s"Predict $numLabels labels for $subClassifierModelName")
            val logSoftMax = batchPrediction.get(index).logSoftmax(-1)
            val predictedDocuments = getSelectedClasses(numLabels.toInt, indexLabelMap, logSoftMax)

            if(predictedDocuments.nonEmpty)
              predictions.append((docIds(index), predictedDocuments.head))
            else
              logger.warn(s"No predicted documents for $subClassifierModelName")
          }
            // If only one class or label associated with this sub-model
            // This case should not happen as such sub-model is treated as Oracle
          else {
            logDebug(logger, msg = "Predict from Kafka with 1 label")
            if(indexLabelMap.contains(0))
              predictions.append((docIds(index), indexLabelMap.get(0).get))
            else {
              logger.error(s"Prediction for ${docIds(index)}")
              predictions.append((docIds(index), "no_label"))
            }
          }
        }
      )
      logDebug(logger, msg = "Predict from source")
      /*
      if(subNDManager.isOpen)
        try {
          batchPrediction.close()
          logDebug(logger, msg = "batchPrediction.close")
          subNDManager.close()
        }
        catch {
          case e: IllegalStateException => logger.error(e.getMessage)
          case e: Exception => logger.error(e.getMessage)
        }

       */
    }

    if(predictions.isEmpty)
      logger.warn(s"No prediction is available for this sub model $subClassifierModelName")
    // close the predictor and the Bert pretrained model
    predictions.toList
  }
}

private[bertspark] final object PredictionFromSource {
  final protected val logger: Logger = LoggerFactory.getLogger("PredictionFromSource")

  private def getSelectedClasses(
    numClasses: Int,
    indexLabelsMap: Map[Int, String],
    ndPrediction: NDArray): Array[String] = {
    import org.bertspark.classifier.training.ClassifierLoss._

    val predictedValues = ndPrediction.toFloatArray
    val batchSize = (predictedValues.size/numClasses).floor.toInt
    val predictedSelectedIndices = getBestPrediction(predictedValues, batchSize, numClasses)
    predictedSelectedIndices
        .filter(indexLabelsMap.contains(_))
        .map(g => indexLabelsMap.get(g).get)
  }
}

