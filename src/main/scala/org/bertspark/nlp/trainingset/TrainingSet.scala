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
 */
package org.bertspark.nlp.trainingset

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql._
import org.bertspark.classifier.dataset.ClassifierDataset
import org.bertspark.predictor.dataset.PredictorDataset
import org.bertspark.transformer.dataset._
import org.bertspark.util.io.S3Util
import org.slf4j._
import scala.collection.mutable.ListBuffer


/**
 * Training set builder: Build a DJL dataset from a Spark dataset of labeled requests
 * @param tokenizedIndexedDS Labeled training set loaded from S3
 * @see org.mlops.medicalcoding.CodingTypes
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class TrainingSet private (
  tokenizedIndexedDS: Dataset[TokenizedTrainingSet],
  initialLabelIndexMap: Map[String, Int]
)(implicit sparkSession: SparkSession, encoder: Encoder[(String, Int)]) {

  // @todo review the issue with distinct labels
  private[this] val labelIndexMap: Map[String, Int] =
    if(initialLabelIndexMap.isEmpty) {
      import sparkSession.implicits._
      tokenizedIndexedDS.map(_.label).distinct().collect.zipWithIndex.toMap
    }
    else
      initialLabelIndexMap

  lazy val indexLabelMap: Map[Int, String] = labelIndexMap.map{ case (k, v) => (v, k)}


  final def getLabeledRequestDS(numRecords: Int): Dataset[TokenizedTrainingSet] =
    if(numRecords <= 0) tokenizedIndexedDS else tokenizedIndexedDS.limit(numRecords)

  final def getLabeledRequests(numRecords: Int): Seq[TokenizedTrainingSet] =
    getLabeledRequestDS(numRecords).collect


  /**
   * Split training and validation data sets
   * @param randomSplit Ratio for random split
   * @param encoder Implicit encoder for the labeled training data
   * @return Pair (Training, Validation) Spark data sets
   */
  final def splitTrainingValidation(
    randomSplit: Double
  )(implicit encoder: Encoder[TokenizedTrainingSet]
  ): (Dataset[TokenizedTrainingSet], Dataset[TokenizedTrainingSet]) = {
    val trainDataDSSplits = tokenizedIndexedDS.randomSplit(Array[Double](randomSplit, 1.0-randomSplit))
    (trainDataDSSplits.head, trainDataDSSplits(1))
  }


  /**
   * Convert Labeled training data loaded from S3 into a pair of training and validation DJL datasets
   * @param randomSplit Random split ratio for training and validation
   * @param subModelName Name or identifier for the sub model
   * @param encoder Implicit encoder for type Labeled Training Data
   * @return Pair of Labeled DJL for Training and Validation
   */
  final def toClassificationDjlDataset(
    randomSplit: Double,
    subModelName: String
  )(implicit encoder: Encoder[TokenizedTrainingSet]
  ): (PretrainingDataset[ContextualDocument], PretrainingDataset[ContextualDocument]) = {
    val (labeledTrainDataDS, labeledValidateDataDS) = splitTrainingValidation(randomSplit)
    extractBertDatasets(labeledTrainDataDS, labeledValidateDataDS, subModelName)
  }


  /**
   * {{{
   * Generate a BERT classifier dataset of type:
   *   (labeledDataDS: Dataset[LabeledEmbedding], numClasses: Long, subModelName: String)
   * }}}
   * @param randomSplit Random split ratio for training and validation
   * @param idPredictionsSeq Sequence of keyed predictor as list of (String, Array[Float])
   * @param subModelName Name of the sub model
   * @param enc1 Implicit encoder for type Labeled Training Data for Labeled embedding  (Array[Float], Int)
   * @return Pair of Labeled DJL for Training and Validation
   */
  final def toClassificationDjlDataset(
    randomSplit: Double,
    idPredictionsSeq: List[KeyedValues],
    subModelName: String
  )(implicit enc1: Encoder[LabelEmbedding],
    enc2: Encoder[TokenizedTrainingSet]): (ClassifierDataset, ClassifierDataset) = {
    // Split into training and validation data set
    val (labeledTrainDataDS, labeledValidateDataDS) = splitTrainingValidation(randomSplit)

    val reducedLabeledTrainDataDS = reduceLabeledDataDS(labeledTrainDataDS, idPredictionsSeq)
    val reducedLabeledValidateDataDS = reduceLabeledDataDS(labeledValidateDataDS, idPredictionsSeq)

    val bertTrainingDataset = new ClassifierDataset(reducedLabeledTrainDataDS, subModelName, labelIndexMap)
    val bertValidationDataset = new ClassifierDataset(reducedLabeledValidateDataDS, subModelName, labelIndexMap)
    (bertTrainingDataset, bertValidationDataset)
  }


  final def toPredictionDataset(
    idPredictionsSeq: List[KeyedValues],
    subModelName: String
  )(implicit enc1: Encoder[KeyedValues],
    enc2: Encoder[TokenizedTrainingSet]): PredictorDataset = {
    val reducedDataDS = reduceDataDS(tokenizedIndexedDS, idPredictionsSeq)
    new PredictorDataset(reducedDataDS)
  }

  /**
   *
   * @param idPredictionsSeq
   * @param subModelName
   * @param enc1
   * @param enc2
   * @return
   */
  final def toClassificationDjlDataset(
    idPredictionsSeq: List[KeyedValues],
    subModelName: String
  )(implicit enc1: Encoder[LabelEmbedding],
    enc2: Encoder[TokenizedTrainingSet]): ClassifierDataset = {
    // Split into training and validation data set

    val reducedDataDS = reduceLabeledDataDS(tokenizedIndexedDS, idPredictionsSeq)
    new ClassifierDataset(reducedDataDS, subModelName, labelIndexMap)
  }



      // ----------------------   Supporting methods --------------------------------

  private def reduceLabeledDataDS(
    labeledDataDS: Dataset[TokenizedTrainingSet],
    idNdPredictions: Seq[KeyedValues]
  )(implicit encoder: Encoder[LabelEmbedding]): Dataset[LabelEmbedding] = {
    val idNdPredictionsBrdCast: Broadcast[Seq[KeyedValues]] =
      sparkSession.sparkContext.broadcast[Seq[KeyedValues]](idNdPredictions)

    labeledDataDS.mapPartitions(
      (labeledDataIter: Iterator[TokenizedTrainingSet]) => {
        val idNdPredictionsMap = idNdPredictionsBrdCast.value.toMap
        val collector = ListBuffer[LabelEmbedding]()

        while(labeledDataIter.hasNext) {
          val labeledData: TokenizedTrainingSet = labeledDataIter.next
          if(idNdPredictionsMap.contains(labeledData.contextualDocument.id)) {
            val clsEmbedding = idNdPredictionsMap.get(labeledData.contextualDocument.id).get
            collector.append((clsEmbedding, labeledData.label))
          }
        }
        collector.iterator
      }
    )
  }

  private def reduceDataDS(
    tokenizerIndexedInputDataDS: Dataset[TokenizedTrainingSet],
    idNdPredictions: Seq[KeyedValues]
  )(implicit encoder: Encoder[KeyedValues]): Dataset[KeyedValues] = {
    val idNdPredictionsBrdCast: Broadcast[Seq[KeyedValues]] =
      sparkSession.sparkContext.broadcast[Seq[KeyedValues]](idNdPredictions)

    tokenizerIndexedInputDataDS.mapPartitions(
      (tokenizerIndexedInputDataIter: Iterator[TokenizedTrainingSet]) => {
        val idNdPredictionsMap = idNdPredictionsBrdCast.value.toMap
        val collector = ListBuffer[KeyedValues]()

        while(tokenizerIndexedInputDataIter.hasNext) {
          val labeledData: TokenizedTrainingSet = tokenizerIndexedInputDataIter.next
          if(idNdPredictionsMap.contains(labeledData.contextualDocument.id)) {
            val clsEmbedding = idNdPredictionsMap.get(labeledData.contextualDocument.id).get
            collector.append((labeledData.contextualDocument.id, clsEmbedding))
          }
        }
        collector.iterator
      }
    )
  }


  private def extractBertDatasets(
    trainingData: Dataset[TokenizedTrainingSet],
    validationData: Dataset[TokenizedTrainingSet],
    subModelName: String
  )(implicit sparkSession: SparkSession): (PretrainingDataset[ContextualDocument], PretrainingDataset[ContextualDocument]) = {
    import sparkSession.implicits._

    val bertDatasetConfig = TDatasetConfig(false)
    val trainingProcDataset: PretrainingDataset[ContextualDocument] = PretrainingDataset(
      trainingData.map(_.contextualDocument),
      bertDatasetConfig)
    val validationProcDataset: PretrainingDataset[ContextualDocument] = PretrainingDataset(
      validationData.map(_.contextualDocument),
      bertDatasetConfig)
    (trainingProcDataset, validationProcDataset)
  }

  /**
   * Compute the number of unique labels for this training set
   * @return Number of unique labels
   */
  final def getLabelIndexMap: Map[String, Int] = labelIndexMap

  override def toString: String = tokenizedIndexedDS.take(20).mkString("\n")
}


/**
 * Singleton for various constructor
 */
private[bertspark] final object TrainingSet {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[TrainingSet])


  /**
    * Constructor using current data set and pre-defined label - index map
    * @param tokenizedIndexedDS  Input data set
    * @param initialLabelIndexMap Initial map {label, label index}
    * @param sparkSession Implicit reference to the current Spark context
    * @param encoder Implicit encoder for pair {label, label index}
    * @return Instance of training set
    */
  def apply(
    tokenizedIndexedDS: Dataset[TokenizedTrainingSet],
    initialLabelIndexMap: Map[String, Int]
  )(implicit sparkSession: SparkSession, encoder: Encoder[(String, Int)]): TrainingSet =
    new TrainingSet(tokenizedIndexedDS, initialLabelIndexMap)


  /**
    * Constructor using current data set
    * @param tokenizedIndexedDS  Input data set
    * @param sparkSession Implicit reference to the current Spark context
    * @param encoder Implicit encoder for pair {label, label index}
    * @return Instance of training set
    */
  def apply(
    tokenizedIndexedDS: Dataset[TokenizedTrainingSet]
  )(implicit sparkSession: SparkSession, encoder: Encoder[(String, Int)]): TrainingSet =
    new TrainingSet(tokenizedIndexedDS, Map.empty[String, Int])


    /**
    * Constructor for creating DJL data sets from labeled data stored on S3
    * @param s3TrainingSetFolder Name of folder containing labeled training data
    * @param numRecords Num of records
    * @param initialLabelIndexMap Map of pair {label -> index}
    * @param sparkSession Implicit reference to the current Spark context
    * @param enc Encoder for the labeled training data
    * @param enc2 Encoder for the indexed labels
    * @param enc3 Implicit encoder for the frequency/count of labels
    * @return Instance of Training set builder
    */
  def apply(
    s3TrainingSetFolder: String,
    numRecords: Int,
    initialLabelIndexMap: Map[String, Int]
  )(implicit sparkSession: SparkSession,
    enc: Encoder[TokenizedTrainingSet],
    enc2: Encoder[LabelIndex],
    enc3: Encoder[LabeledCount]): TrainingSet = {
    import org.bertspark.config.MlopsConfiguration._

    // Load the original data set
    val tokenizedIndicesDS = try {
      S3Util.s3ToDataset[TokenizedTrainingSet](
        mlopsConfiguration.storageConfig.s3Bucket,
        s3TrainingSetFolder,
        header = false,
        fileFormat = "json"
      )
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"TrainingSet: ${e.getMessage}")
        sparkSession.emptyDataset[TokenizedTrainingSet]
    }
    new TrainingSet(tokenizedIndicesDS, initialLabelIndexMap)
  }


  /**
    * Create an instance of the training set loaded from S3
    * @param s3TrainingSetFolder Folder containing the training set
    * @param initialLabelIndexMap Map of pair {label -> index}
    * @param sparkSession Implicit reference to the current Spark context
    * @param enc implicit encoder for the training set
    * @param enc2 Implicit encoder for the pair {claim -> Index}
    * @param enc3 Implicit encoder of the frequency {term -> count}
    * @return Instance of the training set
    */
  def apply(
    s3TrainingSetFolder: String,
    initialLabelIndexMap: Map[String, Int]
  )(implicit sparkSession: SparkSession,
    enc: Encoder[TokenizedTrainingSet],
    enc2: Encoder[LabelIndex],
    enc3: Encoder[LabeledCount]): TrainingSet = apply(s3TrainingSetFolder, -1, initialLabelIndexMap)


  /**
    * Create an instance of the training set loaded from S3
    * @param s3TrainingSetFolder Folder containing the training set
    * @param sparkSession Implicit reference to the current Spark context
    * @param enc implicit encoder for the training set
    * @param enc2 Implicit encoder for the pair {claim -> Index}
    * @param enc3 Implicit encoder of the frequency {term -> count}
    * @return Instance of the training set
    */
  def apply(
    s3TrainingSetFolder: String,
  )(implicit sparkSession: SparkSession,
    enc: Encoder[TokenizedTrainingSet],
    enc2: Encoder[LabelIndex],
    enc3: Encoder[LabeledCount]): TrainingSet = apply(s3TrainingSetFolder, -1, Map.empty[String, Int])

}
