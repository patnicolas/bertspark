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
package org.bertspark.nlp.augmentation

import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.nlp.trainingset.{SubModelsTrainingSet, TokenizedTrainingSet}
import org.bertspark.nlp.trainingset.SubModelsTrainingSet.emptySubModelsTrainingSet
import org.slf4j.{Logger, LoggerFactory}


/**
  * Filter labels for which the number of records < minNumRecordsPerLabel as defined in the configuration file
  * @param subModelsTrainingDS Training data loaded from S2
  * @param recordsFrequencyPerLabel Map {label -> Number of records }
  * @param minNumRecordsPerLabel Minimum number of records per label used by the filter
  * @param sparkSession Implicit reference to the current Spark context
  *
  * @author Patrick Nicolas
  * @version 0.8
  */
private[bertspark] final class LabelRecordFrequencyFilter private (
  subModelsTrainingDS: Dataset[SubModelsTrainingSet],
  recordsFrequencyPerLabel: Array[(String, Int)],
  minNumRecordsPerLabel: Int
)(implicit sparkSession: SparkSession) extends RecordsAugmentation {
  import LabelRecordFrequencyFilter._

  override def augment: Dataset[SubModelsTrainingSet] = {
    logDebug(logger, msg = "LabelRecordFrequencyFilter.augment")
    filter(subModelsTrainingDS, recordsFrequencyPerLabel, minNumRecordsPerLabel)
  }
}


/**
  * Singleton for constructor for filter
  */
private[bertspark] object LabelRecordFrequencyFilter {
  final private val logger: Logger = LoggerFactory.getLogger("LabelRecordFrequencyFilter")
  private val defaultMinNumRecordsPerLabel = mlopsConfiguration.classifyConfig.minNumRecordsPerLabel

  def apply(
    subModelsTrainingDS: Dataset[SubModelsTrainingSet],
    recordsFrequencyPerLabel: Array[(String, Int)],
    minNumRecordsPerLabel: Int
  )(implicit sparkSession: SparkSession): LabelRecordFrequencyFilter =
    new LabelRecordFrequencyFilter(subModelsTrainingDS, recordsFrequencyPerLabel, minNumRecordsPerLabel)

  def apply(
    subModelsTrainingDS: Dataset[SubModelsTrainingSet],
    recordsFrequencyPerLabel: Array[(String, Int)]
  )(implicit sparkSession: SparkSession): LabelRecordFrequencyFilter =
    apply(subModelsTrainingDS, recordsFrequencyPerLabel, defaultMinNumRecordsPerLabel)


  /**
    * Remove labels for which the number of records is < mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
    * @return Filtered Sub model training set
    */
  def filter(
    subModelsTrainingDS: Dataset[SubModelsTrainingSet],
    recordsFrequencyPerLabel: Array[(String, Int)]
  )(implicit sparkSession: SparkSession): Dataset[SubModelsTrainingSet] =
    filter(subModelsTrainingDS, recordsFrequencyPerLabel, defaultMinNumRecordsPerLabel)

  /**
    * Remove labels for which the number of records is < minNumRecordsPerLabel
    * @param minNumRecordsPerLabel Minimum number ore records to keep a label valid
    * @return Filtered Sub model training set
    */
  def filter(
    subModelsTrainingDS: Dataset[SubModelsTrainingSet],
    recordsFrequencyPerLabel: Array[(String, Int)],
    minNumRecordsPerLabel: Int
  )(implicit sparkSession: SparkSession): Dataset[SubModelsTrainingSet] = {
    import sparkSession.implicits._

    // Step 1: Compute the labels globally with at least minNumNotesPerLabels records... which are
    //         either Oracle or sub models to train
    val labelsWithMinNumRecords = recordsFrequencyPerLabel.filter{
      case (_, cnt) => cnt >= minNumRecordsPerLabel
    }   .map{ case (label, _) => label.replaceAll(",", " ")}
        .toSet
    logDebug(
      logger,
      msg = s"Num valid labels for $minNumRecordsPerLabel is ${labelsWithMinNumRecords.size} over ${recordsFrequencyPerLabel.length}"
    )

    if(labelsWithMinNumRecords.nonEmpty) {
      val validGroupedSubModelsTrainingSet: Dataset[SubModelsTrainingSet] = subModelsTrainingDS.map(
        subModelTraining => {
          val trainingRecords: Seq[TokenizedTrainingSet] = subModelTraining.labeledTrainingData
          val trainingLabels = trainingRecords.map(_.label).distinct.filter(labelsWithMinNumRecords.contains(_))
          logDebug(logger, msg = s"Original labels for ${subModelTraining.subModel}: ${trainingLabels.size}")

          // Step 4: Eliminate the sub model for which valid label do not have
          // at least minNumNotesPerLabels associated data/notes
          if (trainingLabels.isEmpty)
            emptySubModelsTrainingSet
          else {
            // Step 5: Update the training data set with the limited number of labels.
            val validLabeledTrainingData = subModelTraining
                .labeledTrainingData
                .filter(t => trainingLabels.contains(t.label))

            logDebug(logger, msg = s"Num of filtered labels for ${subModelTraining.subModel}: ${
              validLabeledTrainingData.size
            }")
            subModelTraining.copy(labeledTrainingData = validLabeledTrainingData)
          }
        }
      ).filter(_.nonEmpty)

      logDebug(
        logger,
        msg = s"Num of sub models ${subModelsTrainingDS.count()} reduced to ${validGroupedSubModelsTrainingSet.count()}"
      )
      validGroupedSubModelsTrainingSet
    }
    else {
      import sparkSession.implicits._
      logger.warn(s"No label have more than $minNumRecordsPerLabel records")
      sparkSession.emptyDataset[SubModelsTrainingSet]
    }
  }

}



