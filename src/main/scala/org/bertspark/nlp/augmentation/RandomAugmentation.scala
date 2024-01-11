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
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.nlp.trainingset.{SubModelsTrainingSet, TokenizedTrainingSet}
import org.bertspark.nlp.vocabulary.MedicalCodeDescriptors.CodeDescriptorMap
import org.slf4j.{Logger, LoggerFactory}


/**
  * {{{
  * Wrapper for the data/token augmentation techniques that replace tokens or character in token that are
  * not referenced in the labels
  * There are 3 substitution technique for augmenting data
  * - Replacing, randomly a token (context or text) which is not a descriptor of code in label by [UNK]
  * - Modifying randomly a character on a token (context or text) which is not a descriptor of code in label
  * - Replacing randomly a token (context or text) which is not a descriptor of code in label by a synonym
  * }}}
  *
  * @param subModelsTrainingDS Training set to augment
  * @param minNumRecordsPerLabel Minimum number of notes per label to trigger augmentation
  * @param recordsFrequencyPerLabel Frequencies of records per labels
  * @param substituteFunc Substitute functions for augmentation
  * @param sparkSession Implicit reference to the current Spark context
  *
  * @author Patrick Nicolas
  * @version 0.8
  */
private[bertspark] final class RandomAugmentation private (
  subModelsTrainingDS: Dataset[SubModelsTrainingSet],
  recordsFrequencyPerLabel: Array[(String, Int)],
  minNumRecordsPerLabel: Int,
  substituteFunc: (TokenizedTrainingSet, Set[String], Int) => TokenizedTrainingSet
)(implicit sparkSession: SparkSession) extends RecordsAugmentation {
  import RandomAugmentation._
  import org.bertspark.config.MlopsConfiguration._

  /**
    * Augment the number of records for label which number of records/notes < minNumNotesPerLabels
    * @return Augmented Grouped sub models training set
    */
  override def augment: Dataset[SubModelsTrainingSet] = {
    logDebug(logger, msg = s"RandomSubstituteAugmentation.augment: ${mlopsConfiguration.classifyConfig.augmentation}")
    RandomAugmentation.augment(
      subModelsTrainingDS,
      minNumRecordsPerLabel,
      recordsFrequencyPerLabel,
      substituteFunc
    )
  }
}


/**
  * Singleton for constructors
  */
private[bertspark] object RandomAugmentation {
  final private val logger: Logger = LoggerFactory.getLogger("RandomSubstituteAugmentation")
  import org.bertspark.config.MlopsConfiguration._

  val maxNumSubstituteSearch = 12


    // -------------  Constructors -----------------------


  def apply(
    subModelsTrainingDS: Dataset[SubModelsTrainingSet],
    recordsFrequencyPerLabel: Array[(String, Int)],
    minNumRecordsPerLabel: Int,
    substituteMethod: String
  )(implicit sparkSession: SparkSession): RandomAugmentation =
    new RandomAugmentation(
      subModelsTrainingDS,
      recordsFrequencyPerLabel,
      minNumRecordsPerLabel,
      getSubstituteMethod(substituteMethod)
    )

  def apply(
    subModelsTrainingDS: Dataset[SubModelsTrainingSet],
    recordsFrequencyPerLabel: Array[(String, Int)],
    minNumRecordsPerLabel: Int,
  )(implicit sparkSession: SparkSession): RandomAugmentation =
    new RandomAugmentation(
      subModelsTrainingDS,
      recordsFrequencyPerLabel,
      minNumRecordsPerLabel,
      getSubstituteMethod
    )

  def apply(
    subModelsTrainingDS: Dataset[SubModelsTrainingSet],
    recordsFrequencyPerLabel: Array[(String, Int)],
    substituteMethod: String
  )(implicit sparkSession: SparkSession): RandomAugmentation =
    apply(
      subModelsTrainingDS,
      recordsFrequencyPerLabel,
      mlopsConfiguration.classifyConfig.minNumRecordsPerLabel,
      substituteMethod
    )

  def apply(
    subModelsTrainingDS: Dataset[SubModelsTrainingSet],
    recordsFrequencyPerLabel: Array[(String, Int)]
  )(implicit sparkSession: SparkSession): RandomAugmentation =
    apply(subModelsTrainingDS, recordsFrequencyPerLabel, mlopsConfiguration.classifyConfig.augmentation)


  // ------------------- Supporting/helper methods -----------------------

  private def augment(
    subModelsTrainingDS:  Dataset[SubModelsTrainingSet],
    minNumRecordsPerLabel: Int,
    recordsFrequencyPerLabel: Array[(String, Int)],
    substituteFunc: (TokenizedTrainingSet, Set[String], Int) => TokenizedTrainingSet
  )(implicit sparkSession: SparkSession): Dataset[SubModelsTrainingSet] = {
    import sparkSession.implicits._

    // Step 1: Extract the labels with a minimum number of
    val labelsMissingRecordsFreqMap: Map[String, Int]  = recordsFrequencyPerLabel.filter{
      case (_, cnt) => cnt < minNumRecordsPerLabel && cnt > 1
    }.toMap
    logDebug(
      logger,
      msg = s"${labelsMissingRecordsFreqMap.size} labels augmented to ${recordsFrequencyPerLabel.length} for $minNumRecordsPerLabel"
    )

    val validGroupedSubModelsTrainingSet: Dataset[SubModelsTrainingSet] = subModelsTrainingDS.map(
      subModelTrainingSet => {
        // Extract valid labels from classifiers parameters
        val trainingRecords: Seq[TokenizedTrainingSet] = subModelTrainingSet.labeledTrainingData
        val groupedByLabel: Map[String, Seq[TokenizedTrainingSet]] = trainingRecords.groupBy(_.label)
        val originalLabels = groupedByLabel.keys.toSet
        logDebug(logger, msg = s"${originalLabels.size} original labels for ${subModelTrainingSet.subModel}")

        if (labelsMissingRecordsFreqMap.nonEmpty) {
          val augmentedTokenizedTrainingSet = augment(
            groupedByLabel,
            labelsMissingRecordsFreqMap,
            minNumRecordsPerLabel,
            subModelTrainingSet.labeledTrainingData,
            substituteFunc)
          subModelTrainingSet.copy(labeledTrainingData = augmentedTokenizedTrainingSet)
        }
        else {
          // Update the training data set with the limited number of labels.
          val validLabeledTrainingData = subModelTrainingSet
              .labeledTrainingData
              .filter(t => originalLabels.contains(t.label))

          logDebug(
            logger,
            msg = s"${validLabeledTrainingData.size} filtered labels for ${subModelTrainingSet.subModel}")
          subModelTrainingSet.copy(labeledTrainingData = validLabeledTrainingData)
        }
      }
    )
    logDebug(
      logger,
      msg = s"${subModelsTrainingDS.count()} sub models reduced to ${validGroupedSubModelsTrainingSet.count()}"
    )
    validGroupedSubModelsTrainingSet
  }


  private def augment(
    groupedByLabel: Map[String, Seq[TokenizedTrainingSet]],
    labelsMissingRecordsFreqMap: Map[String, Int],
    minNumRecordsPerLabel: Int,
    labeledTrainingData: Seq[TokenizedTrainingSet],
    substituteFunc: (TokenizedTrainingSet, Set[String], Int) => TokenizedTrainingSet
  ): Seq[TokenizedTrainingSet] =
    groupedByLabel.flatMap {
      case (label, tokenizedRecords) =>
        // If 'label' has not enough records
        if (labelsMissingRecordsFreqMap.contains(label)) {
          // Collect the label code descriptors to avoid conflict
          val labelDescriptionTokens = CodeDescriptorMap.getClaimDescriptors(label).toSet

          val numRecords = labeledTrainingData.size
          val numRecordsToFill = minNumRecordsPerLabel - numRecords

          // We are taking each existing tokenized training set, substitute
          // some of the tokens of the contextual document using substituteFunc argument
          val tokenizedTrainingSetAugmentation = (0 until numRecordsToFill).map(
            index => {
              val rec = labeledTrainingData(index % numRecords)
              substituteFunc(rec, labelDescriptionTokens, index)
            }
          )

          // Add the newly created tokenized training set to the existing one
          tokenizedTrainingSetAugmentation ++ tokenizedRecords
        }
        else
          tokenizedRecords
    }.toSeq


  def augmentId(id: String, index: Int): String = s"${id}_x$index"

  @throws(clazz = classOf[UnsupportedOperationException])
  private def getSubstituteMethod: (TokenizedTrainingSet, Set[String], Int) => TokenizedTrainingSet =
    getSubstituteMethod(mlopsConfiguration.classifyConfig.augmentation)


  @throws(clazz = classOf[UnsupportedOperationException])
  private def getSubstituteMethod(substituteMethod: String): (TokenizedTrainingSet, Set[String], Int) => TokenizedTrainingSet = {
    substituteMethod match {
      case `randomAugUNK` => RandomTokenSubstitute.substitute
      case `randomAugChar` => RandomCharSubstitute.substitute
      case `randomAugSyn`| `randomAugCorrect` | `randomAugSynAndCorrect` => RandomSynonymSubstitute.substitute
      case _ =>
        throw new UnsupportedOperationException(s"Augmentation substitute method $substituteMethod not supported")
    }
  }
}
