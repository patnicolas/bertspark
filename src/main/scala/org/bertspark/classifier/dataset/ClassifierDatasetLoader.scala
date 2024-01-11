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
package org.bertspark.classifier.dataset

import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.modeling.SubModelOperations
import org.bertspark.nlp.trainingset.{SubModelsTrainingSet, TokenizedTrainingSet}
import org.bertspark.util.io.S3Util
import org.slf4j.{Logger, LoggerFactory}



/**
  * Loader for the training set for classifier. The class is also used to evaluate augmentation or
  * filtering techniques on a sub set of sub models if the argument numSubModelsForAugmentation is
  * defined as > 0.
  *
  * @param s3TrainingSetFolder S3 folder for the training set
  * @param subModelNames List of valid sub models
  * @param numSubModelsForAugmentation Number of sub models used for evaluating augmentation strategy,
  *                                    It is -1 for non evaluation
  *
  * @author Patrick Nicolas
  * @version 0.7
  */
@throws(clazz = classOf[IllegalArgumentException])
private[bertspark] final class ClassifierDatasetLoader private (
  s3TrainingSetFolder: String,
  subModelNames: Set[String],
  numSubModelsForAugmentation: Int) {
  import ClassifierDatasetLoader._
  require(subModelNames.nonEmpty, "ClassifierDatasetLoader requires defined subModelNames")

  /**
    * Load the training set for classifier
    * @param sparkSession Implicit reference to the current Spark context
    * @return Dataset of pair {Sub-model, Tokenized training set}
    */
  def apply()(implicit sparkSession: SparkSession): Dataset[(String, Seq[TokenizedTrainingSet])] =
    execute(s3TrainingSetFolder, subModelNames, numSubModelsForAugmentation)
}


/**
  * Singleton for implementing the loading, filtering and/or augmentation of training set loaded
  * from S3 folder
  */
private[bertspark] object ClassifierDatasetLoader {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[ClassifierDatasetLoader])

  val augmentationCriteria = (cnt: Int) => cnt > 1 && cnt < mlopsConfiguration.classifyConfig.maxNumRecordsPerLabel
  val filterCriteria = (cnt: Int) => cnt >= mlopsConfiguration.classifyConfig.minNumRecordsPerLabel

  /**
    * Function to sample 'maxNumSubModels' sub models for evaluating augmentation strategies
    */
  val samplingForAugmentation: (Dataset[SubModelsTrainingSet], Int, Int => Boolean) => Dataset[SubModelsTrainingSet] =
    (subModelsTrainingDS: Dataset[SubModelsTrainingSet], maxNumSubModels: Int, cutOff: Int => Boolean) => {
      import org.bertspark.implicits._
      import sparkSession.implicits._
      import org.bertspark.config.MlopsConfiguration._

      val recordsFrequencyPerLabel: Set[String] =
        subModelsTrainingDS
            .flatMap(_.labeledTrainingData.map(ts => (ts.label, 1)))
            .rdd
            .reduceByKey(_ + _)
            .filter{ case (_, cnt) => cutOff(cnt) }
            .map(_._1)
            .collect
            .toSet

      // If the maximum number of sub models is defined, case of evaluation, select
      // the sub set of sub models randomly ....

      val sampledSubModelsTrainingDS =
        if(maxNumSubModels > 0) {
          val fraction = maxNumSubModels.toFloat/subModelsTrainingDS.count()
          if(fraction < 0.95) subModelsTrainingDS.sample(fraction) else subModelsTrainingDS
        }
        else
          subModelsTrainingDS

      sampledSubModelsTrainingDS.map(
        subModelTraining => {
          val subModelName = subModelTraining.subModel
          val tokenizedTrainingSet = subModelTraining
              .labeledTrainingData
              .filter(ts => recordsFrequencyPerLabel.contains(ts.label))

          if(tokenizedTrainingSet.nonEmpty)
            SubModelsTrainingSet(subModelName, tokenizedTrainingSet, subModelTraining.labelIndices)
          else
            SubModelsTrainingSet.emptySubModelsTrainingSet
        }
      ).filter(_.nonEmpty)
    }




  def apply(
    s3TrainingSetFolder: String,
    subModelNames: Set[String],
    numSubModelsForAugmentation: Int): ClassifierDatasetLoader =
    new ClassifierDatasetLoader(s3TrainingSetFolder, subModelNames, numSubModelsForAugmentation)


  def apply(
    s3TrainingSetFolder: String,
    subModelNames: Set[String]): ClassifierDatasetLoader =
    new ClassifierDatasetLoader(s3TrainingSetFolder, subModelNames, numSubModelsForAugmentation = -1)


  /**
    * Retrieve the frequencies of records per label
    * @param subModelsTrainingDS Data set for training set
    * @return Sequence of pairs {label -> num records}
    */
  def getRecordsFrequencyPerLabel(
    subModelsTrainingDS: Dataset[SubModelsTrainingSet]
  )(implicit sparkSession: SparkSession): Array[(String, Int)] = {
    import sparkSession.implicits._

    subModelsTrainingDS
        .flatMap(_.labeledTrainingData.map(ts => (ts.label, 1)))
        .rdd
        .reduceByKey(_ + _)
        .collect
        .sortWith(_._1 < _._1)
  }


  // --------------------- Supporting/helper methods ---------------------------------

  private def execute(
    s3TrainingSetFolder: String,
    subModelNames: Set[String],
    maxNumSubModels: Int = -1
  )(implicit sparkSession: SparkSession): Dataset[(String, Seq[TokenizedTrainingSet])] = {
    import sparkSession.implicits._
    import org.bertspark.config.MlopsConfiguration._

    try {
      // Retrieve the sub models training set for this set of model names (slice of training set in local files
      // subModels-1.txt,  subModels-2.txt, .....
      val rawSubModelsTrainingSetDS = S3Util
          .s3ToDataset[SubModelsTrainingSet](s3TrainingSetFolder)
          .filter(subModelTrainingSet => subModelNames.contains(subModelTrainingSet.subModel))

      // Load the appropriate criteria for the count of records per label
      val criteria =
        if(mlopsConfiguration.classifyConfig.isAugmentation) {
          logDebug(logger, msg = s"Proceeds with augmentation ${mlopsConfiguration.classifyConfig.augmentation}")
          augmentationCriteria
        }
        else {
          logDebug(logger, msg = s"Proceeds with filter ${mlopsConfiguration.classifyConfig.augmentation}")
          filterCriteria
        }

      val subModelsTrainingSetDS = samplingForAugmentation(rawSubModelsTrainingSetDS, maxNumSubModels, criteria)

      // We need to weed out or augment the labels for which the number of records is less that
      // classifyConfig.minNumRecordsPerLabel
      val subModelLabelsOperations = SubModelOperations(subModelsTrainingSetDS)
      val subModelsWithFilteredLabelsDS = subModelLabelsOperations.process

      // Build the keyed data set of tokenized training set...
      val finalizedTrainingDataDS: Dataset[(String, Seq[TokenizedTrainingSet])] = subModelsWithFilteredLabelsDS
          .map(grouped => (grouped.subModel, grouped.labeledTrainingData))

      logDebug(
        logger,
        msg = s"${subModelsTrainingSetDS.count()} initial subModels, ${subModelsWithFilteredLabelsDS.count()} filtered"
      )
      finalizedTrainingDataDS
    }
    catch {
      case e: IllegalStateException =>
        logger.error(s"DualS3Dataset: ${e.getMessage}")
        sparkSession.emptyDataset[(String, Seq[TokenizedTrainingSet])]
    }
  }
}