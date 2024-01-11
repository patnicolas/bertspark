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
package org.bertspark.classifier.training

import org.apache.spark.sql._
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalFeedback
import org.bertspark.util.SparkUtil
import org.bertspark.util.io.S3Util
import org.slf4j._


/**
  * Filter to select the training set used in the training of the classifier. The filter applies to the
  * labels or feedback records
  *
  * @param customers List of customers  (all customers for empty list)
  * @param subModels  List of targeted submodels  (all sub models for empty list)
  * @param numRecords Num of records used in the classifier training
  * @param minNumRecordPerLabel Minimum of records for valid label
  * @param maxNumRecordPerLabel Maximum of records for valid label (the records will be truncated/randomly sampled)
  *
  * @author Patrick Nicolas
  * @version 0.7
  */
private[bertspark] case class ClassifierTrainingSetFilter(
  customers: Set[String],
  subModels: Set[String],
  numRecords: Int,
  minNumRecordPerLabel: Int,
  maxNumRecordPerLabel: Int) {

  override def toString: String =
    s"""
       |Customers:            ${if(customers.nonEmpty) customers.mkString(" ") else "None"}
       |Sub models:           ${if (subModels.nonEmpty) subModels.mkString(" ") else "None"}
       |Num records           $numRecords
       |Min Num Records/Label $minNumRecordPerLabel
       |Maz Num Records/Label $maxNumRecordPerLabel
       |"""
}


/**
  * Singleton for constructor using parameters in the configuration file.
  * Implementation of filter on feedback
  */
private[bertspark] final object ClassifierTrainingSetFilter {
  final private val logger: Logger = LoggerFactory.getLogger("TrainingSetBuilder")


  /**
    * Constructor using the parameters defined in the configuration file. The filter applies to the feedback or
    * label records. The maximum number of records is the only parameter passed as part of the
    * @param numRecords Number of records
    * @return Instance of the pretraining Set filter
    */
  def apply(numRecords: Int): ClassifierTrainingSetFilter = {
    import org.bertspark.config.MlopsConfiguration._

    val customers = mlopsConfiguration.preProcessConfig.customers
    val subModels = mlopsConfiguration.preProcessConfig.subModels
    val minNumRecordPerLabel = mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
    val maxNumRecordPerLabel = mlopsConfiguration.classifyConfig.maxNumRecordsPerLabel
    ClassifierTrainingSetFilter(customers.toSet, subModels.toSet, numRecords, minNumRecordPerLabel, maxNumRecordPerLabel)
  }

  def filter(
    s3FeedbackFolder: String,
    numRecords: Int)(implicit sparkSession: SparkSession): Dataset[InternalFeedback] =
    filter(s3FeedbackFolder, ClassifierTrainingSetFilter(numRecords))


  def filter(
    s3FeedbackFolder: String,
    classifierTrainedSetFilter: ClassifierTrainingSetFilter
  )(implicit sparkSession: SparkSession): Dataset[InternalFeedback] = {
    import sparkSession.implicits._

    try {
      val ds = S3Util.s3ToDataset[InternalFeedback](s3FeedbackFolder)
      filter(ds, classifierTrainedSetFilter)
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(e.getMessage)
        sparkSession.emptyDataset[InternalFeedback]
    }
  }


  def filter(
    classifierTrainedSetFilter: ClassifierTrainingSetFilter
  )(implicit sparkSession: SparkSession): Dataset[InternalFeedback] =
  filter(S3PathNames.s3FeedbacksPath, classifierTrainedSetFilter)


  /**
    * Filter the original/raw dataset of feedbacks/labels
    * @param internalFeedbackDS Dataset of internal feedbacks
    * @param numRecords Maximum number of records
    * @param sparkSession Implicit reference to the current Spark context
    * @return Filtered dataset of internal feedbacks
    */
  def filter(
    internalFeedbackDS: Dataset[InternalFeedback],
    numRecords: Int
  )(implicit sparkSession: SparkSession): Dataset[InternalFeedback] =
    filter(internalFeedbackDS, ClassifierTrainingSetFilter(numRecords))


  /**
    * Filter the original/raw dataset of feedbacks/labels
    * @param internalFeedbackDS Dataset of internal feedbacks
    * @param classifierTrainedSetFilter Customized training set filter
    * @param sparkSession Implicit reference to the current Spark context
    * @return Filtered dataset of internal feedbacks
    */
  def filter(
    internalFeedbackDS: Dataset[InternalFeedback],
    classifierTrainedSetFilter: ClassifierTrainingSetFilter
  )(implicit sparkSession: SparkSession): Dataset[InternalFeedback] = {
    import sparkSession.implicits._

    var filterFeedbackDS = internalFeedbackDS
    logDebug(logger, msg = s"Original feedback dataset ${filterFeedbackDS.count()}")

    // Filter by number of records
    if(classifierTrainedSetFilter.numRecords > 0) {
      val fraction = classifierTrainedSetFilter.numRecords.toFloat/filterFeedbackDS.count()
      if(fraction < 0.95) {
        filterFeedbackDS = filterFeedbackDS.sample(fraction)
        logDebug(
          logger,
          msg = s"Filtered dataset by records ${classifierTrainedSetFilter.numRecords} ${filterFeedbackDS.count()}"
        )
      }
    }

    // Filter by customer
    if(classifierTrainedSetFilter.customers.nonEmpty) {
      filterFeedbackDS = filterFeedbackDS.filter(
        feedback => classifierTrainedSetFilter.customers.contains(feedback.context.customer)
      )
      logDebug(logger, msg = s"Filtered per customers ${filterFeedbackDS.count()}")
    }

    // Filter by sub models
    if(classifierTrainedSetFilter.subModels.nonEmpty) {
      filterFeedbackDS = filterFeedbackDS.filter(
        feedback => classifierTrainedSetFilter.subModels.contains(feedback.context.emrLabel.trim)
      )
      logDebug(logger, msg = s"Filtered per subModels ${filterFeedbackDS.count()}")
    }

    // Filter by minimum or maximum number of records per label
    val minNumRecordPerLabel = classifierTrainedSetFilter.minNumRecordPerLabel
    val maxNumRecordPerLabel = classifierTrainedSetFilter.maxNumRecordPerLabel

    if(minNumRecordPerLabel > 0 || maxNumRecordPerLabel > 0) {
      val xsFeedbackDS = filterFeedbackDS.map(List[InternalFeedback](_))
      var groupedByLabelRDD = SparkUtil.groupBy[List[InternalFeedback], String](
        (internalFeedback: List[InternalFeedback]) => internalFeedback.head.toFinalizedSpace,
        (xsFeedback1: List[InternalFeedback], xsFeedback2: List[InternalFeedback]) => xsFeedback1 ::: xsFeedback2,
        xsFeedbackDS
      )
      if(minNumRecordPerLabel > 0)
        groupedByLabelRDD = groupedByLabelRDD.filter(_.size > minNumRecordPerLabel)
      if(classifierTrainedSetFilter.maxNumRecordPerLabel > 0 && minNumRecordPerLabel < maxNumRecordPerLabel)
        groupedByLabelRDD = groupedByLabelRDD.filter(_.size < maxNumRecordPerLabel)

      filterFeedbackDS = groupedByLabelRDD.toDS().flatMap(xs => xs)
      logDebug(logger, msg = s"Filtered by records per labels ${filterFeedbackDS.count()}")
    }
    filterFeedbackDS
  }

}
