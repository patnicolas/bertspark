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
package org.bertspark.modeling

import org.apache.spark.sql._
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames
import org.bertspark.delay
import org.bertspark.nlp.augmentation.RecordsAugmentation.NoAugmentation
import org.bertspark.nlp.medical.MedicalCodingTypes._
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.bertspark.util.io._
import org.bertspark.util.SparkUtil
import org.slf4j._


/**
  * {{{
  *  Class to generate a sub model - labels taxonomy using the schema:
  *     [Sub model],[Num labels],[Labels]
  *   Labels are separated by "||" delimiter
  *   The output S3 file with signature subModels-$minNumRecordsPerLabel.csv is used as input to the
  *   training of the classifier
  *
  *  The two main operations are
  *    - Filter (Remove labels which have less than  minNumRecordsPerLabel records
  *    - Augment (Added records to label which have less than  minNumRecordsPerLabel records
  * }}}
  * @param subModelsTrainingDS Training set to be loaded from S3
  * @param customer Customer as part of filter
  * @param sparkSession Implicit reference to the current Spark context
  *
  * @author Patrick Nicolas
  * @version 0.7
  */
private[bertspark] final class SubModelOperations private (
  subModelsTrainingDS: Dataset[SubModelsTrainingSet],
  minNumRecordsPerLabel: Int,
  customer: String
)(implicit sparkSession: SparkSession) {
  import sparkSession.implicits._
  import SubModelOperations._
  import org.bertspark.nlp.augmentation._

  // Sequence of pair {label -> Number of associated records}
  private[this] val recordsFrequencyPerLabel: Array[(String, Int)] =
    subModelsTrainingDS
        .flatMap(_.labeledTrainingData.map(ts => (ts.label, 1)))
        .rdd
        .reduceByKey(_ + _)
        .collect

  private[this] val recordsAugmentation = mlopsConfiguration.classifyConfig.augmentation match {
    case `filterAug` =>
      LabelRecordFrequencyFilter(subModelsTrainingDS, recordsFrequencyPerLabel, minNumRecordsPerLabel)
    case `randomAugUNK` | `randomAugChar` | `randomAugSyn` | `randomAugCorrect` | `randomAugSynAndCorrect` =>
      RandomAugmentation(subModelsTrainingDS, recordsFrequencyPerLabel, minNumRecordsPerLabel)
    case `noAug` =>
      NoAugmentation(subModelsTrainingDS)
    case _ =>
      throw new UnsupportedOperationException(s"Augmentation ${mlopsConfiguration.classifyConfig.augmentation} not supported")

  }

  /**
    * Augment or filter the records for label which number of records/notes is less than
    * mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
    * @return Augmented Grouped sub mmodels training set
    */
  def process: Dataset[SubModelsTrainingSet] = recordsAugmentation.augment


  /**
    * Augment the number of records for label which number of records/notes is less than
    * mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
    * @return Augmented Grouped sub mmodels training set
    */
  def augment(substituteMethod: String): Dataset[SubModelsTrainingSet] = {
    val augmentation = RandomAugmentation(
      subModelsTrainingDS,
      recordsFrequencyPerLabel,
      minNumRecordsPerLabel,
      substituteMethod)
    augmentation.augment
  }

  /**
    * Filter the number of records for label which number of records/notes is less than
    * mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
    * @return filtered Grouped sub models training set
    */
  def filter: Dataset[SubModelsTrainingSet] =  {
    val filter = LabelRecordFrequencyFilter(subModelsTrainingDS, recordsFrequencyPerLabel, minNumRecordsPerLabel)
    filter.augment
  }

  def apply(minNumRecordsPerLabel: Int): Unit = apply(Array[Int](minNumRecordsPerLabel))

  def apply(minNumRecordsPerLabels: Array[Int]): Unit = {
    require(minNumRecordsPerLabels.nonEmpty && minNumRecordsPerLabels.head >= 2,
      "Sub models labels taxonomy number of records per labels is incorrect")

    minNumRecordsPerLabels.foreach(generateTaxonomy(_, customer))
  }


  // ------------------------   Supported Methods -----------------------------------

  private def generateTaxonomy(minNumRecordsPerLabel: Int, customer: String)(implicit sparkSession: SparkSession): Unit = {
    import sparkSession.implicits._

    logDebug(logger, msg = s"Process group sub models training set for $minNumRecordsPerLabel")
    val validLabels = recordsFrequencyPerLabel.filter{ case (_, cnt) => cnt == 1 || cnt >= minNumRecordsPerLabel }
    val validLabelSet = validLabels.map(_._1).toSet

    logDebug(
      logger,
      msg = s"Num valid labels for $minNumRecordsPerLabel is ${validLabelSet.size} over ${recordsFrequencyPerLabel.length}"
    )

    val entries: Dataset[String] = subModelsTrainingDS.map(
      groupSubModel => {
        val originalLabels = groupSubModel.labeledTrainingData.map(_.label).distinct
        val invalidLabels = originalLabels.find(!validLabelSet.contains(_))
          // If we have at least one label from this sub model that has less than minimum number of records...
          // remove it from the training
        if(invalidLabels.isDefined) {
          logDebug(logger, msg = s"Sub model: ${groupSubModel.subModel.trim} removed")
          ""
        }
        // otherwise keep the sub model for training
        else
          s"${groupSubModel.subModel.trim},${originalLabels.size},${originalLabels.mkString("||")}"
      }
    ).filter(_.nonEmpty)

    // Store the sub model used for training the classifier
    S3Util.upload(
      S3PathNames.getS3SubModelsStructure(
        mlopsConfiguration.target,
        mlopsConfiguration.runId,
        customer,
        minNumRecordsPerLabel),
      entries.collect().mkString("\n")
    )
    delay(timeInMillis = 3000L)
  }
}


/**
  * Singleton for constructors
  */
private[bertspark] object SubModelOperations {
  final private val logger: Logger = LoggerFactory.getLogger("SubModelLabelsTaxonomy")

  def apply(
    groupedSubModelTrainingDS: Dataset[SubModelsTrainingSet],
    minNumRecordsPerLabel: Int,
    customer: String
  )(implicit sparkSession: SparkSession): SubModelOperations =
    new SubModelOperations(groupedSubModelTrainingDS, minNumRecordsPerLabel, customer)

  def apply(
    groupedSubModelTrainingDS: Dataset[SubModelsTrainingSet],
    minNumRecordsPerLabel: Int
  )(implicit sparkSession: SparkSession): SubModelOperations =
    apply(groupedSubModelTrainingDS, minNumRecordsPerLabel, "")

  def apply(
    groupedSubModelTrainingDS: Dataset[SubModelsTrainingSet]
  )(implicit sparkSession: SparkSession): SubModelOperations =
    apply(groupedSubModelTrainingDS, mlopsConfiguration.classifyConfig.minNumRecordsPerLabel, "")


  /**
    * Constructor for the  sub model labels taxonomy derived from a given training set
    * @param s3TrainingSetFolder S3 folder containing the training set.
    * @param customer A given customer
    * @return Sub model label taxonomy
    */
  def apply(s3TrainingSetFolder: String, customer: String = ""): SubModelOperations = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    // Extract the request ids associated with a customer if it is defined.
    val customerIds = if(customer.nonEmpty) getCustomerNoteIds(customer) else Array.empty[String]
    val rawGroupedSubModelsDS = S3Util.s3ToDataset[SubModelsTrainingSet](
      s3TrainingSetFolder,
      header = false,
      fileFormat = "json"
    )
    val groupedSubModelsDS = rawGroupedSubModelsDS

    groupedSubModelsDS.map(
      groupSubModel => {
        val newLabeledTrainingData = groupSubModel.labeledTrainingData.map(
          labeledTS => {
            val newLabel = {
              // We filter by customer if needed...
              if(customerIds.isEmpty || customerIds.contains(labeledTS.contextualDocument.id)) {
                if(customerIds.nonEmpty)
                  logDebug(logger, s"${labeledTS.contextualDocument.id} found in customer")
                labeledTS.label.replaceAll(",", " ").replaceAll(" {2}", " ")
              }
              else
                ""
            }
            labeledTS.copy(label = newLabel)
          }
        ).filter(_.label.nonEmpty)

        groupSubModel.copy(labeledTrainingData = newLabeledTrainingData)
      }
    )
    SubModelOperations(groupedSubModelsDS, mlopsConfiguration.classifyConfig.minNumRecordsPerLabel ,customer)
  }

  def apply(): SubModelOperations = apply(S3PathNames.s3ModelTrainingPath)




  // --------------------  Statistics and reporting related methods  --------------

  final class SubModelLabelsTrainingStats(
    labelRecordsFrequencies: Seq[(String, Int)],
    subModelLabelsFrequencies: Seq[(String, Int)]
  ) {

    override def toString: String = {
      val labelRecordsFrequenciesStr = labelRecordsFrequencies.map {
        case (label, numRecords) => s"$label,$numRecords"
      }.mkString("\n")

      val subModelLabelsFrequenciesStr = subModelLabelsFrequencies.map{
        case (subModel, numLabels) => s"$subModel,$numLabels"
      }.mkString("\n")
      s"Label-records frequencies,count\n$labelRecordsFrequenciesStr\nSubModel-labels frequencies,count\n$subModelLabelsFrequenciesStr"
    }

    def save(): Unit = {
      var s3Folder = s"labelsFreq-${mlopsConfiguration.target}-${mlopsConfiguration.classifyConfig.minNumRecordsPerLabel}.csv"
      val labelRecordsFrequenciesStr = labelRecordsFrequencies.map {
        case (label, numRecords) => s"$label,$numRecords"
      }.mkString("\n")
      LocalFileUtil.Save.local(s"output/$s3Folder",labelRecordsFrequenciesStr)

      s3Folder = s"modelsFreq-${mlopsConfiguration.target}-${mlopsConfiguration.classifyConfig.minNumRecordsPerLabel}.csv"
      val subModelLabelsFrequenciesStr = subModelLabelsFrequencies.map{
        case (subModel, numLabels) => s"$subModel,$numLabels"
      }.mkString("\n")
      LocalFileUtil.Save.local(s"output/$s3Folder",subModelLabelsFrequenciesStr)
    }
  }


  final object SubModelLabelsTrainingStats {

    def apply(): SubModelLabelsTrainingStats = {
      import org.bertspark.implicits._
      import sparkSession.implicits._

      val groupedByEmrDS: Dataset[SubModelsTrainingSet] = S3Util.s3ToDataset[SubModelsTrainingSet](
        S3PathNames.s3ModelTrainingPath
      ).persist()

      val subModelLabelsDistribution: Dataset[(String, List[String])] = groupedByEmrDS.flatMap(grouped => {
        val subModel: String = grouped.subModel.trim
        grouped.labeledTrainingData.map(ts => (subModel, List[String](ts.label)))
      })

      val subModelLabelsCountRDD = SparkUtil.groupBy[(String, List[String]), String](
        (input: (String, List[String])) => input._1,
        (in1: (String, List[String]), in2: (String, List[String])) => (in1._1, in1._2 ::: in2._2),
        subModelLabelsDistribution
      ).map{ case (subModel, labels) => (subModel, labels.distinct.size) }

      val subModelLabelsFrequencies = subModelLabelsCountRDD.collect.sortWith(_._2 < _._2)

      val labelFrequenciesDS: Dataset[(String, Int)] = groupedByEmrDS.flatMap(
        _.labeledTrainingData.map(ts => (ts.label, ts.contextualDocument.id))
      ).dropDuplicates("_2").map(ts => (ts._1, 1))

      val labelFrequencies = labelFrequenciesDS.rdd.reduceByKey(_ + _).collect.sortWith(_._2 > _._2)

      new SubModelLabelsTrainingStats(labelFrequencies, subModelLabelsFrequencies)
    }
  }

  def distributionFeedbacksRecordsPerLabels: Seq[(String, Int)] = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val feedbackDS = S3Util.s3ToDataset[InternalFeedback](S3PathNames.s3FeedbacksPath).dropDuplicates("id")
    val labelDS = feedbackDS.map(feedback => (feedback.toFinalizedSpace, 1))
    labelDS.rdd.reduceByKey(_ + _).collect.sortWith(_._2 < _._2)
  }


  def group(labels: Seq[String]): Seq[(String, Int)] = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    labels.toDS.map((_,1)).rdd.reduceByKey(_ + _).collect.sortWith(_._2 > _._2)
  }

  /**
    * Extract the id of requests associated with a give customer
    * @param customer Name of customer
    * @param sparkSession Implicit reference to the current Spark context
    * @return Set of customer ids
    */
  private def getCustomerNoteIds(customer: String)(implicit sparkSession: SparkSession): Array[String] =
    if(customer.nonEmpty) {
      import sparkSession.implicits._

      S3Util.s3ToDataset[InternalRequest](
        mlopsConfiguration.storageConfig.s3Bucket,
        S3PathNames.s3RequestsPath,
        header = false,
        fileFormat = "json"
      )   .filter(_.context.customer == customer)
          .map(_.id)
          .distinct
          .collect()
    }
    else
      Array.empty[String]
}



