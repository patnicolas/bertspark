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
package org.bertspark.analytics

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.util.io.S3Util.s3ToDataset
import org.bertspark.util.SparkUtil
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.bertspark.config.MlopsConfiguration.DebugLog.logTrace
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.nlp.medical.MedicalCodingTypes.{FeedbackLineItem, InternalFeedback, MlEMRCodes}
import org.bertspark.config.{FsPathNames, S3PathNames}
import org.bertspark.nlp.medical.MedicalCodingTypes.FeedbackLineItem.str
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.slf4j.{Logger, LoggerFactory}


/**
 * Categorization/Characterization of the training set
 * @param sparkSession Implicit reference to the current Spark Session
 */
private[bertspark] final class Categorization(implicit sparkSession: SparkSession) extends Analyzer {
  import Categorization._

  /**
   * {{{
   * Compute the distribution (frequency) per EMR and per (EMR, Document label)
   * The key for the distribution per EMR is the name of the sub model (i.e. 77891 26)
   * The key for the distribution per (EMR, Document label) is emrCode#{label without emr code)
   *
   * The document label is the original labeled document - the emr codes that defines the sub model
   *  - Input: Feedback request [mlops/feedbacks/$target]
   *  - Output: JSON representation of the distribution per emr and emr#label  (local file output/distribution)
   *            DistributionByEmrAndLabels
   *               distributionByEmr: Seq[DistributionByKey(key, count)],
   *               distributionByEmrAndLabels: Seq[DistributionByKey(key, count)]
   * }}}
   */
  override def run(analyzerResult: AnalyzerResult): AnalyzerResult = {
    val distributionByEmrAndLabels = run
    AnalyzerResult(distributionByEmrAndLabels.toString)
  }

  /**
    * {{{
    * Build a distribution for a given training set
    *   - Grouped by EMR
    *   - Grouped by Label
    *   - Grouped by EMR/Label
    * }}}
    * @return
    */
  def run: DistributionByEmrAndLabels =  {
    import sparkSession.implicits._
    import FsPathNames._

    val emrLabelDS: Dataset[(String, String)] = loadFeedbackByEmr

    val emrLabelListDS = emrLabelDS.map(feedback => List[(String, String)](feedback))
    val groupedByEmrDS: Dataset[DistributionByKey] = groupByEmr(emrLabelListDS)
    val groupedByEmrAndLabelDS: Dataset[DistributionByKey] = groupByEmrAndLabel(emrLabelListDS)
    val groupedByLabelRDD = getNotesCountPerLabels

    val distributionByEmrAndLabels = DistributionByEmrAndLabels(
      groupedByEmrDS.collect.sortWith(_.count > _.count),
      groupedByEmrAndLabelDS.collect.sortWith(_.count > _.count),
      groupedByLabelRDD.collect.sortWith(_.count > _.count)
    )
    // Save the distribution by EMR and labels into file
    val distribution = LocalFileUtil.Json.mapper.writeValueAsString(distributionByEmrAndLabels)
    LocalFileUtil.Save.local(distributionJsonPath, distribution)
    LocalFileUtil.Save.local(
      getDistributionCsvPath("Emr"),
      distributionByEmrAndLabels.distributionByEmr.map(_.csv).mkString("\n")
    )
    LocalFileUtil.Save.local(
      getDistributionCsvPath("Label"),
      distributionByEmrAndLabels.distributionByLabel.map(_.csv).mkString("\n")
    )
    LocalFileUtil.Save.local(
      getDistributionCsvPath("EmrLabel"),
      distributionByEmrAndLabels.distributionByEmrAndLabels.map(_.csv).mkString("\n")
    )
    distributionByEmrAndLabels
  }


  def loadFeedbackByEmr: Dataset[(String, String)] = {
    import sparkSession.implicits._

    val emrLabelPairsDS: Dataset[(String, String)] =
      try {
        s3ToDataset[InternalFeedback](
          mlopsConfiguration.storageConfig.s3Bucket,
          S3PathNames.s3FeedbacksPath,
          header = false,
          fileFormat = "json").map(
          feedback => {
            val emrCode = getEmrCode(feedback.context.EMRCpts)
            val claimLabel = FeedbackLineItem.str(feedback.finalized.lineItems)
            (emrCode, claimLabel)
          }
        ).persist().cache()
      }
      catch {
        case e: IllegalArgumentException =>
          logger.error(s"Categorization: ${e.getMessage}")
          sparkSession.emptyDataset[(String, String)]
      }
    logTrace(logger, s"Get distribution from ${emrLabelPairsDS.count} raw feedbacks")

    emrLabelPairsDS
  }

  private def groupByEmr(feedbackPairDS: Dataset[List[(String, String)]]): Dataset[DistributionByKey] = {
    import sparkSession.implicits._

    val groupedByEmrDS: Dataset[DistributionByKey] = SparkUtil.groupBy[List[(String, String)], String](
      (xs: List[(String, String)]) => xs.head._1,
      (xs1: List[(String, String)], xs2: List[(String, String)]) => xs1 ::: xs2,
      feedbackPairDS
    ).toDS.map(
      result => {
        val key = result.head._1.replace(",", " ")
        DistributionByKey(key, result.size)
      }
    )
    logTrace(
      logger,
      s"Get distribution from ${groupedByEmrDS.count} grouped by emr: ${groupedByEmrDS.head().toString}"
    )
    groupedByEmrDS
  }

  private def groupByEmrAndLabel(feedbackPairDS: Dataset[List[(String, String)]]): Dataset[DistributionByKey] = {
    import sparkSession.implicits._

    val groupedByEmrAndLabelDS: Dataset[DistributionByKey] = SparkUtil.groupBy[List[(String, String)], String](
      (xs: List[(String, String)]) => s"${xs.head._1}#${xs.head._2}",
      (xs1: List[(String, String)],xs2: List[(String, String)]) => xs1 ::: xs2,
      feedbackPairDS
    ).toDS.map(
      result => {
        val key = result.head._1.replace(",", " ")
        DistributionByKey(s"$key#${result.head._2}", result.size)
      }
    )
    logTrace(
      logger,
      s"Get distribution from ${groupedByEmrAndLabelDS.count} grouped by ${groupedByEmrAndLabelDS.head().toString}"
    )
    groupedByEmrAndLabelDS
  }


  def getNotesCountPerLabels: RDD[DistributionByKey] = {
    import sparkSession.implicits._

    val labelCountsDS: Dataset[(String, Int)] =
      try {
        s3ToDataset[InternalFeedback](
          mlopsConfiguration.storageConfig.s3Bucket,
          S3PathNames.s3FeedbacksPath,
          header = false,
          fileFormat = "json").map(
          feedback => {
            (str(feedback.finalized.lineItems), 1)
          }
        ).persist()
      }
      catch {
        case e: IllegalArgumentException =>
          logger.error(s"Categorization: ${e.getMessage}")
          sparkSession.emptyDataset[(String, Int)]
      }

    SparkUtil.groupByKey[(String, Int), String](
      (entry: (String, Int)) => entry._1,
      (entry1: (String, Int), entry2: (String, Int)) => (entry1._1, entry1._2 + entry2._2),
      labelCountsDS
    ).map{
      case (key, (_, v)) => DistributionByKey(key, v)
    }
  }
}


private[bertspark] final object Categorization {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[Categorization])

  case class DistributionByKey(key: String, count: Int) {
    override def toString: String = s"Key:$key, Count:$count"
    def csv: String = s"$key,$count"
  }

    /**
    * Wrapper for the frequency of training data per sub-model and per sub-model x labels
    * @param distributionByEmr Distribution by sub model (EMR)
    * @param distributionByEmrAndLabels Distribution by sub-model (EMR) x labels
    * @param distributionByLabel Distribution by Label
    */
  case class DistributionByEmrAndLabels(
    distributionByEmr: Seq[DistributionByKey],
    distributionByEmrAndLabels: Seq[DistributionByKey],
    distributionByLabel: Seq[DistributionByKey]
  ) {

    def csv: String =
      s"${distributionByEmr.map(_.csv).mkString("\n")}${distributionByEmrAndLabels.map(_.csv).mkString("\n")}${distributionByLabel.map(_.csv).mkString("\n")}"

    def show(numRecords: Int): String =
      if(numRecords > 0)
        s"""Distribution by EMR
          |${distributionByEmr.take(numRecords).mkString("\n")}
          |
          |Distribution by EMR and Labels
          |${distributionByEmrAndLabels.take(numRecords).mkString("\n")}
          |
          |Distribution by Labels
          |${distributionByLabel.take(numRecords).mkString("\n")}
          |""".stripMargin
      else
        s"""Distribution by EMR
           |${distributionByEmr.mkString("\n")}
           |
           |Distribution by EMR and Labels
           |${distributionByEmrAndLabels.mkString("\n")}
           |
           |Distribution by Labels
           |${distributionByLabel.mkString("\n")}
           |""".stripMargin

    override def toString: String = show(1)
  }


  private def getEmrCode(emrCodes: Seq[MlEMRCodes]): String =
    if(emrCodes != null && emrCodes.nonEmpty)
      if(emrCodes.head.modifiers.nonEmpty)
        s"${emrCodes.head.cpt} ${emrCodes.head.modifiers.mkString(" ")}"
      else
        emrCodes.head.cpt
    else
      "NA"

  def distributionByLabel(s3TrainingPath: String): Unit = try {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val groupedSubModelsDS = S3Util.s3ToDataset[SubModelsTrainingSet](
      s3TrainingPath,
      header = false,
      fileFormat = "json"
    )

  } catch {
    case e: IllegalStateException =>
      logger.error(s"Failed to load Dataset from $s3TrainingPath")
  }
}
