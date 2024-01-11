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

import java.io.File
import org.apache.spark.sql.{Encoder, SparkSession}
import org.bertspark.analytics.Cleanser.logger
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames
import org.bertspark.config.S3PathNames.getS3FolderCompareSource
import org.bertspark.nlp.medical.MedicalCodingTypes.{lineItemSeparator, InternalFeedback, InternalRequest}
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.bertspark.util.SparkUtil
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer


/**
 * {{{
 *   Class that analyze various output from training and validation data
 *   Methods:
 *    - getDistributionPerEmrAndLabel: Compute the distribution (frequency) per EMR and per (EMR, Document label)
 *    - getSubSummaryFields: Compute the statistics for each of the sub-model for the last epoch of the
 *                           training run. (subModelName, epochNo, totalCount for subModel, rate, label)
 *    - getNormalizedAccuracy: Compute the aggregated accuracy of the training run normalized by the
 *                             distribution frequency of (sub-model, label)
 * }}}
 * @param analysisType Type of analysis
 * @param s3SourceFolder S3 source folder: Feedback records for distributionPerEmrAndLabel and compare
 *                       records for normalizedAccuracy
 * @param sparkSession Implicit reference to the current Spark context
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class PredictionAnalysis private (
  analysisType: String,
  s3SourceFolder: String)(implicit sparkSession: SparkSession) {
  import PredictionAnalysis._

  private[this] val analyzers = ListBuffer[Analyzer]()

  def join(analyzer: Analyzer): Unit = analyzers.append(analyzer)

  def execute: String = try {
    var analyzerResult: AnalyzerResult = null
    analyzers.foreach(analyzer => analyzerResult = analyzer.run(analyzerResult))
    if(analyzerResult != null) analyzerResult.toString else ""
  } catch {
    case e: IllegalStateException =>
      logger.error(e.getMessage)
      ""
  }
}


/**
 * Singleton for constructors and computation of distribution
 */
private[bertspark] final object PredictionAnalysis {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[PredictionAnalysis])


  final val distributionPerEmrAndLabelLbl = "distributionPerEmrAndLabel"
  final val accuracyComparisonSummaryLbl = "accuracyComparisonSummary"
  final val comparisonSummaryLbl = "comparisonSummary"
  final val completeNormalizedAccuracyLbl = "completeNormalizedAccuracy"

  def apply(args: Seq[String])(implicit sparkSession: SparkSession): PredictionAnalysis = {
    require(args.size> 2,
      s"Incorrect argument for prediction analysis: [predictionAnalysis analysisType s3SourceFolder]")
    val predictionAnalysisType = args(1)
    val s3SourceFolder = args(2)
    apply(predictionAnalysisType, s3SourceFolder)
  }

  def apply(
    analysisType: String,
    s3CompareFolder: String
  )(implicit sparkSession: SparkSession): PredictionAnalysis = new PredictionAnalysis(analysisType, s3CompareFolder)

  def apply(analysisType: String)(implicit sparkSession: SparkSession): PredictionAnalysis =
    new PredictionAnalysis(analysisType, getS3FolderCompareSource)

  @throws(clazz = classOf[IllegalArgumentException])
  @throws(clazz = classOf[UnsupportedOperationException])
  def newInstance(args: Seq[String])(implicit sparkSession: SparkSession): PredictionAnalysis = {
    require(args.size == 4, s"Failed to instantiate prediction analysis ${args.mkString(" ")}")
    val s3SourceFolder = args(2)
    val predictionAnalysis = apply(args)
    val isPrimaryCodeMatchOnly = args(3).toBoolean

    args(1) match {
      case `distributionPerEmrAndLabelLbl` =>
        val categorization = new Categorization
        predictionAnalysis.join(categorization)
      case `completeNormalizedAccuracyLbl` =>
        val categorization = new Categorization
        predictionAnalysis.join(categorization)
        val summarization = Summarization(isPrimaryCodeMatchOnly)
        predictionAnalysis.join(summarization)
        val reporting = Reporting(isPrimaryCodeMatchOnly)
        predictionAnalysis.join(reporting)
      case _ =>
        throw new UnsupportedOperationException(s"Prediction analysis instance failed ${args(1)} not supported")
    }
    predictionAnalysis
  }


  def buildOracleClassificationModels(s3FeedbackFolder: String): Unit = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val internalFeedbackDS = try {
      S3Util.s3ToDataset[InternalFeedback](
        s3FeedbackFolder,
        false,
        "json"
      ).dropDuplicates("id")
          .map(
            feedback => (
                feedback.context.emrLabel,
                List[String](feedback.finalized.lineItems.map(_.lineItemSpace).mkString(lineItemSeparator))
            )
          ).persist()
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"buildOracleClassificationModels: ${e.getMessage}")
        sparkSession.emptyDataset[(String, List[String])]
    }

    internalFeedbackDS.show()

    val groupInternalFeedbackDS = SparkUtil.groupByKey[(String, List[String]), String](
      (s: (String, List[String])) => s._1,
      (s1: (String, List[String]), s2:(String, List[String])) => (s1._1, s1._2 ::: s2._2),
      internalFeedbackDS
    )
    val oracleFeedbackRDD = groupInternalFeedbackDS
        .map(_._2)
        .filter{ case (_, xs) => xs.size == 1}
        .map{ case (emr, xs) => s"$emr,${xs.head}"}
    println(s"${oracleFeedbackRDD.count()} Oracles for ${groupInternalFeedbackDS.count()} sub-models and ${internalFeedbackDS.count()} unique notes")
    oracleFeedbackRDD.toDS().show()

    S3Util.upload(
      mlopsConfiguration.storageConfig.s3Bucket,
      s"${mlopsConfiguration.storageConfig.s3RootFolder}/${mlopsConfiguration.target}/oracle.csv",
      oracleFeedbackRDD.collect.sortWith(_ < _).mkString("\n")
    )
  }

  def copy[T](s3Folder: String, dir: String, fs: String)(implicit encoder: Encoder[T]): Unit = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val dirFile = new File(dir)
    if (!dirFile.exists())
      dirFile.mkdir()

    val ds = S3Util.s3ToDataset[T](s3Folder, false, "json")
    val iter = ds.toLocalIterator()

    val collector = ListBuffer[T]()
    var index = 0
    while (iter.hasNext) {
      val nextRecord = iter.next
      collector.append(nextRecord)

      if (collector.size() > 2048) {
        index += 1
        LocalFileUtil.Save.local(s"$dir/$fs-$index", collector.mkString("\n"))
        collector.clear()
      }
    }
  }

  final private val headers = Seq[String](
    "FINDINGS",
    "IMPRESSION"
  )

  def analyzeNoteSection(s3Folder: String, maxNumRecords: Int): Unit = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val ds = S3Util.s3ToDataset[InternalRequest](
      s3Folder,
      false,
      "json"
    ).dropDuplicates("id").limit(maxNumRecords).map(_.notes.head)

    val count = ds.count()
    val resultDS = ds.map(
      note => {
        val findFindings = note.contains(headers.head)
        val findImpression = note.contains(headers(1))
        findFindings && findImpression
      }
    )
    val result = resultDS.collect
  }
}