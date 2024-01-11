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

import org.apache.spark.sql.Dataset
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.analytics.Categorization.DistributionByEmrAndLabels
import org.bertspark.util.io.{DualS3Dataset, LocalFileUtil, S3Util}
import org.bertspark.config.{FsPathNames, S3PathNames}
import org.bertspark.config.S3PathNames.getS3FolderCompareSource
import org.bertspark.nlp.medical.{encodeLabeledTraining, filterTrainingSetPerLabelDistribution}
import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalFeedback, InternalRequest}
import org.bertspark.nlp.{medical, tokenSeparator}
import org.bertspark.nlp.trainingset.{ContextualDocument, SubModelsTrainingSet, TokenizedTrainingSet}
import org.bertspark.nlp.vocabulary.MedicalCodeDescriptors.icdsRef
import org.bertspark.util.io.LocalFileUtil.CSV_SEPARATOR
import org.bertspark.util.SparkUtil
import org.slf4j.{Logger, LoggerFactory}


/**
  * Create a report for accuracy of models
  * @param s3CompareFolder S3 folder containing comparison of prediction and labeled data
  * @param isPrimaryCodeMatchOnly Flag to specify that we compare only CPT and primary ICD
  * @author Patrick Nicolas
  * @version 0.4
  */
private[bertspark] final class Reporting private (s3CompareFolder: String, isPrimaryCodeMatchOnly: Boolean) extends Analyzer {
  import Reporting._


  /**
   * Normalize the computation of the accuracy during training
   * {{{
   *   The normalized, aggregated accuracy takes into account the distribution of sub-model#labels to reflect
   *   the actual distribution of data in production
   *   normalized_accuracy = SUM(count*rate)/total_count
   * }}}
   * @return Aggregated normalized accuracy
   */
  override def run(analyzerResult: AnalyzerResult): AnalyzerResult = {
    val allKeys = S3Util.getS3Keys(mlopsConfiguration.storageConfig.s3Bucket, s3CompareFolder).filter(_.contains(s"epoch-"))
    if (allKeys.isEmpty)
      throw new IllegalStateException(s"Failed to retrieve the comparison data for $s3CompareFolder")

    val keys = retrieveLastEpoch(allKeys)
    val subSummaryFields = Summarization.getSubSummaryFields(keys, isPrimaryCodeMatchOnly).filter(_._3 != -1)

    val subSummaryFieldsStr = subSummaryFields.map{
      case (subModel, totalCount, rate, _) => if(totalCount > 0) s"$subModel,$totalCount,$rate" else ""
    }   .filter(_.nonEmpty)
        .mkString("\n")

    val aggregatedAccuracy = LocalFileUtil.Load.local(FsPathNames.distributionJsonPath).map(
      content => {
        // Load the distribution from the file
        val distribution = LocalFileUtil.Json.mapper.readValue[DistributionByEmrAndLabels](
          content, classOf[DistributionByEmrAndLabels]
        )
        val distByEmrAndLabelsMap = distribution.distributionByEmrAndLabels.map(dist => (dist.key, dist.count)).toMap

        // Create a map of sub summary fields with subModel#label as key and rate as value
        val subSummaryFieldsSeq = subSummaryFields.map {
          case (subModel, _, rate, label) => (s"$subModel#$label", rate)
        }

        // Compute the normalized rate  ...
        val normalizedAccuracyBySubModel = subSummaryFieldsSeq.map {
          case (key, rate) => {
            val cnt = distByEmrAndLabelsMap.getOrElse(key, 0)
            (key, cnt, rate, cnt * rate)
          }
        } .filter(_._2 > 0.0)
            .sortWith(_._4 > _._4)

        if (normalizedAccuracyBySubModel.nonEmpty) {
          val totalCount = normalizedAccuracyBySubModel.map(_._2).reduce(_ + _)
          val weighedCount = normalizedAccuracyBySubModel.map(_._4).reduce(_ + _)
          val normalizedAccuracyBySubModelStr = normalizedAccuracyBySubModel.map(
            entry => s"${entry._1},${entry._2},${entry._3}"
          ).mkString("\n")
          (weighedCount / totalCount, normalizedAccuracyBySubModelStr)
        }
        else {
          logger.error(s"Failed to get normalized accuracy for $s3CompareFolder")
          (-1.0F, "")
        }
      }
    ).getOrElse({
      logger.error(s"Failed loading distribution for normalized aggregated accuracy - $s3CompareFolder")
      (-1.0F, "")
    })
    // Extract the comparison prediction - labels
    val comparisonSummary = getComparisonSummary(keys)
    //val distributionPerLabelsSet = aggregatedAccuracy._2.toSet

    // Generate the formatted accuracy report
    val accuracyReport =
      s"""
         |=========================== Configuration =============================
         |${mlopsConfiguration.toString}
         |============================ Distribution per sub-models ==========================
         |$accuracyReportHeader
         |$subSummaryFieldsStr
         |
         |============================ Comparison summary =======================
         |$comparisonSummary
         |
         |========================== Distribution per labels =========================
         |Label,Count,Rate
         |${aggregatedAccuracy._2}
         | -----------------------------------------------
         |Average adjusted for notes distribution: ${aggregatedAccuracy._1}
         |""".stripMargin
    // and Upload to S3
    try {
      S3Util.upload(
        mlopsConfiguration.storageConfig.s3Bucket,
        S3PathNames.getS3AccuracyReportPath,
        accuracyReport
      )
      AnalyzerResult(accuracyReport)
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"Analyzer result ${e.getMessage}")
        AnalyzerResult("failure")
    }
  }


  /**
   * Generate the comparison summary as a table
   * [IsMatch,label,prediction,count]
   * @param s3Keys S3 keys or paths to
   * @return Statistic summary for comparison of prediction and labels
   */
  def getComparisonSummary(s3Keys: Iterable[String]): String =
    if(s3Keys.nonEmpty) {
      val comparisonList = s3Keys.foldLeft(List[Iterable[(Boolean, Boolean, String, String, Int)]]())(
        (xs, key) => {
          S3Util.download(mlopsConfiguration.storageConfig.s3Bucket, key).map(
            content => {
              val compareSummaryFields: Seq[CompareSummaryFields] = content.split("\n").map(
                line => {
                  val fields = line.split(CSV_SEPARATOR).toSeq
                  if (fields.size >= labelIndex) {
                    val subModel = fields(subModelIndex)
                    (
                        subModel,
                        (
                            fields(numRecordsIndex).toFloat/mlopsConfiguration.executorConfig.batchSize).floor.toInt,
                            s"$subModel ${fields(predictionIndex)}",
                            s"$subModel ${fields(labelIndex)}",
                            strict(fields(predictionIndex), fields(labelIndex))
                        )

                  }
                  else {
                    logger.error(s"line $line for S3 key $key is incomplete or empty")
                    nullCompareSummaryFields
                  }
                }
              )   .toSeq
                  .filter(_._3.nonEmpty)

              val (successfulPredictions, failedPredictions) = compareSummaryFields.partition(_._5)
              val successPredictionCount = successfulPredictions.groupBy(_._4).map{
                case (label, xs) => (true, true, label, label, xs.length)
              }
              val failedPredictionCount = failedPredictions.groupBy(record => (record._4, record._3)).map{
                case ((label, prediction), xs) => {
                  val isIcdsMatched = precisionIcds(prediction, label)
                  (false, isIcdsMatched, label, prediction, xs.length)
                }
              }
              successPredictionCount ++ failedPredictionCount
            }
          ).getOrElse(Iterable.empty[(Boolean, Boolean, String, String, Int)]) :: xs
        }
      ).flatten.sortWith(_._4 > _._4)

      val comparisonTable = comparisonList.map {
        case (exactMatch, _, label, prediction, count) =>
          s"$count,${if(exactMatch) "SUCCEED" else "FAILED"},$prediction,$label"
      }.mkString("\n")


      val (exactMatchCount, notExactMatchCount) = matchesCount(comparisonList.partition(_._1))
      val (approxMatches, notApproxMatchCount) = matchesCount(comparisonList.partition(_._2))
      val totalCount = notExactMatchCount + exactMatchCount
      s"""
         |Count,Status,Prediction,Claim
         |$comparisonTable
         |-------------------------------
         |Predictions:$totalCount, Succeeded:$exactMatchCount, Failed:$notExactMatchCount Rate:${exactMatchCount.toDouble/totalCount} Approx. succeeded:$approxMatches, Approx. failed:$notApproxMatchCount, Rate:${approxMatches.toDouble/totalCount}""".stripMargin
    }
    else {
      logger.error(s"Could not find S3 paths for $s3CompareFolder")
      ""
    }

  private def matchesCount(
    comparisonList: (List[(Boolean, Boolean, String, String, Int)],List[(Boolean, Boolean, String, String, Int)])
    ): (Int, Int) = {

    val matchedList = comparisonList._1
    val notMatchedList = comparisonList._2
    val notMatches = notMatchedList.map(_._5)
    val notMatchCount = if(notMatches.nonEmpty) notMatches.reduce(_ + _) else 0
    val matches = matchedList.map(_._5)
    val matchCount = if(matches.nonEmpty) matches.reduce(_ + _) else 0
    (matchCount, notMatchCount)
  }
}

/**
  * Singleton for constructors and computing precision of ICD predictions
  */
private[bertspark] final object  Reporting {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[Reporting])

  final private val accuracyReportHeader = "EMR,Epoch no,Num notes,Accuracy"

  def apply(s3SourceFolder: String, isPrimaryCodeMatchOnly: Boolean): Reporting =
    new Reporting(s3SourceFolder, isPrimaryCodeMatchOnly)

  def apply(isPrimaryCodeMatchOnly: Boolean): Reporting = new Reporting(getS3FolderCompareSource, isPrimaryCodeMatchOnly)

  private def strict(prediction: String, label: String): Boolean = prediction == label

  def precisionIcds(prediction: String, label: String): Boolean = {
    val predictedIcds = extractIcds(prediction)
    val labeledIcds = extractIcds(label)
    predictedIcds.nonEmpty && labeledIcds.nonEmpty && predictedIcds.forall(labeledIcds.contains(_))
  }

  /**
    * Compute the number of tokens per medical document
    * @return Num tokens
    */
  def getNumTokensPerDoc: Int = {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    val contextualDocumentDS = try {
      S3Util.s3ToDataset[ContextualDocument](
        S3PathNames.s3ContextualDocumentPath,
        false,
        "json"
      )
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"getNumTokensPerDoc: ${e.getMessage}")
        sparkSession.emptyDataset[ContextualDocument]
    }

    val count = contextualDocumentDS.count()
    val numTokensPerDoc: Int = contextualDocumentDS.map(
      doc => doc.contextVariables.length + doc.text.split(tokenSeparator).length
    ).reduce(_ + _)
    val averageNumTokens = (numTokensPerDoc.toFloat/count).floor.toInt
    logger.info(s"Average num of tokens for ${mlopsConfiguration.preProcessConfig.vocabularyType}: $averageNumTokens")
    averageNumTokens
  }

  def datasetCounts(args: Seq[String]): String = {
    require(args.size == 3, s"Dataset counts should be 'datasetCounts dataType s3Folder")
    import org.bertspark.implicits._
    import sparkSession.implicits._
    val dataType = args(1)
    val s3Folder = args(2)

    val counts = new StringBuilder
    dataType match {
      case "requests" =>
        val requestDS = S3Util.s3ToDataset[InternalRequest](
          mlopsConfiguration.storageConfig.s3Bucket,
          s3Folder,
          header = false,
          fileFormat = "json"
        ).dropDuplicates("id")

        val numRequests = requestDS.count()
        logger.info(s"Number of items for $s3Folder: $numRequests")
        numRequests.toString

      case "feedbacks" =>
        val feedbackDS = S3Util.s3ToDataset[InternalFeedback](
          mlopsConfiguration.storageConfig.s3Bucket,
          s3Folder,
          header = false,
          fileFormat = "json"
        ).dropDuplicates("id")

        val numFeedbacks = feedbackDS.count
        logger.info(s"Number of items for $s3Folder: $numFeedbacks")
        numFeedbacks.toString

      case "contextualDocuments" =>
        val contextualDocumentDS = S3Util.s3ToDataset[ContextualDocument](
          mlopsConfiguration.storageConfig.s3Bucket,
          s3Folder,
          header = false,
          fileFormat = "json"
        ).dropDuplicates("id")

        val numContextualDocuments = contextualDocumentDS.count()
        logger.info(s"Number of items for $s3Folder: $numContextualDocuments")
        numContextualDocuments.toString

      case "training" =>
        val s3MapStorage = DualS3Dataset[SubModelsTrainingSet, TokenizedTrainingSet](
          s3Folder,
          filterTrainingSetPerLabelDistribution,
          encodeLabeledTraining
        )
        counts.append("Training set count=").append(s3MapStorage.inputDataset.dropDuplicates("subModel").count())
        logger.info(counts.toString)
        counts.toString
    }
  }

  private def extractIcds(claim: String): Array[String] =
    claim.split("-").flatMap(
      lineItem => {
        val codes = lineItem.trim.split(tokenSeparator)
        // We need to replace
        codes.filter(code => icdsRef.contains(code.replace(".", "")))
      }
  )


  def repartition(args: Seq[String]): Unit = {
    val fromS3Folder = args(1)
    val toS3Folder = args(2)
    val recordType = args(3)
    medical.repartition(fromS3Folder, toS3Folder, recordType)
  }
}