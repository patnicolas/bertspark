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
package org.bertspark.nlp.vocabulary

import org.apache.spark.sql._
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames
import org.bertspark.config.S3PathNames.s3VocabularyPath
import org.bertspark.delay
import org.bertspark.nlp.medical.MedicalCodingTypes._
import org.bertspark.nlp.medical.ContextEncoder
import org.bertspark.nlp.medical.NoteProcessors.{logger => _, _}
import org.bertspark.nlp.token.TfIdf.WeightedToken
import org.bertspark.nlp.token.TokensTfIdf
import org.bertspark.util.SparkUtil
import org.bertspark.util.io._
import org.bertspark.util.io.S3Util.s3ToDataset
import org.slf4j._


/**
 * {{{
 * TF-IDF for documents grouped by labeled. The documents associated to a label are aggregated as
 * a corpus for which the TF-IDF is computed to distinguish from documents associated with other labels
 * The Tf-Idf are applied to note or group of note
 *  note:  Tf-Idf per note
 *  emr:   Tf-Idf using corpus generated with notes sharing the same emr CPT and modifier
 *  emrCpt:  Tf-Idf using corpus generated with notes sharing the same emr CPT
 *  label:  Tf-Idf using corpus generated with notes sharing the same label
 * }}}
 *
 * @param s3PredictReqFolder S3 folder for the prediction request
 * @param s3FeedbackFolder S3 folder for feedbacks
 * @param cutOffPercentageForWeightedTokens Defined the number of tokens to be selected for vocabulary as
 *                                          sorted in decreasing order
 * @param numRecords Number of feedback to be considered
 *
 * @author Patrick Nicolas
 * @version 0.4
 */
private[bertspark] final class CodingTermsTfIdf private (
  s3PredictReqFolder: String,
  s3FeedbackFolder: String,
  cutOffPercentageForWeightedTokens: Double,
  groupByMethod: String,
  numRecords: Int
) extends VocabularyComponent {
  import org.bertspark.implicits._
  import CodingTermsTfIdf._

  override val vocabularyName: String = "CodingTermsTfIdf - groupMethod"

  require(cutOffPercentageForWeightedTokens > 0.10 && cutOffPercentageForWeightedTokens < 0.98,
    s"cutOffPercentageForWeightedTokens $cutOffPercentageForWeightedTokens should be [0.1, 0.9]")

  private[this] val aggregator: NoteLabelEmr => String = groupByMethod match {
    case `labelGroup` => (noteLabelEmr: NoteLabelEmr) => noteLabelEmr.label
    case `emrGroup` => (noteLabelEmr: NoteLabelEmr) => noteLabelEmr.emr
    case `emrCptGroup` => (noteLabelEmr: NoteLabelEmr) =>
      if(noteLabelEmr.emr.nonEmpty) noteLabelEmr.emr.split(codeGroupSeparator).head.trim else "NoCPT"
    case _ =>  (noteLabelEmr: NoteLabelEmr) => noteLabelEmr.toString
  }

  override def build(initialTokens: Array[String], requestDS: Dataset[InternalRequest]): Array[String] = {
    import org.bertspark.config.MlopsConfiguration._
    import sparkSession.implicits._

    logDebug(logger, s"Loaded ${requestDS.count()} unique requests")
    val rawFeedbackDS = try {
      s3ToDataset[InternalFeedback](
        mlopsConfiguration.storageConfig.s3Bucket,
        s3FeedbackFolder,
        header = false,
        fileFormat = "json"
      ).dropDuplicates("id")
    } catch {
      case e: IllegalArgumentException =>
        logger.error(s"TfIdf codes: ${e.getMessage}")
        sparkSession.emptyDataset[InternalFeedback]
    }

    val feedbackDS = if(numRecords > 0) rawFeedbackDS.limit(numRecords) else rawFeedbackDS
    logDebug(logger, s"Loaded ${feedbackDS.count()} unique feedbacks")

    // Select random 40 notes for each label after grouping...
    val noteLabelEmrDS: Dataset[NoteLabelEmr] = SparkUtil.sortingJoin[InternalRequest, InternalFeedback](
      requestDS,
      "id",
      feedbackDS,
      "id"
    ).map{
      case (predictReq, feedbackReq) => {
        val textTokens = cleanse(predictReq.notes.head,  specialCharCleanserRegex).filter(!abbrStopTokens.contains(_))
        val label = feedbackReq.finalized.lineItems.head.cptPrimaryIcd
        val emr =
          if(predictReq.context.EMRCpts != null && predictReq.context.EMRCpts.nonEmpty)
            predictReq.context.EMRCpts.head.toCptModifiers
          else
            "na"
        val contextTokens = ContextEncoder.encodeContext(predictReq.context)
        NoteLabelEmr(textTokens, contextTokens, label, emr)
      }
    }
    logDebug(logger, s"Got ${noteLabelEmrDS.count} Note-label-Emr records")

    val noteGroupedByLabelDS: Dataset[NoteLabelEmr] =
      if(groupByMethod != noteGroup) {
        val ds = SparkUtil.groupBy[NoteLabelEmr, String](
          aggregator,
          (noteLabelEmr1: NoteLabelEmr, noteLabelEmr2: NoteLabelEmr) => {
            NoteLabelEmr(
              noteLabelEmr1.textTokens ++ noteLabelEmr2.textTokens,
              noteLabelEmr1.contextTokens ++ noteLabelEmr2.contextTokens,
              noteLabelEmr1.label,
              noteLabelEmr1.emr)
          },
          noteLabelEmrDS
        ).map(
          noteLabelEmr => {
            val sampledNote =
              if (noteLabelEmr.textTokens.length < groupedNoteCutoff) noteLabelEmr.textTokens
              else noteLabelEmr.textTokens.take(groupedNoteCutoff)

            logDebug(logger, s"Sampled note size for ${noteLabelEmr.label}: ${sampledNote.length}")
            NoteLabelEmr(sampledNote, noteLabelEmr.contextTokens, noteLabelEmr.label, noteLabelEmr.emr)
          }
        ).toDS()

        logDebug(logger, s"Got ${ds.count} grouped note-label-Emr records")
        ds
      }
      else
        noteLabelEmrDS

          // Save to S3
    S3Util.datasetToS3[NoteLabelEmr](
        mlopsConfiguration.storageConfig.s3Bucket,
        noteGroupedByLabelDS,
      s"$noteLabelEmrS3File/$groupByMethod",
        header = false,
        fileFormat = "json",
        toAppend = false,
        numPartitions = 8
      )

    delay(5000L)
    // Compute TF-IDF for corpus associated with a label
    val tfIdfKeys = computeTfIdf
    val outputTokens = (initialTokens ++ tfIdfKeys).distinct
    logDebug(logger, s"Vocabulary: Coding TF-IDF adds ${outputTokens.length - initialTokens.length} tokens")
    outputTokens
  }


  private def computeTfIdf: Array[String] = {
    import sparkSession.implicits._

    val s3IO = new S3IOOps[NoteLabelEmr](mlopsConfiguration.storageConfig.s3Bucket, noteLabelEmrS3File)
    val tokenTfIdf = TokensTfIdf(
      s3IO,
      (noteLabelEmr: NoteLabelEmr) => noteLabelEmr.textTokens.mkString(" "),
      if(numRecords == -1) 9999999 else numRecords)

    val outputS3IO = new S3IOOps[WeightedToken](mlopsConfiguration.storageConfig.s3Bucket, s3VocabularyPath)

    val weightedTokens: Array[WeightedToken] = tokenTfIdf(outputS3IO)
    val cutOff = (weightedTokens.length*cutOffPercentageForWeightedTokens).floor.toInt
    val topWeightedTokens: Array[WeightedToken] = weightedTokens.sortWith(_.weight > _.weight).take(cutOff)
    logDebug(
      logger,
      s"Number of top weighted tokens: ${topWeightedTokens.size}\n${topWeightedTokens.take(20).mkString(" ")}"
    )
    weightedTokens.map(_.token)
  }
}


/**
 * Singleton for constructors
 */
private[bertspark] final object CodingTermsTfIdf {
  final private val logger: Logger = LoggerFactory.getLogger("CodingTermsTfIdf")
  final private val maxNumRecordsForTfIdf = 60000

  final private val emrGroup = "emr"
  final private val labelGroup = "label"
  final private val emrCptGroup = "emrCpt"
  final private val noteGroup = "note"

  final val abbrStopTokens = Set[String]("an", "the", "me", "it", "my","of", "or", "and", "you", "they")
  final private val groupedNoteCutoff = 32000    // 60 notes x 9000 characters/note
  final private val noteLabelEmrS3File =
    s"${mlopsConfiguration.storageConfig.s3RootFolder}/${mlopsConfiguration.target}/noteLabelEmr/"
  final private val tfidfPath = S3PathNames.s3VocabularyPath

  case class NoteLabelEmr(
    textTokens: Array[String],
    contextTokens: Array[String],
    label: String,
    emr: String
  )

  def apply(cutOff: Double, groupByMethod: String, numRecords: Int): CodingTermsTfIdf = {
    new CodingTermsTfIdf(
      S3PathNames.s3RequestsPath,
      S3PathNames.s3FeedbacksPath,
      cutOff,
      groupByMethod,
      numRecords)
  }


  def apply(cutOff: Double, groupByMethod: String): CodingTermsTfIdf = {
    new CodingTermsTfIdf(
      S3PathNames.s3RequestsPath,
      S3PathNames.s3FeedbacksPath,
      cutOff,
      groupByMethod,
      maxNumRecordsForTfIdf)
  }

  /**
   * Generic constructor for command line application
   * {{{
   *  Arguments:
   *     S3 folder for prediction requests
   *     S3 folder for feedbacks
   *     Cut-off for the ordered list of weighted terms
   *     Number of feedback records to be used (all records if -1)
   * }}}
   * @param args Argument for the labeled TF-IDF operation
   * @return
   */
  def apply(args: Seq[String]): CodingTermsTfIdf = {
    require(args.size == 4,
      s"""LabeledTfIdf Arguments ${args.mkString} should be
         |'codeTermsTfIdf cut-off groupByMethod numRecords'""".stripMargin)
    apply(args(1).toDouble, args(2), args(3).toInt)
  }
}
