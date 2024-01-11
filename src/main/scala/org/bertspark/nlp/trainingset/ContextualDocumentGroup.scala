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
package org.bertspark.nlp.trainingset

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalFeedback, InternalRequest}
import org.bertspark.nlp.medical.ContextEncoder
import org.bertspark.nlp.medical.NoteProcessors.{cleanse, specialCharCleanserRegex}
import org.bertspark.nlp.token.PretrainingInput
import org.bertspark.util.io.S3Util._
import org.bertspark.util.{CollectionUtil, SparkUtil}
import org.bertspark.Labels.labeledSentencesBuilderLbl
import org.bertspark.config.S3PathNames
import org.bertspark.config.S3PathNames.s3ContextualDocumentGroupPath
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.ContextualDocument.ContextualDocumentBuilder.preSelectRequest
import org.bertspark.nlp.vocabulary.CodingTermsTfIdf.abbrStopTokens
import org.bertspark.util.io.S3Util
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer


/**
 *
 * @param contextualDocuments Sequence of contextual documents associated with either a label or EMR codes
 * @param label Labeled medical codes or document
 * @param emr Stringized EMR data
 *
 * @author Patrick Nicolas
 * @version 0.2
 */
private[bertspark] case class ContextualDocumentGroup (
  contextualDocuments: List[ContextualDocument],
  label: String,
  emr: String) extends PretrainingInput {

  override def getId: String =
    if(mlopsConfiguration.preTrainConfig.sentenceBuilder == labeledSentencesBuilderLbl) emr
    else getContextualDocument.id

  def getContextualDocument: ContextualDocument = contextualDocuments.head
}


/**
 * Singleton wrapper for the builder
 */
private[bertspark] final object ContextualDocumentGroup {
  final private val logger: Logger = LoggerFactory.getLogger("ContextualDocumentGroup")

  final val segmentSeparator = "Seg_Sep"
  private val contextualDocumentSpan = mlopsConfiguration.preTrainConfig.numSentencesPerDoc

  implicit def cluster2contextualDocument(
    contextualDocumentCluster: ContextualDocumentGroup
  ): Array[ContextualDocument] = {
    val id = contextualDocumentCluster.emr
    val contextualTokens = Array[String](id)
    val contextualTextSegments: Array[List[ContextualDocument]] =
      if(contextualDocumentCluster.contextualDocuments.length > contextualDocumentSpan)
        contextualDocumentCluster.contextualDocuments.sliding(contextualDocumentSpan, contextualDocumentSpan).toArray
      else
        Array[List[ContextualDocument]](contextualDocumentCluster.contextualDocuments)

    contextualTextSegments.indices.map(
      index => {
        val text = contextualTextSegments(index).map(
          contextualDocument => contextualDocument.contextVariables.mkString(" ") + " " + contextualDocument.text
        ).mkString(segmentSeparator)
        ContextualDocument(s"$id-$index", contextualTokens, text)
      }
    ).toArray
  }



  implicit def cluster2contextualDocument(contextualDocument: ContextualDocument): Array[ContextualDocument] = {
    val textTokens = contextualDocument.text.split(tokenSeparator)
    // Splits the text of the contextual document into
    val textTokensList = CollectionUtil.split[String](textTokens, (s: String) => s == segmentSeparator)

    val boundedTextTokensList = textTokensList.size match {
      case 0 => throw new IllegalStateException("Contextual document has no segment separator")
      case 1 => padContextualDocuments(textTokensList)
      case _ =>  textTokensList.filter(_.length >= contextualDocumentSpan)
    }

    val idPrefix = contextualDocument.id
    boundedTextTokensList.indices.map(
      index => ContextualDocument(s"$idPrefix-$index", Array.empty[String], boundedTextTokensList(index).mkString(" "))
    ).toArray
  }


  private def padContextualDocuments(textTokensList: List[Array[String]]): List[Array[String]] =
    if(textTokensList.nonEmpty) {
      val buf = ListBuffer[Array[String]]() ++ textTokensList
      var index = 0
      do {
        buf.append(textTokensList(index % textTokensList.length))
        index += 1
      } while (buf.size < contextualDocumentSpan)
      buf.toList
    }
    else
      List.empty[Array[String]]

    /**
   * Builder for the contextual document clustered by EMR
   * @param maxNotesPerKey Maximum number of note per key (i.e. label, EMR codes...)
   * @param groupByMethod Grouping method (lable, emr, emr cpt only)
   * @param maxNumRecords Number of records used to generated the contextual data
   * @param sparkSession Implicit reference to the current Spark context
   */
  final class ContextualDocumentGroupBuilder (
      maxNotesPerKey: Int,
      groupByMethod: String)(implicit sparkSession: SparkSession) {
      import ContextualDocumentGroupBuilder._

      def apply(requestDS: Dataset[InternalRequest], numSubModels: Int): Unit =
        execute(requestDS, maxNotesPerKey, groupByMethod)

      def apply(vocabularyType: String, numSubModels: Int): Unit = {
        import sparkSession.implicits._

        // Step 1: Load the internal requests
        val internalRequestDS = try {
          s3ToDataset[InternalRequest](
            mlopsConfiguration.storageConfig.s3Bucket,
            S3PathNames.s3RequestsPath,
            header = false,
            fileFormat = "json").dropDuplicates("id")
        }
        catch {
          case e: IllegalStateException =>
            ContextualDocumentGroup.logger.error(s"ContextualDocumentGroupBuilder ${e.getMessage}")
            sparkSession.emptyDataset[InternalRequest]
        }

        // Step 2: Pre-select the internal requests according to a predefined number of sub models.
        val selectedInternalRequestDS = preSelectRequest(internalRequestDS, numSubModels)
        apply(selectedInternalRequestDS, numSubModels)
      }
    }


  private[bertspark] final object ContextualDocumentGroupBuilder {
    final private val logger: Logger = LoggerFactory.getLogger("ContextualDocumentClusterBuilder")

    def apply(maxNotesPerKey: Int, groupByMethod: String)(implicit sparkSession: SparkSession) =
      new ContextualDocumentGroupBuilder(maxNotesPerKey, groupByMethod)

    def apply(maxNotesPerKey: Int)(implicit sparkSession: SparkSession) =
      new ContextualDocumentGroupBuilder(maxNotesPerKey, "label")


    def execute(
      requestDS: Dataset[InternalRequest],
      maxNotesPerKey: Int,
      groupByMethod: String
    )(implicit sparkSession: SparkSession): Unit = {
      import org.bertspark.config.MlopsConfiguration._
      import sparkSession.implicits._

      val feedbackDS = try {
        s3ToDataset[InternalFeedback](
          mlopsConfiguration.storageConfig.s3Bucket,
          S3PathNames.s3FeedbacksPath,
          header = false,
          fileFormat = "json").dropDuplicates("id")
      }
      catch {
        case e: IllegalStateException =>
          logger.error(s"Contextual document group execute: ${e.getMessage}")
          sparkSession.emptyDataset[InternalFeedback]
      }

      logDebug(logger, s"ContextualDocumentGroupBuilder loaded ${feedbackDS.count()} unique feedbacks")

      // Select random 40 notes for each label after grouping...
      val tokensLabelEmrDS = SparkUtil.sortingJoin[InternalRequest, InternalFeedback](
        requestDS,
        tDSKey = "id",
        feedbackDS,
        uDSKey = "id"
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
          ContextualDocumentGroup(List[ContextualDocument](
            ContextualDocument(predictReq.id, contextTokens, textTokens.mkString(" "))), label, emr
          )
        }
      }
      logDebug(
        logger,
        s"""ContextualDocumentGroupBuilder got ${tokensLabelEmrDS.count} Contextual Document group records
           |${tokensLabelEmrDS.head.emr} ${tokensLabelEmrDS.head.label}
           |${tokensLabelEmrDS.head.contextualDocuments.head.toString}
           |""".stripMargin
      )

      val _groupedCtxDocDS: RDD[ContextualDocumentGroup] = SparkUtil.groupBy[ContextualDocumentGroup, String](
        getAggregator(groupByMethod),
        (ctxDocCluster1: ContextualDocumentGroup, ctxDocCluster2: ContextualDocumentGroup) => {
          ContextualDocumentGroup(
            ctxDocCluster1.contextualDocuments ::: ctxDocCluster2.contextualDocuments,
            ctxDocCluster1.label,
            ctxDocCluster1.emr)
        },
        tokensLabelEmrDS
      )
      logDebug(logger, {
        val firstEntry: String = _groupedCtxDocDS.take(1).map(
          group => s"${group.emr}-${group.label}\n${group.contextualDocuments.mkString("\n")}"
        ).head
        s"_groupedCtxDocDS: $firstEntry"
      })

      val groupedCtxDocDS = _groupedCtxDocDS.flatMap(
        tokensLabelEmr => {
          (0 until tokensLabelEmr.contextualDocuments.length by maxNotesPerKey).map(
            index => {
              val sampledNoteTokens =
                if(index + maxNotesPerKey < tokensLabelEmr.contextualDocuments.length)
                  tokensLabelEmr.contextualDocuments.slice(index, index + maxNotesPerKey)
                else
                  tokensLabelEmr.contextualDocuments.slice(index, index + tokensLabelEmr.contextualDocuments.length)

              ContextualDocumentGroup(sampledNoteTokens, tokensLabelEmr.label, tokensLabelEmr.emr)
            }
          )
        }
      )   // We want to make sure there is at least 2 segments per document.
          .filter(_.contextualDocuments.size > 1)
          .toDS()

      logDebug(logger, s"ContextualDocumentGroupBuilder got ${groupedCtxDocDS.count} grouped contextual documents")

      // Save to S3
      S3Util.datasetToS3[ContextualDocumentGroup](
        mlopsConfiguration.storageConfig.s3Bucket,
        groupedCtxDocDS,
        s3ContextualDocumentGroupPath,
        header = false,
        fileFormat = "json",
        toAppend = false,
        numPartitions = 8
      )
    }

    private def getAggregator(groupByMethod: String): ContextualDocumentGroup => String = groupByMethod match {
      case "label" => (noteLabelEmr: ContextualDocumentGroup) => noteLabelEmr.label
      case "emr" => (noteLabelEmr: ContextualDocumentGroup) => noteLabelEmr.emr
      case "emrCpt" => (noteLabelEmr: ContextualDocumentGroup) =>
        if(noteLabelEmr.emr.nonEmpty) noteLabelEmr.emr.split(tokenSeparator).head.trim else "NoCPT"
      case _ =>  (noteLabelEmr: ContextualDocumentGroup) => noteLabelEmr.label
    }
  }
}