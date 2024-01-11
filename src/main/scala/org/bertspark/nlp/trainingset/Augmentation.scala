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
package org.bertspark.nlp.trainingset
/*
import org.apache.spark.sql.{Dataset, SparkSession}
import org.mlops.config.MlopsConfiguration.mlopsConfiguration
import org.mlops.config.MlopsConfiguration.DebugLog.logDebug
import org.mlops.config.S3PathNames
import org.mlops.delay
import org.mlops.nlp.medical.MedicalCodingTypes.InternalFeedback
import org.mlops.nlp.tokenSeparator
import org.mlops.nlp.vocabulary.MedicalCodeDescriptors.CodeDescriptorMap
import org.mlops.util.io.S3Util
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer
import scala.util.Random


/**
  * {{{
  *  Augmentation of training set using original context documents and feedbacks using minimum number
  *  of notes per labels. If a label has < classifyConfig.minNumRecordsPerLabel then the contextual
  *  documents are augmented by randomly masking [UNK] one token.
  *  Workflow:
  *    1- Extract the labels for which num notes are < minimum defined in the configuration file
  *    2- Records the encounter ids associated with these labels
  *    3- Duplicate these contextual encounter by id → id_x01, id_x02 and randomly replace one token by [UNK]
  *    4- Store the map {id → id_x0n) into to local file
  *    5- Duplicate Internal feedback with new ids
  *    6- Regenerate training set
  *
  * Command line:  augmentTrainingSet [num_records]
  * }}}
  *
  * @author Patrick Nicolas
  * @version 0.8
  */
private[mlops] final object Augmentation {
  final private val logger: Logger = LoggerFactory.getLogger("Augmentation")
  private val rand = new Random(42L)


  def apply(args: Seq[String]): Unit =
    if(args.size > 1) apply(args(1).toInt) else apply()


  /**
    * Extends the current set of numRecords contextual documents (or all contextual documents if numRecords = -1)
    * see the 6 steps on the singleton comments
    * @param numRecords Number of records (-1) for all contextual documents
    */
  def apply(numRecords: Int): Unit = {
    import org.mlops.implicits._

    val trainingSetDS = Augmentation.loadTrainingSet(numRecords)
    if(trainingSetDS.isEmpty)
      logger.error(s"Load training set is empty!")
    else {
      val idsPerLabelCount = Augmentation.extractSparseLabelNotes(trainingSetDS)
      Augmentation.augmentContextualDocument(idsPerLabelCount)
      Augmentation.augmentFeedback(idsPerLabelCount)
      Augmentation.createTrainingSet
    }
  }

  def apply(): Unit = apply(-1)



  // -------------------  Private/Helper methods --------------------

  private def loadTrainingSet(
    numSubModels: Int
  )(implicit sparkSession: SparkSession): Dataset[SubModelsTrainingSet] = try {
    import sparkSession.implicits._

    val ds = S3Util.s3ToDataset[SubModelsTrainingSet](S3PathNames.s3ModelTrainingPath)
    val groupedSubModelDS = if(numSubModels > 0) ds.limit(numSubModels).persist().cache() else ds.persist().cache()
    logDebug(logger, msg = s"Training set ${S3PathNames.s3ModelTrainingPath} with ${groupedSubModelDS.count()} is loaded")
    groupedSubModelDS
  }
  catch {
    case e: IllegalArgumentException =>
      import sparkSession.implicits._
      logger.error(e.getMessage)
      sparkSession.emptyDataset[SubModelsTrainingSet]
  }


  private def extractSparseLabelNotes(
    groupedSubModelTrainingDS: Dataset[SubModelsTrainingSet]
  )(implicit sparkSession: SparkSession): Seq[(String, (String, Int))] = {
    import sparkSession.implicits._

    val minNumNotesPerLabels = mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
    val idNumIdsPairs = groupedSubModelTrainingDS.map(
      subModel => {
        val labelTrainingData = subModel.labeledTrainingData.groupBy(_.label)
        logDebug(
          logger,
          {
            val labelsNotes = labelTrainingData.map{ case (label, trainingSet) => s"$label:${trainingSet.length}"}
            s"SubModel: $subModel => ${labelsNotes.mkString(" ")}"
          }
        )
        val sparseLabelNotes = labelTrainingData.filter{
          case (_, trainingSet) => trainingSet.length > 1 && trainingSet.length < minNumNotesPerLabels
        }

        if(sparseLabelNotes.nonEmpty)
          sparseLabelNotes.flatMap {
            case (label, tokenizedTrainingSeq) =>
              val ids = tokenizedTrainingSeq.map(_.contextualDocument.id)
              ids.map((_, (label, ids.length)))
          }.toSeq
        else {
          logger.warn(s"Sub model $subModel has not label to be augmented..")
          Seq.empty[(String, (String, Int))]
        }
      }
    )   .filter(_.nonEmpty)
        .collect()
        .flatten

    groupedSubModelTrainingDS.unpersist()
    logDebug(
      logger,
      s"Find the ids->length: ${idNumIdsPairs.map{ case (id, (label, n)) => s"$id:$n-$label"}.mkString("\n")}"
    )
    idNumIdsPairs
  }


  private def augmentContextualDocument(
    idsPerLabelCount: Seq[(String, (String, Int))]
  )(implicit sparkSession: SparkSession): Boolean = try {
    import sparkSession.implicits._

    logDebug(logger, msg = s"Starts augmenting contextual documents for ${idsPerLabelCount.length} ids")

    // Step 1: Load the contextual document
    val contextualDocumentDS = S3Util.s3ToDataset[ContextualDocument](S3PathNames.s3ContextualDocumentPath)
    logDebug(
      logger,
      msg = s"Contextual document for ${S3PathNames.s3ContextualDocumentPath} loaded"
    )
    val idsPerLabelCount_brdCast = sparkSession.sparkContext.broadcast[Seq[(String, (String, Int))]](idsPerLabelCount)

    // Walk through the original, raw contextual document and spawn new one
    // by extending the id and randomly replacing one of the token with '[UNK]'
    val augmentedContextualDocDS = contextualDocumentDS.mapPartitions(
      (iter: Iterator[ContextualDocument]) => {
        val idsPerLabelCountMap: Map[String, (String, Int)] = idsPerLabelCount_brdCast.value.toMap
        val augmentedContextualDocuments = ListBuffer[ContextualDocument]()

        while(iter.hasNext) {
          val contextualDocument = iter.next()

          if(idsPerLabelCountMap.contains(contextualDocument.id)) {
            val numIds = idsPerLabelCountMap(contextualDocument.id)._2
            val minNumRecordsPerLabel = mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
            val numContextualDocToExtendPerIds = (minNumRecordsPerLabel.toFloat/numIds).ceil.toInt
            val label = idsPerLabelCountMap(contextualDocument.id)._1

            logDebug(logger, s"Augments ${contextualDocument.id} with $numIds contextual documents")
            (0 until numContextualDocToExtendPerIds).foreach(
              idx => {
                val augmentedContextualDoc = insertUnknownToken(contextualDocument, label,idx)
                augmentedContextualDocuments.append(augmentedContextualDoc)
              }
            )
          }
        }
        augmentedContextualDocuments.iterator
      }
    )

    // Store the augmented contextual document into S3 along with the original contextual documents
    logDebug(logger, s"Save augmented ${augmentedContextualDocDS.count()} contextual documents")
    S3Util.datasetToS3[ContextualDocument](
      mlopsConfiguration.storageConfig.s3Bucket,
      augmentedContextualDocDS,
      S3PathNames.s3ContextualDocumentPath,
      header = false,
      fileFormat =  "json",
      toAppend = true,
      numPartitions = 4)

    delay(3000L)
    true
  }
  catch {
    case e: IllegalArgumentException =>
      logger.error(e.getMessage)
      false
  }


  private def insertUnknownToken(
    contextualDocument: ContextualDocument,
    label: String,
    idx: Int): ContextualDocument = {
    val codeDescriptorSet = CodeDescriptorMap.getClaimDescriptors(label).toSet

    val augId = augmentId(contextualDocument.id, idx)
    val contextTokens = contextualDocument.contextVariables
    val textTokens = contextualDocument.text.split(tokenSeparator)
    val numTokens = contextTokens.length + textTokens.length

    var augmentedContextualDoc: Option[ContextualDocument] = None

    do {
      val indexUnknownTokens = rand.nextInt(numTokens - 1)
      if (indexUnknownTokens < contextTokens.length)
        if (!codeDescriptorSet.contains(contextTokens(indexUnknownTokens))) {
          contextTokens(indexUnknownTokens) = "[UNK]"
          augmentedContextualDoc = Some(ContextualDocument(augId, contextTokens, contextualDocument.text))
        }
      else {
        if (!codeDescriptorSet.contains(textTokens(indexUnknownTokens - contextTokens.length))) {
          textTokens(indexUnknownTokens - contextTokens.length) = "[UNK]"
          augmentedContextualDoc = Some(ContextualDocument(augId, contextTokens, textTokens.mkString(" ")))
        }
      }
    } while(!augmentedContextualDoc.isDefined)

    augmentedContextualDoc.get
  }



  private def augmentFeedback(
    idsPerLabelCount: Seq[(String, (String, Int))]
  )(implicit sparkSession: SparkSession): Boolean = try {
    import sparkSession.implicits._

    logDebug(logger, msg = s"Augment feedbacks for ${idsPerLabelCount.length} ids ")
    val internalFeedbackDS = S3Util.s3ToDataset[InternalFeedback](S3PathNames.s3FeedbacksPath)
    val idsPerLabelCount_brdCast = sparkSession.sparkContext.broadcast[Seq[(String, (String, Int))]](idsPerLabelCount)
    val augmentedFeedbacks = ListBuffer[InternalFeedback]()

    val augmentedFeedbackDS = internalFeedbackDS.mapPartitions(
      (iter: Iterator[InternalFeedback]) => {

        val idsPerLabelCountMap = idsPerLabelCount_brdCast.value.toMap
        while(iter.hasNext) {
          val internalFeedback = iter.next()

          if(idsPerLabelCountMap.contains(internalFeedback.id)) {
            val minNumRecordsPerLabel = mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
            val numIds = idsPerLabelCountMap(internalFeedback.id)._2
            val numContextualDocToExtendPerIds = (minNumRecordsPerLabel.toFloat/numIds).ceil.toInt

            (0 until numContextualDocToExtendPerIds).foreach(
              idx => {
                val augmentedFeedback = internalFeedback.copy(id = augmentId(internalFeedback.id, idx))
                augmentedFeedbacks.append(augmentedFeedback)
              }
            )
          }
        }
        augmentedFeedbacks.iterator
      }
    )

    logDebug(logger, s"Save augmented ${augmentedFeedbackDS.count()} feedbacks")
    S3Util.datasetToS3[InternalFeedback](
      mlopsConfiguration.storageConfig.s3Bucket,
      augmentedFeedbackDS,
      S3PathNames.s3FeedbacksPath,
      header = false,
      fileFormat =  "json",
      toAppend = true,
      numPartitions = 4)
    delay(3000L)
    true
  }
  catch {
    case e: IllegalArgumentException =>
      logger.error(e.getMessage)
      false
  }

  private def createTrainingSet(implicit sparkSession: SparkSession): Boolean = {
    TrainingSetBuilder.build()
    true
  }


  private def augmentId(id: String, index: Int): String = s"${id}_x$index"
}

 */

