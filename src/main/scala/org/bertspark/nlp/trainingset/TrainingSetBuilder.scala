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

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark._
import org.bertspark.classifier.training.ClassifierTrainingSetFilter
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalFeedback
import org.bertspark.util.SparkUtil
import org.bertspark.util.io._
import org.bertspark.util.io.S3Util.s3ToDataset
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.config.MlopsConfiguration.DebugLog.{logDebug, logTrace}
import org.bertspark.config.S3PathNames
import org.bertspark.modeling.{InputValidation, ModelOutput, SubModelsTaxonomy, TrainingLabelIndexing}
import org.bertspark.nlp.medical.MedicalCodingTypes.FeedbackLineItem.str
import org.bertspark.nlp.token.TokenizerPreProcessor.AbbreviationMap.abbreviationMap
import org.bertspark.nlp.trainingset.ContextualDocument.ContextualDocumentBuilder
import org.bertspark.nlp.vocabulary.MedicalCodeDescriptors.CodeDescriptorMap
import org.slf4j._
import scala.collection.mutable.ListBuffer

/**
 * Singleton to build a customized medical training set
 * {{{
 *   Building steps:
 *    1: Load and generate Labeled requests dataset
 *    2: Optional limit on the labeled data set
 *    3: Filter the labeled with a minimum frequency of occurrences
 *    4: Index the labels/claims
 *    5: Generate the labeled training set
 *    6: Store indexed labels in S3
 *    7: Store the indexed labels and contextual document training set into S3
 * }}}
 *
 * @author Patrick Nicolas
 * @version 0.3
 */
private[bertspark] final object TrainingSetBuilder
    extends InputValidation
        with ModelOutput[Dataset[SubModelsTrainingSet]] {
  final private val logger: Logger = LoggerFactory.getLogger("TrainingSetBuilder")

  // Pair {Sub model -> Training data grouped by label}
  type SubModelClaimNote = (String, Seq[List[TrainingLabel]])
  import org.apache.spark.sql.Dataset

  override protected def output(
    subModelsTrainingDS: org.apache.spark.sql.Dataset[SubModelsTrainingSet]
  )(implicit sparkSession: SparkSession) : Unit = save(subModelsTrainingDS)

  /**
   * Build training set from notes and feedback labels as invoked through the command line for a given
   * vocabulary type.
   *
   * {{{
   *  Break sub models into groups
   *       - buildTrainingSet
   *  Generate sub model taxonomy files (subModels.csv and labelIndexMap.csv
   *       - buildTrainingSet generateSubModelTaxonomy
   *  Create training set of contextual documents and training dat
   *       - buildTrainingSet vocabularyType numTrainingRecords
   *  Create training set only
   *       - buildTrainingSet vocabularyType 0
   *         i.e. buildTrainingSet AMA 0
   *  Create contextual document cluster and training set given a vocabulary
   *      - buildTrainingSet vocabularyType maxNumNotePerKey numRecords
   *        i.e. buildTrainingSet AMA 8 label 650000
   * }}}
   *
   * @param args Command line arguments
   * @param sparkSession Implicit reference to the current Spark context
   * @throws IllegalArgumentException If the list of arguments is incorrect
   */
  @throws(clazz = classOf[IllegalArgumentException])
  def apply(args: Seq[String])(implicit sparkSession: SparkSession): Unit = args.size match {
    case 1 => breakSubModels()
    case 2 => generateSubModelTaxonomy
    case 4 => createContextualDocumentTrainingSet(args)
    case 5 => createContextualDocumentGroupTrainingSet(args)
    case _ =>
      throw new IllegalArgumentException(s"Command line arguments ${args.mkString(" ")} are incorrect\nShould be 'buildTrainingSet Vocabulary SubModels true|false'")
  }

  @throws(clazz = classOf[InvalidParamsException])
  override protected def validate(args: Seq[String]): Unit = {
    require(
      args.size > 2,
      s"Command line arguments should be 'buildTrainingSet vocabulary numSubModels'"
    )
    CodeDescriptorMap.validate

    val vocabularySize = vocabulary.size()
    if(vocabularySize > 1024)
      logger.info(s"$vocabularySize terms in vocabulary")
    else
      throw new InvalidParamsException("Vocabulary undefined")

    val abbreviationMapSize = abbreviationMap.size
    if(abbreviationMapSize > 1)
      logger.info(s"$abbreviationMapSize Abbreviations")
    else
      throw new InvalidParamsException("Abbreviations map undefined")
  }

  def joinContextDocumentAndLabels(
    contextualDocDS: Dataset[ContextualDocument],
    feedbackDS: Dataset[InternalFeedback]
  )(implicit sparkSession: SparkSession): Dataset[LabeledRequest] = {
    import sparkSession.implicits._

    val labeledClaimDS = feedbackDS.map(LabeledClaim(_))
    logDebug(logger, msg = s"Generated ${labeledClaimDS.count()} labeled claims")

    // Step 1.5: Join the request/notes with the labeled claim
    SparkUtil.sortingJoin[ContextualDocument, LabeledClaim](
      contextualDocDS,
      tDSKey = "id",
      labeledClaimDS,
      uDSKey = "id")
        .map {
          case (contextualDocument, labelClaim) =>
            LabeledRequest(contextualDocument, labelClaim.emrCodes, labelClaim.feedbackLineItems)
        }
  }


  /**
    * Group labeled requests by emr
    * {{{
    * LabeledRequest
    *   - contextualDocument Contextual ContextualDocument(id: String, contextVar: Array[String], text: String)
    *   - emrCodes           EMR codes
    *   - lineItems          Labeled line items
    * }}}
    * @param labeledRequestDS Labeled requests
    * @param sparkSession Implicit reference to the current spark context
    * @return Dataset of training labels grouped by EMR
    */
  def groupLabeledRequestsByEmr(
    labeledRequestDS: Dataset[LabeledRequest]
  )(implicit sparkSession: SparkSession): Dataset[SubModelClaimNote] = {
    import sparkSession.implicits._

    val trainingLabelDS = labeledRequestDS.map(
      labelRequest => List[TrainingLabel](TrainingLabel(labelRequest))
    )

    // Step 1.8: Group by EMR codes
    val trainingLabelGroupedByEmrDS: Dataset[(String, List[TrainingLabel])] = SparkUtil
        .groupBy[List[TrainingLabel], String](
          (hClaimStr: List[TrainingLabel]) => (hClaimStr.head.emrStr),
          (hClaimStr1: List[TrainingLabel], hClaimStr2: List[TrainingLabel]) => hClaimStr1 ::: hClaimStr2,
          trainingLabelDS
        ).toDS.map(hClaimStr => (hClaimStr.head.emrStr, hClaimStr))
    logDebug(logger, msg = s"Generated ${trainingLabelGroupedByEmrDS.count} contextual documents grouped by subModels")

    // This section computes the distribution SubModel -> Label -> Notes and stored
    // into local file for debugging purpose
    scanSubModelLabelNoteDist(trainingLabelGroupedByEmrDS)

    // Step 1.9: Sub-group by remaining claim
    // Note: We need to filter out the data set associated with sub model with undefined labels and contextual docs
    val trainingLabelFilteredByNumNotesDS: Dataset[SubModelClaimNote] = trainingLabelGroupedByEmrDS.map {
      case (emr, trainingLabels) => (emr, TrainingSetSampler.filterPerNumNotesLabels(trainingLabels))
    }

    logDebug(logger, msg = s"balancedGroupedOfClaimDS ${trainingLabelFilteredByNumNotesDS.count()}")
    trainingLabelFilteredByNumNotesDS
  }

  final def getLabelsIndicesPath(path: String): String = s"$path-indices"


  // --------------------------   Helper functions --------------------------------


  private def createContextualDocumentTrainingSet(args: Seq[String])(implicit sparkSession: SparkSession): Unit = {
    validate(args)
    val vocabularyType = args(1)
    val numRecords = args(2).toInt
    val trainingDataOnly = args(3).toBoolean

    // Build the contextual document, then training set given a vocabulary
    if(!trainingDataOnly) {
      val builder = ContextualDocumentBuilder()
      builder(vocabularyType, numRecords)
      delay(4000L)
    }
    // Build the training set for classifier
    build(vocabularyType, numRecords)
  }

  private def createContextualDocumentGroupTrainingSet(args: Seq[String])(implicit sparkSession: SparkSession): Unit = {
    import org.bertspark.nlp.trainingset.ContextualDocumentGroup.ContextualDocumentGroupBuilder

    val vocabularyType = args(1)
    val maxNotesPerKey = args(2).toInt
    val numSubModels = args(3).toInt

    val builder = ContextualDocumentGroupBuilder(maxNotesPerKey)
    builder(vocabularyType, numSubModels)
    delay(4000L)
    build(vocabularyType, -1)
  }


  def build()(implicit sparkSession: SparkSession): Boolean =
    build(mlopsConfiguration.preProcessConfig.vocabularyType, -1)



    /**
    * Build the training set from labels/feedbacks and contextual documents
    * @param vocabularyType Type of vocabulary
    * @param sparkSession Implicit reference to the current Spark Context
    */
  private def build(vocabularyType: String, numRecords: Int)(implicit sparkSession: SparkSession): Boolean = {
    import sparkSession.implicits._
    import S3PathNames._

    val (s3ContextualDocFolder, s3TrainingSetFolder) =
      if(mlopsConfiguration.isLabeledSentencesBuilder) (s3ContextualDocumentGroupPath, s"${s3ModelTrainingPath}Cluster")
      else (s3ContextualDocumentPath, s3ModelTrainingPath)
    logDebug(logger, msg = s"ContextualDoc folder: $s3ContextualDocFolder to Training set folder $s3TrainingSetFolder")

    // Step 1: Load and generate Labeled requests dataset
    // (emr, [remaining_claim/labels ->  contextualDocument, emrStr, remainingClaim])
    val keyedTrainingLabelsDS: Dataset[SubModelClaimNote] = extractTrainingLabels(
      mlopsConfiguration.storageConfig.s3Bucket,
      s3ContextualDocFolder,
      s3FeedbacksPath,
      numRecords)

    // Step 2: Group Training set by EMR codes
    val nonKeyedTrainingLabelsDS = keyedTrainingLabelsDS.filter(_._2.nonEmpty)
    val totalNumTrainingSubModel = nonKeyedTrainingLabelsDS.count()
    logDebug(logger, msg = s"Extracted $totalNumTrainingSubModel labeled grouped notes")
    var counter = 0

    val subModelsTrainingDS: Dataset[SubModelsTrainingSet] = nonKeyedTrainingLabelsDS.map {
      case (emr, labeledSubModelsTrainingData) =>
        val indexedRemainingClaims= labeledSubModelsTrainingData
                .map(_.head.claim.replaceAll("  ", " "))
                .zipWithIndex
        counter += 1

        logDebug(
          logger,
          {
            val progress = 100.0F*counter/totalNumTrainingSubModel
            s"$emr has ${indexedRemainingClaims.size} labels $counter/$totalNumTrainingSubModel subModels $progress%"
          }
        )
            // Create the sub models with their associated training data
        val labeledTrainingData = labeledSubModelsTrainingData.flatMap(
          groupedTrainingSet => {
            val trainingLabels: Seq[TrainingLabel] = groupedTrainingSet
            trainingLabels.map(TokenizedTrainingSet(_))
          }
        )
        SubModelsTrainingSet(emr.trim, labeledTrainingData, indexedRemainingClaims)
    }

    logDebug(logger, msg = s"Number of sub models for training: ${subModelsTrainingDS.count()}")
    logTrace(
      logger,
      {
        if (subModelsTrainingDS.isEmpty) {
          logger.error(s"Failed to create a training set for $vocabularyType")
          ""
        } else {
          val subModelsDistribution = subModelsTrainingDS
              .map(subModels => (subModels.subModel, subModels.labeledTrainingData.size))
              .collect()
          val dump = subModelsDistribution.map { case (subModel, sz) => s"$subModel:$sz" }.mkString("\n")
          val totalCount = subModelsDistribution.map(_._2).reduce(_ + _)

          LocalFileUtil.Save.local(
            fsFileName = "output/subModelsDistribution.txt",
            content = s"Sub models distribution $dump for $totalCount notes")
          s"LabeledSubModelsTrainingDS ${subModelsTrainingDS.count()}\n$dump"
        }
      }
    )

    // Step 3: Store the indexed labels and contextual document training set into S3
    try {
      S3Util.datasetToS3[SubModelsTrainingSet](
        mlopsConfiguration.storageConfig.s3Bucket,
        subModelsTrainingDS,
        s3TrainingSetFolder,
        header = false,
        fileFormat = "json",
        toAppend = false,
        numPartitions = 32
      )
      delay(timeInMillis = 12000L)
      true
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"Training set failed to be save: ${e.getMessage}")
        false
    }
  }


  private def extractTrainingLabels(
    s3Bucket: String,
    s3ContextualDocFolder: String,
    s3FeedbackFolder: String,
    numRecords: Int
  )(implicit sparkSession: SparkSession): Dataset[SubModelClaimNote] = {
    import sparkSession.implicits._

    // Step 1 Load contextual documents and feedbacks
    val (contextualDocDS, feedbackDS) = loadContextualDocumentAndFeedbacks(
      s3Bucket,
      s3ContextualDocFolder,
      s3FeedbackFolder,
      numRecords)
    logDebug(logger, msg = s"ExtractTrainingLabels load ${contextualDocDS.count} contextual docs")

    val labeledRequestDS: Dataset[LabeledRequest] = joinContextDocumentAndLabels(contextualDocDS, feedbackDS)
    logDebug(logger, msg = s"Generated ${labeledRequestDS.count} labeled requests")

    // @todo computes the distribution per labels
    val validLabeledRequestDS = extractLabelDistribution(labeledRequestDS)
    if(validLabeledRequestDS.count() > 0)
      groupLabeledRequestsByEmr(validLabeledRequestDS)
    else {
      logger.warn(s"Join of $s3ContextualDocFolder and $s3FeedbackFolder is empty")
      Seq.empty[(String, Seq[List[TrainingLabel]])].toDS()
    }
  }


  private def extractLabelDistribution(
    labeledRequestDS: Dataset[LabeledRequest]
  )(implicit sparkSession: SparkSession): Dataset[LabeledRequest] = {
    import sparkSession.implicits._

    val validLabelsRDD: RDD[String] = labeledRequestDS
        .map(labeledReq => (str(labeledReq.lineItems), 1))
        .rdd
        .reduceByKey(_ + _)
        .filter(_._2 >= mlopsConfiguration.classifyConfig.minNumRecordsPerLabel)
        .map(_._1)

    val validLabels = validLabelsRDD.map(_.replaceAll("  ", " ")).collect.toSet
    labeledRequestDS.filter(labeledReq => validLabels.contains(str(labeledReq.lineItems)))
  }


  private def scanSubModelLabelNoteDist(
    trainingLabelGroupedByEmrDS: Dataset[(String, List[TrainingLabel])]
  )(implicit sparkSession: SparkSession):  Unit = {
    import sparkSession.implicits._

    val trainingLabelFilteredByNumNotes =
      trainingLabelGroupedByEmrDS.map {
        case (emr, trainingLabels) =>
          val claimNotesDistribution = TrainingSetSampler.getNumNotesPerLabels(trainingLabels)
          val claimNotesDistributionStr: Seq[String] = claimNotesDistribution.map{
            case (label, numNotes) => s"$label:$numNotes"
          }
          s"$emr\n${claimNotesDistributionStr.mkString("\n")}"
      }.collect()

    LocalFileUtil.Save.local(
      fsFileName = "output/SubModelClaimNotesDistribution.txt",
      trainingLabelFilteredByNumNotes.mkString("\n\n")
    )
  }


  private def loadContextualDocumentAndFeedbacks(
    s3Bucket: String,
    s3ContextualDocFolder: String,
    s3FeedbackFolder: String,
    numRecords: Int
  )(implicit sparkSession: SparkSession): (Dataset[ContextualDocument], Dataset[InternalFeedback]) = {
    import sparkSession.implicits._

    // Step 1.1: First load the feedback requests
    val feedbackDS = try {
      // Load the raw feedbacks
      val rawInternalFeedbackDS = s3ToDataset[InternalFeedback](
        s3Bucket,
        s3FeedbackFolder,
        header = false,
        fileFormat = "json").dropDuplicates("id")
      // Applies filter if needed
      ClassifierTrainingSetFilter.filter(rawInternalFeedbackDS, numRecords).persist().cache()
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"loadContextualDocumentAndFeedbacks: ${e.getMessage}")
        sparkSession.emptyDataset[InternalFeedback]
    }
    logDebug(logger, s"Loaded ${feedbackDS.count()} unique feedbacks")

    // Step 1.4: Load request/notes
    val rawContextualDocDS: Dataset[ContextualDocument] = {
      // We need to normalize to contextual documents from clusters
      if(mlopsConfiguration.isLabeledSentencesBuilder) {
        try {
          s3ToDataset[ContextualDocumentGroup](
            s3Bucket,
            s3ContextualDocFolder,
            header = false,
            fileFormat = "json").flatMap(_.contextualDocuments)
        }
        catch {
          case e: IllegalArgumentException =>
            logger.error(s"loadContextualDocumentAndFeedbacks: ${e.getMessage}")
            sparkSession.emptyDataset[ContextualDocument]
        }
      } else try {
        s3ToDataset[ContextualDocument](
          s3Bucket,
          s3ContextualDocFolder,
          header = false,
          fileFormat = "json")
      }
      catch {
        case e: IllegalArgumentException =>
          logger.error(s"loadContextualDocumentAndFeedbacks: ${e.getMessage}")
          sparkSession.emptyDataset[ContextualDocument]
      }
    }

    val contextualDocDS = rawContextualDocDS.dropDuplicates("id")
    logDebug(logger, s"Generated ${contextualDocDS.count()} unique contextual documents")
    (contextualDocDS, feedbackDS)
  }


  /**
    * Break down the number of sub models into groups of sub models defined in a file
    * @param numSubModelsPerRuns Maximum number of sub model per file or training
    */
  private def breakSubModels(numSubModelsPerRuns: Int = 512): Unit = {
    import S3PathNames._

    logDebug(logger, msg = "Start breaking sub models ...")
    val subModelsBatch = ListBuffer[String]()
    var count = 1
    S3Util.download(mlopsConfiguration.storageConfig.s3Bucket, s3SubModelsStructure).foreach(
      content => {
        val lines = content.split("\n").sortWith(_ < _)
        logDebug(logger, msg = s"Total number of sub models: ${lines.size}")

        // We only consider the sub models that needs training
        // Not the Oracle that have one label per sub-model
        val predictiveLines = lines.filter(
          line => {
            val fields = line.split(",")
            fields(1).toInt > 1
          }
        )
        logDebug(logger, msg = s"Total number of predictive sub models: ${predictiveLines.size}")

        (0 until predictiveLines.size by numSubModelsPerRuns).foreach(
          index => {
            val limit =
              if(index + numSubModelsPerRuns > predictiveLines.size) predictiveLines.size
              else index + numSubModelsPerRuns

            predictiveLines.slice(index, limit).foreach(
              line => {
                val ar = line.split(",")
                subModelsBatch.append(ar.head.trim)
              }
            )

            val filename = s"temp/subModels-$count.txt"
            LocalFileUtil.Save.local(filename, subModelsBatch.mkString("\n"))
            logDebug(logger, msg = s"Save $numSubModelsPerRuns sub models in file $filename")
            count += 1
            subModelsBatch.clear
          }
        )
      }
    )
    logDebug(logger, msg = ".. breaking sub models completed!")
  }


  private def generateSubModelTaxonomy: Unit = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    logDebug(logger, msg = "Start generation of sub model taxonomy ...")
    val subModelsTrainingDS = S3Util.s3ToDataset[SubModelsTrainingSet](
      mlopsConfiguration.storageConfig.s3Bucket,
      S3PathNames.s3ModelTrainingPath,
      header = false,
      fileFormat = "json"
    )
    save(subModelsTrainingDS)
    delay(timeInMillis = 4000L)
    logDebug(logger, msg = "... generation of sub model taxonomy completed!")
  }

  private def save(
    subModelsTrainingDS: Dataset[SubModelsTrainingSet]
  )(implicit sparkSession: SparkSession): Unit = {
    val indexedLabels = TrainingLabelIndexing.save(subModelsTrainingDS)
    SubModelsTaxonomy.save(subModelsTrainingDS, indexedLabels)
  }



  def estimateAutoCodingRates(
    numRecordsLabels: Int = mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
  ): (Float, Float, Long, Long, String) =
    if(mlopsConfiguration.classifyConfig.minNumRecordsPerLabel > 0) {
      import org.bertspark.implicits._
      import sparkSession.implicits._

      val feedbackDS = S3Util.s3ToDataset[InternalFeedback](S3PathNames.s3FeedbacksPath)
      val xsFeedbackDS = feedbackDS.map(List[InternalFeedback](_)).persist().cache()

      val groupedByLabelRDD = SparkUtil.groupBy[List[InternalFeedback], String](
        (internalFeedback: List[InternalFeedback]) => internalFeedback.head.toFinalizedSpace,
        (xsFeedback1: List[InternalFeedback], xsFeedback2: List[InternalFeedback]) => xsFeedback1 ::: xsFeedback2,
        xsFeedbackDS
      )

      val totalNumRecords = feedbackDS.count()
      val totalNumLabels = groupedByLabelRDD.count()

      val filteredRDD = groupedByLabelRDD.filter(_.size > numRecordsLabels)
      val numFilteredLabels = filteredRDD.count()
      val numFilteredRecords = filteredRDD.map(_.size).reduce(_ + _)
      val str = (0 until 24 by 4).map(
        cnt => {
          val filteredRDD = groupedByLabelRDD.filter(_.size > numRecordsLabels)
          s"Num labels > $cnt notes: ${filteredRDD.count()}"
        }
      )

      (
          numFilteredLabels.toFloat/totalNumLabels,
          numFilteredRecords.toFloat/totalNumRecords,
          totalNumRecords,
          totalNumLabels,
          str.mkString("\n")
      )
    }
    else
      (-1.0F, -1.0F, -1L, -1L, "")


  def estimateAutoCodingRate: String = {
    val (labelCodingRates, recordCodingRates, numRecords, numLabels, test) = estimateAutoCodingRates()

    s"""
       |Transformer model: ${mlopsConfiguration.runId}
       |Target:            ${mlopsConfiguration.target}
       |Number of records  $numRecords
       |Number of labels   $numLabels
       |Vocabulary         ${mlopsConfiguration.preProcessConfig.vocabularyType}
       |Tokenizer          ${mlopsConfiguration.preTrainConfig.tokenizer}
       |BERT encoder       ${mlopsConfiguration.preTrainConfig.transformer}
       |Segments model     ${mlopsConfiguration.preTrainConfig.sentenceBuilder}
       |Num segments       ${mlopsConfiguration.preTrainConfig.numSentencesPerDoc}
       |Classifier model   ${mlopsConfiguration.classifyConfig.modelId}
       |Classifier layout  ${mlopsConfiguration.classifyConfig.dlLayout.mkString("x")}
       |Classifier min     ${mlopsConfiguration.classifyConfig.minNumRecordsPerLabel}
       |Label coding rate  $labelCodingRates
       |Prediction rate:   $recordCodingRates
       |Test:              $test
       |""".stripMargin
  }
}
