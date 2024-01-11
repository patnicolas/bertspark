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
package org.bertspark.analytics

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.bertspark.util.io.S3Util
import org.bertspark.util.SparkUtil
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.HashMap


/**
  * {{{
  * FOR each vocabulary:
  *   Load the training set associated with a given vocabulary
  *   Compute the relative frequency for the tokens which overlap
  *     - across all the notes within each label
  *     - across all the notes
  *
  *   Compute the Pareto distribution for the top 32 most frequent tokens for notes associated with each label
  *   Compute the Average Pareto distribution across all the labels:  ParetoAveLabel
  *   Compute the Pareto distribution with top 32 most frequent tokens across all the notes: ParetoAllNotes
  *   Compute the difference Diff= ParetoAveLabel - ParetoAllNotes
  *
  * Compare Diff for all the vocabulary
  * }}}
  *
  * @param numEMREntries Number of requests used for the analysis
  * @author Patrick Nicolas
  * @version 0.5
  */
private[bertspark] final class VocabularyAnalyzer(numEMREntries: Int) {
  import VocabularyAnalyzer._

  def analyze(isCompareVocabulary: Boolean): String =
    if(isCompareVocabulary) compareVocabularies else compareTokensOutput


  def compareVocabularies: String = {
    import org.bertspark.implicits._

    val analysisResults = vocabularyTypes.map(computeOverlapRate(_))
    val vocabulariesAnalysis = vocabularyTypes.indices.map( index => s"${vocabularyTypes(index)}:--------------------------------------\n${analysisResults(index)}").mkString("\n\n")
    S3Util.upload(
      mlopsConfiguration.storageConfig.s3Bucket,
      s"${mlopsConfiguration.storageConfig.s3RootFolder}/${mlopsConfiguration.target}/vocabularyAnalysis.txt",
      vocabulariesAnalysis
    )
    vocabulariesAnalysis
  }


  def compareTokensOutput: String = {
    import org.bertspark.implicits._
    import org.bertspark.nlp.trainingset.ContextualDocument
    import sparkSession.implicits._

    val collector = HashMap[String, List[(String, Array[String])]]()
    val contextualDocumentDS = try {
      S3Util.s3ToDataset[ContextualDocument](
        mlopsConfiguration.storageConfig.s3Bucket,
        S3PathNames.getS3ContextualDocumentPath(vocabularyTypes.head),
        false,
        "json"
      ).limit(numEMREntries).dropDuplicates("id")
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"compareTokensOutput: ${e.getMessage}")
        sparkSession.emptyDataset[ContextualDocument]
    }

    val ids = contextualDocumentDS.map(_.id).collect()
    contextualDocumentDS.map(
      ctxDoc => (ctxDoc.id, ctxDoc.contextVariables ++ ctxDoc.text.split(tokenSeparator))
    ).collect.foreach {
      case (id, tokens) => {
        val xs = List[(String, Array[String])]()
        collector.put(id, (vocabularyTypes.head, tokens) :: xs)
      }
    }

    vocabularyTypes.tail.foreach(
      vocabularyType => {
        try {
          S3Util.s3ToDataset[ContextualDocument](
            mlopsConfiguration.storageConfig.s3Bucket,
            S3PathNames.getS3ContextualDocumentPath(vocabularyType),
            false,
            "json"
          ).filter(ctxDoc => ids.contains(ctxDoc.id))
              .map(
                ctxDoc => (ctxDoc.id, ctxDoc.contextVariables ++ ctxDoc.text.split(tokenSeparator))
              ).collect.foreach {
            case (id, tokens) => {
              val xs = collector.getOrElse(id, List[(String, Array[String])]())
              collector.put(id, (vocabularyType, tokens) :: xs)
            }
          }
        }
        catch {
          case e: IllegalArgumentException =>
            logger.error(s"compareTokensOutput: ${e.getMessage}")
            sparkSession.emptyDataset[ContextualDocument]
        }
      }
    )

    val idTokenVocabularyTypes: HashMap[String, Iterable[String]] = collector.map{
      case (id, xs) => {
        val tokenVocabularyTypes = xs.foldLeft(HashMap[String, List[String]]())(
          (hMap, labelTokens) => {
            labelTokens._2.foreach(
              token => {
                val vocabularyTypes = hMap.getOrElse(token, List[String]())
                hMap.put(token, labelTokens._1 :: vocabularyTypes)
              }
            )
            hMap
          }
        ).map {
          case (token, vocabularyTypes) => {
            val vocabularyTypeForToken = vocabularyTypes.distinct.sortWith(_ < _).mkString(" ")
            s"$token: $vocabularyTypeForToken"
          }
        }
        (id, tokenVocabularyTypes)
      }
    }

    S3Util.upload(
      mlopsConfiguration.storageConfig.s3Bucket,
      s"${mlopsConfiguration.storageConfig.s3RootFolder}/${mlopsConfiguration.target}/tokensPerVocabulary.csv",
      idTokenVocabularyTypes.map{ case (id, xs) => s"$id,${xs.mkString("\n")}" }.mkString("\n\n")
    )
    val id = collector.head._1
    val vocabDistribution = collector
        .head
        ._2
        .map{ case (vocab, tokens) => s"$vocab: ${tokens.mkString(" ")}"}
        .mkString("\n")
    s"$id\n$vocabDistribution"
  }

  def computeOverlapRate(vocabularyType: String)(implicit sparkSession: SparkSession): VocabularyAnalyzerResult = {
    import org.bertspark.config.MlopsConfiguration.DebugLog
    import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
    import org.bertspark.nlp.trainingset.TokenizedTrainingSet
    import sparkSession.implicits._, S3PathNames._

    val s3Folder = getS3ModelTrainingPath(vocabularyType)
    logDebug(logger, s"Start analyzing $vocabularyType from $s3Folder")
    // Step 1: Load the appropriate training set
    val tokenizedIndexedTrainingSetDS: Dataset[TokenizedTrainingSet] = try {
      S3Util.s3ToDataset[SubModelsTrainingSet](
        mlopsConfiguration.storageConfig.s3Bucket,
        s3Folder,
        header = false,
        fileFormat = "json"
      ).limit(numEMREntries)
          .filter(_.labeledTrainingData.nonEmpty)
          .flatMap(_.labeledTrainingData)
          .cache()
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"computeOverlapRate: ${e.getMessage}")
        sparkSession.emptyDataset[TokenizedTrainingSet]
    }

    val numNotes = tokenizedIndexedTrainingSetDS.count()
    val averageNumNotesPerEMR = numNotes.toDouble/numEMREntries
    DebugLog.logDebug(logger, {s"$vocabularyType: $numNotes notes with  Average number of notes per EMR: $averageNumNotesPerEMR"})

    val labelTokensDS: Dataset[(String, Array[String])] = tokenizedIndexedTrainingSetDS.map(
      labeledData => {
        val ctxDocument = labeledData.contextualDocument
        val tokens = ctxDocument.contextVariables ++ ctxDocument.text.split(tokenSeparator)
        if(tokens.isEmpty)
          logger.warn(s"$vocabularyType:  $labeledData.label has not associated token")
        (labeledData.label, tokens)
      }
    ).filter(_._2.nonEmpty)
    DebugLog.logDebug(logger, s"$vocabularyType:  Number of labels tokens:  ${labelTokensDS.count()}")

    val tokensCountPerLabels: Array[Int] = labelTokensDS.map{ case (_, tokens) => tokens.size }.collect
    val averageNumTokensPerNote =
      if(tokensCountPerLabels.nonEmpty) tokensCountPerLabels.reduce(_ + _).toDouble/tokensCountPerLabels.size
      else 0.0

    DebugLog.logDebug(logger, s"$vocabularyType:  averageNumTokensPerNote:  $averageNumTokensPerNote")
   val groupedRDD: RDD[LabelTokens] = SparkUtil.groupBy[LabelTokens, String](
      (labelToken: LabelTokens) => labelToken._1,
      (labelToken1: LabelTokens, labelToken2: LabelTokens) => (labelToken1._1, labelToken1._2 ++ labelToken2._2),
      labelTokensDS
    )

    val numLabelsPerEMR = groupedRDD.count().toDouble/numEMREntries
    DebugLog.logDebug(logger, s"$vocabularyType:  ${groupedRDD.count()} RDD grouped by Labels")

    val filteredGroupedRDD = groupedRDD.filter(_._2.nonEmpty)
    if(filteredGroupedRDD.count() > 1) {
      val labelTokenGroups: Array[Seq[Float]] = filteredGroupedRDD.map {
        case (_, tokens) => {
          if(tokens.nonEmpty) {
            val tokenFrequencies = tokens.foldLeft(HashMap[String, Int]())(
              (hMap, token) => {
                val frequency = hMap.getOrElse(token, 0)
                hMap += ((token, frequency + 1))
              }
            )
            val allTokenFrequencies = tokenFrequencies.map(_._2)
            val scalingFactor = 1.0F/allTokenFrequencies.sum
            allTokenFrequencies.map(_ *scalingFactor).toSeq.sortWith(_ > _).take(cutOff)
          }
          else
            Seq.empty[Float]
        }

      }.filter(_.nonEmpty).collect

      val allLabelTokens = filteredGroupedRDD.flatMap(_._2).collect.foldLeft(HashMap[String, Int]())(
        (hMap, token) => {
          val frequency = hMap.getOrElse(token, 0)
          hMap += ((token, frequency + 1))
        }
      ).map(_._2)

      val scalingFactor = 1.0F/allLabelTokens.sum
      val allLabelTokensRelFrequencies = allLabelTokens.map(_ * scalingFactor).toSeq.sortWith(_ > _).take(cutOff)

      val transposed = labelTokenGroups.toSeq.transpose
      val averageInLabelRelFrequencies = transposed.map(ar => ar.sum/ar.size)
      val diffRelFrequencies = averageInLabelRelFrequencies.indices.map(index => averageInLabelRelFrequencies(index) - allLabelTokensRelFrequencies(index))

      val vocabularyAnalyzerResult = VocabularyAnalyzerResult(
        vocabularyType,
        labelTokenGroups,
        allLabelTokensRelFrequencies,
        averageInLabelRelFrequencies,
        diffRelFrequencies,
        numLabelsPerEMR,
        averageNumTokensPerNote,
        averageNumNotesPerEMR
      )
      logger.info(vocabularyAnalyzerResult.toString)
      vocabularyAnalyzerResult
    }
    else {
      logger.error(s"$vocabularyType: Failed to group notes per label")
      VocabularyAnalyzerResult(
        vocabularyType,
        Array.empty[Seq[Float]],
        Seq.empty[Float],
        Seq.empty[Float],
        Seq.empty[Float],
        -1.0,
        -1.0,
        -1.0)
    }
  }
}

private[bertspark] final object VocabularyAnalyzer {
  final private val logger: Logger = LoggerFactory.getLogger("VocabularyAnalyzer")

  val vocabularyTypes = Array[String](
    "AMA",
    "Base",
    "ExtNote",
    "TfIdf50Emr",
    "TfIdf80Emr",
    "TfIdf80Note",
    "FullEmr",
    "FullNote"
  )

  final private val cutOff = 32
  type LabelTokens = (String, Array[String])

  case class VocabularyAnalyzerResult(
    vocabularyType: String,
    groups: Array[Seq[Float]],
    all: Seq[Float],
    averageInLabelRelFrequencies: Seq[Float],
    diffRefFrequencies: Seq[Float],
    averageNumLabelsPerEMR: Double,
    averageTokensPerNote: Double,
    averageNumNotesPerEMR: Double) {

    override def toString: String =
      s"""Vocabulary type: $vocabularyType
        |Groups:
         |Overall:
         |${all.mkString(" ")}
         |averageInLabelRelFrequencies
         |${averageInLabelRelFrequencies.mkString(" ")}
         |diffRefFrequencies
         |${diffRefFrequencies.mkString(" ")}
         |averageNumLabelsPerEMR: $averageNumLabelsPerEMR
         |Average tokens per note: $averageTokensPerNote
         |Average num note per EMR: $averageNumNotesPerEMR""".stripMargin
  }
}
