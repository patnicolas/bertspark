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

import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.bertspark.config.MlopsConfiguration.{mlopsConfiguration, ConstantParameters}
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.S3PathNames
import org.bertspark.delay
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.nlp.medical.NoteProcessors.{cleanse, eolCleanserRegex1, eolCleanserRegex2, eolCleanserRegex3, specialCharCleanserRegex}
import org.bertspark.nlp.medical.NoteProcessors
import org.bertspark.nlp.token.TfIdf.{rawFeaturesCol, wordsCol, WeightedToken}
import org.bertspark.nlp.token.TokensTfIdf.{getTfIdf, processTokens, LabelTokens}
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.vocabulary.MedicalTerms.logger
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.bertspark.util.SparkUtil
import org.bertspark.util.io.S3Util.s3ToDataset
import org.slf4j.{Logger, LoggerFactory}


sealed trait MedicalTermsSource
case object AMATermsSource extends MedicalTermsSource
case object CorpusTermsSource extends MedicalTermsSource

/**
  *
  */
private[vocabulary] final class MedicalTerms(termsSource: MedicalTermsSource) extends VocabularyComponent {
  import MedicalTerms._

  override val vocabularyName: String = "MedicalTerms"

  override def build(initialTokens: Array[String], requestDS: Dataset[InternalRequest]): Array[String] = {
    val outputTokens = termsSource match {
      case AMATermsSource => buildFromAMA(initialTokens)
      case CorpusTermsSource => buildFromCorpus(initialTokens)
    }

    logDebug(logger, s"- Medical terms adds ${outputTokens.length} tokens")
    outputTokens
  }
}


private[bertspark] final object MedicalTerms {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[MedicalTerms])

  def apply(): MedicalTerms = new MedicalTerms(AMATermsSource)

  def buildFromTf(inputCorpus: String, numTfidfNotes: Int, tfThreshold: Float): Array[String] = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val fsTokenTfFile = "output/globalRelTf.csv"
    val medicalTerms = LocalFileUtil
        .Load
        .local(ConstantParameters.termsSetFile, (s: String) => s)
        .map(_.map(_.toLowerCase))
        .getOrElse({
          logger.error(s"Vocabulary: Medical terms ${ConstantParameters.termsSetFile} is undefined")
          Array.empty[String]
        })

    val rawNoteDS = S3Util.s3ToDataset[InternalRequest](
      mlopsConfiguration.storageConfig.s3Bucket,
      s"${mlopsConfiguration.storageConfig.s3RequestFolder}/$inputCorpus",
      false,
      "json"
    ).dropDuplicates("id")

    val noteDS =
      if(numTfidfNotes > 0) rawNoteDS.limit(numTfidfNotes).map(List[InternalRequest](_)).persist()
      else rawNoteDS.map(List[InternalRequest](_)).persist()

    logDebug(logger, s"Collected ${noteDS.count()} notes")
    val groupedNotesRDD = SparkUtil.groupBy[List[InternalRequest], String](
      (req: List[InternalRequest]) => req.head.context.emrLabel,
      (r1: List[InternalRequest], r2: List[InternalRequest]) => r1 ::: r2,
      noteDS
    ).repartition(8)
    val totalNumNotes = noteDS.count()

    logDebug(logger, s"Collected ${groupedNotesRDD.count()} group of notes")
    var count = 0
    val tokenPairsRDD: RDD[List[Array[String]]] = groupedNotesRDD.map(
      _.map(_.notes.head)
          .map(
            note => {
              logDebug(logger, s"Filtered $count notes from $totalNumNotes ${count.toFloat/totalNumNotes}%")
              count += 1
              note.split(tokenSeparator).map(_.toLowerCase).filter(token => token.size > 1 && medicalTerms.contains(token))
            }
          )
    ).persist().cache()

    val resultRDD: RDD[Seq[(String, Float)]] = tokenPairsRDD.map{ ar => {
      val cnt = ar.size
      ar.flatten
          .map((_, 1))
          .groupBy(_._1)
          .map{ case (token, seq) => (token, seq.size.toFloat/cnt)}
          .toSeq
          .sortWith(_._2 > _._2)
    }}
    val result: Seq[(Seq[(String, Float)])] = resultRDD.collect()
    val allResultRDD = tokenPairsRDD.flatMap(
      ar => {
        ar.flatten.map((_, 1)).groupBy(_._1).map { case (token, s) => (token, s.size.toFloat/ar.size)}
      }
    )
    val allResults: Map[String, Float] = allResultRDD.collect().toMap
    val labelTokensFreq: Seq[Seq[(String, Float)]] = result.map(
      _.map{
        case (token, f) =>
          val relFreq = if(allResults.contains(token)) f/allResults.get(token).get else 0.0F
          (token, relFreq)
      }.sortWith(_._2 > _._2)
    )

    val labelTokensFreqStr  = labelTokensFreq
        .map{_.map { case (token, f) => s"$token:$f"}.mkString("\n")}
        .mkString("\n\n\n")

    LocalFileUtil.Save.local("output/relFfTokens.csv",  labelTokensFreqStr)

    val globalRanking = labelTokensFreq
        .flatten
        .groupBy(_._1)
        .map{ case (token, seq) => (token, seq.map(_._2).max)}
        .toSeq
        .sortWith(_._2 > _._2)
    val content2 = globalRanking.map{ case (token, f) => s"$token,$f"}.mkString("\n")
    LocalFileUtil.Save.local(fsTokenTfFile,  content2)
    delay(1000L)
    globalRanking.filter(_._2 > tfThreshold).map(_._1).toArray
  }

  def augmentTf(localFile: String, tfThreshold: Float, numWordPiecesNotes: Int): Array[String] = {
    LocalFileUtil.Load.local(localFile).map(
      content => {
        val tokens = content.split("\n").map {
          line => {
            if (line.nonEmpty) {
              val ar = line.split(",")
              if (ar.size == 2) (ar.head, ar(1).toFloat) else ("", -1.0F)
            }
            else
              ("", -1.0F)
          }
        }.filter(_._2 > tfThreshold).map(_._1)
        logDebug(logger, s"Num of tokens in vocabulary: ${tokens.size}")

        MedicalVocabularyBuilder.augmentVocabulary(tokens, "TFIDF", numWordPiecesNotes)
      }
    ).getOrElse(Array.empty[String])
  }


  def buildFromCorpus(initialTokens: Array[String] = Array.empty[String]): Array[String] = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val noteDS = S3Util.s3ToDataset[InternalRequest](
      mlopsConfiguration.storageConfig.s3Bucket,
      s"${mlopsConfiguration.storageConfig.s3RequestFolder}/ALL",
      false,
      "json"
    ).dropDuplicates("id").map(_.notes.head.toLowerCase)

    logDebug(logger, s"${noteDS.count} used for the Corpus vocabulary")
    val allWordDS = noteDS.flatMap(
      note => {
        val cleansedNote = NoteProcessors.cleanse(note, specialCharCleanserRegex)
        cleansedNote
      }).distinct()
    allWordDS.collect()
  }

  def buildFromAMA(initialTokens: Array[String] = Array.empty[String]): Array[String] = LocalFileUtil
      .Load
      .local(ConstantParameters.termsSetFile, (s: String) => s)
      .map(_.map(_.toLowerCase))
      .getOrElse({
        logger.error(s"Vocabulary: Medical terms ${ConstantParameters.termsSetFile} is undefined")
        Array.empty[String]
      }) ++ initialTokens


  def buildFromStrictAMA(initialTokens: Array[String] = Array.empty[String]): Array[String] = LocalFileUtil
      .Load
      .local(ConstantParameters.medicalTermsSetFile, (s: String) => s)
      .map(_.map(_.toLowerCase))
      .getOrElse({
        logger.error(s"Vocabulary: Medical terms ${ConstantParameters.medicalTermsSetFile} is undefined")
        Array.empty[String]
      }) ++ initialTokens
}
