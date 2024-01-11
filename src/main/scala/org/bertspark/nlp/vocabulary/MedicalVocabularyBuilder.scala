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
package org.bertspark.nlp.vocabulary

import org.apache.spark.sql.SparkSession
import org.bertspark.config.MlopsConfiguration.DebugLog._
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.augmentation.RandomSynonymSubstitute
import org.bertspark.nlp.medical.MedicalCodingTypes._
import org.bertspark.nlp.medical.{encodePredictReq, NoteProcessors}
import org.bertspark.nlp.medical.NoteProcessors.specialCharCleanserRegex
import org.bertspark.nlp.token.TokenizerPreProcessor.AbbreviationMap.abbreviationMap
import org.bertspark.nlp.tokenSeparator
import org.bertspark.util.io.S3Util.s3ToDataset
import org.bertspark.util.io.{S3Util, SingleS3Dataset}
import org.slf4j._

/**
 * Workflow for pre-processing medical notes
 * @param s3Dataset Storage S3
 * @param vocabularyComponents Sequence of vocabulary components
 * @author Patrick Nicolas
 * @version 0.5
 */
private[bertspark] final class MedicalVocabularyBuilder private (
  s3Dataset: SingleS3Dataset[InternalRequest],
  vocabularyComponents: Seq[VocabularyComponent]
)(implicit sparkSession: SparkSession) {
  import MedicalVocabularyBuilder._

  def build: Int = {
    import sparkSession.implicits._

    val requestDS = try {
      s3ToDataset[InternalRequest](
        mlopsConfiguration.storageConfig.s3Bucket,
        S3PathNames.s3RequestsPath,
        header = false,
        fileFormat = "json").dropDuplicates("id")
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"Medical vocabulary build: ${e.getMessage}")
        sparkSession.emptyDataset[InternalRequest]
    }


    @annotation.tailrec
     def add(index: Int, tokens: Array[String]): Array[String] =
       if(index >= vocabularyComponents.size) tokens
       else {
         logDebug(logger, msg = s"Vocabulary: size ${tokens.size} for $index before ${vocabularyComponents(index).vocabularyName}")
         val aggregatedTokens = vocabularyComponents(index).build(tokens, requestDS)
         logDebug(logger, msg = s"Vocabulary: size ${aggregatedTokens.size} for ${index+1}")
         add(index+1, aggregatedTokens)
       }

    val terms = add(0, Array.empty[String])
    val distinctTerms = terms.distinct.sortWith(_ < _)

    s3VocabularyStorage.upload(distinctTerms)
    distinctTerms.size
  }
}


/**
 * Singleton for constructors and loading CPT and ICD code descriptors
 */
private[bertspark] final object MedicalVocabularyBuilder {
  final private val logger: Logger = LoggerFactory.getLogger("MedicalVocabularyBuilder")
  final private val maxNumCharsForWordPieces = 12
  final private val wordPiecesSampleSize = 500000

  final def vocabularyComponents(vocabularyType: String): Seq[VocabularyComponent] = vocabularyType match {
    case "AMA" => Seq[VocabularyComponent](
      MedicalTerms(),
      WordPieceExtractor(wordPiecesSampleSize, maxNumCharsForWordPieces))
    case "Base" => Seq[VocabularyComponent](
      MedicalAbbreviations(),
      MedicalTerms(),
      WordPieceExtractor(wordPiecesSampleSize, maxNumCharsForWordPieces))
    case "ExtEmr" => Seq[VocabularyComponent](
      MedicalAbbreviations(),
      MedicalTerms(),
      ContextVocabulary(),
      CodingTermsTfIdf(0.5, "emr"),
      WordPieceExtractor(wordPiecesSampleSize, maxNumCharsForWordPieces))
    case "ExtNote" => Seq[VocabularyComponent](
      MedicalAbbreviations(),
      MedicalTerms(),
      ContextVocabulary(),
      CodingTermsTfIdf(0.5, "note"),
      WordPieceExtractor(wordPiecesSampleSize, maxNumCharsForWordPieces))
    case "TfIdf50Emr" => Seq[VocabularyComponent](
      MedicalAbbreviations(),
      ContextVocabulary(),
      CodingTermsTfIdf(0.5, "emr"),
      WordPieceExtractor(wordPiecesSampleSize, maxNumCharsForWordPieces))
    case "TfIdf80Emr" => Seq[VocabularyComponent](
      MedicalAbbreviations(),
      ContextVocabulary(),
      CodingTermsTfIdf(0.8, "emr"),
      WordPieceExtractor(wordPiecesSampleSize, maxNumCharsForWordPieces))
    case "TfIdf80Note" => Seq[VocabularyComponent](
      MedicalAbbreviations(),
      ContextVocabulary(),
      CodingTermsTfIdf(0.8, "note"),
      WordPieceExtractor(wordPiecesSampleSize, maxNumCharsForWordPieces))
    case "TfIdfNote" => Seq[VocabularyComponent](
      MedicalAbbreviations(),
      ContextVocabulary(),
      CodingTermsTfIdf(0.95, "note"),
      WordPieceExtractor(wordPiecesSampleSize, maxNumCharsForWordPieces))
    case "FullEmr" => Seq[VocabularyComponent](
      MedicalTerms(),
      ContextVocabulary(),
      MedicalCodeDescriptors(),
      CodingTermsTfIdf(0.95, "emr"),
      WordPieceExtractor(wordPiecesSampleSize, maxNumCharsForWordPieces))
    case "FullNote" => Seq[VocabularyComponent](
      MedicalAbbreviations(),
      MedicalTerms(),
      ContextVocabulary(),
      MedicalCodeDescriptors(),
      CodingTermsTfIdf(0.95, "note"),
      WordPieceExtractor(wordPiecesSampleSize, maxNumCharsForWordPieces))
    case _ => throw new UnsupportedOperationException(s"vocabularyType $vocabularyType not supported")
  }

  /**
   * Default constructor
   * @param s3StorageInfo Storage data
   * @param vocabularyComponents     Number of sequential random splits for processing
   * @param sparkSession  Implicit reference to the current Spark context
   * @return instance of the Medical coding vocabulary
   */
  def apply(
    s3StorageInfo: SingleS3Dataset[InternalRequest],
    vocabularyComponents: Seq[VocabularyComponent]
    )(implicit sparkSession: SparkSession): MedicalVocabularyBuilder =
    new MedicalVocabularyBuilder(s3StorageInfo, vocabularyComponents)

  def apply(
    s3StorageInfo: SingleS3Dataset[InternalRequest],
    vocabularyType: String
  )(implicit sparkSession: SparkSession): MedicalVocabularyBuilder =
    new MedicalVocabularyBuilder(s3StorageInfo, vocabularyComponents(vocabularyType))

  def apply(
    vocabularyType: String
  )(implicit sparkSession: SparkSession): MedicalVocabularyBuilder = {
    import sparkSession.implicits._
    val s3StorageInfo = SingleS3Dataset[InternalRequest](S3PathNames.s3RequestsPath, encodePredictReq)
    new MedicalVocabularyBuilder(s3StorageInfo, vocabularyComponents(vocabularyType))
  }


  def apply()(implicit sparkSession: SparkSession): MedicalVocabularyBuilder = {
    import sparkSession.implicits._, org.bertspark.config.MlopsConfiguration._

    val vocabularyType = mlopsConfiguration.preProcessConfig.vocabularyType
    val s3StorageInfo = SingleS3Dataset[InternalRequest](S3PathNames.s3RequestsPath, encodePredictReq)
    new MedicalVocabularyBuilder(s3StorageInfo, vocabularyComponents(vocabularyType))
  }



  def extractPredefinedTerms: Array[String] = {
    val medicalAbbreviations = MedicalAbbreviations()
    val abbreviationsMap = medicalAbbreviations.getAbbreviationsMap
    logInfo(
      logger,
      msg = s"Abbreviation map: ${abbreviationsMap.take(3).map{ case (k,v) => s"$k: $v"}.mkString(", ")}"
    )
    val abbreviationDescriptors = medicalAbbreviations.getAbbreviationDescriptors.map(_.toLowerCase)
    logInfo(logger,  msg = s"Abbreviation descriptors:  ${abbreviationDescriptors.take(8).mkString(", ")}")
    // Retrieve the CPT
    (abbreviationDescriptors ++ MedicalCodeDescriptors.getCptIcdTerms).distinct
  }

  /**
    * Command line argument
    * buildVocabulary AMA tfIdfSource numTfIdfNotes numNotesForWordPiece tfIdfThreshold
    * @param args
    */
  def vocabulary(args: Seq[String]): Unit = {
    require(
      args.size >4,
      s"${args.mkString(" ")} should be 'buildVocabulary TF93 XLARGE numNotesUsedForTF numNotesUsedForWordPiece tfThreshold'"
    )

    val vocabularyType = args(1)  // TfIdf, AMA, Corpus,...
    val tfSource = args(2)     // CMBS, ALL, LARGE
    val numNotesUsedForTF = args(3).toInt
    val numNotesUsedForWordPiece = args(4).toInt
    val tfThreshold = if(args.size > 5) args(5).toFloat else 0.0F
    vocabulary(vocabularyType, tfSource, numNotesUsedForTF, numNotesUsedForWordPiece, tfThreshold)
  }


  def vocabulary(
    vocabularyType: String,
    corpus: String,
    maxNotesUsedForTF: Int,
    numNotesUsedForWordPiece: Int,
    tfThreshold: Float = 0.0F): Unit = vocabularyType match {
      case "Corpus" =>
        val terms = MedicalTerms.buildFromCorpus()
        augmentVocabulary(terms, vocabularyType, numNotesUsedForWordPiece)
      case "AMA" =>
        val terms = MedicalTerms.buildFromAMA()
        augmentVocabulary(terms, vocabularyType, numNotesUsedForWordPiece)
      case "StrictAMA" =>
        val terms = MedicalTerms.buildFromStrictAMA()
        augmentVocabulary(terms, vocabularyType, numNotesUsedForWordPiece)
      case "TfIdf" =>
        val terms = MedicalTerms.buildFromTf(corpus, maxNotesUsedForTF, tfThreshold)
        augmentVocabulary(terms, vocabularyType, numNotesUsedForWordPiece)
      case _ =>
        throw new UnsupportedOperationException(s"$vocabularyType not supported for building vocabulary")
    }


  /**
    * Build vocabulary for 'Corpus' with headers, punctuation, terms extracted from training
    * set, abbreviation and code descriptor ...
    */
  def augmentVocabulary(terms: Array[String], vocabularyType: String, numNotesUsedForWordPiece: Int): Array[String] = {
    val termsWithExtensions = terms ++ Array[String](
      "findings",
      "findings",
      "impression",
      "impression:",
      "ccomma",
      "creturn",
      "cdot",
      "ccol",
      "xnum",
      "ynum",
      "zpercent",
      "s",
      "ed",
      "ing",
      "es",
      "al",
      "acy",
      "ance",
      "ence",
      "fy",
      "ify",
      "esque",
      "dom",
      "ism",
      "ist",
      "er",
      "or",
      "ity",
      "ty",
      "ment",
      "ness",
      "ship",
      "sion",
      "tion",
      "ate",
      "en",
      "ize",
      "ise",
      "able",
      "ible",
      "ful",
      "ic",
      "ical",
      "ish",
      "less",
      "ious",
      "ous",
      "ive",
      "y"
    )

    logDebug(logger, msg = s"Vocabulary size after punctuation: ${termsWithExtensions.size}")
    val abbreviations = abbreviationMap.flatMap(_._2.split(tokenSeparator)).toSeq.distinct.mkString("\n")
    val cleansedAbbreviations = NoteProcessors.cleanse(abbreviations, specialCharCleanserRegex)
    val termsWithAbbreviations: Array[String] = (termsWithExtensions ++ cleansedAbbreviations).distinct
    logDebug(logger, msg = s"Vocabulary size  after abbreviations: ${termsWithAbbreviations.size}")

    val conversionSet = RandomSynonymSubstitute.conversionMap.keySet.toArray
    val termsWithAbbrAndConversion = (termsWithAbbreviations ++ conversionSet).distinct
    logDebug(logger, msg = s"Vocabulary size  after conversion: ${termsWithAbbrAndConversion.size}")

    val cptDescription = MedicalCodeDescriptors.getCptDescriptors.flatMap(_._2).distinct.mkString("\n")
    val cleansedCptDescription = NoteProcessors.cleanse(cptDescription, specialCharCleanserRegex)

    val icdDescription = MedicalCodeDescriptors.getIcdDescriptors.flatMap(_._2).distinct.mkString("\n")
    val cleansedIcdDescription = NoteProcessors.cleanse(icdDescription, specialCharCleanserRegex)

    val termsWithCodeDescription = (termsWithAbbrAndConversion ++ cleansedCptDescription ++ cleansedIcdDescription).distinct
    logDebug(logger, msg = s"Vocabulary size after code description: ${termsWithCodeDescription.size}")

    val wordPieceTokens = finalizeTfIdf(numNotesUsedForWordPiece)
    val termsWithWordPieces = (termsWithCodeDescription ++ wordPieceTokens).distinct

    S3Util.upload(
      s3Folder = s"${mlopsConfiguration.storageConfig.s3RootFolder}/${mlopsConfiguration.target}/vocabulary/$vocabularyType",
      termsWithWordPieces.sortWith(_ < _).mkString("\n")
    )
    termsWithWordPieces
  }

  private def finalizeTfIdf(numNotes: Int): Array[String] = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val maxNumChars = 16
    val requestDS = s3ToDataset[InternalRequest](
      mlopsConfiguration.storageConfig.s3Bucket,
      S3PathNames.s3RequestsPath,
      header = false,
      fileFormat = "json").limit(numNotes).dropDuplicates("id")

    val wordPieceExtractor = WordPieceExtractor(numNotes, maxNumChars)
    val wordPieces = wordPieceExtractor.build(Array.empty[String], requestDS)
    logDebug(logger, msg = s"${wordPieces.size} Word pieces with max $maxNumChars characters")
    wordPieces
  }
}