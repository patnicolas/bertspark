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
package org.bertspark.nlp.token

import ai.djl.modality.nlp._
import java.io.File
import java.nio.file.Paths
import org.bertspark.delay
import org.bertspark.config.MlopsConfiguration.{mlopsConfiguration, vocabulary}
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.ContextualDocument
import org.bertspark.transformer.dataset.{reservedLabels, unkLabel, TDatasetConfig}
import org.bertspark.util.io.{LocalFileUtil, S3Util}


/**
 * Comprehensive wrapper for Document
 * {{{
 *   The purpose for this class is to extract the document components from an existing input document.
 *   The input contextual document is defined as
 *   - Document id
 *   - Content or text
 *   - Contextual information as a set of attribute values
 *
 *   The output is a set of document components as
 *   - document id
 *   - Sentences or segments
 *   - Tokens associated with each sentence
 * }}}
 * @param tokenizer Raw tokenizer
 * @param sentencesBuilder Builder that extracts extractedSentences from a document
 * @tparam T Type of tokenizer used this class
 * @param U type for Sentences builder
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class DomainTokenizer[T<: TokenizerPreProcessor, U <: SentencesBuilder] private (
  tokenizer: T,
  sentencesBuilder: U
) {
  import org.bertspark.implicits._

  /**
   * Apply tokenizer and filtering for a given text, using a pre-defined vocabulary
   * @param contextualDocument Contextual document loaded from the training set
   * @return Processed document components
   */
  def apply(contextualDocument: ContextualDocument): DocumentComponents = {
    val extractedSentences = sentencesBuilder(contextualDocument)
    val tokens = extractedSentences.map {
      case (ctxText, docText) => {
        val ctxTokens = ctxText.split(tokenSeparator).toList

        val convertedTokens: scala.List[String] =
          if(docText.nonEmpty) {
            val docTokens: scala.List[String] = listOf[String](tokenizer(docText))
            // Conversion from Scala to Java
            ctxTokens ++ docTokens
          }
          else
            ctxTokens

        val validTokens = convertedTokens.filter(vocabulary.contains(_))
        validTokens.toArray
      }
    }
    new DocumentComponents(contextualDocument.getId, tokens)
  }

  final def getTokenizer: T = tokenizer

  override def toString: String = s"${tokenizer.toString}\n${sentencesBuilder.toString}"
}


/**
 * Singleton for constructors
 */
private[bertspark] final object DomainTokenizer {

  /**
   * Basic constructor with a Tokenizer pre processor constructor and a setence builder
   * @param cls Constructor for the tokenizer
   * @param sentencesBuilder Extractor for segments and sentences
   * @tparam T Type of the tokenizer
   * @tparam U Type of the sentence extractor
   * @return Instance of Tokenizer
   */
  def apply[T<: TokenizerPreProcessor, U <: SentencesBuilder](cls: => T, sentencesBuilder: U): DomainTokenizer[T, U] =
    new DomainTokenizer[T, U](cls,  sentencesBuilder)


  /**
   * Constructor for tokenizer pre-processor given a tokenizerPreProcessor and a sentence builder
   * @param tokenizerPreProcessor Tokenizer pre-processor
   * @param sentencesBuilder Extractor for segments and sentences
   * @tparam T Type of the tokenizer
   * @tparam U Type of the sentence extractor
   * @return Instance of Tokenizer
   */
  def apply[T <: TokenizerPreProcessor, U <: SentencesBuilder](
    tokenizerPreProcessor: T,
    sentencesBuilder: U): DomainTokenizer[T, U] =
    new DomainTokenizer[T, U](tokenizerPreProcessor, sentencesBuilder)


  /**
   * Minimalist constructor
   * @param sentencesBuilder Extractor for segments and sentences
   * @tparam U Type of the sentence extractor
   * @return Instance of Tokenizer
   */
  def apply[U <: SentencesBuilder](sentencesBuilder: U): DomainTokenizer[ExtWordPiecesTokenizer, U] =
    new DomainTokenizer[ExtWordPiecesTokenizer, U](ExtWordPiecesTokenizer(), sentencesBuilder)


  /**
   * Vocabulary builder for a S3 file containing list of words/tokens. These tokens are filtered
   * by minimum frequency
   * @param minFrequency Minimum frequency for the tokens in the vocabulary
   * @return Default vocabulary builder
   */
  final def vocabularyBuilder(minFrequency: Int): DefaultVocabulary.Builder = {
    import org.bertspark.implicits._
    import org.bertspark.config.MlopsConfiguration._

    val s3VocabularyPath = S3PathNames.s3VocabularyPath
    val directory =  s"conf/vocabulary${mlopsConfiguration.preProcessConfig.vocabularyType}"
    val localVocabularyFile = s"$directory/${mlopsConfiguration.target}.csv"

    // If local file is not created, load from S3 and creates the local file

    val dir = new File(directory)
    if(!dir.exists())
      dir.mkdirs()

    if(!new File(localVocabularyFile).exists()) {
      S3Util.s3ToFs(localVocabularyFile, s3VocabularyPath, true)
      delay(2000L)
    }

    val path = Paths.get(localVocabularyFile)

    val _reservedLabels: java.util.Collection[String] = reservedLabels.toSeq
    val defaultVocabulary = DefaultVocabulary.builder
        .optMinFrequency(minFrequency)
        .optReservedTokens(_reservedLabels)
        .addFromTextFile(path)
        .optUnknownToken(unkLabel)
    defaultVocabulary
  }


  /**
   * Temporary vocabulary builder from a set of tokens
   * @param tokens Existing tokens
   * @return Default vocabulary builder
   */
  final def vocabularyBuilder(tokens: Array[String]): DefaultVocabulary.Builder = {
    val directory = new File("temp")
    if(!directory.exists())
      directory.mkdirs()

    val tempVocabularyFile = s"$directory/tempVocabulary"
    LocalFileUtil.Save.local(tempVocabularyFile, tokens.mkString("\n"))
    val path = Paths.get(tempVocabularyFile)
    DefaultVocabulary.builder
        .optMinFrequency(1)
        .addFromTextFile(path)
        .optUnknownToken(unkLabel)
  }


  final def vocabularyBuilder(bertDatasetConfig: TDatasetConfig): DefaultVocabulary.Builder =
    vocabularyBuilder(bertDatasetConfig.getMinTermFrequency)
}