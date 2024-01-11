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

import ai.djl.modality.nlp.bert.WordpieceTokenizer
import org.apache.spark.sql.Dataset
import org.bertspark.config.{MlopsConfiguration, S3PathNames}
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.nlp.medical.NoteProcessors.{cleanse, specialCharCleanserRegex}
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.implicits.logger
import org.bertspark.nlp.token.DomainTokenizer
import org.bertspark.nlp.vocabulary.CodingTermsTfIdf.abbrStopTokens
import org.bertspark.util.io.S3Util
import org.slf4j.{Logger, LoggerFactory}


/**
  * Extract token pieces from a list of tokens
  * @param maxNumChars Maximum number of character the tokens to be broken
  * @author Patrick Nicolas
  * @version 0.5
  */
private[vocabulary] final class WordPieceExtractor private (
  numInputDocuments: Int,
  maxNumChars: Int) extends VocabularyComponent {
  import WordPieceExtractor._

  override val vocabularyName: String = "WordPieceExtractor"

  /**
   * Replace abbreviations by their description. The detection of abbreviation is case sensitive and
   * needs to be processed before conversion to lower case.
   *
   * @param textTokens tokens extracted from the content or note
   * @return Text tokens with abbreviations
   */
  override def build(textTokens: Array[String], requestDS: Dataset[InternalRequest]): Array[String] = {
    val outputTokens = (textTokens ++ buildWordPiecesVocabulary(requestDS)).distinct
    logDebug(logger, msg = s"Vocabulary: Word piece tokenizer adds ${outputTokens.length - textTokens.length} tokens")
    outputTokens
  }


  // ------------------   Supporting methods ------------------------

  private def buildWordPiecesVocabulary(requestDS: Dataset[InternalRequest]): Array[String] = {
    import org.bertspark.implicits._, sparkSession.implicits._

    val noteDS =
      if(numInputDocuments > 0)
        requestDS.limit(numInputDocuments).dropDuplicates("id").map(_.notes.head.toLowerCase)
      else
        requestDS.dropDuplicates("id").map(_.notes.head.toLowerCase)

    logDebug(logger, msg = s"WordPieces load ${noteDS.count()} notes")

    val tokenDS = noteDS.flatMap(cleanse(_, specialCharCleanserRegex).filter(!abbrStopTokens.contains(_)))
    val cleanedTokens = tokenDS.distinct.collect

    val vocabularyBuilder = DomainTokenizer.vocabularyBuilder(cleanedTokens)
    val tempVocabulary = vocabularyBuilder.build()
    logDebug(logger, msg = s"Build temp vocabulary with ${cleanedTokens.size} tokens")

    val wordPiecesTokenizer = new WordpieceTokenizer(tempVocabulary, unknownToken, maxNumChars)
    val subTokens: java.util.List[String] = wordPiecesTokenizer.tokenize(cleanedTokens.mkString(" "))
    val finalTokens: scala.List[String] = subTokens
    finalTokens.distinct.toArray
  }
}

private[vocabulary] final object WordPieceExtractor {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[WordPieceExtractor])

  final private val unknownToken = "[UNK]"

  def apply(numInputDocuments: Int, maxNumChars: Int): WordPieceExtractor =
    new WordPieceExtractor(numInputDocuments, maxNumChars)

  def apply(maxNumChars: Int): WordPieceExtractor =
    new WordPieceExtractor(-1, maxNumChars)
}
