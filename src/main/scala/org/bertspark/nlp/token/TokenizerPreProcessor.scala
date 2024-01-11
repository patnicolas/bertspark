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

import org.bertspark.Labels._
import org.bertspark.nlp.medical.NoteProcessors
import org.bertspark.nlp.medical.NoteProcessors.specialCharCleanserRegex
import org.bertspark.nlp.token.TokenizerPreProcessor.AbbreviationMap.abbreviationMap
import org.bertspark.nlp.tokenSeparator
import org.bertspark.util.io.LocalFileUtil
import org.slf4j.{Logger, LoggerFactory}


/**
 * Pre-processor for the tokenizer
 * {{{
 *   The sub-classes have to be merged with any of the BERT tokenizer
 * }}}
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] trait TokenizerPreProcessor {
  def tokenize(input: String): java.util.List[String]
  def apply(input: String): java.util.List[String]
}


/**
 * Singleton for the implementing the text pre-processors
 */
private[bertspark] final object TokenizerPreProcessor {
  final private val logger: Logger = LoggerFactory.getLogger("TokenizerPreProcessor")

  /**
   * Constructor to instantiate the appropriate pre-processed tokenizer
   * @param preProcessorTokenizerType Type of processor tokenizer
   * @return Tokenizer pre-processor
   */
  @throws(clazz = classOf[UnsupportedOperationException])
  def apply(
    preProcessorTokenizerType: String
  ): TokenizerPreProcessor = preProcessorTokenizerType match {
    case `bertTokenizerLbl` => ExtBERTTokenizer()
    case `wordPiecesTokenizerLbl` => ExtWordPiecesTokenizer()
    case _ =>
      throw new UnsupportedOperationException(s"Pre processing tokenizer, $preProcessorTokenizerType is not supported")
  }

  def apply(): TokenizerPreProcessor = ExtWordPiecesTokenizer()



  /**
   * Text pre-processor configuration to be used prior of Bert tokenizer
   */
  case object TextPreprocessor {
    final private val aliasesMapFilename = "conf/codes/aliasesMap.csv"

    lazy val aliasesMap: Map[String, String] = LocalFileUtil.Load.local(aliasesMapFilename, (s: String) => s)
      .map(
        lines => {
          lines.map(
            line => {
              val fields = line.split(",")
              if(fields.size == 2) (fields.head, fields(1)) else ("", "")
            }
          ).filter(_._1.nonEmpty).toMap
        }
      ).getOrElse({
        logger.error(s"Failed to load aliases $aliasesMapFilename")
        Map.empty[String, String]
      })

    /**
     * Create a function to preprocess an existing document
     * {{{
     *   - 1: Remove new line characters
     *   - 2: Finally Remove special character
     *   - 3: Get tokens and keep the ones with a minimum frequency of occurrences
     *   - 4: Replace the medical abbreviation with descriptors
     *   - 5: Convert to lower case if needed
     *   - 6: Filter by vocabulary
     * }}}
     * @return Concatenation of tokens including replacement from abbreviation
     */
    def apply(): String => String = (text: String) => {
      val rawTerms = NoteProcessors.cleanse(text, specialCharCleanserRegex)
      val tokens = rawTerms.flatMap(
        rawTerm =>
          if(abbreviationMap.contains(rawTerm))
            abbreviationMap.get(rawTerm).get.split(tokenSeparator)
          else
            Array[String](rawTerm.toLowerCase)
      )

      val finalTokens = tokens.filter(token => token.length > 1 && token.length < 32)
          .map(token => aliasesMap.getOrElse(token, token))
      finalTokens.mkString("\n")
    }

    override def toString: String = "Text tokenizer"
  }



  final class AbbreviationMap  {
    private[this] val abbrMap: Map[String, String] =
      LocalFileUtil.Load.local("conf/abbreviationsMap").map(
        content => {
          val lines = content.split("\n")
          lines.map(line => {
            val ar = line.split(",")
            (ar.head, ar(1))
          }).toMap
        }
      ).getOrElse(Map.empty[String, String])

    final def getAbbreviations: Map[String, String] = abbrMap
  }

  final object AbbreviationMap {
    lazy val abbreviationMap: Map[String, String] = (new AbbreviationMap).getAbbreviations
  }
}
