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

import ai.djl.modality.nlp.bert.WordpieceTokenizer
import java.util.ArrayList
import org.bertspark.nlp.token.TokenizerPreProcessor.TextPreprocessor
import org.bertspark.config.MlopsConfiguration.vocabulary
import org.bertspark.nlp.token.ExtWordPiecesTokenizer.WordPiecesExtractor
import org.bertspark.nlp.tokenSeparator


/**
 * Extended BERT word pieces tokenizer that implements the DJL word piece tokenizer with the
 * tokenizer pre-processor
 * @param textPreprocessor Text pre-processor
 * @param maxInputChars Maximum number of characters for the word pieces tokenizer
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class ExtWordPiecesTokenizer private (
  maxInputChars: Int,
  wordPiecesExtractor: WordPiecesExtractor
) extends WordpieceTokenizer(vocabulary,  "[UNK]", maxInputChars) with TokenizerPreProcessor {

  override def tokenize(input: String): java.util.List[String] = {
    val cleansed = TextPreprocessor()(input)
    val tokens = wordPiecesExtractor(cleansed, maxInputChars)
    tokens
  }

  override def apply(input: String): java.util.List[String] = tokenize(input)
}


/**
 * Singleton for constructors for this tokenizer
  * {{{
  *  - Constructors
  *  -
  * }}}
 */
private[bertspark] object ExtWordPiecesTokenizer {
  final private val defaultMaxInputChars = 32
  type WordPiecesExtractor = (String, Int) => java.util.List[String]

  def apply(maxInputChars: Int, wordPiecesExtractor: String): ExtWordPiecesTokenizer = wordPiecesExtractor match {
    case "wordPiecesFromStems" => new ExtWordPiecesTokenizer(maxInputChars, wordPiecesFromStems)
    case "defaultWordPieces" => new ExtWordPiecesTokenizer(maxInputChars, defaultWordPieces)
    case _ =>
      throw new UnsupportedOperationException(s"Word piece extractor $wordPiecesExtractor is not supported")
  }

  def apply(maxInputChars: Int): ExtWordPiecesTokenizer = new ExtWordPiecesTokenizer(maxInputChars, wordPiecesFromStems)

  def apply(): ExtWordPiecesTokenizer = new ExtWordPiecesTokenizer(defaultMaxInputChars, wordPiecesFromStems)


  private val wordPiecesFromStems: WordPiecesExtractor =
    (sentence: String, maxInputChars: Int)  => {
      val outputTokens = new ArrayList[String]()
      val tokens = sentence.trim.split(tokenSeparator)

      tokens.foreach(
        token => {
          // If the token is too long, ignore it
          if (token.length > maxInputChars) {
            outputTokens.add("[UNK]")
            // If the token belongs to the vocabulary
          } else if (vocabulary.contains(token)) {
            outputTokens.add(token)
            // ... otherwise attempts to break it down
          } else {
            val chars = token.toCharArray
            var start = 0
            var end = 0
            // Walks through the token
            while (start < chars.length - 1) {
              end = chars.length
              while (start < end) {
                // extract the stem
                val subToken = token.substring(start, end)
                // If the sub token is found in the vocabulary
                if (vocabulary.contains(subToken)) {
                  val prefix = token.substring(0, start)
                  // If the substring prior the token is also contained in the vocabulary
                  if (vocabulary.contains(prefix))
                    outputTokens.add(prefix)
                  else if(prefix.nonEmpty)    // Otherwise added as a word piece
                    outputTokens.add(s"##$prefix")

                  outputTokens.add(subToken)
                  // Extract the substring after the token
                  val suffix = token.substring(end)
                  if (suffix.nonEmpty) {
                    // If this substring is already in the vocabulary..
                    if (vocabulary.contains(suffix)) {
                      outputTokens.add(suffix)
                      // otherwise added as a word piece
                    } else if(suffix.nonEmpty)
                      outputTokens.add(s"##$suffix")
                  }
                  end = chars.length
                  start = chars.length
                }
                else
                  end -= 1
              }
              start += 1
            }
          }
        }
      )
      outputTokens
    }



    private val defaultWordPieces: WordPiecesExtractor =
      (sentence: String, maxInputChars: Int)  => {
        val subTokens = new ArrayList[String]()
        val outputTokens = new ArrayList[String]()
        val tokens = sentence.trim.split(tokenSeparator)

        tokens.foreach(
          token => {
            if(token.length > maxInputChars)
              outputTokens.add("[UNK]")
            else if(vocabulary.contains(token))
              outputTokens.add(token)
            else {
              val chars = token.toCharArray
              var start = 0
              subTokens.clear()

              var currentSubString: String = null
              var outerLoop = true
              while (start < chars.length && outerLoop) {
                var end = chars.length
                var toContinue = true

                while (start < end && toContinue) {
                  val root = token.substring(start, end)

                  if (vocabulary.contains(root)) {
                    outputTokens.add(root)
                    val remaining = token.substring(end)
                    if(vocabulary.contains(remaining))
                      outputTokens.add(remaining)
                    else
                      outputTokens.add(s"##$remaining")
                    toContinue = false
                    outerLoop  = false
                  }
                  else
                    currentSubString = null
                  end -= 1
                }
                if (currentSubString == null)
                  outerLoop = false

                subTokens.add(currentSubString)
                if (subTokens.size > maxInputChars)
                  throw new IllegalStateException(s"Too many subTokens for $sentence")
                start = end  // start = if(end > 0) end else start+1
              }
            }
          }
        )
        outputTokens
      }
}
