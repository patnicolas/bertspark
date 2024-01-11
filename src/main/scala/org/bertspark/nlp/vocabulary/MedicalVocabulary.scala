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

import org.slf4j._


/**
 * Vocabulary to be loaded from a file
 * @param terms Terms of the vocabulary
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] case class MedicalVocabulary(terms: Array[String])  {
  require(terms.nonEmpty, "Cannot create a customer vocabulary with undefined terms")

  @inline
  final def contains(term: String): Boolean = terms.contains(term)

  @inline
  final def nonEmpty: Boolean = terms.size > 0

  override def toString: String = terms.mkString("\n")
}


/**
 * Constructors and builder for vocabulary
 */
private[bertspark] final object MedicalVocabulary {
  import org.bertspark.config.MlopsConfiguration._

  final private val logger: Logger = LoggerFactory.getLogger("MedicalVocabulary")

  final val defaultTermsFile = ConstantParameters.termsSetFile
  final lazy val emptyVocabulary: MedicalVocabulary = new MedicalVocabulary(Array.empty[String])


  def apply(): MedicalVocabulary = {
    val terms = s3VocabularyStorage.download
    new MedicalVocabulary(terms)
  }


  /**
   * Build multiple N-Gram including the original tokens from a sequence of tokens
   * @param tokens Tokens
   * @param numGrams Number of nGrams to collect
   * @return Sequence of all tokens
   */
  def buildNGrams(tokens: Array[String], numGrams: Int): Array[String] =
    if(tokens.nonEmpty) {
      import scala.collection.mutable.ListBuffer
      val nGrams = new ListBuffer[String]()
      nGrams.appendAll(tokens)

      // If collecting 2-grams is required
      if(numGrams > 1 && tokens.size > 2) {
        val n2Grams = tokens.tail.foldLeft((scala.List[String](), tokens.head)) {
          case ((nGrams, prevToken), token) => (s"$prevToken $token" :: nGrams, token)
        }._1
        nGrams.appendAll(n2Grams)
      }

      // If collecting 3- grams is required
      if(numGrams > 2 && tokens.size > 2) {
        val n3Grams = tokens.drop(2).foldLeft((scala.List[String](), tokens.head, tokens(1))) {
          case ((nGrams, prevPrevToken, prevToken), token) => (s"$prevPrevToken $prevToken $token" :: nGrams, prevToken, token)
        }._1
        nGrams.appendAll(n3Grams)
      }

      // If collecting 3- grams is required
      if(numGrams > 3 && tokens.size > 3) {
        val n4Grams = tokens.drop(3).foldLeft((scala.List[String](), tokens.head, tokens(1), tokens(2))) {
          case ((nGrams, prevPrevPrevToken, prevPrevToken, prevToken), token) =>
            (s"$prevPrevPrevToken $prevPrevToken $prevToken $token" :: nGrams, prevPrevToken, prevToken, token)
        }._1
        nGrams.appendAll(n4Grams)
      }
      nGrams.toArray
    }
    else
      Array.empty[String]
}