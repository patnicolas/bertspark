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

import ai.djl.modality.nlp.preprocess.UnicodeNormalizer
import org.bertspark.nlp.trainingset.{ContextualDocument, SentencePair}
import org.bertspark.transformer.dataset.TFeaturesInstance.SegmentTokens
import org.slf4j.{Logger, LoggerFactory}


/**
 * Wrapper for the components of a record or document
 * {{{
 *   The same components are used for pre-training and classification. In case of pre-training
 *   labelIndex is set to -1, while for classification, it is the index to a label or class value
     The contextual variables are appended to the segments extracted from the document.
 * The document components are
 * - Document identifier
 * - All segments extracted from text + one segment containing the context variable
 * - All tokens extracted from the text + context variables
 * - Optional index of a label (-1 for pre-training)
 * }}}
 *
 * @param documentId Identifier for this document (usually the name of the file or S3 reference)
 * @param tokens Tokens extracted from the document.
 * @param labelIndex Index of label data used for classification only.
 *
 * @author Patrick Nicolas
 * @version 0.2
 */
private[bertspark] final class DocumentComponents(
  documentId: String,
  tokens: Array[SegmentTokens],
  labelIndex: Long = -1L
) {
  require(tokens.nonEmpty, s"Tokens from content $documentId are undefined")
  require(tokens.head.nonEmpty, s"Tokens from content $documentId are empty")

  /**
   * extract pairs of sentences from this document
   * @return Sequence of sentences pairs
   */
  def extractSentencePairs: Array[SentencePair] =
    if(tokens.size > 1)
      (0 until tokens.size-1).foldLeft(List[SentencePair]())(
        (xs, index) => if((index & 0x01) == 0x0) new SentencePair(tokens(index), tokens(index+1)) :: xs else xs
      )   .reverse
          .toArray
    else
      Array[SentencePair](new SentencePair(tokens.head, Array.empty[String]))


  @inline
  final def getDocId: String = documentId

  @inline
  final def getTokens: Array[SegmentTokens] = tokens

  @inline
  final def getLabelIndex: Long = labelIndex

  @inline
  final def isClassification: Boolean = labelIndex > 0L


  override def toString: String =
    s"Id: $documentId\nTokens: ${tokens.map(_.mkString(", ")).mkString("\n")}"
}



private[bertspark] final object DocumentComponents {
  final val logger: Logger = LoggerFactory.getLogger("DocumentComponents")

  /**
   * Constructs a Document components set using an existing contextual document and domain specific tokenizer
   * {{{
   * The document segment has the following attribute
   * - Document identifier
   * - All segments extracted from text + one segment containing the context variable
   * - All tokens extracted from the text + context variables
   * - Optional index of a label (-1 for pre-training)
   * }}}
   * @param ctxDocument Contextual document
   * @param domainTokenizer Domain specific tokenizer
   * @tparam T Type of pre-processed tokenizer
   * @tparam U Type of Sentences extractor
   * @return Optional Document components
   */
  def apply[T <: TokenizerPreProcessor, U <: SentencesBuilder](
    ctxDocument: ContextualDocument,
    domainTokenizer: DomainTokenizer[T, U]): Option[DocumentComponents] = try {
    Some(domainTokenizer(ctxDocument))
  }
  catch {
    case e: IllegalArgumentException =>
      logger.warn(s"Could not extract segments or tokens from ${ctxDocument.getId} ${e.getMessage}")
      None
  }



  def normalize(line: String): String =
    if(line.nonEmpty) {
      val unicodeNormalized = UnicodeNormalizer.normalizeDefault(line);
      var endIdx = line.length() - 1
      while (endIdx >= 0 && Character.isWhitespace(unicodeNormalized.charAt(endIdx))) {
        endIdx -= 1
      }
      line.substring(0, endIdx + 1);
    }
    else
      line
}
