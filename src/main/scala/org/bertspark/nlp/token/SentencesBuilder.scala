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

import org.bertspark.nlp.medical.NoteProcessors.{findingsReplacement, impressionReplacement}
import org.bertspark.nlp.token.SentencesBuilder.ContextSentence
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.{ContextualDocument, ContextualDocumentGroup}
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer


/**
 * {{{
 * Builder to extract sentences from a given document. The contextual tokens are distributed across the various
 * sentences depending on the model...
 * For instance in the case of Txt_CtxTxtSentenceBuilder.. the schema is
 * [("", "tokens from first half of note), ("contextual tokens", "second half of the note")]
 *}}}
 * @note The sentences have to be concatenated with the contextual data
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] trait SentencesBuilder {
  /**
   * Extract the sequence of segments defined as a sequence (Context variable, sentences) where the context
   * variables are the concatenate and assign to one or more sentences
   * @param contextualDocument Contextual document
   * @return Array of ("contextual variable", "sentence tokens")
   */
  @throws(clazz = classOf[UnsupportedOperationException])
  protected def execute(contextualDocument: ContextualDocument): Array[ContextSentence] =
    throw new UnsupportedOperationException(s"${getClass().getName()} does not support contextual document")

  def apply(contextualDocument: ContextualDocument): Array[ContextSentence] = execute(contextualDocument)
}


/**
 * Singleton that implements the selection of
 */
private[bertspark] final object SentencesBuilder {
  import org.bertspark.Labels._
  import org.bertspark.config.MlopsConfiguration._

  final val logger: Logger = LoggerFactory.getLogger("SentencesBuilder")

  def str(contextSentences: Array[ContextSentence]): String =
    contextSentences.map{
      case (ctx, txt) =>
        if(ctx.nonEmpty && txt.nonEmpty) s"$ctx $txt"
        else if(ctx.nonEmpty) ctx
        else if(txt.nonEmpty) txt
        else ""
    }.mkString("\n")

  def countTokens(contextSentences: Array[ContextSentence]): Array[Int] =
    contextSentences.map{
      case (ctx, txt) =>
        if(ctx.nonEmpty && txt.nonEmpty) ctx.split(tokenSeparator).length + txt.split(tokenSeparator).length
        else if(ctx.nonEmpty) ctx.split(tokenSeparator).length
        else if(txt.nonEmpty) txt.split(tokenSeparator).length
        else 0
    }


  // Contextual tokens as a string,  text tokens as a string
  type ContextSentence = (String, String)

  /**
   * Generic constructor/builder for the various sentence builder. The configuration is loaded from the
   * service configuration file
   * @throws UnsupportedOperationException If the configuration for the sentence builder is unknown
   * @return Appropriate sentence builder
   */
  @throws(clazz = classOf[UnsupportedOperationException])
  def apply(): SentencesBuilder = apply(mlopsConfiguration.preTrainConfig.sentenceBuilder)

/**
 * Generic constructor/builder for the various sentence builder. The configuration is loaded from the
 * service configuration file
 * @param sentenceBuilderLbl Type of sentences builder using the service configuration file
 * @throws UnsupportedOperationException If the configuration for the sentence builder is unknown
 * @return Appropriate sentence builder
 */
  def apply(sentenceBuilderLbl: String): SentencesBuilder = sentenceBuilderLbl match {
    case `labeledSentencesBuilderLbl` => new LabeledSentencesBuilder
    case `ctxTxtNSentencesBuilderLbl` => new CtxTxtNSentencesBuilder
    case `ctxNSentencesBuilderLbl` => new CtxNSentencesBuilder
    case `sectionsSentencesBuilderLbl` => new SectionsSentencesBuilder
    case _ =>
      throw new UnsupportedOperationException(s"Sentence builder $sentenceBuilderLbl is not supported")
  }


  def concatenate(contextDocumentPair: ContextSentence): String =
    if(contextDocumentPair._1.nonEmpty && contextDocumentPair._2.nonEmpty)
      s"${contextDocumentPair._1} ${contextDocumentPair._2}"
    else if(contextDocumentPair._1.isEmpty)
      contextDocumentPair._2
    else if(contextDocumentPair._1.isEmpty)
      contextDocumentPair._1
    else ""
}



/**
  * Segments: [Context & Segment1, Segment2, ...]
  */
final class CtxTxtNSentencesBuilder extends SentencesBuilder {
  import org.bertspark.config.MlopsConfiguration._
  private[this] val numSegments: Int = mlopsConfiguration.preTrainConfig.numSentencesPerDoc

  override protected def execute(contextualDocument: ContextualDocument): Array[ContextSentence] =
    if (contextualDocument.text == null || contextualDocument.text.isEmpty) {
      SentencesBuilder.logger.error("Contextual document for sentences builder is undefined")
      Array.empty[ContextSentence]
    }
    else {
      val textTokens: Seq[String] = contextualDocument.text.split(tokenSeparator)
      val tokens = contextualDocument.contextVariables ++ textTokens
      val numTokensPerSegment = (tokens.length.toFloat / numSegments).ceil.toInt
      val segments = (0 until tokens.length by numTokensPerSegment).foldLeft(ListBuffer[Seq[String]]())(
        (buf, index) => {
          val lastIndex = if (index > tokens.length) tokens.length else index + numTokensPerSegment
          buf += tokens.slice(index, lastIndex)
        }
      )
      val tailContextSentence: Array[ContextSentence] = segments.tail.map(seg => ("", seg.mkString(" "))).toArray
      val (ctxTokens, segTextTokens) = segments.head.splitAt(contextualDocument.contextVariables.size)
      Array[ContextSentence]((ctxTokens.mkString(" "), segTextTokens.mkString(" "))) ++ tailContextSentence
    }

  private def distributeLoad(numCtxTokens: Int, numTextTokens: Int): Array[Int] = {
    val alpha = numCtxTokens.toFloat/numTextTokens
    val beta = (numSegments - alpha)/(numSegments -1)
    Array[Int](numCtxTokens + (numTextTokens*alpha).toInt) ++ Array.fill(numSegments -1)((beta*numTextTokens).toInt)
  }
}


/**
  * Segments: [Context, Segment1, Segment2, ...]
  */
final class CtxNSentencesBuilder extends SentencesBuilder {
  import org.bertspark.config.MlopsConfiguration._
  private[this] val numSegments: Int = mlopsConfiguration.numSegmentsPerDocument

  override protected def execute(contextualDocument: ContextualDocument): Array[ContextSentence] =
    if (contextualDocument.text == null || contextualDocument.text.isEmpty) {
      SentencesBuilder.logger.error("Contextual document for sentences builder is undefined")
      Array.empty[ContextSentence]
    }
    else {
      val textTokens: Seq[String] = contextualDocument.text.split(tokenSeparator)
      val numTokensPerSegment = (textTokens.length.toFloat /numSegments).ceil.toInt

      val segments = (0 until textTokens.length by numTokensPerSegment).foldLeft(ListBuffer[Seq[String]]())(
        (buf, index) => {
          val lastIndex = if (index > textTokens.length) textTokens.length else index + numTokensPerSegment
          buf += textTokens.slice(index, lastIndex)
        }
      )

      val tailContextSentence: Array[ContextSentence] = segments.map(seg => ("", seg.mkString(" "))).toArray
      Array[ContextSentence](("", contextualDocument.contextVariables.mkString(" "))) ++ tailContextSentence
    }
}


/**
 * ContextualDocument definition
 */
final class LabeledSentencesBuilder extends SentencesBuilder {
  import ContextualDocumentGroup._

  override protected def execute(contextualDocument: ContextualDocument): Array[ContextSentence] = {
    val contextualDocuments: Array[ContextualDocument] = contextualDocument
    contextualDocuments.map(_.toContextSentence)
  }
}


/**
 * Class that defines a sentence as a section. The context terms contains the sections information
 * [text tokens, context + section tokens]
 */
final class SectionsSentencesBuilder extends SentencesBuilder {
  private[this] val sections: Array[String] = Array[String](findingsReplacement, impressionReplacement)

  override protected def execute(contextualDocument: ContextualDocument): Array[ContextSentence] = {
    val tokens = contextualDocument.text.split(tokenSeparator)
    val section1 = ListBuffer[String]() ++ contextualDocument.contextVariables
    val section2 = ListBuffer[String]()
    val section3 = ListBuffer[String]()
    val sectionsBuffer = Seq[ListBuffer[String]](section1, section2, section3)
    var index = 0

    tokens.foreach(
      token => {
        if(sections.contains(token))
          index += 1
        sectionsBuffer(index).append(token)
      }
    )
    sectionsBuffer.map(buf => ("", buf.mkString(" "))).toArray
  }
}

/**
 * Define a simple, single sentences building with a dummy text as second segment
 */
final class CtxTxt_SentencesBuilder extends SentencesBuilder {

  override protected def execute(contextualDocument: ContextualDocument): Array[ContextSentence] =
    Array[ContextSentence](
      (contextualDocument.contextVariables.mkString(" "), contextualDocument.text),
      (contextualDocument.contextVariables.mkString(" "), contextualDocument.text)
    )
}

final object CtxTxt_SentencesBuilder {
  final private val dummyText = "anterior application"
}