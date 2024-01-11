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

import org.apache.spark.sql.{Dataset, Encoder, SparkSession}
import org.bertspark.util.io.SingleS3Dataset
import org.bertspark.nlp.token.DocumentComponents.logger
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.nlp.trainingset.{ContextualDocument, SentencePair}
import org.bertspark.transformer.dataset.TFeaturesInstance.SegmentTokens
import org.bertspark.config.MlopsConfiguration.DebugLog.logInfo
import scala.collection.mutable.ListBuffer


/**
 * {{{
 * Define a corpus or training set of document components and vocabulary
 * }}}
 * @param documentComponents Segments extracted from this document
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class DocumentCorpus private (documentComponents: Array[DocumentComponents]) {

  final def getDocumentComponents: Array[DocumentComponents] = documentComponents
  def getSegments: Array[Array[SegmentTokens]] =
    if(documentComponents.nonEmpty) documentComponents.map(_.getTokens) else Array.empty[Array[SegmentTokens]]

  /**
   * Sampling of document components for generating sentence pairs
   * @return Sample of sentence pairs
   */
  def getSentencePairs: Array[SentencePair] =
    if(documentComponents.nonEmpty) documentComponents.map(_.extractSentencePairs).reduce( _ ++ _)
    else Array.empty[SentencePair]

  def getLabelIndices: Array[Long] = documentComponents.map(_.getLabelIndex)
  def docIds: Array[String] = documentComponents.map(_.getDocId)
  override def toString: String = s"${documentComponents.mkString("\n")}"

  @inline
  final def size: Int = documentComponents.size
}


/**
 * Singleton for various constructors
 */
private[bertspark] final object DocumentCorpus {

  /**
   * Constructs a Document corpus using a given storage and domain-based tokenizer
   * @param s3Dataset Storage used to extract the data set
   * @param domainTokenizer Domain specific tokenizer
   * @tparam T Type of pre-processing tokenizer
   * @tparam U Type of sentence extractor or builder
   * @return Instance of Document corpus
   */
  def apply[T <: TokenizerPreProcessor, U <: SentencesBuilder](
    s3Dataset: SingleS3Dataset[InternalRequest],
    domainTokenizer: DomainTokenizer[T, U]): DocumentCorpus = {
    val contextualDocumentIterator = s3Dataset.getContentIterator
    val input: ListBuffer[DocumentComponents] = ListBuffer[DocumentComponents]()

    while(contextualDocumentIterator.hasNext) {
      val ctxDocument = contextualDocumentIterator.next()
      DocumentComponents(ctxDocument, domainTokenizer).foreach(input.append(_))
    }
    logInfo(logger,  s"Document corpus has been initialized")
    new DocumentCorpus(input.toArray)
  }

  /**
   * Constructor using data set loaded from an existing storage (Local files, S3, Database,..)
   * @param s3Dataset Storage descriptor for dataset loaded from S3
   * @param vocabulary Vocabulary used in pre processed tokenizer
   * @param preProcessedTokenizer Pre-processing tokenizer
   * @param sentencesBuilder Sentence extractor for the document
   * @tparam T Type of the pre-processed Tokenizer
   * @tparam U Type of the sentence builder
   * @return Instance of the Document corpus
   */
  def apply[T <: TokenizerPreProcessor, U <: SentencesBuilder](
    s3Dataset: SingleS3Dataset[InternalRequest],
    preProcessedTokenizer: T,
    sentencesBuilder: U): DocumentCorpus =
    apply(s3Dataset, DomainTokenizer[T, U](preProcessedTokenizer, sentencesBuilder))


  /**
   * Constructor using existing data set containing the training (features + labels) data
   * @param inputDS Data set containing training data
   * @param extractor Extractor to Contextual Document
   * @param preProcessedTokenizer Generic tokenizer
   * @param sentencesBuilder Sentence builder
   * @param sparkSession Implicit reference to the current Spark context
   * @param encoder Encoder for type V
   * @tparam T Type of the Pre processed tokenizer
   * @tparam U Type of the sentences builder
   * @tparam V Type of elements of the data set
   * @return Instance of the Document corpus
   */
  def apply[T <: TokenizerPreProcessor, U <: SentencesBuilder, V](
    inputDS: Dataset[V],
    extractor: V => ContextualDocument,
    preProcessedTokenizer: T,
    sentencesBuilder: U)(implicit sparkSession: SparkSession, encoder: Encoder[V]): DocumentCorpus = {
    import sparkSession.implicits._

    val domainTokenizer = DomainTokenizer[T, U](preProcessedTokenizer, sentencesBuilder)
    val contextualDocumentIterator = inputDS.map(extractor(_)).collect.iterator

    val input: ListBuffer[DocumentComponents] = ListBuffer[DocumentComponents]()
    while(contextualDocumentIterator.hasNext) {
      val ctxDocument = contextualDocumentIterator.next()
      val docComponents = DocumentComponents(ctxDocument, domainTokenizer)
      if(!docComponents.isDefined)
        logger.warn(s"Document components for corpus are undefined")
      docComponents.foreach(input.append(_))
    }
    logInfo(logger,  "Training document corpus has been initialized")
    new DocumentCorpus(input.toArray)
  }
}