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
package org.bertspark.transformer.dataset

import ai.djl.modality.nlp.DefaultVocabulary
import org.bertspark.config.ExecutionMode
import org.bertspark.nlp.token._
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.transformer.config.BERTConfig


/**
 * Configuration for building a dataset for BERT transformer
 *
 * @param batchSize Size of the batch
 * @param maxSeqLength Maximum length of a sentence (or number of tokens extracted from it)
 * @param maxMasking Maximum number of mask for the MLM model
 * @param minTermFrequency Minimum frequency of terms
 * @param sentencesBuilderType Type of sentences builder
 * @param preProcessedTokenizerType Type of pre-processor tokenizer
 * @param preTrainingMode Specify if this configuration applies to pre-training or fine-tuning model
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class TDatasetConfig private (
  batchSize: Int,
  maxSeqLength: Int,
  maxMasking: Int,
  minTermFrequency: Int,
  sentencesBuilderType: String,
  preProcessedTokenizerType: String,
  preTrainingMode: Boolean)  {

//  lazy val vocabularyBuilder: DefaultVocabulary.Builder = DomainTokenizer.vocabularyBuilder(getMinTermFrequency)

  final def getBatchSize: Int = batchSize
  final def getMaxSeqLength: Int = maxSeqLength
  final def getMaxMasking: Int = maxMasking
  final def getMinTermFrequency: Int = minTermFrequency
  final def getSentencesBuilder: SentencesBuilder = SentencesBuilder()
  final def isPreTrainingMode: Boolean = preTrainingMode

  final def getPreProcessedTokenizerType: String = preProcessedTokenizerType


  override def toString: String =
    s"BatchSize: $batchSize\nMaxSeqLength: $maxSeqLength\nMinTermFrequency: $minTermFrequency\nSentencesBuilder: $sentencesBuilderType\npreProcessedTokenizer: $preProcessedTokenizerType\nIsPreTraining: $preTrainingMode"
}


/**
 * Singleton for constructors
 */
private[bertspark] final object TDatasetConfig {

  def apply(
    batchSize: Int,
    maxSeqLength: Int,
    maxMasking: Int,
    vocabularyFile: String,
    sentencesBuilderType: String,
    preProcessedTokenizerType: String,
    preTrainingMode: Boolean): TDatasetConfig =
    new TDatasetConfig(
      batchSize,
      maxSeqLength,
      maxMasking,
      minTermFrequency = 1,
      sentencesBuilderType,
      preProcessedTokenizerType,
      preTrainingMode)


  def apply(minTermFrequency: Int, preTrainingMode: Boolean): TDatasetConfig =
    new TDatasetConfig(
      mlopsConfiguration.executorConfig.batchSize,
      BERTConfig.getMinSeqLength(mlopsConfiguration.getTransformer),
      mlopsConfiguration.getMaxMaskingSize,
      minTermFrequency,
      mlopsConfiguration.getSentenceBuilder,
      mlopsConfiguration.getTokenizer,
      preTrainingMode
     )

  def apply(preTrainingMode: Boolean): TDatasetConfig = apply(1, preTrainingMode)

  final val getPreProcessingTokenizer: String => TokenizerPreProcessor =
    (preProcessedTokenizerType: String) => TokenizerPreProcessor(preProcessedTokenizerType)
}