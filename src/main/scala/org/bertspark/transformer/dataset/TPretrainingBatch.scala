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

import ai.djl.ndarray._
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.config.MlopsConfiguration.DebugLog.logTrace
import org.slf4j._


/**
 * Manage the batch of masked instances to apply
 * {{{
 * Features - dimension maxSegmentLength
 *     Tokens from document including MASK, CLS, SEP and PAD token
 *     TokenIds Tokens converted into index using the default vocabulary
 *     TokenTypeIds: Identifier for the order in the pair of sentences (0 first sentence, 1 for second sentence)
 *                   with padding
 *     InputMask: Mask for padding (1 if tokens exists, 0 for padded tokens)
 *     MaskPosition: Indices for the masking token
 *
 * Labels - dimension maxMasking
 *     Label for the order of sentences {0, 1}
 *     MaskIds: Index of tokens in the mask
 *     LabeledMask: 1 for maskId 0 for padding
 * }}}
 *
 * @param instances Array of BERT masked instances
 * @param encoders Encoding functions BERTMaskedInstance => Array[Int]
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class TPretrainingBatch private (
  override val instances: Array[TMaskedInstance],
  override val encoders: Array[EncodingMaskFunc]
) extends TInstancesBatch[TMaskedInstance, Int] {

  /**
   * Retrieve the features for this batch of data
   * @param ndManager Reference to the NDArray manager
   * @return NDList of batch for the 4 features
   */
  override def getFeatures(ndManager: NDManager): NDList = {
    if(encoders.size > 3)
      new NDList(
        createBatch(ndManager, encoders(0)),
        createBatch(ndManager, encoders(1)),
        createBatch(ndManager, encoders(2)),
        createBatch(ndManager, encoders(3))
      )
    else
      throw new IllegalStateException("Bert training batch is missing ")
  }


  /**
   * Retrieve the values for the Next Sentence Prediction (NSP) and Mask Language Model (MLM)
   * related to the auto label.
   * {{{
   *   The 3 encoders are
   *   - Order of the segments/sentences
   *   - MaskIds Index of token in the mask window
   *   - LabeledMask: 1 for maskIds, 0 for padding
   * }}}
   * @param ndManager Reference to the NDArray manager
   * @return Array of 3 labels (NSP binary, masked ids and label masked ids)
   */
  override def getLabels(ndManager: NDManager): NDList = {
    if (encoders.size > 5) {
      val ndNextSentenceLabels = createSentenceLabels(ndManager)
      val ndMaskedIds = createBatch(ndManager, encoders(4))
      val ndLabelMask = createBatch(ndManager, encoders(5))
      new NDList(ndNextSentenceLabels, ndMaskedIds, ndLabelMask)
    }
    else
      throw new IllegalStateException("Missing indexers for extracting labels")
  }

  private def createSentenceLabels(ndManager: NDManager): NDArray = {
    val values = instances.map(_.isSentenceNextValue)

    val ndArray = ndManager.create(values)
    ndArray
  }
}


/**
 * Singleton for constructors, and debugging trace functions
 */
private[bertspark] final object TPretrainingBatch {
  final private val logger: Logger = LoggerFactory.getLogger("TPretrainingBatch")

  /**
   * Default constructor
   * @param instances BERT masked instances
   * @param indexers Indexers for BERT masked instances
   * @return Instances of BERTPretrainingBatch
   */
  def apply(
    instances: Array[TMaskedInstance],
    indexers: Array[EncodingMaskFunc]): TPretrainingBatch =
    new TPretrainingBatch(instances, indexers)


  /**
   * Constructors with predefined indexers
   * @param instances BERT masked instances
   * @return Instances of BERTPretrainingBatch
   */
  def apply(instances: Array[TMaskedInstance]): TPretrainingBatch = {
    val indexers: Array[EncodingMaskFunc] = Array[EncodingMaskFunc](
      (instance: TMaskedInstance) => instance.getTokenIds,
      (instance: TMaskedInstance) => instance.getTokenTypeIds,
      (instance: TMaskedInstance) => instance.getInputMask,
      (instance: TMaskedInstance) => instance.getMaskedPosition,
      (instance: TMaskedInstance) => instance.getMaskedIndices,
      (instance: TMaskedInstance) => instance.attentionMasking
    )
    new TPretrainingBatch(instances, indexers)
  }

  /**
   * Generic constructor of BERT masked instance for pre-training or classification
   * @param maskInstances Masked instance
   * @param ndManager Reference to the NDArray manager
   * @throws IllegalStateException if the batch of documents does not match the number of masked instance for
   *                               classification
   * @return Appropriate batch BERT masked instances
   */
  @throws(clazz = classOf[IllegalStateException])
  def apply(
    maskInstances: Array[TMaskedInstance],
    ndManager: NDManager,
    isClassification: Boolean): (NDList, NDList) = {

    val pretrainingBatch = TPretrainingBatch(maskInstances)
    if(isClassification) (pretrainingBatch.getFeatures(ndManager), null)
    else  (pretrainingBatch.getFeatures(ndManager), pretrainingBatch.getLabels(ndManager))
  }


  /**
   * Debugging/tracing features as NDList
   * @param featureNdList Features for pre-training model
   */
  def traceFeatures(featureNdList: NDList): Unit =
    logTrace(
      logger, {
        // Access NDArray values
        val ndTokenIds = featureNdList.head
        val ndTokenTypeIds = featureNdList.get(1)
        val ndInputMask = featureNdList.get(2)
        val ndMaskedPositions = featureNdList.get(3)

        // Convert to Scala type
        val tokenIdArray = ndTokenIds.toIntArray
        val tokenIdsStr = tokenIdArray.map(tokenId => s"$tokenId:${vocabulary.getToken(tokenId)}").mkString(" ")
        val tokenTypeStr = ndTokenTypeIds.toIntArray.mkString(" ")
        val inputMaskStr = ndInputMask.toIntArray.mkString(" ")
        val maskedPositionStr = ndMaskedPositions.toIntArray.mkString(" ")
        s"""
           |Batch tokens:          $tokenIdsStr
           |Batch token type ids:  $tokenTypeStr
           |Batch input masks:     $inputMaskStr
           |Batch masked position: $maskedPositionStr
           |""".stripMargin
      }
    )

  /**
   * {{{
   *  Trace the output for labels
   *  - Next sentence label {0, 1}
   *  - Labeled mask ids
   *  - Label masks
   * }}}
   * @param labelNdList NDList output for labels
   */
  def traceLabels(labelNdList: NDList): Unit =
    logTrace(
      logger, {
        // Access NDArray values
        val ndNextSentenceLabels = labelNdList.head
        val ndMaskedIds = labelNdList.get(1)
        val ndLabelMask = labelNdList.get(2)

        // Convert to Scala type
        val nextSentenceLabelsStr = ndNextSentenceLabels.toIntArray.mkString(" ")
        val maskedIdsStr = ndMaskedIds.toIntArray.mkString(" ")
        val labelMaskStr = ndLabelMask.toIntArray.mkString(" ")
        s"""
           |Batch sentence order:  $nextSentenceLabelsStr
           |Batch mask ids:        $maskedIdsStr
           |Batch label mask:      $labelMaskStr
           |""".stripMargin
      }
    )
}