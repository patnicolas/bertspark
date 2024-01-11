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

import org.bertspark.config.MlopsConfiguration._
import org.bertspark.nlp.trainingset.SentencePair
import org.slf4j._
import scala.util.Random


/**
 * Embedding generator for pre-training of BERT model.
 * {{{
 *  It convert a pair of sentences extracted from the corpus into segments with embedding tokens
 *
 *  Description of generated embedding components:
 *    - labeledTokens Tokens extracted from document with segments token [CLS], [SEP]
 *
 * }}}
 * @param inputSentencePair Pair of sentences
 * @param maxSegmentLength Maximum number of tokens per sentences
 * @param maxMasking Maximum number of tokens masked
 * @param maskingProb Probability to mask a given token.
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class TMaskedInstance private (
  inputSentencePair: SentencePair,
  maxSegmentLength: Int,
  maxMasking: Int,
  maskingProb: Float) {
  import TMaskedInstance._

  private[this] val rand = new Random(908907L)
  private[this] val truncatedSentencePair = inputSentencePair.truncate(maxSegmentLength - 3)
  private[this] val labeledTokens: Array[String] = createLabeledTokens(truncatedSentencePair)
  private[this] val tokenTypeIds: Array[Int] = createTokenTypeIds(labeledTokens)
  private[this] val maskedIndices: Array[Int] = createMaskedIndices(maxMasking, maskingProb, labeledTokens)
  private[this] val inputMasks: Array[String] = createInputMasks(maskedIndices, labeledTokens, rand)

  @inline
  final def getMaxSeqLength: Int = maxSegmentLength

  @inline
  final def getMaxMasking: Int = maxMasking


  /**
   * Retrieve the indices associated with the tokens for each sentence. Token are converted to indices through
   * the default vocabulary map
   * @return Token for any given token, token for [PAD] for padding to the max segment length
   */
  final def getTokenIds: Array[Int] = {
    val unpaddedTokenIds = inputMasks.map(mask => vocabulary.getIndex(mask).toInt)
    if(inputMasks.size < maxSegmentLength) unpaddedTokenIds.padTo(maxSegmentLength, 0) else unpaddedTokenIds
  }

  /**
   * Retrieve the index associated with the sentence for each segment after padding if the number of token type ids
   * is < maxSegmentLength
   * @return Padded token type id (segment index)
   */
  final def getTokenTypeIds: Array[Int] =
    if(tokenTypeIds.size < maxSegmentLength) tokenTypeIds.padTo(maxSegmentLength, 0) else tokenTypeIds

  /**
   * Set token mask to 1 for input token and 0 for pad from the original token type ids
   * @return Padded token list [1 1 1 1  .. 1 0 ... 0]
   */
  final def getInputMask: Array[Int] = {
    val unpaddedInputMask = Array.fill(tokenTypeIds.size)(1)
    if(tokenTypeIds.size < maxSegmentLength)
      unpaddedInputMask.padTo(maxSegmentLength, 0)
    else
      unpaddedInputMask
  }

  final def getMaskedPosition: Array[Int] =
    if(maskedIndices.size < maxMasking) maskedIndices.padTo(maxMasking, 0) else maskedIndices


  @inline
  final def isSentenceNextValue: Int = if(truncatedSentencePair.isNext) 1 else 0

  /**
   * Get the mask window of size (maxMasking) indices, converts to token indices/ids then pad to 0 if necessary.
   * @return
   */
  final def getMaskedIndices: Array[Int] = {
    val unpaddedMaskedIds = maskedIndices
        .indices
        .map(idx => vocabulary.getIndex(labeledTokens(maskedIndices(idx))).toInt)
        .toArray
    if(maskedIndices.size < maxMasking) unpaddedMaskedIds.padTo(maxMasking, 0) else unpaddedMaskedIds
  }

  /**
   * Retrieve the array of mask indices as 1 for the mask indices or zero of the mask indices is less than maxMasking
   * @return Array of 1 or 0 (if maxMasking is > number of masked indices)
   */
  final def attentionMasking: Array[Int] =
    if(maskedIndices.size < maxMasking)
      Array.fill(maskedIndices.size)(1) ++ Array.fill(maxMasking - maskedIndices.size)(0)
    else
      Array.fill(maskedIndices.size)(1)

  override def toString: String = {
    val tokenIdsStr = getTokenIds.map(tokenId => s"$tokenId:${vocabulary.getToken(tokenId)}").mkString(" ")
    val segmentIdsStr = getTokenTypeIds.mkString(" ")
    val maskedPositionStr = getMaskedPosition.mkString(" ")
    val maskedIndicesStr = getMaskedIndices.mkString(" ")
    val attentionMaskStr = attentionMasking.mkString(" ")
    val inputMaskStr = getInputMask.mkString(" ")
    s"""Token ids-Labels:   $tokenIdsStr
       |Masked indices:     $maskedIndicesStr
       |Masked positions:   $maskedPositionStr
       |Segment ids:        $segmentIdsStr
       |Attention mask:     $attentionMaskStr
       |Input mask:         $inputMaskStr
       |""".stripMargin
  }
}


/**
 * Singleton to create the various component for Masked Instance
 * {{{
 *   - createLabels: Create labels from sentence pairs <cls><aa>...<zz><sep>... <sep>
 *   - createTypeIds: Extract Type ids for label <sep>
 *   - createMaskedLabels: Assign label mask <msk>
 *   - createMaskedIndices: Extract indices {1, 0} for mask
 * }}}
 */
private[bertspark] final object TMaskedInstance {
  final private val logger: Logger = LoggerFactory.getLogger("TMaskedInstance")

  // Random masking for tokens. The masking process consists of randomly selecting maxMasking indices of
  // tokens in the sentence (shuffle)
  // In case of augmentation, the seed is randomly choosen
  private val maskingRand = new Random(42L)


  /**
   * default constructor for Masked instance
   * @param inputSentencePair Sentences pair as
   * @param maxSegmentLength Maximum length of the segments
   * @param maxMasking Maximum number of tokens to be masked
   * @param maskingProb Probability a token is masked
   * @throws IllegalArgumentException Is maximum probability or the maximum number of tokens by segment
   *                                  is out of bounds of the
   * @return Instance of BERTMaskedInstance
   */
  @throws(clazz = classOf[IllegalArgumentException])
  def apply(
    inputSentencePair: SentencePair,
    maxSegmentLength: Int,
    maxMasking: Int,
    maskingProb: Float
  ): TMaskedInstance = {
    require(maxSegmentLength >= 32, s"Max segment length $maxSegmentLength should be >= 32")
    require(maskingProb > 0.0 && maskingProb < 1.0, s"Masking probability $maskingProb should be ]0, 1.0[")

    new TMaskedInstance(inputSentencePair, maxSegmentLength, maxMasking, maskingProb)
  }

  /**
   * Extract labels from a pair of sentences
   * @param sentencePair Pair of sentences
   * @return List of labels
   */
  def createLabeledTokens(sentencePair: SentencePair): Array[String] =
    ((clsLabel :: sentencePair.getThisSentence.toList) :::
    (sepLabel :: sentencePair.getThatSentence.toList) :::
    (sepLabel :: List[String]())).toArray

  /**
   * Extract the type ids from the labels... The value is incremented for each [SEP] token
   * @param labels Labels (terms or tokens)
   * @return Array of type id
   */
  def createTokenTypeIds(labels: Array[String]): Array[Int] =
    if(labels.nonEmpty) {
      var typeId: Int = 0
      labels.foldLeft(List[Int]())(
        (xs, label) =>
            // If we find end of sentence....
          if (sepLabel == label) {
            typeId += 1
            typeId - 1 :: xs
          }
          else
            typeId :: xs
      ).reverse.toArray
    }
    else {
      logger.warn("Failed to extract type ids from undefined labels")
      Array.empty[Int]
    }

  /**
   * Extract the masked labels from masked indices and labels
   * @param maskedIndices Indices of masked token
   * @param labels Labels to match against
   * @param rand Current random generator
   * @return Array of masked labels
   */
  def createInputMasks(maskedIndices: Array[Int], labels: Array[String], rand: Random): Array[String] =
    if(noMask)
      labels
    else
      if(maskedIndices.nonEmpty && labels.nonEmpty) {
        maskedIndices.foldLeft(labels) (
          (xs, maskedIndex) => {
            val r = rand.nextFloat()
            if (r < 0.8F)
              xs(maskedIndex) = mskLabel
            else if (r < 0.9F)
              xs(maskedIndex) = vocabulary.getToken(rand.nextInt(vocabulary.size.toInt))
            xs
          }
        )
      }
      else {
        logger.warn("Failed to create masked labels from undefined masked indices or labels")
        Array.empty[String]
      }


  /**
    * Create indices for the mask tokens  (1 for masked, 0 otherwise)
    * @param maxMasking Maximum number of tokens to be masked for this sentence
    * @param maskingProb Probability for masking a given token  (Default should be 0.2)
    * @param labels Labels or tokens to be masked
    * @return Array of state of masked tokens maskedIndices(tokenJ) = 1 if masked, 0 otherwise
    */
  def createMaskedIndices(maxMasking: Int, maskingProb: Float, labels: Array[String]): Array[Int] =
    if(labels.nonEmpty) {
      val subLabelsCount = (labels.size * maskingProb).floor.toInt
      val maskedCount = if(subLabelsCount < maxMasking) subLabelsCount else maxMasking
      val tempIndices = Seq.tabulate(labels.size)(n => n)
      val shuffledTempIndices = maskingRand.shuffle(tempIndices)
      shuffledTempIndices.slice(0, maskedCount).sortWith(_ < _).toArray
    }
    else {
      logger.warn("Failed to create masked indices from undefined labels")
      Array.empty[Int]
    }
}
