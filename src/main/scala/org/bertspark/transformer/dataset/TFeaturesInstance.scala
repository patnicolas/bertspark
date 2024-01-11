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
import org.bertspark.transformer.dataset.TFeaturesInstance.SegmentTokens
import org.slf4j._


/**
 * Instance of embedding for Segment tokens associated with a given document
 * {{{
 *  A Document has multiple segments
 *  A segment has 3 embeddings of type Array[Int]
 *
 * }}}
 * @param docSegments Segment tokens associated with a document
 * @param maxSegmentLength Maximum number of tokens in a segment/sentence
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class TFeaturesInstance private (docSegments: Array[SegmentTokens], maxSegmentLength: Int) {
  import TFeaturesInstance._

  @inline
  final def getMaxSegmentLength: Int = maxSegmentLength


  @inline
  final def getTokenIds: Array[Array[Int]] = if(embeddings.nonEmpty) embeddings.head else Array.empty[Array[Int]]


  @inline
  final def getTypeIds: Array[Array[Int]] = if(embeddings.size > 1) embeddings(1) else Array.empty[Array[Int]]

  /**
   * Implement padding for indices
   * @return  Array of indices..
   */
  final def getInputMasks: Array[Array[Int]] = if(embeddings.size > 2) embeddings(2) else Array.empty[Array[Int]]


  /**
   * Retrieve the embeddings associated with the segments extracted from this document
   * @return Array of set of segment embeddings
   */
  lazy val embeddings: Array[Array[Array[Int]]] =
    docSegments.map(
      docSegment => {
        // First embedding
        val labels = Array[String](clsLabel) ++ docSegment
        val tokenIds = getTokenIds(labels)
        // Second embedding
        val typeIds = Array.fill(maxSegmentLength)(0)
        // Third embedding: Position of the masked tokens
        val paddedInputMask = getInputMask(docSegment)
        Array[Array[Int]](tokenIds, typeIds, paddedInputMask)
      }
    )


  override def toString: String =
    (0 until getTokenIds.size).map(
      index => {
        val tokenIdsStr = getTokenIds(index).mkString(" ")
        val typeIdsStr = getTypeIds(index).mkString(" ")
        val inputMaskStr = getInputMasks(index).mkString(" ")
        s"""Token ids:          $tokenIdsStr
           |Masked ids:         $typeIdsStr
           |Input mask:         $inputMaskStr
           |""".stripMargin
      }
    ).mkString("\n\n")


  // ------------------------------  Supporting methods -------------------------

  private def getTokenIds(docSegment: SegmentTokens): Array[Int] = {
    val unpaddedTokenIds = docSegment.map(token => vocabulary.getIndex(token).toInt)
    if (docSegment.size < maxSegmentLength)
      unpaddedTokenIds.padTo(maxSegmentLength, padIndex.toInt)
    else if(docSegment.size > maxSegmentLength)
      unpaddedTokenIds.take(maxSegmentLength)
    else
      unpaddedTokenIds
  }

  private def getInputMask(docSegment: SegmentTokens): Array[Int] = {
    val unpaddedInputMask = Array.fill(docSegment.size)(1)
    if (docSegment.size < maxSegmentLength)
      unpaddedInputMask.padTo(maxSegmentLength, 0)
    else if(docSegment.size > maxSegmentLength)
      unpaddedInputMask.take(maxSegmentLength)
    else
      unpaddedInputMask
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
private[bertspark] final object TFeaturesInstance {
  final private val logger: Logger = LoggerFactory.getLogger("TFeaturesInstance")

  type SegmentTokens = Array[String]


  /**
   * Default constructor for the BERT feature instance
   * @param docSegments Segment associated with a given document
   * @param maxSegmentLength Maximum length allowed for a segment
   * @throws IllegalArgumentException If docSegments is empty of maxSegmentLength < 32
   * @return Instance of BERTFeatureInstance
   */
  @throws(clazz = classOf[IllegalArgumentException])
  def apply(docSegments: Array[SegmentTokens], maxSegmentLength: Int): TFeaturesInstance = {
    require(docSegments.nonEmpty, "Cannot instantiate BERT feature instance with undefined segment tokens")
    require(maxSegmentLength >= 32, s"Max segment length $maxSegmentLength should be >= 32")

    new TFeaturesInstance(docSegments, maxSegmentLength)
  }


  /**
   * Extract labels from a pair of sentences
   * @param docSegmentTokens Segment tokens associated with a document
   * @return List of Segment tokens
   */
  def createLabels(docSegmentTokens: Array[SegmentTokens], maxSegmentLength: Int): Array[SegmentTokens] =
    docSegmentTokens.map(docSegment => {
      val tokens = Array[String](clsLabel) ++ docSegment
      if(tokens.size > maxSegmentLength)
        tokens.take(maxSegmentLength)
      else if(tokens.size < maxSegmentLength)
        tokens.padTo(maxSegmentLength, padLabel)
      else
        tokens
    })
}
