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
package org.bertspark.transformer.block

import ai.djl.ndarray.NDList
import org.bertspark.util.NDUtil


/**
 * Methods to aggregate segment embeddings to generate the document embedding
 * {{{
 *   The method to aggregate the segment embeddings are
 *   - Concatenation of segment embeddings
 *   - Summation of segment embeddings
 * }}}
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
trait SegmentEmbeddingAggregation {
  def apply(segmentEmbeddings: Array[NDList]): NDList
}


final object SegmentEmbeddingAggregation {
  /**
   * {{{
   *   Aggregated CLS predictions from each segments into a single document embedding
   *   There are two aggregations
   *   - concatenate  Doc embedding = [Segment1-embedding segment2-embedding  ...]
   *   - sum          Doc embedding = [Segment1_embedding]+[segment2-embedding] + ......
   * }}}
   * @param segmentEmbeddings Array of un-squeezed segment embeddings associated with a document
   * @param isClsConcatenate Is using concatenation for aggregating segment embeddings
   * @return Document embedding using aggregation (Concatenation or sum)
   */
  def apply(segmentEmbeddings: Array[NDList], isClsAggregation: Boolean): NDList =
    if(isClsAggregation) (new SegmentEmbeddingConcatenation)(segmentEmbeddings)
    else (new SegmentEmbeddingSummation)(segmentEmbeddings)
}


/**
 * Implement the document embedding as the concatenation of segment embeddings
 * Dimension document embedding = SUM (dimension of segment embeddings)
 */
final class SegmentEmbeddingConcatenation extends SegmentEmbeddingAggregation {
  override def apply(segmentEmbeddings: Array[NDList]): NDList = {
    val clsPredictions = segmentEmbeddings.map(_.get(0).squeeze(0))
    new NDList(NDUtil.concat(clsPredictions))
  }
}


/**
* Implement the document embedding as the sum of segment embeddings
 * Dimension document embedding = dimension of each segment embedding
*/
final class SegmentEmbeddingSummation extends SegmentEmbeddingAggregation {
  override def apply(segmentEmbeddings: Array[NDList]): NDList =
    new NDList(NDUtil.add(segmentEmbeddings.map(_.get(0))))
}
