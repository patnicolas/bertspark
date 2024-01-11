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
import org.bertspark.util.NDUtil


/**
 * Create a batch of NDArray embeddings for inference dataset
 * {{{
 * Features - dimension maxSegmentLength
 *     Tokens from document including MASK, CLS, SEP and PAD token
 *     TokenIds Tokens converted into index using the default vocabulary
 *     TokenTypeIds: Identifier for the order in the pair of sentences (0 first sentence, 1 for second sentence)
 *                   with padding
 * }}}
 *
 * @param instances Bert feature instances for basic input embeddings
 * @param encoders Set of functions to extract segments related indexing from a feature instance
 *                          BERTFeaturesInstance => Array[Int]
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class TFeaturesBatch private (
  override val instances: Array[TFeaturesInstance],
  override val encoders: Array[EncodingFunc]
  ) extends TInstancesBatch[TFeaturesInstance, Array[Int]] {

  /**
   * Retrieve the features for this batch of data
   * @param ndManager Reference to the NDArray manager
   * @return NDList of batch for the 3 features
   */
  override def getFeatures(ndManager: NDManager): NDList =
    if(encoders.size > 2) {
      val ndSegmentFeatures: Array[NDList] = instances.map(
        instance => {
          val segmentEmbeddings = instance.embeddings.map(
            embedding => {
              val ndArray = ndManager.create(embedding)
              ndArray
            }
          )
          NDUtil.batchify(segmentEmbeddings.map(new NDList(_)))
        }
      )
      NDUtil.batchify(ndSegmentFeatures)
    }
    else
      throw new IllegalStateException("BERT training batch is missing")

  override def getLabels(ndManager: NDManager): NDList = ???
}


/**
 * Singleton for constructors
 */
private[bertspark] final object TFeaturesBatch {
  /**
   * Default constructor
   * @param instances Bert feature instances for basic input embeddings
   * @param encodingFunctions Set of functions to encode the instance features
   * @throws IllegalArgumentException If instance or indexers are empty
   * @return Instances of BERT features batch
   */
  @throws(clazz = classOf[IllegalArgumentException])
  def apply(
    instances: Array[TFeaturesInstance],
    encodingFunctions: Array[EncodingFunc]): TFeaturesBatch = {
    require(instances.nonEmpty, "Cannot instantiate BERT features batch without instances")
    require(encodingFunctions.nonEmpty, "Cannot instantiate BERT features batch with no encoding functions")
    new TFeaturesBatch(instances, encodingFunctions)
  }

  /**
   * Constructor with default indexer for embeddings
   * @param instances Bert feature instances for basic input embeddings
   * @throws IllegalArgumentException If instance or indexers are empty
   * @return Instances of BERT features batch
   */
  @throws(clazz = classOf[IllegalArgumentException])
  def apply(instances: Array[TFeaturesInstance]): TFeaturesBatch = {
    require(instances.nonEmpty, "Cannot instantiate BERT features batch without instances")

    val encodingFunctions: Array[EncodingFunc] =
      Array[EncodingFunc](
        (instance: TFeaturesInstance) => instance.getTokenIds,
        (instance: TFeaturesInstance) => instance.getTypeIds,
        (instance: TFeaturesInstance) => instance.getInputMasks
      )
    new TFeaturesBatch(instances, encodingFunctions)
  }
}