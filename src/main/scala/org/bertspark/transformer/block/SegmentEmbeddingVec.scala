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

import ai.djl.ndarray.{NDArray, NDList}
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.slf4j.{Logger, LoggerFactory}


/**
 * Trait that define the selection of segment embedding given the downstream task
 * {{{
 *   The segment (or sentence) embedding can be extracted from
 *   - CLS token embedding   Configuration:  predictor="clsEmbedding"
 *   - Pooling all token embeddings  Configuration:  predictor="poolingOutput"
 * }}}
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
trait SegmentEmbeddingVec {
  def apply(embedding: NDList): NDList
}

final object SegmentEmbeddingVec {
  final val logger: Logger = LoggerFactory.getLogger("SegmentEmbeddingSelection")

}


final class CLSSegmentEmbedding extends SegmentEmbeddingVec {
  import SegmentEmbeddingVec._

  override def apply(ndBertTokenEmbeddings: NDList): NDList = {
    val embeddedSequence = ndBertTokenEmbeddings.head
    val clsNdEmbedding: NDArray = embeddedSequence.get(":,0,:")

    if(mlopsConfiguration.getEmbeddingsSize != clsNdEmbedding.getShape().get(1))
      logger.error(s"CLS embedding shape ${clsNdEmbedding.getShape()} should be = embedding.size ${
        mlopsConfiguration.getEmbeddingsSize
      }")
    new NDList(clsNdEmbedding)
  }
}



final class PooledOutputSegmentEmbedding extends SegmentEmbeddingVec {

  override def apply(ndBertTokenEmbeddings: NDList): NDList = {
    val pooledOutput = ndBertTokenEmbeddings.get(1)

    // Need to un-squeeze for batch size =1,   (embedding_vector) => (1, embedding_vector)
    val unSqueezePooledOutput =
      if(pooledOutput.getShape().dimension() == 1) pooledOutput.expandDims(0)
      else pooledOutput
    new NDList(unSqueezePooledOutput)
  }
}
