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
package org.bertspark.transformer.representation

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.transformer.representation.EmbeddingSimilarity.ModelSimilarity


/**
 * Generic interface to compute the similarity of embedding (Transformer representation)
 * @author Patrick Nicolas
 * @version 0.1
 */
trait EmbeddingSimilarity {
self =>
  def similarity(numSamples: Int): ModelSimilarity
}


final object EmbeddingSimilarity  {

  /**
   *
   * @param meanWithinLabels
   * @param meanAcrossLabels
   * @param loss
   * @param subModelSimilarities
   * @param config
   */
  case class ModelSimilarity(
    meanWithinLabels: Double,
    meanAcrossLabels: Double,
    loss: Double,
    subModelSimilarities: Seq[(String, Double)] = Seq.empty[(String, Double)],
    config: String
  ) {
    override def toString: String = {
      s"""Target:               ${mlopsConfiguration.target}
         |Mean in labels:       ${meanWithinLabels}
         |Mean between labels:  ${meanAcrossLabels}
         |""".stripMargin
    }
    final def isDefined: Boolean = meanWithinLabels == 1.0
  }

  final object ModelSimilarity {
    def apply(): ModelSimilarity = ModelSimilarity(-1.0, -1.0, -1.0, Seq.empty[(String, Double)], "")

    def apply(meanWithinLabels: Double, meanAcrossLabels: Double): ModelSimilarity =
      ModelSimilarity(meanWithinLabels, meanAcrossLabels, -1.0, Seq.empty[(String, Double)], "")
  }


}