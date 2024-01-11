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
package org.bertspark.transformer

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration


/**
 * {{{
 * Define representation and computation of similarity for
 * - Segment/Sentence embeddings
 * - Document embeddings
 *
 *  Segment embedding similarity correlates the tokens from each segment with its transformer embedding
 *  Document embedding similarity correlates labels with document (aggregated) embedding
 * }}}
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
package object representation {

  def cosine(x: Array[Float], y: Array[Float]): Double = {
    val dot = x.zip(y).map{ case (_x, _y) => _x*_y}.sum
    val xNorm = x.map(_x => _x*_x).sum
    val yNorm = y.map(_y => _y*_y).sum
    dot/Math.sqrt(xNorm*yNorm)
  }

  def similarity(x: Array[Float], y: Array[Float]): Double = {
    val z = 0.5*(cosine(x, y) + 1.0)
    if(z > 1.0) 1.0 else if(z < 0.0) 0.0 else z
  }

  def jaccard[T](x: Array[T], y: Array[T]): Double = 2.0*x.intersect(y).length/(x.length +y.length)

  def orderedJaccard[T](x: Array[T], y: Array[T])(implicit ordered: Ordering[T]): Double = {
    val minSize = Math.min(x.length, y.length)
    val overlap = (0 until minSize).filter(index => x(index) == y(index))
    2.0*overlap.length/(x.length + y.length)
  }
}
