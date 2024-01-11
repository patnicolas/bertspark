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

import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.SparkSession
import org.bertspark.util.io.IOOps

/**
 * Trait for all TFIDF computation
 * @tparam T Type of the TF-IDF computation
 * @see Subclasses TokensTfIdf and ValuesTfIdf
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] trait TfIdf[T <: TfIdf[T]] {
self =>
  import TfIdf._

  protected[this] val countVectorizer = new CountVectorizer().setInputCol(wordsCol).setOutputCol(rawFeaturesCol)

  def apply(ioOps: IOOps[WeightedToken]): Array[WeightedToken]
}


private[bertspark] object TfIdf {
  final val rawFeaturesCol = "rawFeatures"
  final val featuresCol = "features"
  final val wordsCol = "words"

  /**
   * Class for weighted vocabulary key => Weight
   * @param token Category or terms (1 or N-grams)
   * @param weight TF-IDF weight
   */
  case class WeightedToken(token: String, weight: Float) {
    override def toString: String = s"$token,$weight"
  }

  /**
   * Store the TF-IDF weighted token into a S3 bucket:folder
   * @param weightedTokens Tf-Idf weights for the vocabulary tokens
   * @param ioOps Handle to the IO operation
   * @param sparkSession Implicit reference to the current Spark context
   * @return true if save operation is successful, false otherwise
   */
  def save(
    weightedTokens: Array[WeightedToken],
    ioOps: IOOps[WeightedToken])(implicit sparkSession: SparkSession): Boolean = ioOps.save(weightedTokens)
}
