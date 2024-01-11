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

import org.apache.spark.sql._
import org.bertspark.nlp.token.TfIdf.WeightedToken
import org.bertspark.nlp.token.TokensTfIdf.processTokens
import org.bertspark.util.io.IOOps


/**
 * Computes TF-IDF values for a set of type T of categorical and continuous features
 * @param valuesDS Dataset of value of the T
 * @param project Function that project a value of type T to a sequence of String
 * @param sparkSession Implicit reference to the current Spark context
 * @tparam T Type of value
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class ValuesTfIdf[T](
  valuesDS: Dataset[T],
  project: T => Array[String]
)(implicit sparkSession: SparkSession) extends TfIdf[ValuesTfIdf[T]]{
  import ValuesTfIdf._

  override def apply(ioOps: IOOps[WeightedToken]): Array[WeightedToken] = {
    import sparkSession.implicits._

    val featuresDS = valuesDS.map(project(_))

    // Created an indexed category features per category
    // Category index -> List of features associated with this category
    val indexedCategoryFeaturesSeq: IndexedSeq[(Int, Set[String])] = featuresDS.map(
      categories => categories.indices.map(index => (index, Set[String](categories(index))))
    ).reduce(
      (indexedPair1: IndexedSeq[(Int, Set[String])], indexedPair2: IndexedSeq[(Int, Set[String])]) => {
        indexedPair1.indices.map(
          index => {
            (index, indexedPair1(index)._2 ++ indexedPair2(index)._2)
          }
        )
      }
    )

    // Raw TF-IDF values extracted from categories
    val weightedCategories = processTokens(featuresDS, countVectorizer)
    val weightedCategoriesMap = weightedCategories.map(w => (w.token, w)).toMap

    // Extract boundaries
    val normalizedCategoriesWeights = indexedCategoryFeaturesSeq.flatMap{
      case (_, categoryFeaturesSet) => normalizeCategoryFeatures(categoryFeaturesSet, weightedCategoriesMap)
    }.toArray
    // Save the normalized
    ioOps.save(normalizedCategoriesWeights)
    normalizedCategoriesWeights
  }
}


private[bertspark] final object ValuesTfIdf {

  /**
   * Normalization of the weights associated with features from a given category
   * @param categoryFeatures List of features for this given category
   * @param weightedCategoriesMap Generic wighted map of feature -> WeightedVocabulary record
   * @throws IllegalArgumentException if the features is not correct or does not exist
   * @return Normalized weighted features for a given categorty
   */
  @throws(clazz = classOf[IllegalArgumentException])
  @throws(clazz = classOf[IllegalStateException])
  def normalizeCategoryFeatures(
    categoryFeatures: Set[String],
    weightedCategoriesMap: Map[String, WeightedToken]): Set[WeightedToken] = {
    require(categoryFeatures.nonEmpty, "Category features is empty")
    require(weightedCategoriesMap.nonEmpty, "Weighted category map is empty")

    val weights = categoryFeatures
        .map(weightedCategoriesMap.getOrElse(_, throw new IllegalStateException("Category features is undefined")))

    val maxWeights = weights.maxBy(_.weight).weight
    val minWeights = weights.minBy(_.weight).weight
    val delta = maxWeights - minWeights
    weights.map(
      weightVocabulary => WeightedToken(weightVocabulary.token, (weightVocabulary.weight - minWeights)/delta)
    )
  }
}
