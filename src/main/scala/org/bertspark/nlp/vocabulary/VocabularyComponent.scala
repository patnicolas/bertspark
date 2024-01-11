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
package org.bertspark.nlp.vocabulary

import org.apache.spark.sql.Dataset
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest

/**
 *
 * @author Patrick Nicolas
 * @version 0.5
 */
trait VocabularyComponent {
  val vocabularyName: String
  def build(initialTokens: Array[String], requestDS: Dataset[InternalRequest]): Array[String]
}
