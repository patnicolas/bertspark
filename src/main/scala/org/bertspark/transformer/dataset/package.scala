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



/**
 * Classes and methods to create and clean all data sets (training and evaluation sets) require for the
 * training and evaluation of transformer pre-training models
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
package object dataset {
  type EncodingFunc = TFeaturesInstance => Array[Array[Int]]
  type EncodingMaskFunc = TMaskedInstance => Array[Int]

  // Reserved labels for BERT encoder
  final val clsLabel = "[CLS]"
  final val unkLabel = "[UNK]"
  final val sepLabel = "[SEP]"
  final val mskLabel = "[MSK]"
  final val padLabel = "[PAD]"

  final val reservedLabels: Set[String] = Set[String](clsLabel, unkLabel, mskLabel, sepLabel, padLabel)
  final val noMask = false
}
