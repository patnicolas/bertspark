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
  */
package org.bertspark.nlp


package object augmentation {
  final val randomAugUNK  = "randomUNK"
  final val randomAugChar = "randomChar"
  final val randomAugSyn = "randomSyn"
  final val randomAugCorrect = "randomCorrect"
  final val randomAugSynAndCorrect = "randomSynAndCorrect"
  final val filterAug = "filter"
  final val noAug = "none"
}
