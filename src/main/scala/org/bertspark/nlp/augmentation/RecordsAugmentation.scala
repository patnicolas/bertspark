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
package org.bertspark.nlp.augmentation

import org.apache.spark.sql.Dataset
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import scala.util.Random



trait RecordsAugmentation {
  def augment: Dataset[SubModelsTrainingSet]
}


private[bertspark] object RecordsAugmentation {
  final val randTokenIndex = new Random(42L)

  final class NoAugmentation private (subModelsTrainingSetDS: Dataset[SubModelsTrainingSet]) extends RecordsAugmentation {
    override def augment: Dataset[SubModelsTrainingSet] = subModelsTrainingSetDS
  }

  object NoAugmentation {
    def apply(subModelsTrainingSetDS: Dataset[SubModelsTrainingSet]): NoAugmentation =
      new NoAugmentation(subModelsTrainingSetDS)

  }
}