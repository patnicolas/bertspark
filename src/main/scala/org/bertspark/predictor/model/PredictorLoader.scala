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
package org.bertspark.predictor.model

import ai.djl.ndarray.NDList
import ai.djl.repository.zoo.ZooModel
import org.bertspark.classifier.block.ClassificationBlock
import org.bertspark.classifier.model.ClassifierModelLoader
import org.bertspark.transformer.block.BERTFeaturizerBlock
import org.bertspark.transformer.model.TransformerModelLoader
import org.slf4j._


/**
  * Loader for the predictor model
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] trait PredictorLoader  {
  protected[this] val transformerModel: String
  protected[this] val classificationName: String
  protected[this] val subModelNames: Seq[String]
  protected[this] val classifierBlock: ClassificationBlock
  protected[this] val featurizerBlock: BERTFeaturizerBlock

  private[this] val transformerLoader: TransformerModelLoader = new TransformerModelLoader {
    protected[this] val preTrainingBlock: BERTFeaturizerBlock = featurizerBlock
  }

  private[this] val classifierLoaders = subModelNames.map(
    modelName =>
      new ClassifierModelLoader {
        protected[this] val transformerModelName: String = transformerModel
        protected[this] val classificationModel: String = classificationName
        protected[this] val subModelName: String = modelName
        protected[this] val classificationBlock: ClassificationBlock = classifierBlock
      }
  )

  def loadClassifiers: Option[Seq[ZooModel[NDList, NDList]]] = Some(classifierLoaders.flatMap(_.model))
  def loadTransformer: Option[ZooModel[NDList, NDList]] = transformerLoader.model
}


private[bertspark] final object PredictorLoader {
  final private val logger: Logger = LoggerFactory.getLogger("TPredictorModel")
}

