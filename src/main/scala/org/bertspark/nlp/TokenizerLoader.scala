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
package org.bertspark.nlp

import ai.djl.modality.nlp.DefaultVocabulary
import java.nio.file.Paths
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames
import org.bertspark.transformer.dataset._
import org.bertspark.util.io.ModelLoader

/**
  * Load for the tokenizer from vocabulary defined in the configuration file
  * @see mlopsConfiguration.preProcessConfig.vocabularyType
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] trait TokenizerLoader extends ModelLoader[DefaultVocabulary] {
  protected[this] val minFrequency: Int

  override lazy val model: Option[DefaultVocabulary] = {
    import org.bertspark.implicits._

    val directory = s"conf/vocabulary${mlopsConfiguration.preProcessConfig.vocabularyType}"
    val localVocabularyFile = s"$directory/${mlopsConfiguration.target}.csv"

    s3ToFs(directory, localVocabularyFile, S3PathNames.s3VocabularyPath)
    val path = Paths.get(localVocabularyFile)

    val _reservedLabels: java.util.Collection[String] = reservedLabels.toSeq
    val defaultVocabulary = DefaultVocabulary.builder
        .optMinFrequency(minFrequency)
        .optReservedTokens(_reservedLabels)
        .addFromTextFile(path)
        .optUnknownToken(unkLabel)
    Some(defaultVocabulary.build())
  }
}
