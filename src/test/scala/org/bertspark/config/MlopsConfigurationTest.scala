package org.bertspark.config

import org.bertspark.Labels.configValidation
import org.scalatest.flatspec.AnyFlatSpec

private[bertspark] final class MlopsConfigurationTest extends AnyFlatSpec {
  import MlopsConfiguration._



  ignore should "Succeed extracting extension from vocabulary file name" in {
    val extension = FsPathNames.getVocabularyExtension
    println(s"Extension for ${mlopsConfiguration.preProcessConfig.vocabularyType}:  $extension")
  }

  ignore should "Succeed loading Service configuration file" in {
    println(mlopsConfiguration.toString)

    configValidation
    MlopsConfiguration.mlopsConfiguration.isValid
  }

  it should "Succeed describing vocabulary" in {
    (0 until vocabulary.size().toInt).foreach(
      index => println(vocabulary.getToken(index))
    )
  }
}
