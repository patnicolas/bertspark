package org.bertspark.classifier.model

import org.bertspark.config.ExecutionMode
import org.scalatest.flatspec.AnyFlatSpec


private[classifier] final class ClassifierModelSaverTest extends AnyFlatSpec {
  ExecutionMode.setTest

  it should "Succeed saving a model into S3" in {
    ClassifierModelSaver.saveConfiguration
  }
}


