package org.bertspark.config

import org.scalatest.flatspec.AnyFlatSpec


private[config] final class ExecutionModeTest extends AnyFlatSpec {

  it should "Succeed setting and retrieving execution modes" in {
    ExecutionMode.setPretraining
    ExecutionMode.setSimilarity
    println(ExecutionMode.toString)

    assert(ExecutionMode.isPretraining == true)
    assert(ExecutionMode.isClassifier == false)
    assert(ExecutionMode.isSimilarity == true)
  }
}
