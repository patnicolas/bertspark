package org.bertspark.modeling

import org.scalatest.flatspec.AnyFlatSpec

private[bertspark] final class BaseTrainingListenerTest extends AnyFlatSpec{

  it should "Succeed loading training plots" in {
    BaseTrainingListener.createPlots("Pre-trained-Bert")
  }
}
