package org.bertspark.util

import org.scalatest.flatspec.AnyFlatSpec

private[util] final class TProgressTest extends AnyFlatSpec {

  it should "Succeed instantiating progress bar" in {
    val tProgress = new TProgress[Int, Int] {
      protected[this] val maxValue: Int = 12
      protected[this] val progress: Int => Int = (n: Int) => (n*100.0/maxValue).floor.toInt
    }
    tProgress.show(3, "Progress")
  }
}
