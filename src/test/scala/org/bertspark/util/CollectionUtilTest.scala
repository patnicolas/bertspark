package org.bertspark.util

import org.scalatest.flatspec.AnyFlatSpec


private[util] final class CollectionUtilTest extends AnyFlatSpec {

  it should "Succeed splitting an array into multiple segments according to a predicate" in {
    val predicate = (w: String) => w == "STOP"
    val input = Array[String](
      "John", "arrived", "late", "STOP", "but", "was", "unable", "to", "complete", "STOP", "Well", "Done"
    )

    val result = CollectionUtil.split[String](input, predicate)
    println(result.map(_.mkString(" ")).mkString("\n"))
  }
}
