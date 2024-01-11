package org.bertspark.hpo.ga

import org.bertspark.config.ExecutionMode
import org.scalatest.flatspec.AnyFlatSpec

private[ga] final class GASearchTest extends AnyFlatSpec {

  it should "Succeed triggering GA search" in {
    import org.bertspark.implicits._
    ExecutionMode.setHpo
    val gaSearch = new GASearch
    gaSearch()
  }
}
