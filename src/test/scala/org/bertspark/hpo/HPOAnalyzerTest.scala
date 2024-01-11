package org.bertspark.hpo

import org.scalatest.flatspec.AnyFlatSpec


private[hpo] final class HPOAnalyzerTest extends AnyFlatSpec {

  it should "Succeed load data for Classifier HPO analysis" in {
   // val hpoAnalyzer1 = new HPOAnalyzer(false, "169", "1", false)
   // hpoAnalyzer1.analyze

    val hpoAnalyzer2 = new HPOAnalyzer(false, "205", "2", false)
    hpoAnalyzer2.analyze
  }
}
