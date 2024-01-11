package org.bertspark.analytics

import org.bertspark.analytics.Reporting.precisionIcds
import org.bertspark.analytics.ReportingTest.TrainingTokensStats
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.ContextualDocument
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.scalatest.flatspec.AnyFlatSpec

private[analytics] final class ReportingTest extends AnyFlatSpec {

  ignore should "Succeed evaluating the min, max and average number of tokens per request" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val toRemove = Set[String]("xnum")
    val contextualDocStatsDS= S3Util.s3ToDataset[ContextualDocument](
      "mlops/ALL/contextDocument/TFIDF", false, "json"
    )   .limit(36000)
        .map(ctxDocument => {
      val numTextTokens = ctxDocument.text.split(tokenSeparator).size
      TrainingTokensStats(ctxDocument.contextVariables.size, numTextTokens)
    })

    val contextualDocStats = contextualDocStatsDS.collect()
    val maxNumCtxTokens = contextualDocStats.maxBy(_.numCtxTokens).numCtxTokens
    val maxNumTextTokens = contextualDocStats.maxBy(_.numTextTokens).numTextTokens
    val minNumTextTokens = contextualDocStats.minBy(_.numTextTokens).numTextTokens
    val aveNumTextTokens = contextualDocStats.map(_.numTextTokens).sum/contextualDocStats.size
    val sortedDesNumTextTokens = contextualDocStats.map(_.numTextTokens).sortWith(_ > _)
    val top5NumTextTokensIndex = (0.05*contextualDocStats.size).toInt
    val top10NumTextTokensIndex = (0.10*contextualDocStats.size).toInt

    val threshold5NumTextTokens: Int = sortedDesNumTextTokens(top5NumTextTokensIndex)
    val threshold10NumTextTokens: Int = sortedDesNumTextTokens(top10NumTextTokensIndex)
    val statsDump =
      s"""
         |Max num ctx tokens:  $maxNumCtxTokens
         |Max num text tokens: $maxNumTextTokens
         |Min num text tokens: $minNumTextTokens
         |Ave num text tokens: $aveNumTextTokens
         |95% num text tokens: $threshold5NumTextTokens
         |90% num text tokens: $threshold10NumTextTokens
         |""".stripMargin

    println(statsDump)
  }

  ignore should "Succeed applying approximated match" in {
    val prediction1 = "I67.82 26 - G4562"
    val label1 = "I67.82 26 - G4562"
    val isApproxMatch1 = precisionIcds(prediction1, label1)
    assert(isApproxMatch1 == true)

    val prediction2 = "3100F I65.23 - I65.23 - G9637 I65.23"
    val label2 = "3100F R29.818 - 70496 26 R29.818 - R29.818 - G9637 R29.818 - G9637 R29.818"
    val isApproxMatch2 = precisionIcds(prediction2, label2)
    assert(isApproxMatch2 == false)

    val prediction3 = "3100F R29.818"
    val label3 = "3100F R29.818 - 70496 26 R29.818 - R29.818 - G9637 R29.818 - G9637 R29.818"
    val isApproxMatch3 = precisionIcds(prediction3, label3)
    assert(isApproxMatch3 == true)

    val prediction4 = "3100F R29.818"
    val label4 = "3100F R29.818 - 70496 26 R29.818 - R90.82 - G9637 R29.818 - G9637 Z01.89"
    val isApproxMatch4 = precisionIcds(prediction4, label4)
    assert(isApproxMatch4 == true)
  }
}


private[analytics] final object ReportingTest {
  case class TrainingTokensStats(numCtxTokens: Int, numTextTokens: Int)

}