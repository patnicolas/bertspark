package org.bertspark.analytics

import org.bertspark.analytics.Categorization.DistributionByKey
import org.bertspark.util.io.LocalFileUtil
import org.scalatest.flatspec.AnyFlatSpec


private[analytics] final class CategorizationTest extends AnyFlatSpec {

  it should "Succeed scanning feedback labels" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    val categorization = new Categorization

    val emrLabelPairDS = categorization.loadFeedbackByEmr.map { case (emr, label) => s"$emr,$label"}
    println(emrLabelPairDS.collect().sortWith(_ < _ ).mkString("\n"))
  }

  ignore should "Succeed categorizing by labels" in {
    import org.bertspark.implicits._
    val categorization = new Categorization
    val labelsDistribution: Seq[DistributionByKey] = categorization
        .getNotesCountPerLabels
        .collect()
        .sortWith(_.count > _.count)
    LocalFileUtil.Save.local("output/groupByLabels", labelsDistribution.mkString("\n"))
  }

  ignore should "Succeed create distribution by emr, labels and emr/labels" in {
    import org.bertspark.implicits._

    val categorization = new Categorization
    categorization.run
  }

}
