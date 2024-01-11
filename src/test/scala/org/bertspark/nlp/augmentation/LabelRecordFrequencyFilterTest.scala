package org.bertspark.nlp.augmentation

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.bertspark.util.io.S3Util
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

private[augmentation] final class LabelRecordFrequencyFilterTest extends AnyFlatSpec {
  import LabelRecordFrequencyFilterTest._

  it should "Succeed apply filter to labels with minimum number of records" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "mlops/TEST/training/TF95"
    val subModelsTrainingSetDS = S3Util.s3ToDataset[SubModelsTrainingSet](s3Folder)
    val labelsSetDS = subModelsTrainingSetDS.flatMap(_.labeledTrainingData.map(_.label)).distinct()

    val minNumRecordsPerLabel: Int = mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
    val recordsFrequencyPerLabel = labelsSetDS.map((_, randomCount(minNumRecordsPerLabel))).collect()

    val labelsFilter = LabelRecordFrequencyFilter(subModelsTrainingSetDS, recordsFrequencyPerLabel)
    val filteredTrainingSetDS = labelsFilter.augment
    println(filteredTrainingSetDS.head.toString)
  }
}


private[augmentation] object LabelRecordFrequencyFilterTest {
  def randomCount(minNumRecordsPerLabel: Int): Int = Random.nextInt(minNumRecordsPerLabel) +(minNumRecordsPerLabel>>1)
}
