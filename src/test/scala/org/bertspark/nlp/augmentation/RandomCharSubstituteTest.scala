package org.bertspark.nlp.augmentation

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.bertspark.util.io.S3Util
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random


private[augmentation] final class RandomCharSubstituteTest extends AnyFlatSpec {
  import RandomCharSubstituteTest._

  it should "Succeed augmenting data set with token replacement with random character" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "mlops/TEST/training/TF95"
    val subModelsTrainingSetDS = S3Util.s3ToDataset[SubModelsTrainingSet](s3Folder)
    val labelsSet = subModelsTrainingSetDS.flatMap(_.labeledTrainingData.map(_.label))

    val minNumRecordsPerLabel: Int = mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
    val recordsFrequencyPerLabel = labelsSet.map((_, randomCount(minNumRecordsPerLabel))).collect()

    val nonLabelReplacingAugmentation = RandomAugmentation(
      subModelsTrainingSetDS,
      recordsFrequencyPerLabel,
      "randomChar")

    val augSubModelsTrainingSetDS = nonLabelReplacingAugmentation.augment
    println(augSubModelsTrainingSetDS.head.toString)
  }
}

private[augmentation] object RandomCharSubstituteTest {
  def randomCount(minNumRecordsPerLabel: Int): Int = {
    val slice = minNumRecordsPerLabel >> 1
    Random.nextInt (slice) + slice
  }
}
