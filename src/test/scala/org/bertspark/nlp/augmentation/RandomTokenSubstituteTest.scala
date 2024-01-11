package org.bertspark.nlp.augmentation

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.bertspark.util.io.S3Util
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random


private[augmentation] final class RandomTokenSubstituteTest extends AnyFlatSpec {
  import RandomTokenSubstituteTest._

  it should "Succeed augmenting data set with token replacement with [UNK]" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "mlops/TEST/training/TF95"
    val subModelsTrainingSetDS = S3Util.s3ToDataset[SubModelsTrainingSet](s3Folder)
    val labelsSetDS = subModelsTrainingSetDS.flatMap(_.labeledTrainingData.map(_.label)).distinct()

    val minNumRecordsPerLabel: Int = mlopsConfiguration.classifyConfig.minNumRecordsPerLabel

    val recordsFrequencyPerLabel = labelsSetDS.map((_, randomCount(minNumRecordsPerLabel))).collect()
    val randomSubstituteAugmentation = RandomAugmentation(
      subModelsTrainingSetDS,
      recordsFrequencyPerLabel,
      "randomUNK")

    val augSubModelsTrainingSetDS = randomSubstituteAugmentation.augment
    println(augSubModelsTrainingSetDS.head.toString)
  }
}




private[augmentation] object RandomTokenSubstituteTest {
  def randomCount(minNumRecordsPerLabel: Int): Int = {
    val slice = minNumRecordsPerLabel >> 1
    Random.nextInt(minNumRecordsPerLabel) + slice
  }
}

