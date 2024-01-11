package org.bertspark.nlp.augmentation

import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.bertspark.util.io.S3Util
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

private[augmentation] final class RandomSubstituteAugmentationTest extends AnyFlatSpec{

  it should "Succeed augmenting data set with token replacement with [UNK]" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "mlops/TEST/training/TF95"
    val subModelsTrainingSetDS = S3Util.s3ToDataset[SubModelsTrainingSet](s3Folder)
    val labelsSetDS = subModelsTrainingSetDS.flatMap(_.labeledTrainingData.map(_.label)).distinct()
    val minNumRecordsPerLabel: Int = mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
    val randNumRecords = Random.nextInt(minNumRecordsPerLabel*3)+(minNumRecordsPerLabel>>1)

    val recordsFrequencyPerLabel = labelsSetDS.map((_, randNumRecords)).collect()

    val randSubstituteAugmentation = RandomAugmentation(subModelsTrainingSetDS,
      recordsFrequencyPerLabel,
      "randomToken"
    )
    val augSubModelsTrainingSetDS = randSubstituteAugmentation.augment
    println(augSubModelsTrainingSetDS.head.toString)
  }
}
