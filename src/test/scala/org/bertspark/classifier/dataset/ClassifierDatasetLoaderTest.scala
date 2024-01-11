package org.bertspark.classifier.dataset

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.bertspark.util.io.S3Util
import org.scalatest.flatspec.AnyFlatSpec

private[dataset] final class ClassifierDatasetLoaderTest extends AnyFlatSpec {

  it should "Succeed sampling training set for evaluating augmentation techniques" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    import ClassifierDatasetLoader._

    val s3Folder = "mlops/TEST/training/TF95"
    val subModelsTrainingSetDS = S3Util.s3ToDataset[SubModelsTrainingSet](s3Folder)
    val maxNumSubModelsForAugmentation = 5
    val labelRecordFreq = getRecordsFrequencyPerLabel(subModelsTrainingSetDS.limit(maxNumSubModelsForAugmentation))
    val augSubModelsTrainingSetDS = samplingForAugmentation(
      subModelsTrainingSetDS,
      maxNumSubModelsForAugmentation,
      augmentationCriteria)

    val augLabelRecordFreq = getRecordsFrequencyPerLabel(augSubModelsTrainingSetDS)
    println(s"Minimum num records per label: ${mlopsConfiguration.classifyConfig.minNumRecordsPerLabel}")
    println(s"Original training set:\n${labelRecordFreq.map{ case (k, v) => s"$k:$v"}.mkString("  ")}")
    println(s"Augmented training set:\n${augLabelRecordFreq.map{ case (k, v) => s"$k:$v"}.mkString("  ")}")
  }

  it should "Succeed sampling training set for evaluating filter" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    import ClassifierDatasetLoader._

    val s3Folder = "mlops/TEST/training/TF95"
    val subModelsTrainingSetDS = S3Util.s3ToDataset[SubModelsTrainingSet](s3Folder)
    val maxNumSubModelsForAugmentation = 5
    val labelRecordFreq = getRecordsFrequencyPerLabel(subModelsTrainingSetDS.limit(maxNumSubModelsForAugmentation))
    val augSubModelsTrainingSetDS = samplingForAugmentation(
      subModelsTrainingSetDS,
      maxNumSubModelsForAugmentation,
      filterCriteria)

    val augLabelRecordFreq = getRecordsFrequencyPerLabel(augSubModelsTrainingSetDS)
    println(s"Minimum num records per label: ${mlopsConfiguration.classifyConfig.minNumRecordsPerLabel}")
    println(s"Original training set:\n${labelRecordFreq.map{ case (k, v) => s"$k:$v"}.mkString("  ")}")
    println(s"Augmented training set:\n${augLabelRecordFreq.map{ case (k, v) => s"$k:$v"}.mkString("  ")}")
  }
}
