package org.bertspark.modeling

import org.bertspark.modeling.SubModelOperations.SubModelLabelsTrainingStats
import org.bertspark.nlp.augmentation.randomAugUNK
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.bertspark.util.io.S3Util
import org.scalatest.flatspec.AnyFlatSpec


private[modeling] final class SubModelOperationsTest extends AnyFlatSpec {

  ignore should "Succeed filtering sub model training set" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = s"mlops/TEST/training/TF92"
    val rawDS = S3Util.s3ToDataset[SubModelsTrainingSet](s3Folder)
    val rawRecords = rawDS.collect()
    println(s"\n----Original ----\n${rawRecords.mkString("\n\n")}\n-------------------\n")

    val minNumRecordsPerLabel = 2
    val subModelOperations = SubModelOperations(rawDS, minNumRecordsPerLabel)
    val filteredDS = subModelOperations.process
    if(!filteredDS.isEmpty) {
      val filteredRecords = filteredDS.collect()
      println(s"\n----Filtered----\n${filteredRecords.mkString("\n\n")}\n-------------------\n")
    }
  }


  it should "Succeed augmenting training data set" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = s"mlops/TEST/training/TF92"
    val rawDS = S3Util.s3ToDataset[SubModelsTrainingSet](s3Folder)
    val rawRecords = rawDS.collect()
    println(s"\n----Original ----\n${rawRecords.mkString("\n\n")}\n-------------------\n")

    val subModelOperations = SubModelOperations(rawDS)

    val augmentedDS = subModelOperations.augment(randomAugUNK)
    if(!augmentedDS.isEmpty) {
      val augmentedRecords = augmentedDS.collect()
      println(s"\n---- Augmented ----\n${augmentedRecords.mkString("\n\n")}\n-------------------\n")
    }
  }


  ignore should "Succeed extracting frequency statistics for sub model and labels" in {
    val subModelLabelsTrainingStats = SubModelLabelsTrainingStats()
    subModelLabelsTrainingStats.save()
  }
}
