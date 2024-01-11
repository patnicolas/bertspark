package org.bertspark.util.io

import org.bertspark.config.MlopsConfiguration
import org.bertspark.modeling.SubModelsTaxonomy
import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalFeedback, InternalRequest}
import org.bertspark.nlp.trainingset.{ContextualDocument, SubModelsTrainingSet}
import org.scalatest.flatspec.AnyFlatSpec

private[io] final class S3UtilTest extends AnyFlatSpec {


  it should "Succeed moving S3 files into local" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Source = s"mlops/XLARGE2/training/TF92"
    val fsTargetPath = s"/Users/patricknicolas/s3Data/$s3Source"
    val numRecordsPerFile = 400
    val numRecords = S3Util.s3DatasetToFs[SubModelsTrainingSet](s3Source, fsTargetPath, numRecordsPerFile)
    println(s"Number records: $numRecords")
  }


  ignore should "Succeed transferring S3 file to local file" in {
    val s3Path = "mlops/Cornerstone/models/410/Pre-trained-Bert-0000.params"
    val destFileName = "models/temp/test.params"
    S3Util.cpS3ToFs(
      MlopsConfiguration.mlopsConfiguration.storageConfig.s3Bucket,
      s3Path,
      destFileName)
  }


  ignore should "Succeed transferring S3 file to local file 2" in {
    val s3Path = s"mlops/Cornerstone/vocabulary.csv"
    val targetLocalFile = "output/vocabulary.csv"
    S3Util.s3ToFs(
      targetLocalFile,
      MlopsConfiguration.mlopsConfiguration.storageConfig.s3Bucket,
      s3Path,
      true)
  }

  ignore should "Succeed extracting highest digit contained in a set of keys" in {
    val keys = Seq[String](
      "aaa/bb/cc/key-1",
        "aaa/bb/cc/key-2",
      "aaa/bb/cc/key-3"
    )
    val highestDigit = S3Util.getHighestCount(keys, 1)
    assert(highestDigit == 3)
  }
}
