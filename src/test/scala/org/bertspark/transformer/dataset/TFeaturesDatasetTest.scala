package org.bertspark.transformer.dataset

import ai.djl.ndarray.NDManager
import ai.djl.training.util.ProgressBar
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.util.io.SingleS3Dataset
import org.bertspark.Labels._
import org.bertspark.nlp.medical.encodePredictReq
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.convert.ImplicitConversions.`iterable AsScalaIterable`


private[dataset] final class TFeaturesDatasetTest extends AnyFlatSpec {

  it should "Succeed retrieving data from data source" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val ndManager = NDManager.newBaseManager

    val s3Folder = "requests/Cornerstone"
    val maxNumRecords = 64

    val storage = SingleS3Dataset[InternalRequest](s3Folder, encodePredictReq, maxNumRecords)

    val batchSize = 4
    val maxSeqLength = 32
    val maxMasking = 3
    val vocabularyFile = "conf/vocabulary/Cornerstone.csv"

    val bertDatasetConfig = TDatasetConfig(
      batchSize,
      maxSeqLength,
      maxMasking,
      vocabularyFile,
      ctxTxtNSentencesBuilderLbl,
      bertTokenizerLbl,
      true
    )

    val bertDataset = FeaturesDataset(storage, bertDatasetConfig)
    bertDataset.prepare(new ProgressBar)
    val batches = bertDataset.getData(ndManager).toSeq
    println(s"${batches.size} batches")
    println(s"First batch label: ${batches(0).getData.mkString("\n")}")
    println(s"Second batch data: ${batches(1).getData.mkString("\n")}")
  }
}
