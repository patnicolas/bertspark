package org.bertspark.transformer.dataset

import ai.djl.modality.nlp.Vocabulary
import ai.djl.ndarray.NDManager
import ai.djl.training.util.ProgressBar
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.util.io.SingleS3Dataset
import org.bertspark.config.MlopsConfiguration.vocabulary
import org.bertspark.nlp.medical.{encodePredictReq, noContextualDocumentEncoding}
import org.bertspark.nlp.trainingset.ContextualDocument
import org.bertspark.transformer.dataset.PretrainingDatasetTest.displayVocab
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.convert.ImplicitConversions.`iterable AsScalaIterable`


private[dataset] final class PretrainingDatasetTest extends AnyFlatSpec {
  import org.bertspark.Labels._

  it should "Succeed creating a dataset for BERT model" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "requests/40/7/Cornerstone"
    val maxNumRecords = 32

    val storage = SingleS3Dataset[ContextualDocument](s3Folder, noContextualDocumentEncoding, maxNumRecords)

    val batchSize = 8
    val maxSeqLength = 48
    val maxMasking = 6
    val vocabularyFile = "conf/vocabulary/Cornerstone.csv"
    val bertDatasetConfig = TDatasetConfig(
      batchSize,
      maxSeqLength,
      maxMasking,
      vocabularyFile,
      ctxTxtNSentencesBuilderLbl,
      bertTokenizerLbl,
      true)

    val bertDataset = PretrainingDataset[ContextualDocument](storage, bertDatasetConfig)
    bertDataset.prepare(new ProgressBar)

    println(s"First few tokens:\n${displayVocab(vocabulary, 64)}")
  }

  ignore should "Succeed retrieving data" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val ndManager = NDManager.newBaseManager

    val s3Folder = "requests/40/7/Cornerstone"
    val maxNumRecords = 256

    val storage = SingleS3Dataset[InternalRequest](s3Folder, encodePredictReq, maxNumRecords)

    val batchSize = 6
    val maxSeqLength = 128
    val maxMasking = 2
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
    val bertDataset = PretrainingDataset(storage, bertDatasetConfig)
    bertDataset.prepare(new ProgressBar)
    val batches = bertDataset.getData(ndManager).toList
    println(s"${batches.size} batches")
    println(s"First batch label: ${batches(0).getLabels.mkString("\n")}")
    println(s"First batch data: ${batches(0).getData.mkString("\n")}")
  }
}



private[bertspark] final object PretrainingDatasetTest {
  def displayVocab(vocab: Vocabulary, limit: Int): String =
    (0 until limit).map(vocab.getToken(_)).mkString("\n")
}

