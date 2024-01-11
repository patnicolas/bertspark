package org.bertspark.transformer.model


import ai.djl.training.util.ProgressBar
import org.bertspark._
import org.bertspark.util.io.SingleS3Dataset
import org.bertspark.config.MlopsConfiguration.vocabulary
import org.bertspark.modeling.TrainingContext
import org.bertspark.nlp.medical.noContextualDocumentEncoding
import org.bertspark.transformer.block.PretrainingModule
import org.bertspark.transformer.config.BERTConfig
import org.bertspark.transformer.dataset.{PretrainingDataset, TDatasetConfig}
import org.bertspark.nlp.trainingset.ContextualDocument
import org.scalatest.flatspec.AnyFlatSpec


private[model] final class TPretrainingModelTest extends AnyFlatSpec {

  ignore should "Succeed training a BERT model given a training set" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    System.setProperty("collect-memory", "true")
    val s3Folder = "requests/Cornerstone"
    val maxNumRecords = 32

    val storage = SingleS3Dataset[ContextualDocument](s3Folder, noContextualDocumentEncoding, maxNumRecords)
    val embeddingsSize = 512
    val bertDatasetConfig = TDatasetConfig(true)
    val bertDataset = PretrainingDataset[ContextualDocument](storage, bertDatasetConfig)
    bertDataset.prepare(new ProgressBar)
    val modelName = "Pretrained-Bert"
    val trainingContext = TrainingContext()
    val bertConfig = BERTConfig("BERT", "BERT-micro", vocabulary.size)
    val pretrainingModule = PretrainingModule(bertConfig)
    val bertModel = new TPretrainingModel(pretrainingModule, embeddingsSize, modelName)

    try {
      bertModel.train(trainingContext, bertDataset, null, "")
    }
    catch {
      case e: DLException =>
        org.bertspark.printStackTrace(e)
        println(s"Error => ${e.getMessage}")
    }
  }
}