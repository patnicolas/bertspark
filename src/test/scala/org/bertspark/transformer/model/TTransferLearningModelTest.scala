package org.bertspark.transformer.model

import ai.djl.training.util.ProgressBar
import org.bertspark.config.MlopsConfiguration.{mlopsConfiguration, vocabulary}
import org.bertspark.modeling.TrainingContext
import org.bertspark.nlp.medical.noContextualDocumentEncoding
import org.bertspark.nlp.trainingset.ContextualDocument
import org.bertspark.transformer.block.{BERTFeaturizerBlock, PretrainingModule}
import org.bertspark.transformer.config.BERTConfig
import org.bertspark.transformer.dataset.{PretrainingDataset, TDatasetConfig}
import org.bertspark.util.io.SingleS3Dataset
import org.scalatest.flatspec.AnyFlatSpec


private[bertspark] final class TTransferLearningModelTest extends AnyFlatSpec {

  it should "Succeed apply transfer learning to an existing model" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val loader = new TransformerModelLoader {
      override protected[this] val preTrainingBlock: BERTFeaturizerBlock = BERTFeaturizerBlock()
    }
    val transferLearningModel =  new TTransferLearningModel(
      BERTConfig.getEmbeddingsSize(mlopsConfiguration.preTrainConfig.transformer),
      loader.model.get)

    System.setProperty("collect-memory", "true")
    val s3Folder = "mlops/CornerstoneTest/contextDocument/AMA"
    val maxNumRecords = 32

    val storage = SingleS3Dataset[ContextualDocument](s3Folder, noContextualDocumentEncoding, maxNumRecords)
    val bertDatasetConfig = TDatasetConfig(true)
    val contextualDocuments = storage.inputDataset.collect()
    val bertDataset = PretrainingDataset[ContextualDocument](storage, bertDatasetConfig)
    bertDataset.prepare(new ProgressBar)
    val modelPath = transferLearningModel.train(TrainingContext(), bertDataset, null, "")
    println(modelPath)
  }
}
