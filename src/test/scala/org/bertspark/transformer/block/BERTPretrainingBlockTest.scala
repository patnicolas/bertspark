package org.bertspark.transformer.block

import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.types.Shape
import ai.djl.nn.transformer.BertPretrainingLoss
import ai.djl.training.Trainer
import ai.djl.training.initializer.NormalInitializer
import java.nio.file.Paths
import org.bertspark.config.FsPathNames
import org.bertspark.dl.model.NeuralModel
import org.bertspark.getPretrainingModelPath
import org.bertspark.transformer.config.BERTConfig
import org.scalatest.flatspec.AnyFlatSpec

private[block] final class BERTPretrainingBlockTest extends AnyFlatSpec{

  it should "Succeed loading then saving a BERT pretraining block in local directory" in {
    import org.bertspark.config.MlopsConfiguration._
    val modelName = FsPathNames.getModelName
    val transformerModel: String = mlopsConfiguration.getTransformer
    val bertConfig = BERTConfig("BERT-1", transformerModel, 13451)
    val bertPreTrainingBlock = CustomPretrainingBlock(bertConfig)

    val model = Model.newInstance(modelName)
    model.setBlock(bertPreTrainingBlock)
    model.load(Paths.get(getPretrainingModelPath), modelName)

    val trainingCtx = NeuralModel.buildTrainingContext(
      new NormalInitializer(),
      new BertPretrainingLoss(),
      "Pretraining")
    val trainer: Trainer = model.newTrainer(trainingCtx.getDefaultTrainingConfig)
    trainer.setMetrics(new Metrics())

    val inputShape = new Shape(mlopsConfiguration.getMinSeqLength, mlopsConfiguration.getEmbeddingsSize)
    trainer.initialize(inputShape, inputShape, inputShape, inputShape)
    println(model.toString)
    val inputPairs = model.describeInput()
    println(s"Artifacts: ${model.getArtifactNames.mkString(" ")}\nInput pairs $inputPairs")
    model.save(Paths.get(getPretrainingModelPath), modelName)

    val model3 = Model.newInstance(modelName)
    model3.setBlock(bertPreTrainingBlock)
    model3.load(Paths.get(getPretrainingModelPath), modelName)
    println(model3.toString)
  }
}

