package org.bertspark.predictor.model

import ai.djl.metric.Metrics
import ai.djl.ndarray.NDManager
import ai.djl.training.listener.MemoryTrainingListener
import org.bertspark.classifier.block.ClassificationBlock
import org.bertspark.transformer.representation.PretrainingInference
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random


private[bertspark] final class TPredictorModelTest extends AnyFlatSpec{
  import TPredictorModelTest._

  System.setProperty("collect-memory", "true")

  ignore should "Succeed releasing ND resources" in {
    def compute(input: Array[Float], ndManager: NDManager, metricName: String): Float = {
      val manager = ndManager.newSubManager()
      val metrics = new Metrics()
      MemoryTrainingListener.collectMemoryInfo(metrics)
      val originalValue = metrics.latestMetric(metricName).getValue.longValue

      val ndInput = ndManager.create(input)
      val ndMean = ndInput.mean()
      val mean = ndMean.toFloatArray.head
      collectMetric(metrics, originalValue, metricName)

      ndMean.close()
      collectMetric(metrics, originalValue, metricName)
      ndInput.close()
      collectMetric(metrics, originalValue, metricName)
      ndManager.close()
      collectMetric(metrics, originalValue, metricName)
      MemoryTrainingListener.dumpMemoryInfo(metrics, "output")
      mean
    }

    val ndManager = NDManager.newBaseManager()
    val input = Array.fill(10000)(Random.nextFloat())
    val metricName = "rss"

    compute(input, ndManager, metricName)
  }

  ignore should "Succeed loading the classification model from S3" in {
    val preTrainingModel = "314"
    val classificationModel = "C-314"
    val subModel = "70450_26"
    val fsModelUrl = PredictorModel.localModelURLFromS3(preTrainingModel, classificationModel, subModel)
    println(s"fsModelUrl: $fsModelUrl")
  }

  it should "Succeed loading classification model for a given sub-model" in {
    import org.bertspark.implicits._
    val preTrainingModel = "314"
    val classificationModel = "C-314"
    val subModelName = "70450_26"

    val preTrainingInference = PretrainingInference()
    val classificationBlock = new ClassificationBlock(20)
    val predictorModel = PredictorModel(
      preTrainingInference,
      classificationBlock,
      preTrainingModel,
      classificationModel,
      subModelName
    )

    val subModel = predictorModel.getClassificationSubModel
    subModel.foreach(_.getArtifactNames.foreach(println(_)))
  }

}


private[bertspark] final object TPredictorModelTest {
  private def collectMetric(metrics: Metrics, startValue: Long, metricName: String): Unit = {
    MemoryTrainingListener.collectMemoryInfo(metrics)
    val newValue = metrics.latestMetric(metricName).getValue.longValue
    println(s"$metricName: ${(newValue - startValue)/1024.0} KB")
  }
}
