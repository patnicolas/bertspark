package org.bertspark.config

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.scalatest.flatspec.AnyFlatSpec



private[config] final class DynamicConfigurationTest extends AnyFlatSpec {

  it should "Increment the classifier model id"  in {
    import MlopsConfiguration._, DynamicConfiguration._
    val oldModelId = mlopsConfiguration.classifyConfig.modelId
    ++ (false)
    val newModelId = mlopsConfiguration.classifyConfig.modelId
    assert(newModelId.toInt == oldModelId.toInt + 1)
  }

  it should "Increment the pre-training model id"  in {
    import MlopsConfiguration._, DynamicConfiguration._
    val oldRunId = mlopsConfiguration.runId
    ++ (true)
    val newRunId = mlopsConfiguration.runId
    assert(newRunId.toInt == oldRunId.toInt + 1)
  }

  it should "Initialize the run id and model id for configuration" in {
    val runId = "30098"
    val modelId = "8931"
    DynamicConfiguration(runId, modelId)

    assert(mlopsConfiguration.runId == runId)
    assert(mlopsConfiguration.classifyConfig.modelId == modelId)
  }
}
