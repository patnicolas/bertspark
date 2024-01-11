package org.bertspark.classifier.config

import ai.djl.basicmodelzoo.basic.Mlp
import ai.djl.ndarray.types.Shape
import ai.djl.ndarray.NDList
import ai.djl.nn._
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.classifier.block.ClassificationBlock.batchifierLambda

/**
 * Singleton wrapper for configuration of DL classification models
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final object MlopsClassifierConfig {
  type ClassifierBlocks = Array[Int] => Array[(String, Block)]

  def getClassifierBlocks: ClassifierBlocks = mlopsConfiguration.classifyConfig.dlModel match {
    case "MLP2" => mlp2
    case _ =>
      throw new UnsupportedOperationException(s"DL model ${mlopsConfiguration.classifyConfig.dlModel} is not supported")
  }

  final private val mlp2: ClassifierBlocks = {
    (args: Array[Int]) => {
      Array[(String, Block)](
        ("MLP2-Batchifier",
            new LambdaBlock(batchifierLambda) {
              override def getOutputShapes(inputShapes: Array[Shape]): Array[Shape] = {
                val outputShape = new Shape(inputShapes.size, inputShapes.head.size)
                Array[Shape](outputShape)
              }
            }
        ),
        ("MLP2-MLP",
            {
              val activation: java.util.function.Function[NDList, NDList] = ai.djl.nn.Activation.relu
              val hiddenLayers = mlopsConfiguration.classifyConfig.dlLayout
              val inputLayerSize = mlopsConfiguration.getPredictionOutputSize
              new Mlp(inputLayerSize, args(0), hiddenLayers, activation)
            }
        )
      )
    }
  }
}
