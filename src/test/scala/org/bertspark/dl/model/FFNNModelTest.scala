package org.bertspark.dl.model

import ai.djl.ndarray._
import ai.djl.ndarray.types.Shape
import ai.djl.nn.BlockList
import ai.djl.Model
import ai.djl.translate.NoopTranslator
import org.bertspark.dl.block.BaseHiddenLayer
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random



private[model] final class FFNNModelTest extends AnyFlatSpec {

  ignore should "Succeed configuring a RBM model" in {
    val ndManager = NDManager.newBaseManager()
    val numFeatures = 32
    val hiddenLayerConfig = Seq[BaseHiddenLayer](
      (16, "relu"),
      (6, "relu")
      )
    val rbmBlock = FFNNModel(numFeatures, hiddenLayerConfig, 2)
    rbmBlock.initialize(ndManager, new Shape(1, numFeatures))

    val blockList: BlockList = rbmBlock.getChildren
    val blockIterator = blockList.iterator()

    while(blockIterator.hasNext) {
      val block = blockIterator.next
      println(s"${block.getKey}: ${block.getValue}")
    }
    ndManager.close()
  }
}
