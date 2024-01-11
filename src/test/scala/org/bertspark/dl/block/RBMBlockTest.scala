package org.bertspark.dl.block

import ai.djl.ndarray._
import ai.djl.ndarray.types.Shape
import ai.djl.nn.BlockList
import ai.djl.training.initializer.NormalInitializer
import org.scalatest.flatspec.AnyFlatSpec


private[block] final class RBMBlockTest extends AnyFlatSpec {

  ignore should "Succeed configuring a RBM model" in {
    val ndManager = NDManager.newBaseManager()
    val rbmBlock = RBMBlock((16, "relu"))
    rbmBlock.initialize(ndManager, new Shape(1, 32))

    val blockList: BlockList = rbmBlock.getChildren
    val blockIterator = blockList.iterator()

    while(blockIterator.hasNext) {
      val block = blockIterator.next
      println(s"${block.getKey}: ${block.getValue}")
    }
    ndManager.close()
  }
}
