package org.bertspark.dl.block

import ai.djl.ndarray._
import ai.djl.nn.BlockList
import ai.djl.translate.NoopTranslator
import ai.djl.Model
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

private[block] final class MLPBlockTest extends AnyFlatSpec {
/*
  it should "Succeed configuring a RBM model" in {
    val ndManager = NDManager.newBaseManager()
    val numFeatures = 32
    val x = NDArrayScala.floatConvert(ndManager, Array.fill(numFeatures)(Random.nextFloat()))

    val mlpBlock = new MLPBlock(numFeatures, (16, "relu"), 2)
    mlpBlock.initialize(ndManager, x.getShape())

    val blockList: BlockList = mlpBlock.getChildren
    val blockIterator = blockList.iterator()

    while(blockIterator.hasNext) {
      val block = blockIterator.next
      println(s"${block.getKey}: ${block.getValue}")
    }
    ndManager.close()
  }


  it should "Succeed forward a RBM model" in {
    val ndManager = NDManager.newBaseManager()
    val numFeatures = 32
    val x = NDArrayScala.floatConvert(ndManager, Array.fill(numFeatures)(Random.nextFloat()))
    val mlpBlock = new MLPBlock(numFeatures,    (16,"relu"), 2)
    mlpBlock.initialize(ndManager, x.getShape())

    val model =  Model.newInstance("MLP")
    model.setBlock(mlpBlock)
    val inputNDList = new NDList(x)
    val translator = new NoopTranslator
    val predictor = model.newPredictor(translator)
    val result = predictor.predict(inputNDList)
    println(result.toString)
    ndManager.close
  }

 */
}
