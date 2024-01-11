package org.bertspark.classifier.block

import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.BatchNorm.batchNorm
import ai.djl.training.ParameterStore
import ai.djl.translate.StackBatchifier
import ai.djl.util.PairList
import org.bertspark.dl.block.BaseNetBlock
import org.scalatest.flatspec.AnyFlatSpec


private[block] final class ClassificationBlockTest extends AnyFlatSpec {
  final private val batchSize = 4
  final private val embeddingSize = 8

  it should "Succeed evaluating batchFlatten block for batch processing" in {
    val ndManager = NDManager.newBaseManager()
    val input = Array.fill(batchSize)(Array.fill(embeddingSize)(0.5F))
    val ndInput = ndManager.create(input)
    val ndList = new NDList(ndInput)
    println(s"Input to BatchFlatten: ${ndList.getShapes().mkString(" ")}")
    val ndOutput = Blocks.batchFlatten(ndInput)
    println(s"Output from BatchFlatten: ${ndOutput.getShape()}  => ${ndOutput.toFloatArray.mkString(" ")}")

    ndManager.close()
  }

  it should "Succeed evaluating Linear block" in {
    val ndManager = NDManager.newBaseManager()
    val input = Array.fill(batchSize)(Array.fill(embeddingSize)(0.5F))
    val ndInput = ndManager.create(input)
    val ndList = new NDList(ndInput)
    println(s"Input to Linear block: ${ndList.getShapes().mkString(" ")} =>\n${
      input.map(_.mkString(" ")).mkString("\n")
    }")
    val linBlock = Linear.builder().setUnits(10).build()
    val ps: ParameterStore = new ParameterStore()
    val ndOutput = linBlock.forward(ps, ndList, true)
    println(s"Output from Linear: ${ndOutput.getShapes().mkString(" ")} =>\nNum output: ${
      ndOutput.get(0).size()
    }\n${ndOutput.get(0).toFloatArray.mkString("\n")}")

    ndManager.close()
  }

  it should "Succeed applying stack batchifier" in {
    import ClassificationBlockTest._

    val stackBatchifier = new StackBatchifier
    val inputToBatchifier = Array[NDList](
      singleNDList(embeddingSize, 0.25F),
      singleNDList(embeddingSize, 0.89F)
    )
    println(s"Input to Batchifier:\n${inputToBatchifier.map(_.get(0).toFloatArray.mkString(" ")).mkString("\n")}")
    val batchifiedNDList = stackBatchifier.batchify(inputToBatchifier)
    println(s"Output from Batchifier: ${batchifiedNDList.getShapes().mkString(" ")} =>\nNum output: ${
      batchifiedNDList.get(0).size()
    }\n${batchifiedNDList.get(0).toFloatArray.mkString(" ")}")
    val unBatchiedNDList = stackBatchifier.unbatchify(batchifiedNDList)
    println(s"Output from UnBatchifier: ${
      unBatchiedNDList.map(_.getShapes().mkString(" ")).mkString("\n")
    }\n${unBatchiedNDList.map(_.get(0).toFloatArray.mkString("    ")).mkString("\n")}")

  }

  ignore should "Succeed evaluating Batch norm block" in {
    val ndManager = NDManager.newBaseManager()
    val input = Array.fill(4)(Array.fill(256)(0.5F))
    val ndInput = ndManager.create(input)
    val ndList = new NDList(ndInput)
    println(ndList.getShapes().mkString(" "))
    val runningMean = Array.fill(256)(0.2F)
    val runningVar = Array.fill(256)(0.1F)
    val ndOutput = batchNorm(ndInput, ndManager.create(runningMean), ndManager.create(runningVar))
    println(ndOutput.getShapes())

    ndManager.close()
  }
}


private[block] final object ClassificationBlockTest   {

  def singleNDList(embeddingSize: Int, value: Float): NDList = {
    val ndManager = NDManager.newBaseManager()
    val input = Array.fill(embeddingSize)(value)
    val ndInput = ndManager.create(input)
    val ndList = new NDList(ndInput)

    ndManager.close()
    ndList
  }

  final class TestBlock extends BaseNetBlock {

   add("BERT-decoder-Flatten", Blocks.batchFlattenBlock())
    /**
     * This method delegates processing to the block that actually implements the recursive
     * initialization of child block
     * @param ndManager Reference to the ND array manager
     * @param dataType data type (Default Float 32)
     * @param shapes Shape for the 4 embedding (batch size x embedding size)
     */
    override def initializeChildBlocks(ndManager: NDManager, dataType: DataType, shapes: Shape*): Unit =
      super.initializeChildBlocks(ndManager, dataType, shapes:_*)

    override protected def forwardInternal(
      parameterStore: ParameterStore,
      inputNDList: NDList,
      training : Boolean,
      params: PairList[String, java.lang.Object]): NDList =
      sequentialBlock.forward(parameterStore, inputNDList, training,  params)

    private def add(name: String, block: Block): Unit = {
      sequentialBlock.add(block)
      addChildBlock(name, block)
    }
  }
}
