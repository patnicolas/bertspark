package org.bertspark.transformer.block

import ai.djl.ndarray.{NDList, NDManager}
import ai.djl.training.ParameterStore
import ai.djl.Device
import org.bertspark.util.NDUtil
import org.bertspark.Labels.microBertLbl
import org.bertspark.dl
import org.bertspark.transformer.block.BERTFeaturizerBlockTest.MyBERTFeaturizerBlock
import org.bertspark.transformer.config.BERTConfig
import org.scalatest.flatspec.AnyFlatSpec

private[block] final class BERTFeaturizerBlockTest extends AnyFlatSpec{

  it should "Succeed extracting embedding from a batch input" in {
    val featuresEmbeddings1 = Array[Array[Int]](
      Array[Int](1, 8, 9, 6, 1),
      Array[Int](2, 9, 10, 7, 2),
      Array[Int](3, 10, 11, 8, 3)
    )
    val featuresEmbeddings2 = featuresEmbeddings1.map(_.map(- _))
    val ndManager = NDManager.newBaseManager()
    val ndSegmentEmbedding1 = ndManager.create(featuresEmbeddings1)
    val ndSegmentEmbedding2 = ndManager.create(featuresEmbeddings2)
    val expanded1 = new NDList(ndSegmentEmbedding1)
    val expanded2 = new NDList(ndSegmentEmbedding2)
    val batchified = NDUtil.batchify(Array[NDList](expanded1, expanded2))

    val segmentNDEmbeddings = BERTFeaturizerBlock.getSegmentEmbeddings(batchified.get(0), ndManager)
    segmentNDEmbeddings.foreach(
      segmentNDEmbedding => {
        val tokenIds = segmentNDEmbedding.get(0).toIntArray
        val typeIds = segmentNDEmbedding.get(1).toIntArray
        val maskId = segmentNDEmbedding.get(2).toIntArray
        println(s"TokenIds: ${tokenIds.mkString(" ")}")
        println(s"TypeIds: ${typeIds.mkString(" ")}")
        println(s"MaskIds: ${maskId.mkString(" ")}")
      }
    )
    ndManager.close()
  }



  it should "Succeed forward computation for a segment embedding" in {
    val ndManager = NDManager.newBaseManager()

    val featuresEmbeddings1 = Array[Array[Float]](
      Array[Float](1, 8, 9, 6, 1),
      Array[Float](2, 9, 10, 7, 2),
      Array[Float](3, 10, 11, 8, 3)
    )
    val embeddings = featuresEmbeddings1.map(vec => ndManager.create(vec).expandDims(0))

    val bertConfig = BERTConfig(dl.pretrainedBertLbl, microBertLbl, 13456)
    val myBERTFeaturizerBlock = new MyBERTFeaturizerBlock(bertConfig)
    val parameterStore = new ParameterStore(ndManager, false)
    parameterStore.setParameterServer(null, Array[Device](Device.of("cpu", 0)))
    parameterStore.updateAllParameters()
    parameterStore.sync()

    val ndSegmentEmbedding = myBERTFeaturizerBlock.forwardSegment(
      ndManager,
      parameterStore,
      new NDList(embeddings:_*),
      false)
    val values = ndSegmentEmbedding.get(0).toFloatArray
    ndManager.close()
  }
}



private[block] final object BERTFeaturizerBlockTest {

  final class MyBERTFeaturizerBlock(bertConfig: BERTConfig) extends BERTFeaturizerBlock(bertConfig) {
    override def forwardSegment(
      ndChildManager: NDManager,
      parameterStore: ParameterStore,
      segmentNdEmbeddings: NDList,
      training: Boolean): NDList =  super.forwardSegment(ndChildManager, parameterStore, segmentNdEmbeddings, training)
  }

}

