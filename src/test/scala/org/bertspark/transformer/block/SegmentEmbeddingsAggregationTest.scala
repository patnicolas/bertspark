package org.bertspark.transformer.block

import ai.djl.ndarray.{NDList, NDManager}
import org.scalatest.flatspec.AnyFlatSpec


private[block] final class SegmentEmbeddingsAggregationTest extends AnyFlatSpec{

  it should "Succeed aggregating segment embeddings with concatenation" in {
    val ndManager = NDManager.newBaseManager()
    val inputSegment1 = Array[Float](1.0F, 0.5F, 0.5F, 0.0F, 0.1F, 1.0F, 1.0F)
    val ndInputSegment1 = new NDList(ndManager.create(inputSegment1).expandDims(0))

    val inputSegment2 = Array[Float](0.6F, 0.2F, 0.0F, 1.0F, 0.8F, 0.0F, 0.9F)
    val ndInputSegment2 = new NDList(ndManager.create(inputSegment2).expandDims(0))

    val docEmbedding = SegmentEmbeddingAggregation(Array[NDList](ndInputSegment1, ndInputSegment2), true)
    val docEmbeddingValues = docEmbedding.get(0).toFloatArray
    assert(docEmbeddingValues.size == inputSegment1.size + inputSegment2.size)
    println(s"Doc embedding through concatenation: ${docEmbeddingValues.mkString(" ")}")
    assert(docEmbeddingValues(1) == inputSegment1(1))
    assert(docEmbeddingValues(3+inputSegment1.size) == inputSegment2(3))
    ndManager.close()
  }

  it should "Succeed aggregating segment embeddings with summation" in {
    val ndManager = NDManager.newBaseManager()
    val inputSegment1 = Array[Float](1.0F, 0.5F, 0.5F, 0.0F, 0.1F, 1.0F, 1.0F)
    val ndInputSegment1 = new NDList(ndManager.create(inputSegment1))

    val inputSegment2 = Array[Float](0.6F, 0.2F, 0.0F, 1.0F, 0.8F, 0.0F, 0.9F)
    val ndInputSegment2 = new NDList(ndManager.create(inputSegment2))

    val docEmbedding = SegmentEmbeddingAggregation(Array[NDList](ndInputSegment1, ndInputSegment2), false )
    val docEmbeddingValues = docEmbedding.get(0).toFloatArray
    assert(docEmbeddingValues.size == inputSegment1.size)
    println(s"Doc embedding through summation ${docEmbeddingValues.mkString(" ")}")
    assert(docEmbeddingValues(1) == inputSegment1(1) + inputSegment2(1))
    ndManager.close()
  }


}
