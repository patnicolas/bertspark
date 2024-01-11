package org.bertspark.transformer.dataset

import ai.djl.ndarray._

/**
 * Generic batch of instances (Embedding components)
 * {{{
 *   instances Are embedding
 * }}}
 *
 * @tparam T Type of Instances in a batch
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] trait TInstancesBatch[T, U] {
self =>
  protected[this] val instances: Array[T]
  protected[this] val encoders: Array[T => Array[U]]

  def getFeatures(ndManager: NDManager): NDList

  def getLabels(ndManager: NDManager): NDList

  override def toString: String = instances.mkString("\n\n")


  /**
   * Create a batch of converted/indexed entities
   *
   * @param ndManager Current NDManager
   * @param encoder   Specific encoder function that convert input type into array of integers
   * @return NDArray of the batch
   */
  protected def createBatch(ndManager: NDManager, encoder: T => Array[Int]): NDArray = {
    var values = instances.map(encoder(_).toArray)
    val ndArray = ndManager.create(values)
    values = null  // To force GC ???
    ndArray
  }

  /**
   * Create a batch of converted/indexed entities
   *
   * @param ndManager Current NDManager
   * @param encoder   Specific indexing function that convert input type into array of array ofintegers
   * @return NDArray of the batch
   */
  protected def createSegmentsBatch(ndManager: NDManager, encoder: T => Array[Array[Int]]): Array[NDArray] = {
    instances.map(
      instance => {
        var values = encoder(instance)
        val ndArray = ndManager.create(values)
        // To force GC
        values = null
        ndArray
      }
    )
  }
}
