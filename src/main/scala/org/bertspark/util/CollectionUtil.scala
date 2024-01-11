/**
 * Copyright 2022,2023 Patrick R. Nicolas. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package org.bertspark.util

import java.util.Comparator
import scala.collection.mutable.HashMap
import scala.reflect.ClassTag


/**
  * Convenient wrapper for mutable and parameterized collection
  *
  * @author Patrick Nicolas
  * @version 0.2.8.5
  */
private[bertspark] final object CollectionUtil {


  /**
   * Split an array into multiple sub array given a predicate-marker
   * @param items Original array of elements
   * @param predicateMarker Predicate to break the array using a marker
   * @tparam T Type of elements
   * @return List of sub-array delimited by the marker
   */
  def split[T](items: Array[T], predicateMarker: T => Boolean): List[Array[T]] =
    split[T](items, predicateMarker, List[Array[T]]())

  @annotation.tailrec
  private def split[T](items: Array[T], matcher: T => Boolean, xs: List[Array[T]]): List[Array[T]] = {
    var index = 0
    while(index < items.length && !matcher(items(index)))
      index += 1
    if(index >= items.length)
      (items :: xs).reverse
    else {
      val (firstSegment, lastSegment) = items.splitAt(index)
      split[T](lastSegment.drop(1), matcher, firstSegment :: xs)
    }
  }

  abstract class CustomerIterator[T, U <: Iterator[T]](thisIterator: => U) extends Seq[T] {
    override def iterator: Iterator[T] = thisIterator
  }

  /**
    * Generic mutable map
    * @tparam T
    * @tparam U
    */
  @SerialVersionUID(-2L)
  final class MutableMap[T, U] extends HashMap[T, U] with Serializable {
    def += (key: T, value: U): Unit = put(key, value)

    final def getOrElse(key: T): U = getOrElse(key, null.asInstanceOf[U])
  }

  @SerialVersionUID(-3L)
  final class MultiMutableMap[T, U] extends HashMap[T, Map[T, U]] with Serializable {
    def += (key: T, value: Map[T, U]): Unit = put(key, value)

    final def += (key: T, subKey: T, value: U): Unit = put(key, getOrElse(key, Map[T, U](subKey->value)))

    final def getOrElse(key: T): Map[T, U] =  getOrElse(key, Map.empty[T, U])
  }


  /**
    * Generic priority queue wrapper to the priority queue in Java
    * @param score Scoring method used in the comparison
    * @tparam T Type for object to compare
    */
  final class PQueue[T](score: T => Int, maxHeap: Boolean = true)  {
    private[this] val comparator = new Comparator[T] {
      override def compare(t1: T, t2: T): Int =
        if(maxHeap) score(t2) - score(t1) else score(t1) - score(t2)
    }
    private[this] val pq = new java.util.PriorityQueue[T](comparator)

    @inline
    def add(t: T): Unit = pq.add(t)

    def deQueue(numElements: Int): Seq[T] = {
      // Make sure we do not return number of elements
      // exceeding the current size of the queue
      val numElementsToReturn = if(pq.size < numElements) pq.size else numElements

      // Retrieve the elements in the queue
      (0 until numElementsToReturn).foldLeft(List[T]()) (
        (xs, _) => pq.poll :: xs
      ).reverse
    }


    /**
      * This method is read-only. It retrieves the top numElements of
      * the queue then add them back to maintain the queue. This
      * operation is efficient as long as the number of elements is small (< 10)
      * @param numElements Number of elements to retrieve
      * @return The top elements in the queue
      */
    final def get(numElements: Int): Seq[T] = {
      val collected = deQueue(numElements)
      collected.foreach( pq.add(_))
      collected
    }
  }


  /**
    * Join two sequence with different element typs.
    * @param src The sequence to filter by
    * @param filterSeq Filtering Sequence
    * @param srcKey Method to extract key from source sequence/collection
    * @param byKey Method to extract key from filtering (by) sequence/collection
    * @tparam T Type of element of source sequence
    * @tparam U Type of element of filtering sequence
    * @return Filtered source sequence
    */
  @throws(clazz = classOf[IllegalStateException])
  def filter[T, U](src: Seq[T], filterSeq: Seq[U], srcKey: T => String, byKey: U => String): Seq[T] =
    if(src.nonEmpty && filterSeq.nonEmpty) {
      val sourceMap = src.map(t => (srcKey(t), t)).toMap
      filterSeq.map(
        t => {
          val key = byKey(t)
          sourceMap.getOrElse(key, throw new IllegalStateException(s"Key $key is not found in source"))
        }
      )
    }
    else if(src.nonEmpty)
      src
    else
      Seq.empty[T]


  /**
    * Join two sequence with different element typs.
    * @param condition The sequence to filter by
    * @param input Filtering Sequence
    * @param srcKey Method to extract key from source sequence/collection
    * @tparam T Type of element of source sequence
    * @throws IllegalStateException if source for key is not found
    * @return Sequence of rejected entities not meeting filter condition
    */
  @throws(clazz = classOf[IllegalStateException])
  def filter[T](condition: Seq[T], input: Seq[String], srcKey: T => String): Seq[String] =
    if(condition.nonEmpty && input.nonEmpty) {
      val conditionSet = condition.map(srcKey(_))
      input.filter(!conditionSet.contains(_))
    }
    else if(condition.nonEmpty)
      Seq.empty[String]
    else
      input


  /**
    * HashMap that load value from file, HDFS or S3 folder, dynamically. This is an immutable
    * data structure that will throw a UnsupportedOperationException for any attempt to add
    * a new item.
    *
    * @param loader Loading function
    * @param toKey Extractor of key
    * @tparam T Type of the key
    * @tparam U Type of the value.
    * @note Both the key and value should be serialized (case classes)
    */
  final class LazyHashMap[T, U](loader: String => Option[U], toKey: String => T) extends HashMap[T, U] {
    def get(item: String): Option[U] = synchronized {
      val key = toKey(item)
      if(super.contains(key))
        super.get(key)
      else
       loader(item).map(
         l => {
          super.put(key, l)
          l
        }
       )
    }
      // This map is immutable
    @throws(clazz = classOf[UnsupportedOperationException])
    override def put(key: T, value: U): Option[U] =
      throw new UnsupportedOperationException("Lazy HashMap is immutable")
  }


  /**
    * Merge two map
    * @param xs First map to be merged
    * @param ys Second map to be merge
    * @param f Merging function
    * @tparam K Type Key for the map
    * @tparam X Type of value for first map
    * @tparam Y Type of value for second map
    * @tparam Z Type of output value
    * @return Map with consolidated keys
    */
  def mergeWith[K, X, Y, Z](xs: Map[K, X], ys: Map[K, Y])(f: (X, Y) => Z): Map[K, Z] =
    xs.flatMap { case (k, x) => ys.get(k).map(k -> f(x, _)) }


  /**
    * Draw a sample of random elements from a sequence
    * @param input Input sequence
    * @param numSamples Number of samples
    * @tparam T Type of elements in the sequence
    * @return Original sequence if size sequence < numSamples, random elements otherwise
    */
  def drawSample[T: ClassTag](input: Seq[T], numSamples: Int): Seq[T] =
    if(input.size < numSamples)
      input
    else {
      import scala.util.Random
      val rand = new Random(42L)
      Seq.fill(numSamples)(input(rand.nextInt(input.size)))
    }


  implicit class SeqExt[T](seq: Seq[T]) {
    def dedup(key: T => String): Seq[T] = {
      seq.foldLeft(new HashMap[String, T]())( (hmap, t) => hmap += ((key(t), t))).values.toSeq
    }
  }


  final def permutate[T]: List[T] => Iterable[List[T]] = {
    case Nil => List(Nil)
    case xs => {
      for {
        (x, i) <- xs.zipWithIndex
        ys <- permutate(xs.take(i) ++ xs.drop(1 + i))
      } yield {
        x :: ys
      }
    }
  }

  /**
   * Extract the combinatorial subsets of an unordered sequence of items with permutation. The output is ordered
   * by increasing size of the sub collections
   * @param input Sequence of type T
   * @param singleElementIncluded Flag to include sub lists should contain a single element
   * @param maxCollectionSize Maximum size of each ordered sub list
   * @tparam T Parameterized
   * @return Sequence of Sequences
   */
  final def combinatorialPermutations[T](
    input: Seq[T],
    singleElementIncluded: Boolean = true,
    maxCollectionSize: Int = -1
  ): Seq[Seq[T]] = {
    import scala.collection.mutable.ListBuffer

    val allPermutations = input.permutations
    val collector = new ListBuffer[Seq[T]]()
    while (allPermutations.hasNext) {
      val permutation = allPermutations.next()
      combinatorial[T](permutation, singleElementIncluded, maxCollectionSize).foreach(collector.append(_))
    }
    collector.toSeq
  }

  /**
   * Extract the combinatorial subsets of an ordered sequence of items without order permutation. The output is
   * ordered by increasing size of the sub collections.
   * @param input Sequence of type T
   * @param singleElementIncluded Flag to include sub lists should contain a single element
   * @param maxCollectionSize Maximum size of each ordered sub list
   * @tparam T Parameterized
   * @return Sequence of Sequences
   */
  final def combinatorial[T](
    input: Seq[T],
    singleElementIncluded: Boolean = true,
    maxCollectionSize: Int = -1
  ): Seq[Seq[T]] = {

    @scala.annotation.tailrec
    def traverse(list: Seq[T], index: Int, span: Int, collector: List[Seq[T]]): List[Seq[T]] =
      if (index + span > list.size)
        collector
      else {
        val xss = (index until list.size - span + 1).foldLeft(collector)(
          (xs, n) => (Seq[T](list(index)) ++ list.slice(n + 1, n + span)) :: xs
        )
        traverse(list, index + 1, span, xss)
      }

    @scala.annotation.tailrec
    def combinatorial(list: Seq[T], span: Int, _collector: List[Seq[T]]): List[Seq[T]] =
      if (span <= 1)
        _collector
      else {
        val xss = traverse(list, 0, span, _collector)
        combinatorial(list, span - 1, xss)
      }

    if (input.nonEmpty)
      if (maxCollectionSize == 1)
        input.map(Seq[T](_))

      else {
        val _list = combinatorial(input, input.size - 1, List[Seq[T]](input))
        val list = if (singleElementIncluded) input.foldLeft(_list)((xss, t) => Seq[T](t) :: xss)
        else _list

        if (maxCollectionSize > 0 && maxCollectionSize < input.size) list.takeWhile(_.length <= maxCollectionSize)
        else list
      }
    else
      Seq.empty[Seq[T]]
  }

  /**
   * Extract all ordered sub-lists of contiguous elements from a give list
   * @param values Input list
   * @tparam T Type of elements of the list
   * @return Groups of sublists, including the list itself
   */
  def getOrderedSubLists[T](values: Seq[T], minSize: Int = 1): Seq[Seq[T]] = {

    @scala.annotation.tailrec
    def getOrderedSubLists[T](values: Seq[T], len: Int, collector: List[Seq[T]]): Seq[Seq[T]] =
      if (len == minSize - 1)
        collector
      else
        getOrderedSubLists[T](
          values: Seq[T],
          len - 1,
          (0 until values.size - len + 1).map(index => values.slice(index, index + len)).toList ::: collector
        )

    values.size match {
      case 0 => List.empty[Seq[T]]
      case 1 => if (minSize == 1) Seq[Seq[T]](values) else List.empty[Seq[T]]
      case _ =>
        getOrderedSubLists(values, values.size - 1, List[Seq[T]](values))
    }
  }

  /**
   * Cleanse content by replacing some characters or string using regular expression
   * @param content Original content
   * @param fromToRegex Sequence of pairs of regex
   * @return Cleansed content
   */
  def cleanseContent(content: String, fromToRegex: Seq[(String, String)]): String =
    if (fromToRegex.nonEmpty)
      fromToRegex.foldLeft(content) {
        case (str, (fromRegex, toRegex)) => str.replaceAll(fromRegex, toRegex)
      }
    else
      content

}
