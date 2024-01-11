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
package org.bertspark.nlp.trainingset

import scala.util.Random
import org.slf4j.{Logger, LoggerFactory}


/**
 * Wrapper for manipulating pair of sentence within a corpus
 * @param thisSentence First sentence
 * @param thatSentence Second sentence. It can be empty if a single sentence is defined
 * @param isConsecutive Boolean that specify if this
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class SentencePair(
  thisSentence: Array[String],
  thatSentence: Array[String],
  isConsecutive: Boolean = true) {
  require(thisSentence.nonEmpty, "First sentence is empty")
  import SentencePair._

  /**
   * Swap the sentence within a sentence pairs
   * @return Pair with swapped sentences.
   */
  def swapEveryOther: SentencePair =
    if(thatSentence.nonEmpty)
      if(rand.nextBoolean()) new SentencePair(thatSentence, thisSentence, false) else this
    else
      new SentencePair(thisSentence, Array.empty[String], false)

  final def getTotalLength: Int = thisSentence.length + thatSentence.length

  /**
   * truncate the sentences of a Pair, not to exceed max allowed length
   * @param maxAllowedLength Maximum number of tokens allowed for each of the sentences of the pair
   * @return Truncated sentences pairs
   */
  def truncate(maxAllowedLength: Int): SentencePair = {
    val totalLength = thisSentence.length + thatSentence.length

    // If the sum of two sentences are larger that maximum allowd then we need to reduce
    // (truncate) both sentence using the ratio of sentenceA.length/sentenceB.length
    if(maxAllowedLength < totalLength) {
      val surplus = totalLength - maxAllowedLength
      val combinedHalfSize = surplus >> 1
      val newThisSentenceLength = thisSentence.length - (if(isToSwap(surplus)) combinedHalfSize else combinedHalfSize + 1)
      val newThatSentenceLength = thatSentence.length  - combinedHalfSize

      // Needs to keep at list minTokensPerSentence tokens for each sentence
      val (toRemoveFromThis, toRemoveFromThat) =
        if(newThisSentenceLength < minTokensPerSentence)
          (minTokensPerSentence, newThatSentenceLength - minTokensPerSentence + newThisSentenceLength)
        else if(newThatSentenceLength < minTokensPerSentence)
          (newThisSentenceLength - minTokensPerSentence + newThatSentenceLength, minTokensPerSentence)
        else
          (newThisSentenceLength, newThatSentenceLength)

      // Applies truncation
      val thisNewSentence = thisSentence.take(toRemoveFromThis)
      val thatNewSentence = thatSentence.take(toRemoveFromThat)
      new SentencePair(thisNewSentence, thatNewSentence, isNext)
    }
    else
      this
  }

  @inline
  final def isNext: Boolean = isConsecutive

  @inline
  final def getThisSentence: Array[String] = thisSentence

  @inline
  final def getThatSentence: Array[String] = thatSentence

  override def toString: String = s"[${thisSentence.mkString(" ")}] [${thatSentence.mkString(" ")}]"
}


/**
 * Singleton for manipulating sequences of sentence pairs and constructor
 */
private[bertspark]  final object SentencePair {
  final val logger: Logger = LoggerFactory.getLogger("BERTDataset")
  private val rand = new Random(42L)
  private val minTokensPerSentence = 12

  /**
   * Constructor to create a Sentence pair by splitting randomly a sentence
   * @param sentence Original sentence
   * @param splitRand Random split
   * @throws IllegalArgumentException If sentence has a single token or split ratio is out of range
   * @return a SentencePair instance
   */
  @throws(clazz = classOf[IllegalArgumentException])
  def apply(sentence: Array[String], splitRand: Random): SentencePair =
    split(sentence, ((splitRand.nextFloat()*sentence.size).floor).toInt)

  /**
   * Constructor to create a Sentence pair by splitting an existing sentence
   * @param sentence Original sentence
   * @param splitRatio Splitting ratio
   * @throws IllegalArgumentException If sentence has a single token or split ratio is out of range
   * @return a SentencePair instance
   */
  @throws(clazz = classOf[IllegalArgumentException])
  def apply(sentence: Array[String], splitRatio: Float): SentencePair = {
    require(sentence.size > 1, "Cannot split a sentence with a single token")
    require(splitRatio > 0.0F && splitRatio < 1.0F, s"Split ratio $splitRatio should be ]0, 1[")

    split(sentence, ((splitRatio*sentence.size).floor).toInt)
  }

  private def split(sentence: Array[String], splitIndex: Int): SentencePair = {
    if(splitIndex == 0)
      new SentencePair(Array[String](sentence.head), sentence.tail)
    else if(splitIndex >= sentence.size-1)
      new SentencePair(sentence.dropRight(1), Array[String](sentence.last))
    else
      new SentencePair(sentence.take(splitIndex), sentence.drop(splitIndex))
  }


  private def isToSwap(index: Int): Boolean = (index & 0x01) == 0x00

  /**
   * Shuffle all the sentence pairs then shuffle the 2 sentences of the pair, every other pair
   * @param sentencePairs Sentence pairs to shuffle then shuffle sentences in each other pair
   * @return Shuffled sentence pairs
   */
  def swapEveryOther(sentencePairs: Array[SentencePair]): Seq[SentencePair] =
    if(sentencePairs.size > 1) {
      val shuffledPairs = Random.shuffle(sentencePairs.toSeq)
      shuffledPairs.indices.foldLeft(List[SentencePair]()) (
        (xs, index) => {
          val newSentencePair =
            if(isToSwap(index) && index < sentencePairs.size-1)  shuffledPairs(index).swapEveryOther
            else shuffledPairs(index)
          newSentencePair :: xs
        }
      ).reverse
    }
    else {
      logger.warn("Cannot swap undefined sequence of sentence pairs")
      Seq.empty[SentencePair]
    }
}
