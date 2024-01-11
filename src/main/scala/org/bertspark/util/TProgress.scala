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


/**
  * Trait to show progress bar
  * @tparam T Type of input variable
  * @tparam U Type of progress variable
  *
  * @author Patrick Nicolas
  * @version 0.7
  */
private[bertspark] trait TProgress[T, U] {
  protected[this] val maxValue: T
  protected[this] val progress: U => Int

  /**
    * Use Unicode characters to display progress
    * @param u Input value - current progress
    * @param descriptor Descriptor of variables
    */
  def show(u: U, descriptor: String): Unit = TProgress.show[U](u, descriptor, progress)
}

private[bertspark] final object TProgress {
  private val filler: Char = '\u2588'
  private val empty: Char = '\u2581'

  def toString[U](u: U, descriptor: String, progress: U => Int): String = {
    val perCent = progress(u)
    val filledChars = Seq.fill(perCent)(filler)

    if(100 > perCent) {
      val numRemainingChars = 100-perCent
      val toFillChars = Seq.fill(numRemainingChars)(empty)
      s"$descriptor: $perCent% ${(filledChars ++ toFillChars).mkString}"
    } else
      s"$descriptor: 100% ${filledChars.mkString}"
  }

  def show[U](u: U, descriptor: String, progress: U => Int): Unit = println(toString(u, descriptor, progress))
}
