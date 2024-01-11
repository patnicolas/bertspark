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
package org.bertspark.config


/**
 * Wrapper for the execution modes (bitwise)
 * @author Patrick Nicolas
 * @version 0.4
 */
private[bertspark] final object ExecutionMode {
  private final val undefinedLbl = 0
  private final val pretrainingLbl = 1
  private final val classifierLbl = 2
  private final val similarityLbl = 4
  private final val transferLearningLbl = 8
  private final val evaluationLbl = 16
  private final val hpoLbl = 32
  private final val testLbl = 64

  var mode: Int = undefinedLbl

  // Set bit-wise model
  def setPretraining: Unit = mode += pretrainingLbl
  def setClassifier: Unit = mode += classifierLbl
  def setSimilarity: Unit = mode += similarityLbl
  def setEvaluation: Unit = mode += evaluationLbl
  def setTransferLearning: Unit = mode += transferLearningLbl
  def setTest: Unit = mode += testLbl
  def setHpo: Unit = mode += hpoLbl
  final def reset: Unit = mode = undefinedLbl


  // Get bit-wise model
  @inline
  final def isValid: Boolean = mode != undefinedLbl

  @inline
  final def isPretraining: Boolean = isMode(pretrainingLbl)

  @inline
  final def isSimilarity: Boolean = isMode(similarityLbl)

  @inline
  final def isEvaluation: Boolean = isMode(evaluationLbl)

  @inline
  final def isClassifier: Boolean = isMode(classifierLbl)

  final def isHpo: Boolean = isMode(hpoLbl)

  final def isTransferLearning: Boolean = isMode(transferLearningLbl)

  final def isTest: Boolean = isMode(testLbl)

  private def isMode(label: Int): Boolean = (mode & label) == label

  def convertModelName(modelName: String): String =
    if(isPretraining) modelName else modelName.replace("Pre-t", "T")

  override def toString: String = {
    val sb = new StringBuilder
    if(isPretraining) sb.append("Pretraining").append(" ")
    if(isClassifier) sb.append("Classifier").append(" ")
    if(isSimilarity) sb.append("Similarity").append(" ")
    if(isEvaluation) sb.append("Evaluation").append(" ")
    if(isTransferLearning) sb.append("TransferLearning").append(" ")
    if(isTest) sb.append("Test").append(" ")
    if(isHpo) sb.append("HPO").append(" ")
    sb.toString
  }
}