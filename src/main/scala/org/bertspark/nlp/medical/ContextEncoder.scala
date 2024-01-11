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
 */
package org.bertspark.nlp.medical

import org.bertspark.config.MlopsConfiguration
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalContext
import org.bertspark.nlp.vocabulary.MedicalCodeDescriptors.CodeDescriptorMap
import org.bertspark.transformer.dataset.reservedLabels


/**
 * Methods to encode a context from a request
 * {{{
 *    Tag then encode the variable
 * }}}
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final object ContextEncoder {
  final val modifierTag = "_mod"
  final val ageBucketTag = "_age"
  final val genderTag = "_gender"
  final val cptTag = "_cpt"
  final val clientTag = "_client"
  final val customerTag = "_cust"
  final val modalityTag = "_modality"
  final val posTag = "_pos"

  final private val minAge = 0
  final private val maxAge = 120
  final private val numSteps = 6

  private def setTag(tag: String, value: String = "no"): String = s"$value$tag"


  final def getEncodedGender: Array[String] = Array[String]("m_gender", "f_gender", "u_gender")
  final def getEncodedAge: Array[String] = Array.tabulate(13)(n => s"${n}_age") ++ Array[String]("u_age")

  /**
   * Embedding of numerical values using bucketing technique
   *
   * @param min      Minimum value
   * @param max      Maximum value
   * @param numSteps Number of buckets
   * @param value    Value to encoder
   * @return Encoded string value
   */
  final def encodeNumeric(min: Int, max: Int, numSteps: Int, value: Int): String =
    if (value >= min && value <= max) {
      val bucket = (numSteps * (value - min).toDouble / (max - min)).floor.toInt.toString
      setTag(ageBucketTag, bucket)
    } else
      setTag(ageBucketTag)


  /**
   * Encode the context (a subset of its attributes)
   *
   * @param context Context from request
   * @return Array of categories encoding
   */
  def encodeContext(context: InternalContext): Array[String] = {
    import MlopsConfiguration._
    // Encode CPT and modifier
    val (encodedCpt, encodedModifier) = encodeEMRCodes(context)

    // Encode gender
    val encodedGender = if (context.gender != null) setTag(genderTag, context.gender.toLowerCase) else setTag(genderTag)
    val encodedCustomer = encodeCustomer(context)
    val encodedClient = encodeClient(context)
    val encodedModality = encodeModality(context)
    val encodedPos = encodePos(context)

    // Default encoding for the context
    val primaryEncoding = Array[String](
      encodeNumeric(minAge, maxAge, numSteps, context.age),
      encodedGender,
      encodedCustomer,
      encodedClient,
      encodedModality,
      encodedPos,
      encodedCpt,
      encodedModifier
    )

    encodeSemantic(context, primaryEncoding)
  }

  def encodeSemantic(context: InternalContext, primaryEncoding: Array[String]): Array[String] =
    if (context.EMRCpts.nonEmpty) {
      val codeSemanticEncoding =
        if (context.EMRCpts.head.modifiers.nonEmpty) {
          val key = s"${context.EMRCpts.head.cpt} ${context.EMRCpts.head.modifiers.mkString(" ")}"
          CodeDescriptorMap.getDescriptors(key)
        } else
          CodeDescriptorMap.getDescriptors(context.EMRCpts.head.cpt)

      // We make sure that the code semantic encoder does not refer to one of the reserved tokens
      if (codeSemanticEncoding.nonEmpty)
        primaryEncoding ++ codeSemanticEncoding.filter(!reservedLabels.contains(_))
      else
        primaryEncoding
    } else
      primaryEncoding


  def encodeCustomer(context: InternalContext): String =
    if (context.customer != null && context.customer.nonEmpty) setTag(customerTag, context.customer.toLowerCase)
    else setTag(customerTag)

  def encodeModality(context: InternalContext): String =
    if (context.modality != null && context.modality.nonEmpty) setTag(modalityTag, context.modality.toLowerCase)
    else setTag(modalityTag)

  def encodeClient(context: InternalContext): String =
    if (context.client != null && context.client.nonEmpty) setTag(clientTag, context.client.toLowerCase)
    else setTag(clientTag)

  def encodePos(context: InternalContext): String =
    if (context.placeOfService != null && context.placeOfService.nonEmpty) setTag(posTag, context.placeOfService
        .toLowerCase)
    else setTag(posTag)

  def encodeEMRCodes(context: InternalContext): (String, String) =
    if (context.EMRCpts.nonEmpty) {
      val encCpt =
        if (context.EMRCpts.head.cpt != null && context.EMRCpts.head.cpt.nonEmpty) setTag(cptTag, context.EMRCpts.head
            .cpt)
        else setTag(cptTag)
      val encModifier =
        if (context.EMRCpts.head.modifiers.nonEmpty) setTag(modifierTag, context.EMRCpts.head.modifiers.head)
        else setTag(modifierTag)
      (encCpt, encModifier)
    }
    else
      (setTag(cptTag), setTag(modifierTag))
}
