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

import org.bertspark.nlp.medical.MedicalCodingTypes.InternalContext.{getEmrCodesComma, getEmrCodesSpace}
import org.bertspark.nlp.medical.MedicalCodingTypes.lineItemSeparator
import org.slf4j.{Logger, LoggerFactory}


/**
  * Hierarchical label container
  * @param contextualDocument Contextual document
  *                           ContextualDocument(id: String, contextVariables: Array[String], text: String)
  * @param emrStr             Stringized EMR codes
  * @param claim     Claim without EMR codes
  *
  * @author Patrick Nicolas
  * @version 0.3
  */
private[bertspark] case class TrainingLabel(contextualDocument: ContextualDocument, emrStr: String, claim: String) {
  override def toString: String = s"${contextualDocument.toString}\nEmr: $emrStr Claim: $claim"
  final def getLabel: String = s"$emrStr: $claim"
}


/**
  * {{{
  *  Conversion from
  *  LabeledRequest:
  *      contextualDocument: ContextualDocument
  *      emrCodes: Seq[MlEMRCodes]
  *      lineItems: Seq[FeedbackLineItem]
  *  HierarchicalLabels
  *       contextualDocument: ContextualDocument
  *       emrStr: String
  *       remainingClaim: String*
  * }}}
  * @author Patrick Nicolas
  * @version 0.3
  */
private[bertspark] final object TrainingLabel {
  final private val logger: Logger = LoggerFactory.getLogger("TrainingLabel")

  /**
    * Conversion of a labeledRequest record into a hierarchical labeled claim
    * @param labeledRequest Labeled records
    * @return Instance of hierarchical labeled claim
    */
  def apply(labeledRequest: LabeledRequest): TrainingLabel = {
    val emrWithComma = getEmrCodesComma(labeledRequest.emrCodes)
    val emrWithSpace = getEmrCodesSpace(labeledRequest.emrCodes)

    val claim = mkStringLineItems(labeledRequest, emrWithComma)
    TrainingLabel(labeledRequest.contextualDocument, emrWithSpace, claim)
  }


  def mkStringLineItems(labeledRequest: LabeledRequest, emrWithComma: String): String = {
    val labeledLineItems = labeledRequest.lineItems.map(
      lineItem => {
        val lineItemStr = lineItem.lineItemComma
        (lineItemStr.contains(emrWithComma), lineItemStr)
      }
    )
    val (emrLineItem, otherLineItems) = labeledLineItems.partition(_._1)
    (emrLineItem.map(_._2) ++ otherLineItems.map(_._2).sortWith(_ < _)).mkString(lineItemSeparator)
  }
}