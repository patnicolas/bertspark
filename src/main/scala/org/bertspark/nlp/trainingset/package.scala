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
package org.bertspark.nlp

import ai.djl.ndarray.NDArray
import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.nlp.medical.MedicalCodingTypes.{csvCodeSeparator, lineItemSeparator, FeedbackLineItem, InternalFeedback, InternalRequest, MlEMRCodes}
import org.bertspark.nlp.trainingset.ContextualDocument.nullContextualDocument

/**
 * Data types/classes used in the various training set and builder
 * @author Patrick Nicolas
 * @version 0.1
 */
package object trainingset {
  type LabelEmbedding = (Array[Float], String)

  /**
   * Pair {[CLS} token embedding, claim/document index}
   */
  type LabelIndexEmbedding = (Array[Float], Int)
  /**
   * Pair {Document_id, [CLS] token embedding}
   */
  type KeyedValues = (String, Array[Float])
  type KeyedNDArrays = (String, NDArray)

  type TrainData = (InternalRequest, Int)
  type TrainDataDS = Dataset[TrainData]
  /**
   * Count for labeled claim
   */
  type LabeledCount = (String, Int)


  case class LabelIndex(claim: String, index: Int)

  /**
   * Simplified LabeledClaim
   * @param id                Identifier for the document or claim
   * @param emrCodes          EMR codes from the request
   * @param feedbackLineItems Line items as labels
   */
  case class LabeledClaim(id: String, emrCodes: Seq[MlEMRCodes], feedbackLineItems: Seq[FeedbackLineItem]) {
    import LabeledClaim._

    def getClaim: String = feedbackLineItems.map(_.lineItemComma).mkString(lineItemSeparator)
    override def toString: String = s"Id:$id EMR:${getEmrCodesStr(emrCodes)} ${feedbackLineItems.mkString("\n")}"
  }



  final object LabeledClaim {

    def apply(feedback: InternalFeedback): LabeledClaim =
      LabeledClaim(feedback.id, feedback.context.EMRCpts, feedback.finalized.lineItems)

    def getEmrCptModifier(emrCodes: Seq[MlEMRCodes]): (String, Seq[String]) = {
      if (emrCodes == null || emrCodes.isEmpty) ("", Seq.empty[String])
      else if (emrCodes.head.modifiers.isEmpty) (emrCodes.head.cpt, Seq.empty[String])
      else (emrCodes.head.cpt, emrCodes.head.modifiers)
    }

    def getEmrCodesStr(emrCodes: Seq[MlEMRCodes]): String = {
      val (cpt, modifiers) = getEmrCptModifier(emrCodes)
      if (modifiers.nonEmpty) s"$cpt ${modifiers.mkString(csvCodeSeparator)}" else cpt
    }
  }

  /**
   * Structure for of labeled training data
   * @param contextualDocument Contextual document (tokenized claim and context)
   * @param label              Label (sub claim)
   * @param clsEmbedding       [CLS] token embedding for classification
   */
  case class TokenizedTrainingSet(
    contextualDocument: ContextualDocument,
    label: String,
    clsEmbedding: Array[Float]) {
    override def toString: String = {
      val embeddingStr = if (clsEmbedding != null && clsEmbedding.nonEmpty) clsEmbedding.mkString(" ") else "None"
      s"\nLabel:$label\n${contextualDocument.toString}nEmbedding: $embeddingStr"
    }
  }

  final object TokenizedTrainingSet {

    def apply(contextualDocument: ContextualDocument): TokenizedTrainingSet =
      TokenizedTrainingSet(contextualDocument, "", Array.empty[Float])

    def apply(trainingLabel: TrainingLabel): TokenizedTrainingSet =
      TokenizedTrainingSet(trainingLabel.contextualDocument, trainingLabel.claim, Array.empty[Float])

    final lazy val nullTokenizedIndexedTrainingSet =
      TokenizedTrainingSet(nullContextualDocument, "", Array.empty[Float])

    def groupByLabel(tokenizedTrainingSets: Seq[TokenizedTrainingSet]): Seq[(String, Seq[TokenizedTrainingSet])] = {
      tokenizedTrainingSets.groupBy(_.label)
    }.toSeq
  }

  case class EmrGroupedLabeledRequest(emr: String, labeledRequests: Seq[LabeledRequest])

  case class EmrGroupedLabeledClaim(emr: String, labeledClaims: Seq[LabeledClaim])

  case class LabeledClaimStr(id: String, remainingClaim: String)


  /**
   * Labeled training set associated to a give EMR specification
   * Stored in target/training
   * @param subModel            stringized EMR codes
   * @param labeledTrainingData Training Data associated with this EMR
   * @param labelIndices        Indices Sub-claim to Index
   */
  private[bertspark] case class SubModelsTrainingSet(
    subModel: String,
    labeledTrainingData: Seq[TokenizedTrainingSet],
    labelIndices: Seq[(String, Int)]
  ) {
    final def nonEmpty: Boolean = subModel.nonEmpty

    override def toString: String = {
      val labeledTSStr = labeledTrainingData.map(ts => s"${ts.label}:${ts.contextualDocument.id}").mkString("\n")
      s"$subModel\n$labeledTSStr"
    }

    def count(): String =
      labeledTrainingData
          .groupBy(_.label)
          .map{ case (label, tokenizedTS) => s"$label,${tokenizedTS.size}"}
          .mkString("\n")
  }

  final object SubModelsTrainingSet {

    def count(ds: Dataset[SubModelsTrainingSet])(implicit sparkSession: SparkSession): String = {
      import sparkSession.implicits._
      ds.map(subModelTS => s"${subModelTS.subModel}:${subModelTS.labeledTrainingData.size}\n${subModelTS.count()}")
          .collect()
          .mkString("\n")
    }

    final val emptySubModelsTrainingSet =
      SubModelsTrainingSet("", Seq.empty[TokenizedTrainingSet], Seq.empty[(String, Int)])

    final def nonEmpty(subModelsTrainingSet: SubModelsTrainingSet): Boolean =
      subModelsTrainingSet.subModel.nonEmpty
  }




  /**
   * Intermediary structure used to extract contextual document and line items from raw labeled
   * data loaded form S3
   *
   * @param contextualDocument Contextual document of type
   *                           ContextualDocument(id: String, contextVariables: Array[String], text: String)
   * @param emrCodes           EMR codes
   * @param lineItems          Labeled line items
   */
  case class LabeledRequest(
    contextualDocument: ContextualDocument,
    emrCodes: Seq[MlEMRCodes],
    lineItems: Seq[FeedbackLineItem]) {
    override def toString: String = s"$contextualDocument\n${lineItems.mkString("\n")}"

    def claimStr: String = lineItems.map(_.lineItemComma).mkString(" - ")
  }
}
