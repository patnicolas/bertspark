package org.bertspark.nlp.medical

import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalContext, InternalFeedback, InternalRequest, MlClaimEntriesWithCodes, MlEMRCodes}
import org.bertspark.util.io.LocalFileUtil


private[bertspark] final object LegacyCodingTypes {

  case class OldContext(
    claimType: String,
    sectionMode: String,
    sectionGroups: Seq[String],
    age: Int,
    gender: String,
    taxonomy: String,
    placeOfService: String,
    dateOfService: String,
    EMCode: String,
    EMRCpts: Seq[MlEMRCodes],
    providerId: String,
    patientId: String,
    sectionHeaders: Seq[String],
    planId: String
  )


  implicit def old2NewContext(oldContext: OldContext, customer: String): InternalContext = {
    val modality = modalityFromCPT.getOrElse(oldContext.EMRCpts.head.cpt, "unknown")
    InternalContext(
      oldContext.claimType,
      oldContext.age,
      oldContext.gender,
      oldContext.taxonomy,
      customer,
      "no_client",
      modality,
      oldContext.placeOfService,
      oldContext.dateOfService,
      oldContext.EMRCpts,
      oldContext.providerId,
      oldContext.patientId,
      oldContext.planId,
      "",
      "",
      ""
    )
  }

  case class OldRequest(id: String, context: OldContext, notes: Seq[String])

  case class OldFeedback(
    id: String,
    autoCodable: Boolean,
    context: OldContext,
    autocoded: MlClaimEntriesWithCodes,
    finalized: MlClaimEntriesWithCodes,
    audited: MlClaimEntriesWithCodes
  )


  implicit def old2NewFeedback(oldFeedback: OldFeedback, customer: String): InternalFeedback = {
    val newContext: InternalContext = old2NewContext(oldFeedback.context, customer)
    InternalFeedback(
      oldFeedback.id,
      oldFeedback.autoCodable,
      newContext,
      oldFeedback.autocoded,
      oldFeedback.finalized,
      oldFeedback.audited
    )
  }

  implicit def old2NewRequest(oldRequest: OldRequest, customer: String): InternalRequest = {
    val newContext: InternalContext = old2NewContext(oldRequest.context, customer)
    InternalRequest(oldRequest.id, newContext, oldRequest.notes)
  }

  val modalityFromCPT = LocalFileUtil.Load.local(s"conf/cpt-modality.csv")
      .map(
        content => {
          val lines = content.split("\n")
          lines.map(
            line => {
              val ar = line.split(",")
              (ar.head.trim, ar(1).trim)
            }
          ).toMap
        }
      ).getOrElse(Map.empty[String, String])
}
