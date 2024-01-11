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

import org.apache.spark.sql.SparkSession
import org.bertspark.modeling.SubModelsTaxonomy.subModelTaxonomy
import org.bertspark.nlp.medical.MedicalCodingTypes.MlClaimResp.zeroMlClaimResp
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.vocabulary.ContextVocabulary
import org.bertspark.predictor.TPredictor.{oracleState, unsupportedState}
import org.bertspark.util.io._
import org.slf4j._
import scala.collection.mutable.ListBuffer

/**
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] object MedicalCodingTypes {
  final private val logger: Logger = LoggerFactory.getLogger("MedicalCodingTypes")

  final val lineItemSeparator = "-"
  final val codeGroupSeparator = tokenSeparator
  final val csvCodeSeparator = ","
  final val spaceCodeSeparator = " "
  final val comparisonSeparator = " || "

  /**
    * retrieve the S3 storage associated with medical coding
    *
    * @param s3Folder     Name of the S3 folder (absolute path)
    * @param numRecords   Number of records to be processed
    * @param sparkSession Implicit reference to the current Spark context
    * @return Instance of S3 Storage info for prediction requests
    */
  final def getPredictReqS3Storage(
    s3Folder: String,
    numRecords: Int = -1
  )(implicit sparkSession: SparkSession): SingleS3Dataset[InternalRequest] = {
    import sparkSession.implicits._
    SingleS3Dataset[InternalRequest](s3Folder, encodePredictReq, minSampleSize = -1, numRecords)
  }

  case class MlEMRCodes(
    order: Int,
    cpt: String,
    modifiers: Seq[String],
    icds: Seq[String],
    quantity: Int = 1,
    unit: String = "UN"
  ) {
    override def toString: String =
      s"Cpt: $cpt, modifiers: ${modifiers.mkString(csvCodeSeparator)}, icds: ${icds.mkString(csvCodeSeparator)}"

    def toCptModifiers: String = if (modifiers.nonEmpty) s"$cpt ${modifiers.mkString(csvCodeSeparator)}" else cpt

    final def isZero: Boolean = cpt.isEmpty
  }

  final object MlEMRCodes {
    def apply(): MlEMRCodes = MlEMRCodes(0, "", Seq.empty[String], Seq.empty[String], 0, "")

    def apply(emrCodes: String): MlEMRCodes = {
      val codes = emrCodes.split(codeGroupSeparator)
      codes.size match {
        case 0 => apply()
        case 1 => MlEMRCodes(0, codes.head.trim, Seq.empty[String], Seq.empty[String], 0, "")
        case _ => MlEMRCodes(0, codes.head.trim, codes(1).split(csvCodeSeparator), Seq.empty[String], 0, "")
      }
    }
  }

  case class Pipe(data: String, name: String) {
    override def toString: String = s"        Pipe data: $data\n         Pipe name: $name"
  }


  case class PRequestContext(
    claimType: String,
    age: scala.Int,
    gender: String,
    taxonomy: String,
    customer: String,
    client: String,
    procedureCategory: String,
    placeOfService: String,
    dateOfService: String,
    enableMentions: Boolean,
    EMRCpts: Seq[MlEMRCodes],
    providerId: String,
    patientId: String,
    planId: String,
    extraDict: String,
    unused: String,
    unused2: String,
    notesLocation: Seq[String],
    pipes: Seq[Pipe]
  ) {
    override def toString: String =
      s"""
         |claimType:         $claimType
         |Age:               $age
         |Gender:            $gender
         |Taxonomy:          $taxonomy
         |PoS:               $placeOfService
         |DoS:               $dateOfService
         |Customer:          $customer
         |Client:            $client
         |procedureCategory: $procedureCategory
         |enableMentions:    $enableMentions
         |EMR Cpts:          ${EMRCpts.mkString("\n")}
         |Provider:          $providerId
         |Patient:           $patientId
         |Plan id:           $planId
         |Meta data:         $extraDict
         |notes location:    ${notesLocation.mkString(" ")}
         |Pipes:             ${pipes.mkString("\n")}
         |""".stripMargin

    def emrLabel: String =
      if (EMRCpts.nonEmpty) s"${EMRCpts.head.cpt} ${EMRCpts.head.modifiers.mkString(spaceCodeSeparator)}"
      else "no_emr"
  }

  final object PRequestContext {
    /**
      * Conversion from internal context to API/Kafka context
      *
      * @param internalContext Internal context
      * @return Request context
      */
    def apply(internalContext: InternalContext): PRequestContext =
      PRequestContext(
        internalContext.claimType,
        internalContext.age,
        internalContext.gender,
        internalContext.taxonomy,
        internalContext.customer,
        internalContext.client,
        internalContext.modality,
        internalContext.placeOfService,
        internalContext.dateOfService,
        false,
        internalContext.EMRCpts,
        internalContext.providerId,
        internalContext.patientId,
        internalContext.planId,
        internalContext.extraDict,
        internalContext.unused,
        internalContext.unused2,
        Seq.empty[String],
        Seq.empty[Pipe]
      )

    def apply(): PRequestContext = nullContextReq

    def apply(emrCodes: Seq[MlEMRCodes]): PRequestContext = apply(InternalContext(emrCodes))

    final lazy val nullContextReq = PRequestContext(
      "",
      0,
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      false,
      Seq.empty[MlEMRCodes],
      "",
      "",
      "",
      "",
      "",
      "",
      Seq.empty[String],
      Seq.empty[Pipe])
  }


  /**
    * Definition of the context in the request as passed along various data pipelines
    *
    * @param claimType      Type of claim (CMS,...)
    * @param age            Age of the patient
    * @param gender         Age of the patient
    * @param taxonomy       Taxonomy associated with this request
    * @param customer       Customer associated with this request
    * @param client         Client associated with this request
    * @param modality       Modality associated with this request
    * @param placeOfService Place of the service or encounter
    * @param dateOfService  Date of the encounter
    * @param EMRCpts        CPT and Modifier structures provided by the user
    * @param providerId     Identifier for the provider
    * @param patientId      Identifier for the patient
    * @param planId         Customer or plan identification
    * @param extraDict      List of optional key value pairs
    * @param unused         For future use
    * @param unused2        For future use
    */
  case class InternalContext(
    claimType: String,
    age: scala.Int,
    gender: String,
    taxonomy: String,
    customer: String,
    client: String,
    modality: String,
    placeOfService: String,
    dateOfService: String,
    EMRCpts: Seq[MlEMRCodes],
    providerId: String,
    patientId: String,
    planId: String,
    extraDict: String,
    unused: String,
    unused2: String
  ) {
    override def toString: String =
      s"""
         |claimType:         $claimType
         |Age:               $age
         |Gender:            $gender
         |Taxonomy:          $taxonomy
         |PoS:               $placeOfService
         |DoS:               $dateOfService
         |Customer:          $customer
         |Client:            $client
         |Modality:          $modality
         |EMR Cpts:          ${EMRCpts.mkString("\n")}
         |Provider:          $providerId
         |Patient:           $patientId
         |Plan id:           $planId
         |Meta data:         $extraDict
         |""".stripMargin

    def emrLabel: String =
      if (EMRCpts.nonEmpty)
        EMRCpts.map {
          emrCode =>
            if(emrCode.modifiers.nonEmpty)
              s"${emrCode.cpt} ${emrCode.modifiers.mkString(spaceCodeSeparator)}".trim
            else
              emrCode.cpt.trim
        }.mkString(lineItemSeparator)
      else
        "no-emr"


    def emrComma: String =
      if (EMRCpts.nonEmpty)
        EMRCpts.map {
          emrCode =>
            if(emrCode.modifiers.nonEmpty)
              s"${emrCode.cpt} ${emrCode.modifiers.mkString(csvCodeSeparator)}"
            else
              emrCode.cpt
        }.mkString(lineItemSeparator)
      else
        "no-emr"
  }


  final object InternalContext {
    def apply(requestContext: PRequestContext): InternalContext =
      InternalContext(
        requestContext.claimType,
        requestContext.age,
        requestContext.gender,
        requestContext.taxonomy,
        requestContext.customer,
        requestContext.client,
        requestContext.procedureCategory,
        requestContext.placeOfService,
        requestContext.dateOfService,
        requestContext.EMRCpts,
        requestContext.providerId,
        requestContext.patientId,
        requestContext.planId,
        requestContext.extraDict,
        requestContext.unused,
        requestContext.unused2
      )

    def apply(EMRCpts: Seq[MlEMRCodes]): InternalContext =
      InternalContext(
        "",
        0,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        EMRCpts,
        "",
        "",
        "",
        "",
        "",
        "")

    def apply(): InternalContext = nullInternalContextReq

    final private val nullInternalContextReq = InternalContext(
      "",
      0,
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      Seq.empty[MlEMRCodes],
      "",
      "",
      "",
      "",
      "",
      "")


    def getEmrCptModifier(emrCode: MlEMRCodes, separator: String): String = {
      if (emrCode.modifiers.isEmpty) emrCode.cpt
      else s"${emrCode.cpt}$separator${emrCode.modifiers.mkString(separator)}"
    }

    def getEmrCodesSpace(emrCodes: Seq[MlEMRCodes]): String =
      emrCodes.map(getEmrCptModifier(_, spaceCodeSeparator)).mkString(lineItemSeparator)

    def getEmrCodesComma(emrCodes: Seq[MlEMRCodes]): String =
      emrCodes.map(getEmrCptModifier(_, csvCodeSeparator)).mkString(lineItemSeparator)
  }


  case class InternalRequest(id: String, context: InternalContext, notes: Seq[String])

  final object InternalRequest {
    def apply(pRequest: PRequest): InternalRequest =
      InternalRequest(pRequest.id, InternalContext(pRequest.context), pRequest.notes)

    def apply(): InternalRequest = InternalRequest("", InternalContext(), Seq.empty[String])
  }



  case class PRequest(id: String, context: PRequestContext, notes: Seq[String])

  final object PRequest {
    def apply(): PRequest =  PRequest("", PRequestContext(), Seq.empty[String])
    def apply(internalRequest: InternalRequest): PRequest =
      PRequest(internalRequest.id, PRequestContext(internalRequest.context), internalRequest.notes)

    def apply(id: String, emrCpts: Seq[MlEMRCodes], note: String): PRequest =
      PRequest(id, PRequestContext(emrCpts), Seq[String](note))
  }

  /**
    * Representation of a line item with an order, CPT code with optional modifiers and
    * one or more ICD codes, ....
    *
    * @param order      Order of this line item in the claim
    * @param cpt        CPT code
    * @param modifiers  Array of modifiers
    * @param icds       Array of ICD 10 codes
    * @param quantity   Number of unit if CPT requires it
    * @param unit       Type of units
    * @param score      Score for this line item
    * @param providerId Line item
    */
  case class MlLineItem(
    order: Int,
    cpt: String,
    modifiers: Seq[String],
    icds: Seq[String],
    quantity: Int,
    unit: String,
    score: Double,
    providerId: String
  ) {

    def toCodesSpace: String =
      if (modifiers.nonEmpty) s"$cpt ${modifiers.mkString(spaceCodeSeparator)} ${icds.mkString(spaceCodeSeparator)}"
      else s"$cpt ${icds.mkString(spaceCodeSeparator)}"

    def toCodesComma: String =
      if (modifiers.nonEmpty) s"$cpt ${modifiers.mkString(csvCodeSeparator)} ${icds.mkString(csvCodeSeparator)}"
      else s"$cpt ${icds.mkString(csvCodeSeparator)}"

    override def toString: String = {
      val provider = if (providerId == null || providerId.isEmpty) "0" else providerId
      s"$cpt : ${modifiers.mkString(csvCodeSeparator)} : ${icds.mkString(csvCodeSeparator)} : $quantity : $unit : $score : $provider"
    }

    def isEqual(mlLineItem: MlLineItem): Boolean = {
      val primarySimilarity = cpt == mlLineItem.cpt &&
          icds.size == mlLineItem.icds.size &&
          icds.indices.forall(index => icds(index) == mlLineItem.icds(index))
      primarySimilarity &&
        (modifiers.nonEmpty &&
            mlLineItem.modifiers.nonEmpty &&
            modifiers.size == mlLineItem.modifiers.size &&
            modifiers.indices.forall(index => modifiers(index) == mlLineItem.modifiers(index))) ||
            (modifiers.isEmpty && mlLineItem.modifiers.isEmpty)
    }
  }


  final object MlLineItem {


    def getLineItem(lineItemSpace: String): MlLineItem = {
      val medicalCodes = lineItemSpace.split(codeGroupSeparator)
      val cpt = medicalCodes.head
      val modifiers = ListBuffer[String]()
      val icds = ListBuffer[String]()
      medicalCodes.tail.foreach(
        medicalCode =>
          if(medicalCode.size == 2) modifiers.append(medicalCode)
          else icds.append(medicalCode)
      )
      MlLineItem(0, cpt, modifiers, icds, 1, "UN", 0.0, "0")
    }

    // Generate a line item instance from a string representation
    def apply(lineItemComma: String): MlLineItem = {
      val medicalCodes = lineItemComma.split(codeGroupSeparator)

      val (cpt, modifiers, icds): (String, Seq[String], Seq[String]) =
        medicalCodes.size match {
          case 3 => (medicalCodes.head, medicalCodes(1).split(csvCodeSeparator), medicalCodes(2).split(csvCodeSeparator))
          case 2 => (medicalCodes.head, Seq.empty[String], medicalCodes(1).split(csvCodeSeparator))
          case 1 => (medicalCodes.head, Seq.empty[String], Seq.empty[String])
          case _ => ("", Seq.empty[String], Seq.empty[String])
        }
      MlLineItem(0, cpt, modifiers, icds, 1, "UN", 0.0, "0")
    }

    def apply(feedbackLineItem: FeedbackLineItem): MlLineItem =
      MlLineItem(
        feedbackLineItem.order,
        feedbackLineItem.cpt,
        feedbackLineItem.modifiers,
        feedbackLineItem.icds,
        feedbackLineItem.quantity,
        feedbackLineItem.unit,
        feedbackLineItem.score,
        "0")
    final val allRadiologyModalities = Seq[String](
      "MRI",
      "XRAY",
      "CT",
      "PET",
      "MAMMOGRAPHY",
      "NUCLEAR MEDICINE",
      "INTERVENTIONAL",
      "ULTRASOUND",
      "DIAGNOSTIC",
      "ANESTHESIA",
      "INTERV BREAST",
      "E & M",
      "PQRS",
      "US",
      "XR",
      "DEXA",
      "UNKNOWN"
    ).map(_.toLowerCase)
  }


  final object Modality {
    final private val cptModalityFile = "conf/codes/cpt-modality.csv"

    final val unknownModality = "unknown"
    lazy val cptModalitiesMap: Option[Map[String, String]] =
      LocalFileUtil.Load.local(cptModalityFile, (s: String) => {
        val ar = s.split(csvCodeSeparator)
        (ar.head, ar(1).trim.toLowerCase)
      }).map(_.toMap)

    /**
     * Extract Modality from CPT
     *
     * @param cptCode Input CPT code
     * @return Modality
     */
    def getModalityFromCpt(cptCode: String): String =
      cptModalitiesMap.map(_.getOrElse(cptCode, unknownModality)).getOrElse(unknownModality)
  }


  /**
    * Codes from all sources {Clinical notes, rules engine, claim...}
    * {{{
    *   Nomenclature for mention
    *   - offset/length for NLP output
    *   - start/end for API response
    * }}}
    * @param codeId Identifier for the code
    * @param codeSet Type for the code {ICD1-, ..}
    * @param mention Mention in note associated with this code
    * @param offset Position of the first character of the mention
    * @param length Number of characters for the mentions
    * @param score Latest confidence score
    */
  case class MlCodeResp(
    codeId: String,
    codeSet: String,
    mention: String,
    offset: Int,
    length: Int,
    score: Double
  ) {
    override def toString: String = s"$codeId, $codeSet, $mention, $offset, $length, $score"
  }

  def zeroMlCodeResp: MlCodeResp = MlCodeResp("", "", "", -1, -1, -1.0)
  final def isZero(mlCodeResp: MlCodeResp): Boolean = mlCodeResp.codeId.isEmpty

  /**
    *
    * @param noteId
    * @param codes
    * @param sections
    */
  case class MlNoteResp(noteId: String, codes: Seq[MlCodeResp]) {
    override def toString: String = s"Codes: \n${codes.mkString("\n")}"
  }
  def buildNoteResp(noteId: String): MlNoteResp = MlNoteResp(noteId, Seq.empty[MlCodeResp])
  def buildNoteResp(noteId: String, codes: Seq[MlCodeResp]): MlNoteResp = MlNoteResp(noteId, codes)

  /**
    * Retrieve the error message associated to a prediction response
    * @param predictResp Instance of prediction response
    * @return Either noteId if no error or an error message
    */
  final def getError(predictResp: PResponse): String = predictResp.id

  final def isZero(mlNoteResp: MlNoteResp): Boolean = mlNoteResp.codes.isEmpty
  final def hasCodes(mlNoteResp: MlNoteResp): Boolean = mlNoteResp.codes.nonEmpty


  /**
    * Definition of the response for claim
    * @param lineItems Line items generated by a given model
    * @param notes Sequence of bag of codes..
    * @param completeNote Note modified by model
    */
  case class MlClaimResp(lineItems: Seq[MlLineItem], notes: Seq[MlNoteResp]) {
    @inline
    final def hasLineItems: Boolean = lineItems.nonEmpty

    @inline
    final def hasNotes: Boolean = notes.nonEmpty

    final def toCodes: String = lineItems.map(_.toCodesComma).mkString(lineItemSeparator)

    final def str: String = lineItems.map(_.toCodesSpace).mkString(lineItemSeparator)

    override def toString: String = s"Predictions: ${lineItems.mkString("\n")}\nNotes: ${notes.mkString("\n")}"
  }


  final object MlClaimResp {
    def apply(): MlClaimResp = MlClaimResp(Seq.empty[MlLineItem], Seq.empty[MlNoteResp])

    // Generate a Claim response instance from a string representation
    def apply(claimResp: String): MlClaimResp = {
      val lineItemStr = claimResp.split(lineItemSeparator).map(_.trim)
      val lineItems = lineItemStr.map(MlLineItem.getLineItem(_))
      MlClaimResp(lineItems, Seq.empty[MlNoteResp])
    }

    def apply(feedbackLineItems: Seq[FeedbackLineItem]): MlClaimResp =
      MlClaimResp(feedbackLineItems.map(MlLineItem(_)), Seq.empty[MlNoteResp])

    def zeroMlClaimResp: MlClaimResp = MlClaimResp(Seq.empty[MlLineItem], Seq.empty[MlNoteResp])

    final def isZero(mlClaimResp: MlClaimResp): Boolean = mlClaimResp.lineItems.isEmpty && mlClaimResp.notes.isEmpty
  }





  /**
    * Response to the prediction request
    * @param id Id of the claim, note or request
    * @param autoCodable Flag that indicates that the response is auto-codable
    * @param autoCodingState State 0 Rejected, State 1: Oracle, State 2: Predicted
    * @param claim Claim (line items and code) components
    * @param status Status
    */
  case class PResponse(
    id: String,
    autoCodable: Boolean,
    autoCodingState: Int,
    claim: MlClaimResp,
    completeNote: String,
    status: String
  )  {
    override def toString: String =
      s"$id $autoCodable $autoCodingState ${claim.toCodes} $status"

    def toCodes: String = claim.toCodes

    def str = claim.str
  }


  final object PResponse {
    def apply(id: String): PResponse = PResponse(id, false, unsupportedState, MlClaimResp(), "Unsupported request", "400")

    // Generate a PR response instance from a string representation
    def apply(id: String, label: String): PResponse = {
      val claim = MlClaimResp(label)
      PResponse(id, true, oracleState, claim, "", "200")
    }

    def apply(id: String, state: Int, label: String): PResponse = {
      val claim = MlClaimResp(label)
      PResponse(id, true, state, claim, "", "200")
    }

    def zeroPResponse(id: String, status: Int): PResponse = {
      val statusStr = status.toString
      PResponse(id, false, 0, zeroMlClaimResp, statusStr, statusStr)
    }

    def failedMlPredictResp(id: String, status: Int, errorMessage: String): PResponse =
      PResponse(id, false, 0, zeroMlClaimResp, errorMessage, status.toString)

    @inline
    final def isZero(pResponse: PResponse): Boolean = MlClaimResp.isZero(pResponse.claim)
  }



  /**
   * Representation of a line item with an order, CPT code with optional modifiers and
   * one or more ICD codes, ....
   *
   * @param order     Order of this line item in the claim
   * @param cpt       CPT code
   * @param modifiers Array of modifiers
   * @param icds      Array of ICD 10 codes
   * @param quantity  Number of unit if CPT requires it
   * @param unit      Type of units
   * @param score     Score for this line item
   */
  case class FeedbackLineItem(
    order: Int,
    cpt: String,
    modifiers: Seq[String],
    icds: Seq[String],
    quantity: Int,
    unit: String,
    score: Double
  ) {
    override def toString: String = s"$lineItemComma : $quantity : $unit : $score"

    def cptPrimaryIcd: String = if(icds.nonEmpty) s"$cpt ${icds.head}" else cpt

    def lineItemComma: String =
      if (modifiers.nonEmpty) s"$cpt ${modifiers.mkString(csvCodeSeparator)} ${icds.mkString(csvCodeSeparator)}"
      else s"$cpt ${icds.mkString(csvCodeSeparator)}"

    def lineItemSpace: String =
      if (modifiers.nonEmpty) s"$cpt ${modifiers.mkString(" ")} ${icds.mkString(" ")}"
      else s"$cpt ${icds.mkString(" ")}"
  }



  final object FeedbackLineItem {

    def apply(lineItemStr: String): FeedbackLineItem = {
      val codes = lineItemStr.split(codeGroupSeparator)
      val cpt = codes.head
      val remainingCodes = codes.tail
      val (modifiers, icds) = remainingCodes.partition(ContextVocabulary.modifiers.contains(_))
      FeedbackLineItem(0, cpt, modifiers, icds, 1, "UN", 1.0)
    }

    def apply(lineItem: MlLineItem): FeedbackLineItem =
      FeedbackLineItem(0, lineItem.cpt, lineItem.modifiers, lineItem.icds, lineItem.quantity, lineItem.unit, 0.0)

    def toLineItems(lineItemsStr: String): Seq[FeedbackLineItem] = {
      val lineItems = lineItemsStr.split(lineItemSeparator).map(_.trim)
      lineItems.map(apply(_))
    }

    def str(lineItems: Seq[FeedbackLineItem]): String =
      lineItems.map(_.lineItemSpace).mkString(lineItemSeparator)
  }



  /**
   * Feedback on codes with mention
   *
   * @param codeId   Code identifier
   * @param codeSet  Code set (ICD10, CPT, ..)
   * @param mentions List of mentions associated with the code
   */
  case class MlCodeWithMention(codeId: String, codeSet: String, mentions: String) {
    override def toString: String = {
      val mentionsStr = if (mentions != null && mentions.nonEmpty) mentions else ""
      s"id: $codeId, codeSet: $codeSet, mentions: $mentionsStr"
    }
  }


  /**
   * Wrapper to a set line items associated to a claim and one or more notes
   *
   * @param lineItems List of claim entry
   * @param codes     List of code with mention
   */
  case class MlClaimEntriesWithCodes(codes: Seq[MlCodeWithMention], lineItems: Seq[FeedbackLineItem]) {
    override def toString: String = {
      val codesStr = if (codes != null && codes.nonEmpty) codes.mkString("\n") else ""
      val lineItemsStr = if (lineItems != null && lineItems.nonEmpty) lineItems.mkString("\n") else ""
      s"$codesStr\n$lineItemsStr"
    }
  }

  final object MlClaimEntriesWithCodes {
    def apply(): MlClaimEntriesWithCodes = MlClaimEntriesWithCodes(
      Seq.empty[MlCodeWithMention],
      Seq.empty[FeedbackLineItem]
    )

    def apply(lineItems: Seq[FeedbackLineItem]): MlClaimEntriesWithCodes = MlClaimEntriesWithCodes(
      Seq.empty[MlCodeWithMention],
      lineItems
    )
  }



  /**
   * Request for Feedback from coder, and auditor.
   *
   * @param id          Identifier for the encounter or claim
   * @param autoCodable flag that specifies that the prediction request was originally set as auto-codable
   * @param context     Wrapper for contextual information
   * @param autocoded   Auto-coded (or predicted) line items
   * @param finalized   Finalized line items (prior to 837)
   * @param audited     Audited line items, if audit is performed
   */
  case class InternalFeedback(
    id: String,
    autoCodable: Boolean,
    context: InternalContext,
    autocoded: MlClaimEntriesWithCodes,
    finalized: MlClaimEntriesWithCodes,
    audited: MlClaimEntriesWithCodes
  ) {
    override def toString: String =
      s"""
         |Context: ${context.toString}
         |AutoCoded: ${if (autocoded != null) autocoded.toString else ""}
         |Finalized: ${finalized.toString}
         |Audited ${if (audited != null) audited.toString else ""}""".stripMargin

    final def hasFeedback: Boolean = finalized.lineItems.nonEmpty

    final def isSupported: Boolean = subModelTaxonomy.isSupported(context.emrLabel)

    final def isOracle: Boolean = subModelTaxonomy.isOracle(context.emrLabel)

    final def isTrained: Boolean = subModelTaxonomy.isTrained(context.emrLabel)


    def toFinalizedSpace: String = finalized.lineItems.map(_.lineItemSpace).mkString(lineItemSeparator).replace("  "," ")
  }

  final object InternalFeedback {
    def apply(): InternalFeedback = apply("", false, InternalContext(), MlClaimEntriesWithCodes())

    def apply(
      id: String,
      autoCodable: Boolean,
      context: InternalContext,
      finalized: MlClaimEntriesWithCodes): InternalFeedback =
      InternalFeedback(
        id,
        autoCodable,
        context,
        MlClaimEntriesWithCodes(),
        finalized,
        MlClaimEntriesWithCodes()
      )


    def apply(
      id: String,
      autoCodable: Boolean,
      context: InternalContext,
      finalizedLineItem: FeedbackLineItem): InternalFeedback =
      InternalFeedback(
        id,
        autoCodable,
        context,
        MlClaimEntriesWithCodes(),
        MlClaimEntriesWithCodes(Seq[FeedbackLineItem](finalizedLineItem)),
        MlClaimEntriesWithCodes()
      )

    def apply(
      id: String,
      autoCodable: Boolean,
      context: InternalContext,
      autocodedLineItem: FeedbackLineItem,
      finalizedLineItem: FeedbackLineItem): InternalFeedback =
      InternalFeedback(
        id,
        autoCodable,
        context,
        MlClaimEntriesWithCodes(Seq[FeedbackLineItem](autocodedLineItem)),
        MlClaimEntriesWithCodes(Seq[FeedbackLineItem](finalizedLineItem)),
        MlClaimEntriesWithCodes()
      )


    def apply(
      id: String,
      emrCodes: Seq[MlEMRCodes],
      finalized: MlClaimEntriesWithCodes): InternalFeedback =
      InternalFeedback(
        id,
        false,
        InternalContext(emrCodes),
        MlClaimEntriesWithCodes(),
        finalized,
        MlClaimEntriesWithCodes()
      )

    def apply(
      id: String,
      emrCodes: Seq[MlEMRCodes],
      autoCodedLineItems: Seq[FeedbackLineItem],
      finalizedLineItems: Seq[FeedbackLineItem]): InternalFeedback =
      InternalFeedback(
        id,
        false,
        InternalContext(emrCodes),
        MlClaimEntriesWithCodes(autoCodedLineItems),
        MlClaimEntriesWithCodes(finalizedLineItems),
        MlClaimEntriesWithCodes()
      )


    def apply(
      id: String,
      emrCodes: Seq[MlEMRCodes],
      finalizedLineItems: Seq[FeedbackLineItem]): InternalFeedback =
      InternalFeedback(
        id,
        false,
        InternalContext(emrCodes),
        MlClaimEntriesWithCodes(),
        MlClaimEntriesWithCodes(finalizedLineItems),
        MlClaimEntriesWithCodes()
      )

    def apply(
      id: String,
      emrCodes: Seq[MlEMRCodes],
      finalizedLineItem: FeedbackLineItem): InternalFeedback =
      apply(id, emrCodes, Seq[FeedbackLineItem](finalizedLineItem))



    def removeDuplicateLineItems(internalFeedback: InternalFeedback): InternalFeedback = {
      val finalizedLineItems = internalFeedback
          .finalized
          .lineItems
          .groupBy(_.lineItemSpace)
          .map(_._2.head)
          .toSeq
      val updateFinalized = internalFeedback.finalized.copy(lineItems = finalizedLineItems)
      internalFeedback.copy(finalized = updateFinalized)
    }
  }
}
