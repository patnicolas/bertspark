package org.bertspark.nlp.vocabulary

import org.apache.spark.sql.Dataset
import org.bertspark.config.MlopsConfiguration.DebugLog.logInfo
import org.bertspark.config.MlopsConfiguration.vocabulary
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.medical.MedicalCodingTypes.{FeedbackLineItem, InternalFeedback, InternalRequest}
import org.bertspark.nlp.tokenSeparator
import org.bertspark.util.io.LocalFileUtil
import org.bertspark.util.io.S3Util.S3UploadDownloadSeq
import org.bertspark.InvalidParamsException
import org.slf4j._


private[vocabulary] final class MedicalCodeDescriptors extends VocabularyComponent {
  import MedicalCodeDescriptors._

  override val vocabularyName: String = "MedicalCodeDescriptors"

  override def build(initialTokens: Array[String], requestDS: Dataset[InternalRequest]): Array[String] = build(initialTokens)

  def build(initialTokens: Array[String]): Array[String] = {
    // MedicalCodeDescriptors.build
    initialTokens ++ getCptIcdTerms
  }
}

/**
 * {{{
 * Implement semantic mapping for CPT and ICD codes: {code -> Descriptive terms}. The semantic is used for
 * - descriptive terms are added to the vocabulary
 * - coding map are potentially used in token embedding beside context and note
 * }}}
 *
 * @author Patrick Nicolas
 * @version 0.2
 */
private[bertspark] final object MedicalCodeDescriptors {
  final private val logger: Logger = LoggerFactory.getLogger("MedicalCodeDescriptors")

  final val fsCptCodesPath: String = "conf/codes/ClinicianDescriptor.txt"
  final private val fsIcdCodesPath: String = "conf/codes/icdCodes.txt"
  type CodeDescriptor = (String, Array[String])

  def apply(): MedicalCodeDescriptors = new MedicalCodeDescriptors


  def build: Unit = {
    val codeDescriptors: Array[CodeDescriptor] = getCptDescriptors
    CodeDescriptorMap.s3UploadDownloadIcdCptDescriptors.upload(codeDescriptors)
  }


  def getCptDescriptors: Array[CodeDescriptor] =
      LocalFileUtil.Load.local(fsCptCodesPath, (s: String) => {
        val ar: Array[String] = s.split("\t")
        if (ar.size > 3) {
          val cptCode = ar(1)
          val descriptor = ar(3).toLowerCase
          (cptCode, descriptor)
        } else
          ("", "")
      }, header = true)
          .getOrElse(throw new IllegalStateException("Failed loading CPT descriptors"))
          .filter(_._1.nonEmpty)
          .map {
            case (cpt, description) =>
              (cpt, description.split(tokenSeparator).filter(_.size > 1).distinct)
          }


  /**
   * Extract pair ICD -> ICD terms
   *
   * @return Map[ICD, ICD terms]
   */
  @throws(clazz = classOf[IllegalStateException])
  def getIcdDescriptors: Array[CodeDescriptor] =
    LocalFileUtil.Load.local(fsIcdCodesPath, (s: String) => {
      val ar = s.split("   ")
      if (ar.size > 1) (ar.head.trim, ar(1).trim) else ("", "")
    }, header = true)
        .getOrElse(throw new IllegalStateException("Failed loading Icd descriptors"))
        .filter(_._1.nonEmpty)
        .map {
          case (icd, description) => {
            val icdTerms = description.split(tokenSeparator)
                .filter(_.size > 3)
                .map(_.toLowerCase)
                .filter(_.forall(ch => ch > 96 && ch < 127))
            (icd, icdTerms.distinct)
          }
        }

  /**
    * Extract the ICDs related to left or right body parts
    */
  lazy val (leftIcds, rightIcds) = {
    import IcdConversion._

    val icdDescriptors = getIcdDescriptors
    val _leftIcds = icdDescriptors.filter(_._2.contains("left")).map(_._1).map{toComma(_)}
    val _rightIcds = icdDescriptors.filter(_._2.contains("right")).map(_._1).map(toComma(_))
    (_leftIcds, _rightIcds)
  }

  final object IcdConversion {
    def toComma(icd: String): String = s"${icd.substring(0, 3)}.${icd.substring(3)}"
    def fromComma(icd: String): String = icd.replace(".", "")
  }


  lazy val icdsRef = LocalFileUtil.Load.local(fsIcdCodesPath, (s: String) => {
    val ar = s.split(tokenSeparator)
    if (ar.nonEmpty) ar.head.trim else "9999"
  }, header = true)
      .getOrElse(throw new IllegalStateException(s"Could not loaded ICD codes reference file $fsIcdCodesPath"))


  /**
   * Extract a combination of CPT and ICD terms descriptors terms
   *
   * @return Array of unique terms used in the descriptors of
   */
  @throws(clazz = classOf[IllegalStateException])
  def getCptIcdTerms: Array[String] = {
    // Load the cpt descriptors
    val cptDescriptors = LocalFileUtil.Load.local(fsCptCodesPath, (s: String) => {
      val ar = s.split("\t")
      if (ar.size > 2) ar(3).toLowerCase else ""
    }, header = true)
        .getOrElse(throw new IllegalStateException("Failed loading CPT descriptors"))
        .filter(_.nonEmpty)
        .flatMap(_.split(tokenSeparator).filter(_.size > 3).map(_.toLowerCase))
        .distinct
        .filter(_.forall(ch => ch > 96 && ch < 127))

    // Load the ICD descriptors
    val icdDescriptors = LocalFileUtil.Load.local(fsIcdCodesPath, (s: String) => {
      val ar = s.split("   ")
      if (ar.size > 1) ar(1).trim else ""
    }, header = true)
        .getOrElse(throw new IllegalStateException("Failed loading Icd descriptors"))
        .filter(_.nonEmpty)
        .flatMap(_.split(tokenSeparator).filter(_.size > 3).map(_.toLowerCase))
        .distinct
        .filter(_.forall(ch => ch > 96 && ch < 127))
    (cptDescriptors ++ icdDescriptors).distinct
  }


  /**
    * Code descriptor map as loaded from S3
    */
  final object CodeDescriptorMap {
    val s3UploadDownloadIcdCptDescriptors = new S3UploadDownloadSeq[CodeDescriptor](
      S3PathNames.s3CodeDescriptorFile,
      (str: String) => {
        val ar = str.split(",")
        (ar.head, ar.tail.map(_.trim))
      },
      (codeDescriptor: CodeDescriptor) => s"${codeDescriptor._1},${codeDescriptor._2.mkString(",")}"
    )

    final private val noCptDescriptorLbl = "no_cpt_descriptor"

    private val descriptorMap: Map[String, Array[String]] = {
      val codeDescriptors = s3UploadDownloadIcdCptDescriptors.download
      if (codeDescriptors.isEmpty)
        throw new InvalidParamsException(s"No code descriptors from ${S3PathNames.s3CodeDescriptorFile}")

      val descriptorsMap = codeDescriptors.toMap
      logInfo(logger, msg = s"Created a descriptor map for ${descriptorsMap.size} codes")
      descriptorsMap
    }

    final def validate: Boolean = descriptorMap.nonEmpty

    final def size: Int = descriptorMap.size

    /**
      * Get descriptors associated with this code
      * {{{
      *   First attempt to extract the associated token for the entire code which may contain modifiers
      *   If failed, then select only the CPT and find the associated tokens.
      * }}}
      *
      * @param code CPT, CPT with modifiers or ICD code
      * @return List of terms associated with this code or CPT
      */
    def getDescriptors(code: String): Array[String] =
      if (descriptorMap.nonEmpty) {
        val codeDescriptors = descriptorMap.getOrElse(code, {
          val individualCodes = code.split(" ")
          if (individualCodes.size > 1)
            descriptorMap.getOrElse(individualCodes.head, Array.empty[String])
          else
            Array.empty[String]
        })
        if (codeDescriptors.isEmpty)
          logger.warn(s"Could not find associated descriptors for $code")
        codeDescriptors
      }
      else
        Array[String](noCptDescriptorLbl)


    /**
      * Retrieve the descriptor labels from claim
      * @param lineItemsStr Claim or line items a string
      * @return Set of unique tokens/words associated with the codes of the claim
      */
    def getClaimDescriptors(lineItemsStr: String): Seq[String] = {
      val feedbackLineItems = FeedbackLineItem.toLineItems(lineItemsStr)
      feedbackLineItems.flatMap(
        lineItem => {
          getDescriptors(lineItem.cpt) ++ lineItem.icds.flatMap(icd => getDescriptors(icd))
        }
      ).distinct.map(_.trim)
    }

    def loadMedicalDescriptorsMap: Map[String, Array[String]] =
      s3UploadDownloadIcdCptDescriptors.download.toMap
  }
}
