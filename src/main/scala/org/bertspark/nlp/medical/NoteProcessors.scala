package org.bertspark.nlp.medical

import org.bertspark.config.MlopsConfiguration
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.nlp.medical.MedicalCodingTypes.Modality.getModalityFromCpt
import org.bertspark.nlp.medical.ContextEncoder.encodeContext
import org.bertspark.nlp.tokenSeparator
import org.bertspark.util.io.S3Util._
import org.slf4j._
import scala.collection.mutable.ListBuffer

/**
 * Utilities to process document or notes
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final object NoteProcessors {
  final private val logger: Logger = LoggerFactory.getLogger("NoteProcessors")

  final private val tokenRegex = "(\\S+?)([,?!])?(\\s+|$)".r
  final val findingsReplacement = "sectionfindings"
  final val impressionReplacement = "sectionimpression"
  val numRegexPerCent = "([0-9]+%)"
  val numLabelPerCent = "zpercent"
  val numRegex = "([0-9.%]+)"
  val numAtLeastOneChar = ".*[A-Za-z]+.*"
  val numLabel = "xnum"

  val eolCleanserRegex1 = "\\\\n"
  val eolCleanserRegex2 = "\\n"
  val eolCleanserRegex3 = "\\\\"
  final val specialCharCleanserRegex = "[^(a-zA-Z0-9)']"


  /**
   * Extract tokens from a document and convert to lower case
   *
   * @param text Content of the document
   * @return Array of tokens
   */
  def extractTokens(text: String): Array[String] = {
    val matcher = tokenRegex.findAllIn(text)
    val collector = ListBuffer[String]()
    while (matcher.hasNext) {
      collector.append(matcher.next().toLowerCase)
    }
    collector.toArray
  }

  type TokenizeFunc = InternalRequest => Array[String]


  /**
   * Cleanser function for end of line and extraction of terms
   */
  final val eolCleanserFunc: TokenizeFunc = (request: InternalRequest) =>
    if (request.notes.nonEmpty)
      cleanse(request.notes.head)
    else {
      logger.warn(s"Could cleanse an empty note for ${request.id}")
      Array.empty[String]
    }

  /**
   * Cleanser function for end of line and special characters and extraction of terms
   */
  final val eolAndSpecialCharCleanserFuncA: TokenizeFunc = (request: InternalRequest) =>
    if (request.notes.nonEmpty)
      cleanse(request.notes.head, specialCharCleanserRegex)
    else {
      logger.warn(s"Could cleanse an empty note for ${request.id}")
      Array.empty[String]
    }

  final val contextualValuesFunc: TokenizeFunc = (request: InternalRequest) => encodeContext(request.context)

    /**
    * {{{
    * - Cleanse a content by replacing a list of special characters with empty space.
    * - Replace new line character with 'creturn'
    * - Replace any numeric 3.5 by 'x.x'
    * - Replace any percentage 3.6x by 'z.z%'
    * - Replace any FINDINGS  by 'sectionfindings'
    * - Replace any IMPRESSION  by 'sectionimpression'
    * - Allowed on string characters
    * }}}
    * @param content                  Text note
    * @param specialCharacterToRemove List of special characters to replace
    * @return Return terms
    */
  @throws(clazz = classOf[IllegalArgumentException])
  def cleanse(content: String, specialCharacterToRemove: String = ""): Array[String] = {

    val contentWithoutEndOfLine = content
        .replaceAll(eolCleanserRegex1, " ")
        .replaceAll(eolCleanserRegex2, " ")
        .replaceAll(eolCleanserRegex3, " ")

    if (contentWithoutEndOfLine.nonEmpty)
      contentWithoutEndOfLine.split(tokenSeparator)
          .map(token =>
            if(token.matches(numRegex)) "xnum" else  removeSpecialCharsFromToken(token.toLowerCase)
          )
          .filter(_.size > 1)
    else
      Array.empty[String]
  }

  final private val specialCharsMap = Set[Char]('%', '(', ')', '.', ':', '!', '?', '-', ';', ',', '\'')
  private def removeSpecialCharsFromToken(token: String): String =
    new String(token.toCharArray.filter(!specialCharsMap.contains(_)))


  def normalizeRawRequest(
    s3InputFolder: String,
    s3OutputFolder: String,
    thisCustomer: String): Unit = {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    import MlopsConfiguration._

    val inputDS = try {
      s3ToDataset[InternalRequest](
        mlopsConfiguration.storageConfig.s3Bucket,
        s3InputFile = s3InputFolder,
        header = false,
        fileFormat = "json")
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"normalizeRawRequest: ${e.getMessage}")
        sparkSession.emptyDataset[InternalRequest]
    }

    val outputDS = inputDS.map(
      predict => {
        val inferredModality =
          if (predict.context.EMRCpts != null && predict.context.EMRCpts.nonEmpty)
            getModalityFromCpt(predict.context.EMRCpts.head.cpt)
          else
            "unknown"
        val updatedContext = predict.context.copy(modality = inferredModality, customer = thisCustomer)
        predict.copy(context = updatedContext)
      }
    )
    datasetToS3[InternalRequest](
      mlopsConfiguration.storageConfig.s3Bucket,
      outputDS,
      s3OutputFolder,
      header = false,
      fileFormat = "json",
      toAppend = true,
      numPartitions = 8)
  }
}
