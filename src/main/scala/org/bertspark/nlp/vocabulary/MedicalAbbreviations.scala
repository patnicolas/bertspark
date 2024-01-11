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
package org.bertspark.nlp.vocabulary

import org.apache.spark.sql.Dataset
import org.bertspark.config.MlopsConfiguration.DebugLog.logInfo
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.nlp.tokenSeparator
import org.bertspark.util.io.LocalFileUtil
import org.slf4j._


/**
 * Manage medical abbreviations for any given medical notes. The medical abbreviations have to be replacede
 * in the clinical note as part of the note preprocessing
 * @param abbreviationsMap Abbreviation map { abbreviations -> descriptor }
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class MedicalAbbreviations private (
  abbreviationsMap: Map[String, String]
) extends VocabularyComponent {
  require(abbreviationsMap.nonEmpty, "Abbreviations map is undefined")


  override val vocabularyName: String = "MedicalAbbreviations"

  /**
   * Replace abbreviations by their description. The detection of abbreviation is case sensitive and
   * needs to be processed before conversion to lower case.
   * @param textTokens tokens extracted from the content or note
   * @return Text tokens with abbreviations
   */
  override def build(textTokens: Array[String], requestDS: Dataset[InternalRequest]): Array[String] = build(textTokens)

  def build(textTokens: Array[String]): Array[String] =
    (textTokens ++ abbreviationsMap.flatMap{ case (k, v) => v.split(tokenSeparator) :+ k }).distinct

  @inline
  final def getAbbreviationsMap: Map[String, String] = abbreviationsMap


  final def getExpansionMap: Map[String, String] = abbreviationsMap.map{ case (k, v) => (v, k)}

  final def getAbbreviationDescriptors: Array[String] =
    abbreviationsMap.values.flatMap(_.split(tokenSeparator)).toArray.distinct
}


/**
 * Singleton for constructor and method to create and load medical abbreviations
 */
final object MedicalAbbreviations {
  final private val logger: Logger = LoggerFactory.getLogger("MedicalAbbreviations")

  final private val defaultMedicalAbbreviationsFile = "conf/medicalAbbreviations.csv"

  /**
   * Default, global instance of medical abbreviation
   */
  final lazy val medicalAbbreviationsInstance = create


  def apply(abbreviationsMap: Map[String, String]): MedicalAbbreviations =
    new MedicalAbbreviations(abbreviationsMap)


  def apply(medicalAbbreviationsFile: String): MedicalAbbreviations = {
    val abbreviationsMap = load(medicalAbbreviationsFile)
    logInfo(logger,  msg = s"Abbreviation map is loaded")
    new MedicalAbbreviations(abbreviationsMap)
  }

  def apply(): MedicalAbbreviations = apply(defaultMedicalAbbreviationsFile)

  /**
   * Load the medical abbreviations map. These abbreviations have to be replaced by their associated definitions
   * @param medicalAbbreviationsFile Medical abbreviation mapping file
   * @return Map {abbreviations -> Definition }
   */
  @throws(clazz = classOf[IllegalStateException])
  def load(medicalAbbreviationsFile: String): Map[String, String] =
    LocalFileUtil.Load.local(
      medicalAbbreviationsFile,
      (s: String) =>
        if(s.nonEmpty) {
          val fields = s.split("\t")
          if (fields.length > 1)
            (fields.head.trim, fields(1).trim)
          else {
            logger.warn(s"Abbreviation fields $s is incomplete")
            ("", "")
          }
        }
        else
          ("", "")
    ).getOrElse(
      throw new IllegalStateException(s"$medicalAbbreviationsFile not found or improperly formatted")
    ).filter(_._1.nonEmpty)
        .toMap



  /**
   * Generate a medical abbreviation map for medical terms from a default list of abbreviations
   * Only abbreviation with at least 2 characters are considered.
   */
  @throws(clazz = classOf[IllegalStateException])
  def create: Unit = {
    // Load generic medical abbreviations.
    val abbreviationsMap = LocalFileUtil.Load.local(
      defaultMedicalAbbreviationsFile,
      (s: String) => {
        val fields = s.trim.split(tokenSeparator)
        if (fields.size < 2 || fields.head.size < 2)
          ("", "")
        else {
          (fields.head, fields.tail.mkString(" "))
        }
      }
    ) .map(_.filter(_._1.nonEmpty))
      .getOrElse(
        throw new IllegalStateException(s"$defaultMedicalAbbreviationsFile not found or improperly formatted")
      )

    LocalFileUtil.Save.local(
      defaultMedicalAbbreviationsFile,
      abbreviationsMap.map { case (abbr, desc) => s"$abbr\t$desc" }.mkString("\n")
    )
  }
}
