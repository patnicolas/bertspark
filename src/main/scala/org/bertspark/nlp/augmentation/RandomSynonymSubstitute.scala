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
package org.bertspark.nlp.augmentation

import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.nlp.augmentation.RandomAugmentation.{augmentId, maxNumSubstituteSearch}
import org.bertspark.nlp.augmentation.RecordsAugmentation.randTokenIndex
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.{ContextualDocument, TokenizedTrainingSet}
import org.bertspark.util.io.LocalFileUtil
import org.slf4j.{Logger, LoggerFactory}




/**
  * Wrapper for the data/token augmentation techniques that replace randomly a token (context or text)
  * which is not a descriptor of code in label by a synonym loaded from a local file
  * @author Patrick Nicolas
  * @version 0.8
  */
private[bertspark] object RandomSynonymSubstitute {
  final private val logger: Logger = LoggerFactory.getLogger("RandomSynonymSubstitute")
  import org.bertspark.config.MlopsConfiguration._
  final private val augmentationFilename = "conf/codes/synonyms.csv"
  final private val correctedTokenFilename = "conf/codes/similartokens.csv"

  lazy val conversionMap = mlopsConfiguration.classifyConfig.augmentation match {
    case `randomAugSyn` => augmentationMap
    case `randomAugCorrect` => correctedTokenMap
    case `randomAugSynAndCorrect` => augmentationMap ++ correctedTokenMap
    case _ => Map.empty[String, String]
  }

  private lazy val augmentationMap: Map[String, String] = {
    val initial = LocalFileUtil.Load.local(augmentationFilename).map(
      content =>
        content
            .split("\n")
            .map(
              line => {
                val fields = line.split(",")
                if (fields.size == 2)
                  (fields.head, fields(1))
                else
                  ("", "")
              }
            ).toMap
    ).getOrElse({
      logger.error("Failed to load conf/codes/synonyms.csv")
      Map.empty[String, String]
    })

    initial ++ initial.map { case (k, v) => (v, k) }
  }

  private lazy val correctedTokenMap: Map[String, String] =
    LocalFileUtil.Load.local(correctedTokenFilename).map(
      content =>
        content
            .split("\n")
            .map(
              line => {
                val fields = line.split(",")
                if (fields.size == 2) {
                  val subFields = fields(1).split("\\|")
                  (fields.head, subFields.head)
                } else
                  ("", "")
              }
            ).toMap
    ).getOrElse({
      logger.error("Failed to load conf/codes/synonyms.csv")
      Map.empty[String, String]
    })


  /**
    * Replace an existing token (not defined in description of label codes) by pseudo-synonym from a list
    * loaded from a local file
    */
  val substitute: (TokenizedTrainingSet, Set[String], Int) => TokenizedTrainingSet  =
    (tokenizedTrainingSet: TokenizedTrainingSet, _: Set[String], idx: Int) => {
      val contextualDocument = tokenizedTrainingSet.contextualDocument
      val textTokens = contextualDocument.text.split(tokenSeparator)
      val numTokens = textTokens.length

      var augmentedContextualDoc: Option[ContextualDocument] = None
      val augId = augmentId(contextualDocument.id, idx)
      var count = 0

      do {
        val substituteTokenIndex = randTokenIndex.nextInt(numTokens - 1)
        conversionMap.get(textTokens(substituteTokenIndex)).map(
          converted => {
            logDebug(logger, msg = s"Token ${textTokens(substituteTokenIndex)} replaced by $converted")
            textTokens(substituteTokenIndex) = converted
            augmentedContextualDoc = Some(
              ContextualDocument(augId, contextualDocument.contextVariables, textTokens.mkString(" "))
            )
          }
        )
        count += 1
      } while(augmentedContextualDoc.isEmpty && count < maxNumSubstituteSearch)

      augmentedContextualDoc
          .map(augmentedCtxDoc => tokenizedTrainingSet.copy(contextualDocument = augmentedCtxDoc))
          .getOrElse({
            logger.warn(
              s"Failed to randomly substitute token by synonym for ${tokenizedTrainingSet.label}"
            )
            tokenizedTrainingSet
          })
    }

}
