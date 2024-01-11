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
import org.slf4j.{Logger, LoggerFactory}
import scala.util.Random


/**
  * Wrapper for the data/token augmentation techniques that replace randomly a character on a token (context or text)
  * which is not a descriptor of code in label
  * @author Patrick Nicolas
  * @version 0.8
  */
private[bertspark] object RandomCharSubstitute {
  final private val logger: Logger = LoggerFactory.getLogger("RandomCharSubstitute")
  private val randCharIndex = new Random(90343L)

  /**
    * Replace an existing token (not defined in description of label codes) by same token with a randomly
    * altered character (value -> vxlue)
    */
  val substitute: (TokenizedTrainingSet, Set[String], Int) => TokenizedTrainingSet  =
    (tokenizedTrainingSet: TokenizedTrainingSet, labelDescriptionTokens: Set[String], idx: Int) => {
      val contextualDocument = tokenizedTrainingSet.contextualDocument
      val contextTokens = contextualDocument.contextVariables
      val textTokens = contextualDocument.text.split(tokenSeparator)
      val numTokens = contextTokens.length + textTokens.length
      val augId = augmentId(contextualDocument.id, idx)

      var augmentedContextualDoc: Option[ContextualDocument] = None
      var count = 0
      do {
        // Select a random index
        val indexUnknownTokens = randTokenIndex.nextInt(numTokens - 1)
        val relativeIndex = indexUnknownTokens - contextTokens.length

        // If the substitution should happen in the context tokens....
        if (indexUnknownTokens < contextTokens.length &&
            !labelDescriptionTokens.contains(contextTokens(indexUnknownTokens))) {

          val modifiedToken = replaceChar(contextTokens(indexUnknownTokens))
          logDebug(logger, msg = s"Token ${contextTokens(indexUnknownTokens)} replaced by $modifiedToken")
          contextTokens(indexUnknownTokens) = modifiedToken
          augmentedContextualDoc = Some(ContextualDocument(augId, contextTokens, contextualDocument.text))
        }

        // .. or in the text ...
        else if (relativeIndex > 0 &&
            !labelDescriptionTokens.contains(textTokens(relativeIndex))) {

          val modifiedToken = replaceChar(textTokens(relativeIndex))
          logDebug(logger, msg = s"Token ${textTokens(relativeIndex)} replaced by $modifiedToken")
          textTokens(relativeIndex) = modifiedToken
          augmentedContextualDoc = Some(ContextualDocument(augId, contextTokens, textTokens.mkString(" ")))
        }
        count += 1
      } while(augmentedContextualDoc.isEmpty && count < maxNumSubstituteSearch)

      augmentedContextualDoc
          .map(augmentedCtxDoc => tokenizedTrainingSet.copy(contextualDocument = augmentedCtxDoc))
          .getOrElse({
            logger.warn(
              s"Failed to randomly modify token random character for ${tokenizedTrainingSet.label}"
            )
            tokenizedTrainingSet
          })
    }


  private def replaceChar(selectedToken: String): String = {
    val chars = selectedToken.toCharArray
    val randomCharIndex = randCharIndex.nextInt(chars.length-1)
    chars(randomCharIndex) = 'x'
    new String(chars)
  }

}
