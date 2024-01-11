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
import org.bertspark.nlp.augmentation.RandomAugmentation.{augmentId, logger, maxNumSubstituteSearch}
import org.bertspark.nlp.augmentation.RecordsAugmentation.randTokenIndex
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.{ContextualDocument, TokenizedTrainingSet}
import org.slf4j.{Logger, LoggerFactory}


/**
  * Wrapper for the data/token augmentation techniques that replace randomly a token (context or text)
  * which is not a descriptor of code in label by [UNK]
  * @author Patrick Nicolas
  * @version 0.8
  */
private[bertspark] object RandomTokenSubstitute {
  final private val logger: Logger = LoggerFactory.getLogger("RandomTokenSubstitute")

  /**
    * Replace an existing token (not defined in description of label codes) by [UNK]
    * altered character (value -> [UNK])
    */
  val substitute: (TokenizedTrainingSet, Set[String], Int) => TokenizedTrainingSet =
    (tokenizedTrainingSet: TokenizedTrainingSet, labelDescriptionTokens: Set[String], idx: Int) => {
      val contextualDocument = tokenizedTrainingSet.contextualDocument
      val contextTokens = contextualDocument.contextVariables
      val textTokens = contextualDocument.text.split(tokenSeparator)
      val numTokens = contextTokens.length + textTokens.length

      var augmentedContextualDoc: Option[ContextualDocument] = None
      val augId = augmentId(contextualDocument.id, idx)
      var count = 0
      do {
        val indexUnknownTokens = randTokenIndex.nextInt(numTokens - 1)
        val relativeIndex = indexUnknownTokens - contextTokens.length

        // If the substitution should happen in the context tokens....
        if (indexUnknownTokens < contextTokens.length &&
            !labelDescriptionTokens.contains(contextTokens(indexUnknownTokens))) {
          logDebug(logger, msg = s"Token ${contextTokens(indexUnknownTokens)} replaced by [UNK]")

          contextTokens(indexUnknownTokens) = "[UNK]"
          augmentedContextualDoc = Some(ContextualDocument(augId, contextTokens, contextualDocument.text))
        }

        // .. or in the text ...
        else if (relativeIndex > 0 &&
            !labelDescriptionTokens.contains(textTokens(relativeIndex))) {

          logDebug(logger, msg = s"Token ${textTokens(relativeIndex)} replaced by [UNK]")
          textTokens(relativeIndex) = "[UNK]"
          augmentedContextualDoc = Some(ContextualDocument(augId, contextTokens, textTokens.mkString(" ")))
        }
        count += 1
      } while(augmentedContextualDoc.isEmpty && count < maxNumSubstituteSearch)

      augmentedContextualDoc
          .map(augmentedCtxDoc => tokenizedTrainingSet.copy(contextualDocument = augmentedCtxDoc))
          .getOrElse({
            logger.warn(
              s"Failed to randomly substitute token by [UNK] for ${tokenizedTrainingSet.label}"
            )
            tokenizedTrainingSet
          })
    }


}
