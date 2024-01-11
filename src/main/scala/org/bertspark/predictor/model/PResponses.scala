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
package org.bertspark.predictor.model

import org.bertspark.modeling.SubModelsTaxonomy.subModelTaxonomy
import org.bertspark.nlp.medical.MedicalCodingTypes._
import org.slf4j._



  /**
    * {{{
    *  Breaks down predictions into 3 categories
    *  - Oracle or deterministic  1 Sub-model - 1 label
    *  - Predictive  1 Sub-model - 2+ labels
    *  - Unsupported 1 Sub-model - 0 label
    * }}}
    *
    * @param oracleResponses      Deterministic responses (label) associated with a sub-model
    * @param predictedResponses   Set of predictions
    * @param unsupportedResponses Un supported (either Oracle nor predictions)
    *
    * @author Patrick Nicolas
    * @version 0.7
    */

final class PResponses private(
    oracleResponses: Seq[PResponse],
    predictedResponses: Seq[PResponse],
    unsupportedResponses: Seq[PResponse]) {
  import PResponses._

    def allResponses: Seq[PResponse] = oracleResponses ++ predictedResponses ++ unsupportedResponses

    final def getStats: String =
      if(allResponses.nonEmpty) {
        val numResponses = allResponses.size
        val scaleFactor = 100.0F / numResponses
        val codingRate = (oracleResponses.size + predictedResponses.size) * scaleFactor
        s"Coding rate: $codingRate% Oracle rate: ${oracleResponses.size * scaleFactor}% Prediction rate: ${predictedResponses.size * scaleFactor}%"
      }
      else {
        logger.error(s"Cannot extract coding rates from empty responses")
        ""
      }
}


/**
  * Singleton for constructors
  */
private[bertspark] final object PResponses {
  final private val logger: Logger = LoggerFactory.getLogger("PResponses")

  def apply(
    oracleResponses: Seq[PResponse],
    predictedResponses: Seq[PResponse],
    unsupportedResponses: Seq[PResponse]): PResponses =
      new PResponses(oracleResponses, predictedResponses, unsupportedResponses)

  def apply(): PResponses = new PResponses(Seq.empty[PResponse], Seq.empty[PResponse], Seq.empty[PResponse])

  def apply(inputRequests: Seq[PRequest]): PResponses = subModelTaxonomy(inputRequests)
}

