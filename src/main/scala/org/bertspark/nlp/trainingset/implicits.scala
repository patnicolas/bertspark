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
package org.bertspark.nlp.trainingset

import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.nlp.trainingset.ContextualDocument.ContextualDocumentBuilder
import org.bertspark.util.rdbms.PredictionsTbl


/**
 * Implicit conversions from storage types (S3, RDBMS) into contextual document
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final object implicits {

  implicit def request2ContextualDocument(request: InternalRequest): ContextualDocument =
    ContextualDocumentBuilder.extractContextualDocument(request)

  implicit def requests2ContextualDocuments(requests: Seq[InternalRequest]): Array[ContextualDocument] =
    requests.map(ContextualDocumentBuilder.extractContextualDocument(_)).filter(_.id.nonEmpty).toArray


  implicit def prediction2ContextualDocument(prediction: Seq[String]): ContextualDocument =
    ContextualDocumentBuilder.extractContextualDocument(prediction)

  implicit def prediction2ContextualDocumentLimit(predictionsTblCondition: (PredictionsTbl, String, Int)): Seq[ContextualDocument] = {
    val predictions = predictionsTblCondition._1.defaultQuery(predictionsTblCondition._3, predictionsTblCondition._2)
    predictions.map(prediction2ContextualDocument(_)).filter(_.id.nonEmpty)
  }

  implicit def prediction2ContextualDocument(predictionsTblCondition: (PredictionsTbl, String)): Seq[ContextualDocument] = {
    val predictions = predictionsTblCondition._1.defaultQuery(-1, predictionsTblCondition._2)
    predictions.map(prediction2ContextualDocument(_)).filter(_.id.nonEmpty)
  }
}
