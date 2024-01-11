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
package org.bertspark.predictor

import org.bertspark.analytics.MetricsCollector
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalFeedback
import org.slf4j.{Logger, LoggerFactory}


/**
  * Simple processor for feedback (mainly statistics and quality metrics)
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] final class TFeedback extends MetricsCollector {
  protected[this] val lossName: String = "predictionLoss"
  final val logger: Logger = LoggerFactory.getLogger("TFeedback")

  /**
    * Update statistics with a new batch of feedbacks
    * @param feedbacks Sequence of feedbacks
    */
  def update(feedbacks: Seq[InternalFeedback]): Unit =
    feedbacks.map(feedback => updateMetrics(feedback))
}
