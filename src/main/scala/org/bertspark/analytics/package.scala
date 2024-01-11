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
package org.bertspark

package object analytics {
  type SubSummaryFields = (String, Int, Float, String)
  type CompareSummaryFields = (String, Int, String, String, Boolean)

  final val nullCompareSummaryFields = ("", -1, "", "", false)
  final val nullSubSummaryFields = ("", -1, -1.0F, "")

  // SubModelName,NumSuccesses,NumRecords,Rate,isMatch,Prediction,Label
  final val subModelIndex = 0
  final val numRecordsIndex = 2
  final val RateIndex = 3
  final val predictionIndex = 5
  final val labelIndex = 6

  trait Analyzer {
    def run(analyzerResult: AnalyzerResult): AnalyzerResult
  }

  case class AnalyzerResult(result: String, subSummaryFields: Seq[SubSummaryFields])

  final object AnalyzerResult {
    def apply(result: String): AnalyzerResult = AnalyzerResult(result, Seq.empty[SubSummaryFields])
    def apply(subSummaryFields: Seq[SubSummaryFields]): AnalyzerResult = AnalyzerResult("", subSummaryFields)
  }



  /**
   * Retrieve the keys with the last epoch for analysis of predictions
   * @param s3Keys List of S3 keys containing comparison statistics
   * @return List of S3 key sub-model with the latest epoch
   */
  def retrieveLastEpoch(s3Keys: Iterable[String]): Iterable[String] =
    if(s3Keys.nonEmpty)
      s3Keys.map(
        s3Key => {
          val index = s3Key.indexOf("/epoch-")
          if(index != -1) s3Key.splitAt(index + "/epoch-".length)
          else ("", "")
        }
      ).filter(_._1.nonEmpty)
          .groupBy(_._1)
          .map{ case (prefix, seq) => s"$prefix${seq.maxBy(_._2.toInt)._2}"}
    else
      Iterable.empty[String]


}
