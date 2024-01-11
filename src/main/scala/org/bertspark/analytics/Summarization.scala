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
package org.bertspark.analytics

import org.bertspark.util.io.S3Util
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames.getS3FolderCompareSource
import org.bertspark.nlp.medical.MedicalCodingTypes.codeGroupSeparator
import org.bertspark.util.io.LocalFileUtil.CSV_SEPARATOR
import org.slf4j.{Logger, LoggerFactory}


/**
  *
  * @param s3CompareFolder Source folder for the summarization
  * @param isPrimaryCodeMatchOnly
  * @author Patrick Nicolas
  * @version 0.4
  */
private[bertspark] final class Summarization private (
  s3CompareFolder: String,
  isPrimaryCodeMatchOnly: Boolean) extends Analyzer {
  import Summarization._

  override def run(analyzerResult: AnalyzerResult): AnalyzerResult = {
    val allKeys = S3Util.getS3Keys(mlopsConfiguration.storageConfig.s3Bucket, s3CompareFolder).filter(_.contains(s"epoch-"))
    if(allKeys.isEmpty)
      throw new IllegalStateException(s"Failed to retrieve the comparison data for $s3CompareFolder")

    val keys = retrieveLastEpoch(allKeys)
    AnalyzerResult(getSubSummaryFields(keys, isPrimaryCodeMatchOnly).filter(_._3 != -1))
  }
}


private[bertspark] final object Summarization {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[Summarization])

  // Summary fields with (Sub-model, count, rate, label)

  def apply(s3SourceFolder: String, isPrimaryCodeMatchOnly: Boolean): Summarization =
    new Summarization(s3SourceFolder, isPrimaryCodeMatchOnly)

  def apply(isPrimaryCodeMatchOnly: Boolean): Summarization =
    new Summarization(getS3FolderCompareSource, isPrimaryCodeMatchOnly)

  /**
   * {{{
   * Retrieve the sub summary fields from the compare files located in S3 (mlops/$target/compare)
   * The compare has 8 fields from which 5 are extracted
   *        SubSummaryFields: (subModelName, epochNo, totalCount for subModel, rate, label)
   * Input: Name of the S3 folder containing the comparison between prediction and label
   *        Epoch no (usually the last epoch)
   * Output:  List of Sub summary fields associated with the compare file
   * }}}
   * @return List of Sub summary fields associated with the compare file
   */
  def getSubSummaryFields(keys: Iterable[String], isPrimaryCodeMatchOnly: Boolean): Seq[SubSummaryFields] =
    if(keys.nonEmpty)
      keys.foldLeft(List[Seq[SubSummaryFields]]())(
        (xs, key) => {
          S3Util.download(mlopsConfiguration.storageConfig.s3Bucket, key).map(
            content => {
              val subSummaryFields: Seq[SubSummaryFields] = content.split("\n").map(
                line => {
                  val fields = line.split(CSV_SEPARATOR).toSeq
                  if (fields.size >= labelIndex)
                    (
                        fields(subModelIndex),
                        fields(numRecordsIndex).toInt,
                        computeRate(fields, isPrimaryCodeMatchOnly),
                        fields(labelIndex)
                    )
                  else {
                    logger.error(s"line $line for S3 key $key is incomplete or empty")
                    nullSubSummaryFields
                  }
                }
              )   .toSeq
                  .filter(_._1.nonEmpty)
              if(subSummaryFields.nonEmpty) subSummaryFields else Seq[SubSummaryFields](nullSubSummaryFields)
            }
          ).getOrElse(Seq[SubSummaryFields](nullSubSummaryFields)) :: xs
        }
      ).flatten
    else {
      logger.error("Could not find S3 paths")
      Seq.empty[SubSummaryFields]
    }


  private def computeRate(fields: Seq[String], isPrimaryCodeOnly: Boolean): Float =
    if(isPrimaryCodeOnly) {
      val labeledFieldIndex =  if(fields.size == labelIndex+1) labelIndex else labelIndex+1
      val predictedFields = fields(predictionIndex).split(codeGroupSeparator)
      val labeledFields = fields(labeledFieldIndex).split(codeGroupSeparator)

      if (predictedFields.head == labeledFields.head) 1.0F else 0.0F
    }
    else
      fields(RateIndex).toFloat
}
