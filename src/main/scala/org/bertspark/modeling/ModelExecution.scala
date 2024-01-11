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
package org.bertspark.modeling

import org.bertspark.config.MlopsConfiguration._
import org.bertspark.config.MlopsConfiguration.DebugLog._
import org.bertspark.util.io._
import org.slf4j._


/**
 * Generic training session for any BERT model
 * {{{
 *   Arguments:  [S3Bucket] [S3Folder] [isTraining]
 * }}}
 * @param isTraining Specify training if true, prediction otherwise.
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] abstract class ModelExecution {
  import ModelExecution._
  /**
   * Execute the training or prediction for any given model
   */
  def apply(): Float = {
    // It is assumed that validation has been initialized
    logInfo(logger,  msg = s"Is service configured: ${mlopsConfiguration.isValid}")

    val start = System.currentTimeMillis()
    val accuracy = train
    logInfo(logger,  msg = s"Total duration = ${(System.currentTimeMillis() - start)*0.001} secs. ")
    accuracy
  }

  protected[this] def train(): Float
}


private[bertspark] final object ModelExecution {
  final val logger: Logger = LoggerFactory.getLogger("ModelExecution")

  def createRequestRouting: Unit = {
    logDebug(logger, msg = s"Update RequestRouting for ${mlopsConfiguration.target}")
    val subModelDescriptor = S3Util.download(
      mlopsConfiguration.storageConfig.s3Bucket,
      s"${mlopsConfiguration.storageConfig.s3RootFolder}/${mlopsConfiguration.target}/models/${mlopsConfiguration.runId}/subModels.csv"
    )
    subModelDescriptor.map(
      content => {
        val lines = content.split("\n")
        val entries = lines.map(
          line => {
            val fields = line.split(",")
            val subModelName = fields.head
            val count = fields(1).toInt
            val property = if(count == 1) "Oracle" else "Predictive"
            s"${mlopsConfiguration.target},$subModelName,$property"
          }
        )
        LocalFileUtil.Save.local(
          fsFileName = "conf/customerSubModels.csv",
          entries.mkString("\n")
        )
      }
    )
  }
}