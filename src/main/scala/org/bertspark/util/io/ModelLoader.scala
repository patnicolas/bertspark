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
package org.bertspark.util.io

import java.io.File
import org.bertspark.delay
import org.slf4j.{Logger, LoggerFactory}


/**
  * Generic loader for Deep learning models such as Transformer and Feed forward neural networks
  * By default, all models are loaded from S3 into local file before instantiated through the Criteria class
  * @tparam T Type of DL model to load
  *
  * @author Patrick Nicolas
  * @version 0.6
  */
trait ModelLoader[T] {
self =>
  val model: Option[T]

  protected def s3ToFs(
    fsModelDir: String,
    fsModelFile: String,
    s3ModelFile: String,
    deleteExistingFile: Boolean = false): Boolean = try {
    val dir = new File(fsModelDir)
    if(!dir.exists())
      dir.mkdirs()

    val existingFile = new File(fsModelFile)
    // If the existing file has to be deleted
    if(deleteExistingFile) {
      if(existingFile.exists()) {
        existingFile.delete()
        delay(1000L)
      }
      S3Util.s3ToFs(fsModelFile, s3ModelFile, isText = false)
      true
    }
    else if(!existingFile.exists()) {
      S3Util.s3ToFs(fsModelFile, s3ModelFile, isText = false)
      true
    }
    else
      true
  } catch {
    case e: Exception =>
      ModelLoader.logger.error(e.getMessage)
      false
  }

  def isReady: Boolean = model.isDefined
}


private[bertspark] final object ModelLoader {
  final private val logger: Logger = LoggerFactory.getLogger("ModelLoader")

}