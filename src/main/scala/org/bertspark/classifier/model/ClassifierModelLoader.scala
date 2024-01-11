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
package org.bertspark.classifier.model

import ai.djl.ndarray.NDList
import ai.djl.repository.zoo._
import ai.djl.Application
import ai.djl.engine.Engine
import ai.djl.training.util.ProgressBar
import java.io.File
import java.nio.file._
import org.bertspark.classifier.block.ClassificationBlock
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.S3FsPathMapping._
import org.bertspark.util.io.ModelLoader
import org.slf4j._


/**
  * Loader for the classification model
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] trait ClassifierModelLoader extends ModelLoader[ZooModel[NDList, NDList]] {
  import ClassifierModelLoader._

  protected[this] val transformerModelName: String
  protected[this] val classificationModel: String
  protected[this] val subModelName: String
  protected[this] val classificationBlock: ClassificationBlock

  private def extractPath: Path = {
    val correctedFsModelDir: String = s3FsMap(classifierModelLbl)._2(subModelName).replace(" ", "_")
    val fsModelFolder = new File(correctedFsModelDir)

    if(!fsModelFolder.exists()) {
      val (s3ModelFile, _, fsModelFile) = paths(classifierModelLbl, subModelName)
      // We need to convert ' ' into '_' for the path in local directory
    //  val correctedFsModelDir = fsModelDir.replace(" ", "_")
      val correctedFsModelFile = fsModelFile.replace(" ", "_")

      // Move the files from S3 to local directory
      val succeeded = s3ToFs(correctedFsModelDir, correctedFsModelFile, s3ModelFile)
      if (succeeded) Paths.get(correctedFsModelDir) else null
    }
    else
      Paths.get(correctedFsModelDir)
  }

  /**
    * Load for the classifier model which copies the model from S3 to local file then load it
    * as a Zoo model
    * @return Option of ZooModel
    */
  override lazy val model: Option[ZooModel[NDList, NDList]] = {
    import org.bertspark.implicits._

    val path = extractPath
    if(path != null) {
      logDebug(logger, s"Loaded classifier from ${path.toAbsolutePath}")
      val optionsMap = Map[String, String]("epoch" -> "0")
      val options: java.util.Map[String, String] = optionsMap

      // Build the criteria for predictor
      val classificationCriteria: Criteria[NDList, NDList] = Criteria.builder()
          .optApplication(Application.UNDEFINED)
          .setTypes(classOf[NDList], classOf[NDList])
          .optOptions(options)
          .optModelUrls(s"file://${path.toAbsolutePath}")
          //   .optTranslator(new ClassifierTranslator)
          .optBlock(classificationBlock)
          .optEngine(Engine.getDefaultEngineName())
          .optProgress(new ProgressBar())
          .build()
      try {
        val thisModel = classificationCriteria.loadModel()
        Some(thisModel)
      }
      catch {
        case e: ModelNotFoundException =>
          logger.error(s"Failed loading training model ${path.toAbsolutePath}: ${e.getMessage}")
          None
        case e: java.nio.file.NoSuchFileException =>
          logger.error(s"IO error ${path.toAbsolutePath}: ${e.getMessage}")
          None
        case e: UnsupportedOperationException =>
          logger.error(s"Unsupported operation ${path.toAbsolutePath}: ${e.getMessage}")
          None
        case e: Exception =>
          logger.error(s"Undefined error ${path.toAbsolutePath}: ${e.getMessage}")
          None
      }
    }
    else
      None
  }
}


final object ClassifierModelLoader {
  final private val logger: Logger = LoggerFactory.getLogger("ClassifierModelLoader")

}