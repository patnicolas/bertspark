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
package org.bertspark.transformer.model

import ai.djl.engine.Engine
import ai.djl.repository.zoo._
import ai.djl.training.util.ProgressBar
import ai.djl.Application
import ai.djl.ndarray.NDList
import java.nio.file.Paths
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.S3FsPathMapping
import org.bertspark.transformer.block.BERTFeaturizerBlock
import org.bertspark.util.io.ModelLoader
import org.slf4j._


/**
  * Load the transformer model from S3, copy to the local file system then initialized the
  * relevant Zoo model.
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] trait TransformerModelLoader extends ModelLoader[ZooModel[NDList, NDList]] {
  protected[this] val preTrainingBlock: BERTFeaturizerBlock

  /**
    * Load for the transformer model which copies the model from S3 to local file then load it
    * as a Zoo model
    * @return Option of ZooModel
    */
  override lazy val model: Option[ZooModel[NDList, NDList]] = {
    import S3FsPathMapping._
    import TransformerModelLoader._

    val (s3ModelFile, fsModelDir, fsModelFile) = paths(transformerModelLbl, "")
    s3ToFs(fsModelDir, fsModelFile, s3ModelFile, deleteExistingFile = true)

    // Load model from S3 to local file if it does not exist yet
    val path = Paths.get(fsModelDir)
    val modelUrls = s"file://${path.toAbsolutePath}"
    logDebug(logger, msg = s"Start loading Transformer model from s3://$s3ModelFile to $modelUrls")

    // Build the criteria for predictor
    val criteria = Criteria.builder()
        .optApplication(Application.UNDEFINED)
        .setTypes(classOf[NDList], classOf[NDList])
        .optModelUrls(modelUrls)
        .optBlock(preTrainingBlock)
        .optEngine(Engine.getDefaultEngineName())
        .optProgress(new ProgressBar())
        .build()
    // Load the embedding vector through the criteria
    try {
      val thisModel = criteria.loadModel()
      Some(thisModel)
    }
    catch {
      case e: ModelNotFoundException =>
        logger.error(s"Failed loading Pre training model $modelUrls: ${e.getMessage}")
        None
    }
  }
}




private[bertspark] final object TransformerModelLoader {
  final private val logger: Logger = LoggerFactory.getLogger("TransformerLoader")
}
