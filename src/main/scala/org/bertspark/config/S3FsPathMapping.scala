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
package org.bertspark.config


/**
  * Singleton helper that match local and S3 folders, addressing issue of characters between path
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] final object S3FsPathMapping {
  final val transformerModelLbl = "transformerModel"
  final val classifierModelLbl = "classifierModel"

  val s3FsMap = Map[String, (String => String, String => String, String => String)](
    `transformerModelLbl` ->
        (
            (s: String) => S3PathNames.getS3TransformerModelPath,
            (s: String) => FsPathNames.getFsTransformerModelPath,
            (s: String) => FsPathNames.getFsTransformerModelFile
        ),
        `classifierModelLbl` ->
         (
             (s: String) => S3PathNames.getS3ClassifierPath(s),
             (s: String) => FsPathNames.getFsClassifierModelPath(s),
             (s: String) => FsPathNames.getFsClassifierModelFile(s)
         )
  )


  def fsPath(modelType: String, subModelName: String): String = modelType match {
    case `transformerModelLbl` | `classifierModelLbl` => s3FsMap(modelType)._2(subModelName)
    case _ => throw new UnsupportedOperationException(s"$modelType is not a supported model name")
  }

  def fsFile(modelType: String, subModelName: String): String = modelType match {
    case `transformerModelLbl` | `classifierModelLbl` => s3FsMap(modelType)._3(subModelName)
    case _ => throw new UnsupportedOperationException(s"$modelType is not a supported model name")
  }

  def s3Path(modelType: String, subModelName: String): String = modelType match {
    case `transformerModelLbl` | `classifierModelLbl` => s3FsMap(modelType)._1(subModelName)
    case _ => throw new UnsupportedOperationException(s"$modelType is not a supported model name")
  }


  /**
    * {{{
    * Retrieve all appropriate path...
    * 1- S3 path
    * 2- FS directory/folder
    * 3- FS filename
    *
    * }}}
    * @param modelType Name of model
    * @param subModelName Name of sub model
    * @return Tuple as described above.
    */
  def paths(modelType: String, subModelName: String): (String, String, String) = modelType match {
    case `transformerModelLbl` | `classifierModelLbl` =>
      (s3FsMap(modelType)._1(subModelName), s3FsMap(modelType)._2(subModelName), s3FsMap(modelType)._3(subModelName))
    case _ => throw new UnsupportedOperationException(s"$modelType is not a supported model name")
  }
}
