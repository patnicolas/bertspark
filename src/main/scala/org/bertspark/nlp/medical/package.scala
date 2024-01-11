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
package org.bertspark.nlp

import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalFeedback, InternalRequest}
import org.bertspark.config.{ExecutionMode, MlopsConfiguration}
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.nlp.medical.ContextEncoder.encodeContext
import org.bertspark.nlp.trainingset.{ContextualDocument, SubModelsTrainingSet, TokenizedTrainingSet}
import org.bertspark.transformer.representation.SegmentEmbeddingSimilarity.segmentEmbeddingsSimilarity
import org.bertspark.util.io.S3Util
import org.slf4j.{Logger, LoggerFactory}


/**
 * Classes and methods that are specific to medical codig
 * @author Patrick Nicolas
 * @version 0.1
 */
package object medical {
  import MlopsConfiguration._

  final private val logger: Logger = LoggerFactory.getLogger("medical")

  final val encodePredictReq = (req: InternalRequest) =>
    ContextualDocument(req.id, encodeContext(req.context), req.notes.head)


  /**
    * Compute training minimum and maximum training size for each label
    * @param trainingData Training set associated with this sub model {sub model -> Sequence (label/tokens) }
    * @return Balanced training set
    */
  def limitTrainingData(
    trainingData: Map[String, Seq[TokenizedTrainingSet]]
  ): Map[String, Seq[TokenizedTrainingSet]] = {
    val config = mlopsConfiguration.classifyConfig

    val minNumRecords = config.minNumRecordsPerLabel
    val maxNumRecords = config.maxNumRecordsPerLabel
    val validTrainingData = trainingData.forall(_._2.size >= minNumRecords)
    if(validTrainingData)
      trainingData.map{ case (label, records) => (label, records.take(maxNumRecords))}
    else
      Map.empty[String, Seq[TokenizedTrainingSet]]
  }

  /**
    * {{{
    * Extract and filter the training records for classifier
    * - Remove label with low number of associated notes
    * - Sample the notes if the number of these notes associated with the label exceeds the threshold
    * }}}
    */
  final val filterTrainingSetPerLabelDistribution:
    (SubModelsTrainingSet, Set[String]) => (String, Seq[TokenizedTrainingSet]) =
    (data: SubModelsTrainingSet, subModels: Set[String]) =>
      if (subModels.nonEmpty)
        if (subModels.contains(data.subModel)) {
          val trainingDataGroupedByLabel = data.labeledTrainingData.groupBy[String](_.label)
          if(trainingDataGroupedByLabel.nonEmpty)
            (data.subModel, trainingDataGroupedByLabel.flatMap(_._2).toSeq)
          else {
            logger.warn(s"${data.subModel} has no associated labels")
            ("", Seq.empty[TokenizedTrainingSet])
          }
        }
        else {
          logger.error(s"Sub models do not contain ${data.subModel}")
          ("", Seq.empty[TokenizedTrainingSet])
        }
      else {
        logger.error("Sub models are empty")
        ("", data.labeledTrainingData)
      }


  final val encodeLabeledTraining: TokenizedTrainingSet => ContextualDocument =
    (data: TokenizedTrainingSet) => data.contextualDocument

  /**
   * Default conversion of contextual documents. In 'similarity' mode is register the
   * contextual document for analyzing the similarity of segment embedding (representation layer)
   */
  final val noContextualDocumentEncoding = (contextualDocument: ContextualDocument) => {
    // if the mode is similarity (computation of similarity of segment/sentence)
    // then add the contextual document to the segment embedding similarity model
    if(ExecutionMode.isSimilarity)
      segmentEmbeddingsSimilarity += contextualDocument
    contextualDocument
  }



  def repartition(fromS3Folder: String, toS3Folder: String, recordType: String): Unit = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    recordType match {
      case "requests" =>
        val ds = S3Util.s3ToDataset[InternalRequest](
          fromS3Folder,
          header = false,
          fileFormat ="json"
        ).dropDuplicates("id")

        S3Util.datasetToS3[InternalRequest](
          mlopsConfiguration.storageConfig.s3Bucket,
          ds,
          toS3Folder,
          false,
          "json",
          false,
          64
        )

      case "feedbacks" =>
        val ds = S3Util.s3ToDataset[InternalFeedback](
          fromS3Folder,
          false,
          "json"
        ).dropDuplicates("id")

        S3Util.datasetToS3[InternalFeedback](
          mlopsConfiguration.storageConfig.s3Bucket,
          ds,
          toS3Folder,
          false,
          "json",
          false,
          64
        )
    }
  }
}
