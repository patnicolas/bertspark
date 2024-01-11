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

import org.apache.spark.sql.{Dataset, Encoder, SparkSession}
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.nlp.medical.{encodeLabeledTraining, filterTrainingSetPerLabelDistribution, noContextualDocumentEncoding}
import org.bertspark.nlp.trainingset.{ContextualDocument, SubModelsTrainingSet, TokenizedTrainingSet}
import org.bertspark.util.io.DualS3Dataset.logger
import org.slf4j.{Logger, LoggerFactory}
import scala.util.Random


/**
 * Generic dataset extracted from S3
 * @param s3Path Path or folder in S3 bucket
 * @param extractCtxDocument Extraction method
 * @param minSampleSize minimum number of records to be retrieved. -1 for all records*
 * @param targetSampleSize target or maximum number of records to be retrieved. -1 for all records*
 * @tparam T Type of dataset loaded from S3
 *
 * @author Patrick Nicolas
 * @version 0.2
 */
trait S3Dataset[T] {
  val s3Path: String
  val minSampleSize: Int
  val maxSampleSize: Int
  val extractCtxDocument: T => ContextualDocument
}


/**
 * Wrapper for data collection on S3
 * @param s3Path Path or folder in S3 bucket
 * @param extractCtxDocument Extraction method
 * @param minSampleSize minimum number of records to be retrieved. -1 for all records*
 * @param maxSampleSize target or maximum number of records to be retrieved. -1 for all records*   *
 * @param sparkSession Implicit reference to the current Spark context
 * @param encoderT Encoder for type of element or records
 * @param encoderString Encoder for String
 * @param encoderCtxDocument Encoder for Contextual Document
 * @tparam T Type of element or records in S3
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
case class SingleS3Dataset[T](
  override val s3Path: String,
  override val extractCtxDocument: T => ContextualDocument,
  override val minSampleSize: Int = -1,
  override val maxSampleSize: Int = -1,
)(implicit sparkSession: SparkSession,
  encoderT: Encoder[T]
) extends S3Dataset[T] {
  import sparkSession.implicits._
  import org.bertspark.config.MlopsConfiguration._

  require(minSampleSize == -1 || minSampleSize > 0, s"Sample size for S3Store $minSampleSize > 0}")

  final def getPath: String = s3Path

  /**
   * Retrieve the data set of elements of type T associated with this S3 store
   * @return Data set extracted from the S3 storage
   */
  lazy val inputDataset = this.getDataset

  def sampleDataset: Dataset[T] = {
    val ds = inputDataset
    val fraction = 1.0/8
    ds.sample(fraction)
  }

  /**
   * Create an iterator for all the document loaded from this particular S3 folder
   * @return Iterator
   */
  def getContentIterator: Iterator[ContextualDocument] = {
    val ds = getDataset
    ds.show()
    ds.map(extractCtxDocument(_)).collect().iterator
  }

  def check: Unit = mlopsConfiguration.check(!getDataset.isEmpty, s"S3StorageInfo is incorrect for ${mlopsConfiguration.storageConfig.s3Bucket}/$s3Path")

  override def toString: String = {
    val ds = getDataset
    if(ds.isEmpty) s"S3 storage for ${mlopsConfiguration.storageConfig.s3Bucket}/$s3Path is undefined" else ds.take(3).mkString("\n")
  }

  @inline
  final def getSampleSize: Int = maxSampleSize

  // --------------------   Supporting method ------------------------

  private def getDataset: Dataset[T] = {
    val rawDataset = S3Util.s3ToDataset[T](
      mlopsConfiguration.storageConfig.s3Bucket,
      s3Path,
      header = false,
      fileFormat = "json"
    )
    // Sample the large dataset
    if(maxSampleSize > 0 && maxSampleSize < rawDataset.count()) {
      val fraction = maxSampleSize.toFloat/rawDataset.count()
      rawDataset.sample(fraction)
    }
    else
      rawDataset
  }
}


/**
 * Singleton for constructors
 */
private[bertspark] final object SingleS3Dataset {

  def apply(
    s3Folder: String,
    numSamples: Int
  )(implicit sparkSession: SparkSession): SingleS3Dataset[ContextualDocument] = {
    import sparkSession.implicits._
    new SingleS3Dataset[ContextualDocument](s3Folder, noContextualDocumentEncoding, numSamples)
  }

  def apply(s3Folder: String)(implicit sparkSession: SparkSession): SingleS3Dataset[ContextualDocument] = {
    import sparkSession.implicits._
    new SingleS3Dataset[ContextualDocument](s3Folder, noContextualDocumentEncoding)
  }
}


/**
 * S3 storage for composite (or map) data structure {sub-model key -> training records }
 * @param s3Path Path of the training data
 * @param extractSubTrainingSet Function to extract the map {key -> training records }
 * @param extractCtxDocument Function to extract the contextual document [context, text]
 * @param sparkSession Implicit reference to the current Spark context
 * @param encoderT Implicit encoder for the type of the data source
 * @param encoderCtxDocument Implicit encoder for contextual document
 * @tparam T Type of data source loaded from S3
 * @tparam U Type of records or training data
 *
 * @author Patrick Nicolas
 * @version 0.2
 */
case class DualS3Dataset[T, U] (
  override val s3Path: String,
  extractSubTrainingSet: (T, Set[String]) => (String, Seq[U]),
  override val extractCtxDocument: U => ContextualDocument
)(implicit sparkSession: SparkSession,
  encoderT: Encoder[T],
  encoderCtxDocument: Encoder[Seq[ContextualDocument]]) extends S3Dataset[U] {

  override val minSampleSize: Int = -1
  override val maxSampleSize: Int = -1

  lazy val inputDataset: Dataset[T] = try {
    S3Util.s3ToDataset[T](
      mlopsConfiguration.storageConfig.s3Bucket,
      s3Path,
      header = false,
      fileFormat = "json"
    )
  } catch {
    case e: IllegalStateException =>
      logger.error(s"DualS3Dataset: ${e.getMessage}")
      sparkSession.emptyDataset[T]
  }


  /**
   * {{{
   * Retrieve the training data set organized as {subModel, Sequence of records}
   * The sequence of records associated with each sub-model is initially restricted by sample size if != -1
   * The sub-sampling is done randomly
   * }}}
   * @param encoder implicit encoder for the pair (Sub model, associated training set)
   * @return Data set of records (subModel, sequence of requests} ready
   */
  def getLabelTrainingData(
    subModels: Set[String]
  )(implicit encoder: Encoder[(String, Seq[U])], enc2: Encoder[Seq[U]]): Dataset[(String, Seq[U])] = {
    import DualS3Dataset._

    val filterFunc = (a: (String, Seq[U])) => a._1.nonEmpty
    val completeTrainingDataDS: Dataset[(String, Seq[U])] = inputDataset
        .map(extractSubTrainingSet(_, subModels))
        .filter(filterFunc)
    logDebug(logger, s"${completeTrainingDataDS.count()} available sub-models out of ${inputDataset.count()}")
    completeTrainingDataDS
  }
}


/**
 * Singleton for constructor
 */
private[bertspark] final object DualS3Dataset {
  final private val logger: Logger = LoggerFactory.getLogger("DualS3Dataset")

  def apply(
    s3Path: String
  )(implicit sparkSession: SparkSession,
    enc1: Encoder[SubModelsTrainingSet],
    enc2: Encoder[Seq[ContextualDocument]]): DualS3Dataset[SubModelsTrainingSet, TokenizedTrainingSet] =
    DualS3Dataset[SubModelsTrainingSet, TokenizedTrainingSet](
      s3Path,
      filterTrainingSetPerLabelDistribution,
      encodeLabeledTraining)

}
