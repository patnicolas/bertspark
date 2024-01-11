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
package org.bertspark.transformer.representation

import ai.djl.ndarray._
import org.apache.spark.sql._
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.nlp.medical._
import org.bertspark.nlp.trainingset._
import org.bertspark.transformer.representation.EmbeddingSimilarity.ModelSimilarity
import org.bertspark.util._
import org.bertspark.util.NDUtil.FloatNDArray.fromVec
import org.bertspark.util.io._
import org.bertspark.config.MlopsConfiguration.DebugLog.{logDebug, logTrace}
import org.bertspark.config.S3PathNames
import org.bertspark.RuntimeSystemMonitor
import org.slf4j._
import scala.collection.mutable.ListBuffer

/**
 * {{{
 *  Singleton to compute the similarity of document representation. The similarity uses either cosine or
 *  Euclidean metric.
 *  The similarity is stored into S3 in mlops/$target/similarity/$runId folder
 * }}}
 * @param s3Path Path for S3 folder containing the input data
 * @param sparkSession Implicit reference to the current Spark context
 *
 * @author Patrick Nicolas
 * @version 0.1
 **/
private[bertspark] final class DocumentEmbeddingSimilarity private (
  s3Path: String
)(implicit sparkSession: SparkSession) extends EmbeddingSimilarity  {
  import DocumentEmbeddingSimilarity._

  /**
   * Computes the similarity between CLS prediction (representation) given labels
   * {{{
   *   The CLS prediction for document sharing the same labels should be similar (cosine ~ 1)
   *   CLS predictions for document associated with different labels should be different
   *   We select random CLS prediction from documents sharing the same label and across labels.
   *
   *   The similarity is stored into S3 in mlops/$target/similarity/$runId folder
   * }}}
   *
   * @param maxNumLabels Maximum number of labels to processed (-1 for all labels)
   * @throws IllegalArgumentException if max number of labels is not > 1 or -1
   * @return Model similarity
   */
  @throws(clazz = classOf[IllegalArgumentException])
  override def similarity(maxNumLabels: Int): ModelSimilarity = {
    require(maxNumLabels > 1 || maxNumLabels == -1,
      s"Max number of labels for similarity $maxNumLabels should be > 1 or -1")
    val ndManager = NDManager.newBaseManager()

    val modelSimilarity = computeSimilarity(ndManager, s3Path, maxNumLabels)
    // Generate a JSON representation ...
    val jsonModelSimilarity = LocalFileUtil.Json.mapper.writeValueAsString(modelSimilarity)
    // and stored into S3
    S3Util.upload(
      mlopsConfiguration.storageConfig.s3Bucket,
      S3PathNames.getS3SimilarityOutput,
      jsonModelSimilarity
    )
    ndManager.close()
    modelSimilarity
  }
}

/**
 * Singleton for constructors and supporting routines for computation of similarity
 */
private[bertspark] final object DocumentEmbeddingSimilarity extends RuntimeSystemMonitor {
  final private val logger: Logger = LoggerFactory.getLogger("DocumentEmbeddingSimilarity")


  /**
   * Customizable constructor with run-time command line arguments
   * @param args Command line arguments
   * @return Instance of DocumentEmbeddingSimilarity
   */
  def apply(args: Seq[String]): DocumentEmbeddingSimilarity = {
    import org.bertspark.implicits._
    new DocumentEmbeddingSimilarity(args(1))
  }

  /**
   * Default constructor
   * @return Instance of DocumentEmbeddingSimilarity
   */
  def apply(): DocumentEmbeddingSimilarity  = {
    import org.bertspark.implicits._
    new DocumentEmbeddingSimilarity(S3PathNames.s3ModelTrainingPath)
  }

  /**
   * Compute the similarity between two representation of document sharing the same label
   *
   * @param ndManager           Reference to the current ND Manager
   * @param s3TrainingSetFolder S3 folder containing the training set used for the similarity
   * @param maxNumLabels        Maximum number of labels to be processed, -1 for processing all labels
   * @param sparkSession        Implicit reference to the current Spark context
   * @return Model similarity
   */
  def computeSimilarity(
    ndManager: NDManager,
    s3TrainingSetFolder: String = S3PathNames.s3ModelTrainingPath,
    maxNumLabels: Int
  )(implicit sparkSession: SparkSession): ModelSimilarity = {
    import sparkSession.implicits._

    // Load the labeled training data from the appropriate Storage
    val s3MapStorage = DualS3Dataset(s3TrainingSetFolder)
    logDebug(logger, s"Loaded training set with a vocabulary size ${vocabulary.size}")

    val tokenizedIndexedDS = s3MapStorage.getLabelTrainingData(Set.empty[String])
    computeSimilarity(ndManager, tokenizedIndexedDS, maxNumLabels)
  }


  /**
   * Compute similarity for data loaded from training set
   *
   * @param ndManager          Explicit reference to the current NDManager
   * @param tokenizedIndexedDS Tokenized input
   * @param maxNumLabels       Maximum number of labels used in the computation of similarity
   * @param sparkSession       Implicit reference to the current Spark context
   * @return Model similarity
   */
  def computeSimilarity(
    ndManager: NDManager,
    tokenizedIndexedDS: Dataset[(String, Seq[TokenizedTrainingSet])],
    maxNumLabels: Int
  )(implicit sparkSession: SparkSession): ModelSimilarity = {
    import sparkSession.implicits._

    logTrace(
      logger,
      msg = s"${tokenizedIndexedDS.count()}\n${allMetrics("Similarity")}"
    )
    val labeledTokenizedDocDS = tokenizedIndexedDS.flatMap(_._2).map(Seq[TokenizedTrainingSet](_))

    // Group tokenized indexed training set by label
    val tokenizedDocGroupedByLabelRDD = SparkUtil.groupBy[Seq[TokenizedTrainingSet], String](
      (tokenizedTS: Seq[TokenizedTrainingSet]) => tokenizedTS.head.label,
      (tokenizedTS1: Seq[TokenizedTrainingSet], tokenizedTS2: Seq[TokenizedTrainingSet]) =>
        tokenizedTS1 ++ tokenizedTS2,
      labeledTokenizedDocDS
    )

    // Extract the iterator for a sequence of contextual documents associated with a label
    val contextualDocumentIterator: Iterator[(String, Seq[ContextualDocument])] = tokenizedDocGroupedByLabelRDD
        .map(tokenizedDocGroupedByLabel =>
          (tokenizedDocGroupedByLabel.head.label, tokenizedDocGroupedByLabel.map(_.contextualDocument))
        )
        .collect
        .iterator

    // To collect sequence of embeddings per each label
    val accClsNdPredictions = ListBuffer[Seq[Array[Float]]]()
    val bertPreTrainingPredictor = PretrainingInference()
    val subNDManager = ndManager.newSubManager()

    // Walk across all labels if defined as -1
    val numOfLabels = if (maxNumLabels == -1) 9999999 else maxNumLabels

    var labelCount = 0
    var contextualDocumentCount = 0

    // We do not assume that we have to process all the labels to compute the similarity
    while (contextualDocumentIterator.hasNext && labelCount < numOfLabels) {
      logTrace(logger, s"${tokenizedIndexedDS.count()}\n${allMetrics(descriptor = "Similarity")}")
      val (label, contextualDocuments) = contextualDocumentIterator.next()
      labelCount += 1
      contextualDocumentCount += contextualDocuments.size
      logDebug(logger,
        msg = s"Process $label with $labelCount labels and $contextualDocumentCount contextual documents")

      val ndKeyedDocEmbedding = bertPreTrainingPredictor.predict(subNDManager, contextualDocuments.toDS())
      accClsNdPredictions.append(ndKeyedDocEmbedding.map(_._2))
      logTrace(logger, allMetrics(descriptor = "Similarity"))
    }
    logDebug(logger, msg = s"Predicted ${mlopsConfiguration.classifyConfig.maxNumRecordsPerLabel} records")

    val modelSimilarity = computeModelSimilarity(subNDManager, accClsNdPredictions)
    subNDManager.close()
    modelSimilarity
  }


  // ---------------------------  Supporting methods ------------------------------------------

  private def computeModelSimilarity(
    ndManager: NDManager,
    accClsNdPredictions: ListBuffer[Seq[Array[Float]]]): ModelSimilarity = {
    import org.bertspark.util.NDUtil.FloatNDArray._

    val filteredClsNdPredictions = accClsNdPredictions.filter(_.size > 1)
    val inClassSimilarities = filteredClsNdPredictions.flatMap(
      clsNdPrediction =>
        (0 until clsNdPrediction.size - 1).map(
          index =>
            NDUtil.computeSimilarity(
              fromVec(ndManager, clsNdPrediction(index)),
              fromVec(ndManager, clsNdPrediction(index + 1)),
              ConstantParameters.simFactor)
        )
    )
    val inClassSimilarity = inClassSimilarities.sum / inClassSimilarities.size

    var index = 0
    val outClassSimilarities = ListBuffer[Double]()
    do {
      // We make sure that we have at least two documents per label
      outClassSimilarities ++= computeExtraLabelSimilarity(ndManager, accClsNdPredictions, index)
      index += 1
    } while (index < accClsNdPredictions.size - 1)

    val outClassSimilarity = outClassSimilarities.sum / outClassSimilarities.size

    ModelSimilarity(
      inClassSimilarity.toFloat,
      outClassSimilarity.toFloat,
      0.0F,
      Seq.empty[(String, Double)],
      "NA"
    )
  }

  private def computeExtraLabelSimilarity(
    ndManager: NDManager,
    accClsNdPredictions: ListBuffer[Seq[Array[Float]]],
    index: Int): Seq[Double] = {

    val frstPredictions = getPredictions(ndManager, accClsNdPredictions, index)
    val sndPredictions = getPredictions(ndManager, accClsNdPredictions, index + 1)
    val simFactor = ConstantParameters.simFactor

    val similarityValues = ListBuffer[Double]()
    similarityValues.append(NDUtil.computeSimilarity(frstPredictions(0), sndPredictions(0), simFactor))
    if (frstPredictions.size > 1)
      similarityValues.append(NDUtil.computeSimilarity(frstPredictions(1), sndPredictions(0), simFactor))
    if (sndPredictions.size > 1)
      similarityValues.append(NDUtil.computeSimilarity(frstPredictions(0), sndPredictions(1), simFactor))
    if (frstPredictions.size > 1 && sndPredictions.size > 1)
      similarityValues.append(NDUtil.computeSimilarity(frstPredictions(1), sndPredictions(1), simFactor))

    similarityValues.toSeq
  }

  private def getPredictions(
    ndManager: NDManager,
    accClsNdPredictions: ListBuffer[Seq[Array[Float]]],
    index: Int): Array[NDArray] =
    if (accClsNdPredictions(index).size > 1)
      Array[NDArray](
        fromVec(ndManager, accClsNdPredictions(index).head),
        fromVec(ndManager, accClsNdPredictions(index)(1))
      )
    else
      Array[NDArray](fromVec(ndManager, accClsNdPredictions(index).head))
}