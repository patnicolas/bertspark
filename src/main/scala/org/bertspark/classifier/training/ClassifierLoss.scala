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
package org.bertspark.classifier.training

import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import java.util.concurrent.atomic.AtomicInteger
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.config.MlopsConfiguration.DebugLog._
import org.bertspark.config.S3PathNames
import org.bertspark.util.io.S3Util
import org.bertspark.RuntimeSystemMonitor
import org.bertspark.analytics.MetricsCollector
import org.bertspark.nlp.medical.MedicalCodingTypes
import org.slf4j._
import scala.collection.mutable.ListBuffer


/**
 * {{{
 *   Customize softmax cross entropy loss to support direction computation of accuracy, precision, recall..
 *   This loss function can be
 *   - weighted average over each sample of a batch
 *   - average (mean) across the entire batch
 *   It supports
 *   - sparse label index of the value 1.0 or
 *   - dense labels (0, 0, ... 1, ..0)
 *
 *   Comparison between prediction and labels are stored in S3 mlops/$target/compare folder in CSV format
 *   [SubModelName,Epoch#,NumSuccesses,NumRecords,Rate,isMatch,Prediction,Label]
 * }}}
 *
 * @param indexLabelsMap Map of index to labels tokens
 * @param lossName Name for the loss (for analytics purpose)
 * @param weight Weight used to compute the
 * @param classAxis Axis or dimension of the data
 * @param sparseLabel Is it a sparse label?
 * @param fromLogit Are these values processed through logit
 * @param subModelName Name of the sub-model
 *
 * @see ai.djl.training.loss.SoftmaxCrossEntropyLoss
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] class ClassifierLoss protected (
  indexLabelsMap: Map[Int, String],
  override val lossName: String,
  weight: Float,
  classAxis: Int,
  sparseLabel: Boolean,
  fromLogit: Boolean,
  subModelName: String)
    extends SoftmaxCrossEntropyLoss(lossName, weight, classAxis,sparseLabel, fromLogit)
    with MetricsCollector
    with RuntimeSystemMonitor {
  import ClassifierLoss._

  /**
   * Evaluate the loss, comparing the label with the predicted CLS embedding
   * @param label Either dense [0,..1 ..0] or sparse (index of 1)
   * @param prediction CLS embedding associated with the document
   * @return Loss as a NDArray value => as Scalar
   */
  override def evaluate(label: NDList, prediction: NDList): NDArray = {
    import org.bertspark.implicits._

    var headPrediction = prediction.head()
    if(fromLogit)
      headPrediction = headPrediction.logSoftmax(classAxis)

    val shape = headPrediction.getShape
    val lab = label.singletonOrThrow()
    val errorRatio =
      if(indexLabelsMap.nonEmpty) evaluateDocument(shape.getShape()(1).toInt, headPrediction, lab)
      else 0.0F

    // Use direct evaluation as difference between prediction and label
    if(mlopsConfiguration.classifyConfig.isSuccessRateLoss)
      label.head().getManager().create(errorRatio)
    else {
      val loss = {
        // Use index of the label as target
        if (sparseLabel) {
          val pickIndex = new NDIndex()
              .addAllDim(Math.floorMod(classAxis, headPrediction.getShape().dimension()))
              .addPickDim(lab)
          logInfo(logger,  s"Pick index: ${pickIndex.getIndices.map(_.toString).mkString(" ")}")
          headPrediction.get(pickIndex).neg()
        }
          // Use output layer as dense values (0, 0, ... 1, ..0)
        else {
          val _lab = lab.reshape(headPrediction.getShape())
          headPrediction.mul(_lab).neg().sum(Array[Int](classAxis), true)
        }
      }
      // Compute the mean of the loss across the batch
      val outputLoss = if (weight != 1.0F) loss.mul(weight).mean() else loss.mean()
      if(outputLoss.hasGradient && Math.abs(outputLoss.toFloatArray.head) < 1e-28F)
        logger.error(s"Output loss is null")
      outputLoss
    }
  }

  /**
   * Save the content of the comparison between prediction and labels on S3
   * on S3: mlops/$target/compare/$runId/$subModelName/epoch-$epochNo
   * @param epochNo Current epoch number
   */
  def save(epochNo: Int): Unit = {
    val s3Folder = S3PathNames.getS3ComparePath(subModelName, epochNo)
    S3Util.upload(mlopsConfiguration.storageConfig.s3Bucket, s3Folder, comparisonBuffer.mkString("\n"))
    logDebug(logger, s"Save $subModelName comparison buffer to s3:/$s3Folder")
  }

  // --------------------------------  Supporting methods ------------------------------

  /**
   * Evaluate the prediction of a claim
   * Schema SubModelName,Epoch#,NumSuccesses,NumRecords,Rate,isMatch,Prediction,Label
   * @param numClasses Number of classes (output layer)
   * @param ndPrediction Prediction using [CLS] embedding
   * @param ndLabel Either dense [0,..1 ..0] or sparse (index of 1)
   * @return Accuracy
   */
  private def evaluateDocument(numClasses: Int, ndPrediction: NDArray, ndLabel: NDArray): Float = {
    import ClassifierLoss._

    val predictedValues = ndPrediction.toFloatArray
    // We need to adjust the batch size to the number of segments extracted per document.
    val batchSize = (predictedValues.size/numClasses).floor.toInt

    // Extract document prediction with the highest logSoftmax
    val predictedSelectedIndices = getBestPrediction(predictedValues, batchSize, numClasses)

    val labels = ndLabel.toFloatArray
    val labeledSelectedIndices = getLabelIndices(labels, batchSize, numClasses)

    // Make sure there was no leak in converting labels into indices and vice versa
    if(labeledSelectedIndices.size != batchSize || predictedSelectedIndices.size != batchSize)
      logger.error(
        s"Label ${labeledSelectedIndices.size} or prediction ${predictedSelectedIndices.size} indices should be $batchSize")
    else
      updateMetrics(predictedSelectedIndices, labeledSelectedIndices)
    failure
  }

  private def updateMetrics(predictedSelectedIndices: Array[Int], labeledSelectedIndices: Array[Int]): Unit = {
    if(mlopsConfiguration.evaluationConfig.compareEnabled)
      updateBatchMetrics(predictedSelectedIndices, labeledSelectedIndices, indexLabelsMap, subModelName)
    else
      updateBatchMetrics(predictedSelectedIndices, labeledSelectedIndices)

    val labeledCoreElements = getCoreElements(labeledSelectedIndices, indexLabelsMap)
    val predictedCoreElements = getCoreElements(predictedSelectedIndices, indexLabelsMap)
    updateCoreMetrics(predictedCoreElements, labeledCoreElements)
  }
}



/**
 * Singleton for constructors
 */
private[bertspark] final object ClassifierLoss {
  final private val logger: Logger = LoggerFactory.getLogger("ClassifierLoss")

  def apply(
    indexLabelsMap: Map[Int, String],
    lossName: String,
    weight: Float,
    classAxis: Int,
    sparseLabel: Boolean,
    fromLogit: Boolean,
    subModelName: String
  ): ClassifierLoss =
    new ClassifierLoss(
      indexLabelsMap,
      lossName,
      weight,
      classAxis,
      sparseLabel,
      fromLogit,
      subModelName)

  def apply(
    indexLabelsMap: Map[Int, String],
    lossName: String,
    weight: Float,
    sparseLabel: Boolean,
    fromLogit: Boolean,
    subModelName: String
  ): ClassifierLoss =
    apply(indexLabelsMap, lossName, weight, -(1), sparseLabel, fromLogit, subModelName)

  def apply(
    indexLabelsMap: Map[Int, String],
    lossName: String,
    sparseLabel: Boolean,
    fromLogit: Boolean,
    subModelName: String
  ): ClassifierLoss =
    apply(indexLabelsMap, lossName, 1.0F, sparseLabel, fromLogit, subModelName)

  /**
   * Select the predicted document given the log softmax probabilities, the size of this
   * batch and the number of classes
   * @param logSoftMax Log softmax for each prediction of the batch
   * @param batchSize Size of the batch (number of log softmax prediction)
   * @param numClasses Number of classes
   * @return Array of indices of the classes with the highest log softmax
   */
  def getBestPrediction(logSoftMax: Array[Float], batchSize: Int, numClasses: Int): Array[Int] =
    (0 until batchSize).foldLeft(List[Int]()) (
      (xs, index) => {
        val startIndex = index*numClasses
        val values = logSoftMax.slice(startIndex, startIndex + numClasses)
        val res = values.zipWithIndex.maxBy(_._1)._2
        res :: xs
      }
    ).reverse.toArray

  /**
   *
   * @param logSoftMax
   * @param batchSize
   * @param numClasses
   * @return
   */
  def getLabelIndices(logSoftMax: Array[Float], batchSize: Int, numClasses: Int): Array[Int] =
    (0 until batchSize).foldLeft(List[Int]()) (
      (xs, index) => {
        val startIndex = index*numClasses
        val values = logSoftMax.slice(startIndex, startIndex + numClasses)
        val res: Int = values.zipWithIndex.find( _._1 != 0.0).map(_._2).getOrElse(-1)
        if(res == -1) xs else res :: xs
      }
    ).reverse.toArray


  @throws(clazz = classOf[IllegalArgumentException])
  def getCoreElements(labelIndices: Array[Int], indexLabelsMap: Map[Int, String]): Array[String] =
    labelIndices.map(
      labelIndex => {
        val label = indexLabelsMap.getOrElse(
          labelIndex,
          throw new IllegalArgumentException(s"Label index $labelIndex is not valid")
        )

        val lineItems: Array[String] = label.split(MedicalCodingTypes.lineItemSeparator)
        if(lineItems.nonEmpty) {
          val codeGroups = lineItems.head.split(MedicalCodingTypes.codeGroupSeparator)
          val secondaryCode = codeGroups.size match {
            case 3 => codeGroups(2).split(MedicalCodingTypes.csvCodeSeparator).head.trim
            case 2 => codeGroups(1).split(MedicalCodingTypes.csvCodeSeparator).head.trim
            case _ => codeGroups.head.trim
          }
          s"${codeGroups.head.trim} ${secondaryCode}"
        }
        else
          throw new IllegalArgumentException("Cannot get core elements, without line items")
      }
    )

}