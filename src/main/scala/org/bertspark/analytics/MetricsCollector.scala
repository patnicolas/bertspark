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

import java.util.concurrent.atomic.AtomicInteger
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.S3PathNames
import org.bertspark.modeling.SubModelsTaxonomy.subModelTaxonomy
import org.bertspark.nlp.medical.MedicalCodingTypes.{comparisonSeparator, lineItemSeparator, logger, InternalFeedback}
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.bertspark.util.DateUtil
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer


/**
  * Run-time collector for quality metrics and comparison
  * @author Patrick Nicolas
  * @version 0.6
  */
trait MetricsCollector {
  import MetricsCollector._

  protected[this] val lossName: String

  protected[this] val comparisonBuffer = ListBuffer[String]()
  protected[this] val allRecords = new MetricCollectorRecord
  protected[this] val firstLineRecords = new MetricCollectorRecord
  protected[this] val coreElementRecords = new MetricCollectorRecord

  final def getMetrics: (CustomMetric, CustomMetric) = (allRecords.getMetrics, coreElementRecords.getMetrics)

  def failure: Float = allRecords.failure

  override def toString: String =
    s"""Label records: ${allRecords.toString}
       |First line item: ${firstLineRecords.toString}
       |Core element records:${coreElementRecords.toString}""".stripMargin

  /**
    * Save the various metrics, Strict, Unordered and core to S3
    */
  def save: Unit = {
    import org.bertspark.config.MlopsConfiguration._
    val thisDate = DateUtil.longToDate
    val classifyConfig = mlopsConfiguration.classifyConfig
    val preTrainConfig = mlopsConfiguration.preTrainConfig

    val introduction =
      s"""
         |Date:              $thisDate
         |Transformer model: ${mlopsConfiguration.runId}
         |Target:            ${mlopsConfiguration.target}
         |Number of records: ${allRecords.size}
         |Vocabulary:        ${mlopsConfiguration.preProcessConfig.vocabularyType}
         |Tokenizer          ${preTrainConfig.tokenizer}
         |BERT encoder       ${preTrainConfig.transformer}
         |Segments model     ${preTrainConfig.sentenceBuilder}
         |Num segments       ${preTrainConfig.numSentencesPerDoc}
         |Classifier model   ${classifyConfig.modelId}
         |Classifier layout  ${classifyConfig.dlLayout.mkString("x")}
         |Classifier band    [${classifyConfig.minNumRecordsPerLabel}, ${classifyConfig.maxNumRecordsPerLabel}]
         |""".stripMargin

    val predictorMetrics =
      s"""$introduction
         |$metricHeader
         |${allRecords.getMetrics.toCsv(evaluationType = "Strict match")}
         |${firstLineRecords.getMetrics.toCsv(evaluationType ="Core codes")}
         |${coreElementRecords.getMetrics.toCsv(evaluationType ="CPT-Primary ICD")}""".stripMargin

    val content =
      if(mlopsConfiguration.evaluationConfig.compareEnabled)
        s"$predictorMetrics\n\nComparison:\nStatus,Type,Predicted,Actual,NumClaims\n${comparisonBuffer.mkString("\n")}"
      else
        predictorMetrics

    S3Util.upload(
      mlopsConfiguration.storageConfig.s3Bucket,
      s"${S3PathNames.s3PredictionMetricPath}-$thisDate",
      content
    )
    logDebug(logger, s"Metrics saved into ${S3PathNames.s3PredictionMetricPath}\n$content")
  }


      // ------------ Method available to sub classes --------------------------------

  protected def updateBatchMetrics(
    predictedSelectedIndices: Seq[Int],
    labeledSelectedIndices: Seq[Int],
    indexLabelsMap: Map[Int, String] = Map.empty[Int, String],
    subModelName: String = ""): Unit = {

    val batchSize = predictedSelectedIndices.size
    if(batchSize > 0) {
      val singleCompare = compare(
        predictedSelectedIndices.head,
        labeledSelectedIndices.head,
        indexLabelsMap,
        subModelName
      )
      val comparisonInBatch = Seq[String](singleCompare)
      comparisonBuffer.append(comparisonInBatch.mkString("\n"))
    }
    else
      logger.warn("Batch for updating metrics is empty")
  }

  import org.bertspark.config.MlopsConfiguration._


  /**
    * Compute the metrics associated with the feedback (or labeled claim)
    * and update the various collector
    * @param internalFeedbacks Feedback record
    * @return True if successful false, otherwise
    */
  protected def updateMetrics(internalFeedbacks: Seq[InternalFeedback]): Boolean = {
    internalFeedbacks.foreach(updateMetrics(_))
    true
  }

  /**
    * Compute the metrics associated with the feedback (or labeled claim)
    * and update the various collector
    * @param internalFeedback Feedback record
    * @return True if successful false, otherwise
    */
  protected def updateMetrics(internalFeedback: InternalFeedback): Boolean =
      // If the internal feedback relates to an oracle model.....

    if(internalFeedback.isOracle) {
      val subModelName = internalFeedback.context.emrLabel.trim
      logger.info(s"$subModelName is an Oracle model")

      coreElementRecords += MetricsCollector.tpLbl
      firstLineRecords += MetricsCollector.tpLbl
      allRecords += (MetricsCollector.tpLbl)
      val oraclePrediction = subModelTaxonomy.getOracleLabel(internalFeedback.context.emrLabel.trim).getOrElse("")
      val comparison = s"$oraclePrediction$comparisonSeparator$oraclePrediction"
      val dump = metricsDescriptor("S", "_O_", subModelName, comparison)
      if(dump.nonEmpty)
        comparisonBuffer.append(dump)
      true
    }
        // If the internal feedback relates to a trained model.....

    else if(internalFeedback.isTrained) {
      metrics(internalFeedback).map {
        case (result, comparison) => {
          val subModelName = internalFeedback.context.emrLabel.trim
          coreElementRecords += (if(result.isCore) MetricsCollector.tpLbl else MetricsCollector.fpLbl)
          firstLineRecords += (if (result.isFirstLine) MetricsCollector.tpLbl else MetricsCollector.fpLbl)

          val status =
            if (result.isStrict) {
              allRecords += MetricsCollector.tpLbl
              "S"
            }
            else {
              allRecords += MetricsCollector.fpLbl
              "F"
            }

          val dump = metricsDescriptor(status, modelType = "_T_", subModelName, comparison)
          if(dump.nonEmpty)
            comparisonBuffer.append(dump)
          true
        }
      }.getOrElse({
        logger.error(s"Metrics for ${internalFeedback.context.emrLabel} is undefined")
        false
      })
    }
    else {
      logger.warn(s"${internalFeedback.context.emrLabel} is an unsupported model")
      false
    }



  private def metricsDescriptor(
    status: String,
    modelType: String,
    subModelName: String,
    comparison: String): String =
    if(mlopsConfiguration.evaluationConfig.compareEnabled) {
      val numLabels: Int = subModelTaxonomy.getPredictiveNumLabels(subModelName).getOrElse(0)
      val correctedNumLabels = if(numLabels == 0) 1 else numLabels
      val dump = s"$status,$modelType,$subModelName,${comparison.replace(comparisonSeparator, ",")},$correctedNumLabels"
      logDebug(logger, dump)
      dump
    }
    else
      ""



  protected def updateCoreMetrics(predictedCoreElements: Seq[String], labeledCoreElements: Seq[String]): Unit =
    predictedCoreElements.indices.foreach(
      index => updateCoreMetrics(predictedCoreElements(index), labeledCoreElements(index))
    )

  protected def updateStrictMetrics(predictedCoreElements: Seq[String], labeledCoreElements: Seq[String]): Unit =
    predictedCoreElements.indices.foreach(
      index => updateStrictMetrics(predictedCoreElements(index), labeledCoreElements(index))
    )

  private def updateStrictMetrics(predictedIndex: String, labeledIndex: String): Unit =
    updateStrictMetrics[String](predictedIndex, labeledIndex)

  private def updateCoreMetrics(predictedCoreElement: String, labeledCoreElement: String): Unit =
    if(predictedCoreElement == labeledCoreElement) coreElementRecords += (MetricsCollector.tpLbl)
    else coreElementRecords += (MetricsCollector.fpLbl)




      // --------------- Helper methods ----------------------

  private def updateStrictMetrics(predictedIndex: Int, labeledIndex: Int): Unit =
    updateStrictMetrics[Int](predictedIndex, labeledIndex)

  private def updateStrictMetrics[T](predictedIndex: T, labeledIndex: T): Unit =
    if(predictedIndex == labeledIndex) allRecords += MetricsCollector.tpLbl
    else allRecords  += MetricsCollector.fpLbl



  private def compare(
    predictedIndex: Int,
    labeledIndex: Int,
    indexLabelsMap: Map[Int, String],
    subModelName: String): String = {
    updateStrictMetrics(predictedIndex, labeledIndex)

    if(indexLabelsMap.nonEmpty) {
      val isMatched = if (predictedIndex == labeledIndex) "Y" else "N"
      // Compute the statistics, number of records, successes
      val predicted = indexLabelsMap.getOrElse(predictedIndex, "NA")
      val label = indexLabelsMap.getOrElse(labeledIndex, "NA")

      val comparedOutput = s"$subModelName,${allRecords.numSuccesses},${allRecords.numRecords},${allRecords.accuracy},$isMatched,$predicted,$label"
      logDebug(logger, msg = s"*$isMatched  ${comparedOutput}")
      comparedOutput
    }
    else
      ""
  }
}


private[bertspark] final object MetricsCollector {
  final private val logger: Logger = LoggerFactory.getLogger("MetricsCollector")

  final val tpLbl = "TP"
  final val fpLbl = "FP"
  final val fnLbl = "FN"
  final val successLbl = "SUCCESS"
  final private val metricHeader = s"Evaluation\tAccuracy\tF1"

  /**
    * Class to collect metrics for label and core eleents
    */
  final class MetricCollectorRecord {
    val numRecords = new AtomicInteger(0)
    val numSuccesses = new AtomicInteger(0)
    private[this] val numTP = new AtomicInteger(0)
    private[this] val numFP = new AtomicInteger(0)
    private[this] val numFN = new AtomicInteger(0)

    final def getMetrics: CustomMetric = {
      val tpFactor = numTP.get().toFloat
      val accuracy = if(numRecords.get() < 1) 0.0F else numSuccesses.get().toFloat/numRecords.get()
      val p = tpFactor/(numTP.get() + numFP.get())
      val r = tpFactor/(numTP.get() + numFN.get())
      val f1 = 2.0F*p*r/(p + r)
      CustomMetric(accuracy, p, r, f1)
    }

    def += (customMetricType: String): Int = {
      customMetricType match {
        case `tpLbl` =>
          numTP.incrementAndGet()
          numSuccesses.incrementAndGet()
        case `fpLbl` => numFP.incrementAndGet()
        case `fnLbl` => numFN.incrementAndGet()
        case _ => throw new UnsupportedOperationException(s"Custom metric $customMetricType is not supported")
      }
      numRecords.incrementAndGet()
    }

    def accuracy: Float = if(numRecords.get() < 1) 0.0F else numSuccesses.get().toFloat / numRecords.get()
    def precision: Float = {
      val den = numTP.get() + numFP.get()
      if(den > 0) numTP.get().toFloat/den else 0.0F
    }

    def recall: Float = {
      val den = numTP.get() + numFN.get()
      if(den > 0) numTP.get().toFloat/den else 0.0F
    }

    def f1: Float = {
      val p = precision
      val r = recall
      if(p + r > 0.0F) 2.0F*p*r/(p + r) else 0.0F
    }


    def getNumRecords: Int = numRecords.get()

    def size: Int = numRecords.get()

    def failure: Float = if(numRecords.get() < 1) 0.0F else (1.0F - numSuccesses.get().toFloat / numRecords.get())

    override def toString: String = s"Count: ${numRecords.get()} Precision: $precision, Recall: $recall, F1: $f1"
  }



  case class CustomMetric(accuracy: Float, precision: Float, recall: Float, f1: Float) {
    override def toString: String = s"Accuracy: $accuracy, Precision: $precision, Recall: $recall, F1: $f1"

    def toCsv(evaluationType: String): String = s"$evaluationType\t$accuracy\t$f1"

    def toMap(id: String): Map[String, Float] = Map[String, Float](
      s"${id}Accuracy" -> accuracy,
      s"${id}Precision" -> precision,
      s"${id}Recall" -> recall,
      s"${id}F1"-> f1
    )
  }


  /**
    * Compute the sorted distribution of label -> accuracy from the logs of training of classifier
    * @param fsNohupFilename Name of the dump of the training of classifier (i.e.  nohup.out)
    * @return Ranked sequence (label, accuracy) in decreasing order
    */
  def collectTrainingLabelDistribution(fsNohupFilename: String): Seq[(String, Float)] = {
    import org.bertspark.config.MlopsConfiguration._

    val filter = (line: String) => line.contains("MetricsCollector: DEBUG")
    val parse = (line: String) => {
      val fields = line.split(",")
      val status = if(fields(4) == "Y") 1 else 0
      val label = fields(6)
      (label, (status, 1))
    }

    LocalFileUtil.Load.local[(String, (Int, Int))](
      fsNohupFilename,
      parse,
      from = -1,
      to = -1,
      Some(filter)
    ).map(
      _.groupBy(_._1).map{
          case (label, seq) =>
            val positiveCount = seq.filter(_._2._1 == 1)
            (label, positiveCount.length.toFloat/seq.length)
        } .toSeq
          .filter(_._2 > mlopsConfiguration.evaluationConfig.subModelFilterThreshold)
          .sortWith(_._2 > _._2)

    ).getOrElse({
      logger.error(s"Could not load classifier training logs from $fsNohupFilename")
      Seq.empty[(String, Float)]
    }
    )
  }

  /**
    * Wraps the various claim matching algorithm
    * @param isCoreMatch Same CPT, Modifiers with one 1 ICD overlapping
    * @param isFirstLineMatch Exact match for the first line item
    * @param isStrictMatch Exact match for the entire claim
    */
  final class ClaimMatchResult(isCoreMatch: Boolean, isFirstLineMatch: Boolean, isStrictMatch: Boolean){
    override def toString: String =
      s"Core: ${if(isCoreMatch) "matched" else ""}, first line ${if(isFirstLineMatch) "matches" else ""}, strict ${if(isStrictMatch) "matched" else ""}"

    final def isCore: Boolean = isCoreMatch
    final def isFirstLine: Boolean = isFirstLineMatch
    final def isStrict: Boolean = isStrictMatch
  }


  /**
    * Extract the various match and comparison of predicted and labeled claims
    * @param internalFeedback Feedback containing at a minimum prediction and label claims
    * @return Optional pair (Matching results, comparison)
    */
  def metrics(internalFeedback: InternalFeedback): Option[(ClaimMatchResult, String)] =
    if(internalFeedback.autocoded.lineItems.nonEmpty) {
      val predictedLineItems: Seq[String] = internalFeedback.autocoded.lineItems.map(_.lineItemSpace)
      val labeledLineItems: Seq[String] = internalFeedback.finalized.lineItems.map(_.lineItemSpace.replace("  ", " "))

      val strictPredictLineItems = predictedLineItems.sortWith(_ < _)
      val strictLabeledLineItems = labeledLineItems.sortWith(_ < _)

      val strictPredictClaim = strictPredictLineItems.mkString(lineItemSeparator)
      val strictLabeledClaim = strictLabeledLineItems.mkString(lineItemSeparator)
      val comparisonOutput = s"$strictPredictClaim$comparisonSeparator$strictLabeledClaim"

      val intersect = internalFeedback.finalized.lineItems.head.icds.intersect(internalFeedback.autocoded.lineItems.head.icds)
      Some(
        new ClaimMatchResult(
          intersect.nonEmpty,  // Core labels
          strictPredictLineItems.head == strictLabeledLineItems.head,  // First line match
          strictPredictClaim == strictLabeledClaim  // Strict match
        ),
        comparisonOutput
      )
    }
    else {
      logger.warn(s"Predicted ${internalFeedback.id} has no line Item")
      None
    }
}