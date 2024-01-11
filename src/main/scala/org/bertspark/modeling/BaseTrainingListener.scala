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

import ai.djl.training.listener.LoggingTrainingListener
import ai.djl.training.Trainer
import ai.djl.training.listener.TrainingListener.BatchData
import ai.djl.util.Utils
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicInteger
import org.bertspark._
import org.bertspark.config._
import org.bertspark.config.MlopsConfiguration.DebugLog._
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.KeyedValues
import org.bertspark.util.io._
import org.bertspark.util.plot._
import org.bertspark.util.DateUtil
import org.slf4j._
import scala.collection.mutable.HashMap


/**
 * Customized training listener to support collection of metrics in progress for future plots
 * @param trainingContext Training context
 * @param modelName Description of the test
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] abstract class BaseTrainingListener protected (
  trainingContext: TrainingContext,
  modelName: String
) extends LoggingTrainingListener {
  import BaseTrainingListener._
  import org.bertspark.config.MlopsConfiguration._

  private[this] val startTime = System.currentTimeMillis()
  private[this] var filteredMetricNames: scala.collection.Set[String] = scala.collection.Set.empty[String]

  protected[this] val metricAccumulator = HashMap[String, List[Float]]()
  protected[this] val epochNo = new AtomicInteger(0)


  final def getEpochNo: Int = epochNo.get()

  final def trainEpochAccuracy: Float = metricAccumulator.get("train_epoch_Accuracy").map(_.last).getOrElse(-1.0F)
  final def validEpochAccuracy: Float = metricAccumulator.get("validate_epoch_Accuracy").map(_.last).getOrElse(-1.0F)

  // Abstract methods...
  protected def record(trainer: Trainer, enabled: Boolean): Unit

  /**
   * Update various statistics and model files at the end of the epoch and store into a local file if the
   * saveEachEpoch is set to true
   * {{{
   *   The 2 files to be stored are
   *   - Model parameters
   *   - Statistics for plots
   * }}}
   * @param trainer Reference to the current trainer
   */
  override def onEpoch(trainer: Trainer): Unit = {
    // Invoke the default statistics gathering from the parent class
    super.onEpoch(trainer)
    updateMetrics(trainer)
    record(trainer, mlopsConfiguration.executorConfig.saveOnEachEpoch)
  }


  override def onTrainingBatch(trainer: Trainer, batchData: BatchData): Unit =
    super.onTrainingBatch(trainer, batchData)

  /**
   * Save the statistics for all the training run
   * @param trainer Trainer instance managing the training run
   */
  override def onTrainingEnd(trainer: Trainer): Unit = {
    super.onTrainingEnd(trainer)
    record(trainer, mlopsConfiguration.executorConfig.saveEndOfTraining)
  }

  def add(metricKey: String, values: List[Float]): Unit = metricAccumulator.put(metricKey, values)

  def add(metricKey: String, value: Float): Unit = {
    val values = metricAccumulator.getOrElse(metricKey, List[Float]())
    metricAccumulator.put(metricKey, value :: values)
  }

  def add(metricMap: Map[String, Float]): Unit =
    if(metricMap.nonEmpty)
      metricMap.foreach{ case (key, value) => add(key, value)}


  // --------------------  Shared Methods ------------------------------

  /**
    *  Collect the list of metrics accumulated so far. The elements in the list are
    *  pre-pended so they should be reversed when loading
    * @param modelType Either Transformer/pre-training or classifier
    * @return List of historical metrics (reverse order) fot
    */
  protected def getMetrics(modelType: String = ""): List[String] =
    metricAccumulator.toList.sortWith(_._1 < _._1).map{
      case (metricName, values) => s"$metricName,${values.reverse.toArray.mkString(" ")}"
    }


  protected def getCurrentEpoch(path: Path, modelName: String): String = {
    val currentEpoch = Utils.getCurrentEpoch(path, modelName)
    if (currentEpoch > 99) s"0$currentEpoch"
    else if (currentEpoch > 9) s"00$currentEpoch"
    else s"000$currentEpoch"
  }


  // --------------------  Supporting/Helper Methods ------------------------------


  private def updateMetrics(trainer: Trainer): Unit = {
    epochNo.incrementAndGet()

    logInfo(logger,  s"Duration of epoch: ${(System.currentTimeMillis() - startTime)*0.001} secs.")
    initializeMetricNames(trainer)
    filteredMetricNames.map( updateMetricAccumulator(_, trainer))
  }

  private def initializeMetricNames(trainer: Trainer): Unit =
    if(filteredMetricNames.isEmpty) {
      import org.bertspark.implicits._
      val metricNames: scala.collection.Set[String] = trainer.getMetrics.getMetricNames()
      filteredMetricNames = metricNames.filter(validTrainingMetricNames.contains(_))
    }


  private def updateMetricAccumulator(metricName: String, trainer: Trainer): Unit = {
    val values = metricAccumulator.getOrElse(metricName, List[Float]())
    val newValue = trainer.getMetrics.latestMetric(metricName).getValue.toFloat
    val scaledValue = scalingMetrics(metricName, newValue)
    // If memory value exceeds max allowed, shutdown the application
    if(scaledValue == -1.0F) {
      logger.error(s"Could not found accumulator for $metricName")
      implicits.close
    }
    metricAccumulator.put(metricName, scaledValue :: values)
  }
}


/**
 * Singleton for constructor
 */
private[bertspark] final object BaseTrainingListener {
  import org.bertspark.config.MlopsConfiguration._
  final val logger: Logger = LoggerFactory.getLogger("BaseTrainingListener")

  final private val currentDate = DateUtil.longToDate

  final private val validTrainingMetricNames =
    if(ExecutionMode.isPretraining)
      Array[String](
        "train_epoch_Accuracy",
        "train_epoch_BertPretrainingLoss",
        "train_all_BertPretrainingLoss",
        "train_all_Accuracy",
        "train_all_loss",
        "Heap",
        "NonHeap",
        "rss"
      )
    else
      Array[String](
        "train_epoch_Accuracy",
        "train_epoch_loss",
        "validate_epoch_Accuracy",
        "validate_epoch_loss",
        "train_all_loss",
        "train_all_Accuracy",
        "Heap",
        "NonHeap",
        "rss"
      )

  private val invMbytes = 1.0/(1024*1024)



    // ---------------- Plotting methods -------------------------------
  /**
   * Create a line plot from the file containing training progress data
   * {{{
   *   The CSV file format is:
   *   Header:  Descriptor,*
   *   Rows: Metric,value1 value2 value3 ....
   * }}}
   * @param trainingRunName Name of the training variables
   */
  def createPlots(trainingRunName: String): Unit = {
    var descriptor: String = ""
    LocalFileUtil.Load
        .local(trainingRunName, (s: String) => s.split(","), false)
        .map(_.map(ar => {
          if(ar(1) == "*") {
            descriptor = ar.head
            ("", Array.empty[Float])
          }
          else
            (ar.head, ar(1).split(tokenSeparator).map(_.toFloat))}
        ))
        .map(_.filter(_._1.nonEmpty).partition(_._1.contains("Accuracy")))
        .map{
          case (accuracyStats, lossStats) => {
            plotStats("Accuracy", descriptor, accuracyStats)
            plotStats("Losses", descriptor, lossStats)
          }
        }.getOrElse(
      throw new DLException(s"Failed to found output plots for $trainingRunName")
    )
  }


  private def plotStats(metricCategory: String, descriptor: String, stats: Array[(String, Array[Float])]): Unit =
    if(stats.nonEmpty) {
      val plotConfig = MPlotConfiguration(
        s"Train-$metricCategory-$currentDate",
        s"Epochs $descriptor",
        metricCategory,
        "Training session"
      )
      val linePlot = new MLinePlot(plotConfig)
      val size = stats.head._2.size
      val x = Array.tabulate(size)(n => (n + 1).toFloat)

      val labeledValues: Seq[KeyedValues] = stats.map { case (metricName, values) => (metricName, values.reverse)}.toSeq
      linePlot(x, labeledValues)
    }
    else
      logger.error("Cannot plot empty stats")



    // -----------------  Private supporting methods ---------------------

  private def scalingMetrics(thisMetricName: String, value: Float): Float = thisMetricName match {
    case "Heap" => monitoryValue(
      (value*invMbytes).floor.toInt,
      mlopsConfiguration.executorConfig.maxHeapMemMB,
      desc = "Heap")
    case "NonHeap" => monitoryValue(
      (value*invMbytes).floor.toInt,
      mlopsConfiguration.executorConfig.maxNonHeapMemMB,
      desc = "NonHeap")
    case "rss" => monitoryValue(
      (value*invMbytes).floor.toInt,
      mlopsConfiguration.executorConfig.maxRssMemMB,
      desc = "RSS")
    case _ => value
  }

  private def monitoryValue(value: Int, maxValue: Int, desc: String): Float = {
    if(value > maxValue) {
      logger.error(s"$desc memory $value exceeds $maxValue")
      -1.0F
    }
    else if(value > (maxValue>>1)) {
      logger.warn(s"$desc memory $value exceeds ${maxValue>>1}")
      value
    }
    else
      value
  }
}
