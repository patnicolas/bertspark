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

import ai.djl.metric.Metrics
import ai.djl.training.evaluator.Evaluator
import ai.djl.training.Trainer
import org.bertspark.config.MlopsConfiguration.DebugLog.logInfo
import org.bertspark.dl._
import org.bertspark.implicits.collection2Scala
import org.bertspark.util.plot._
import org.slf4j._
import scala.collection.mutable.HashMap


/**
 * Model metrics display in various format CSV, JSON, HTML....
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] trait TrainingSummary {
  import TrainingSummary._

  protected[this] val evaluatorMetrics = new HashMap[String, Array[Float]]()
  protected[this] val metricsAccumulator = HashMap[String, List[Float]]()

  def apply(trainer: Trainer): Unit = {
    val metrics = trainer.getMetrics
    trainer.getEvaluators.foreach(
      evaluator => {
        updateEvaluatorMetrics(evaluator, metrics, label = trainLabel)
        updateEvaluatorMetrics(evaluator, metrics, label = validLabel)
      }
    )
  }


  /**
   * Mainly used for testing
   * @param metricName Name of the metrics
   * @param values Values of the metrics
   */
  def += (metricName: String, values: Array[Float]): Unit =
    evaluatorMetrics.put(metricName, values)

  def toPlot(plotConfiguration: MPlotConfiguration): Unit = {
    val linePlot = new MLinePlot(plotConfiguration)
    if(evaluatorMetrics.nonEmpty) {
      val epochNumber = (0 until evaluatorMetrics.head._2.length).map(_.toFloat).toArray
      linePlot(epochNumber, evaluatorMetrics.toSeq)
    }
    else
      logger.error("Evaluator metrics are empty for plotting")
  }

  def toRawText: String =
    textSummary.map{
      case (variable, values) => s"$variable: ${values.map("%.3f".format(_)).mkString(", ")}"
    }.mkString("\n")

  def toHtml: String  = ???

  def textSummary: Seq[(String, Array[Float])] = evaluatorMetrics.toSeq

  def textSummary(trainContext: TrainingContext): Seq[(String, Array[Float])] =
    trainContext.getEvaluationMetrics.flatMap(
      key => {
        key match {
         case "loss" =>
            val trainKey = s"${trainLabel}_${trainContext.getLossName}"
            val validKey = s"${validLabel}_${trainContext.getLossName}"
            Seq[(String, Array[Float])](
              (trainKey, evaluatorMetrics.get(trainKey).getOrElse(Array.empty[Float])),
              (validKey, evaluatorMetrics.get(validKey).getOrElse(Array.empty[Float]))
            )
         case _ =>
            val trainKey = s"${trainLabel}_${key}"
            val validKey = s"${validLabel}_${key}"
            Seq[(String, Array[Float])](
              (trainKey, evaluatorMetrics.get(trainKey).getOrElse(Array.empty[Float])),
              (validKey, evaluatorMetrics.get(validKey).getOrElse(Array.empty[Float]))
            )
        }
      }
    )

  private def updateEvaluatorMetrics(evaluator: Evaluator, metrics: Metrics, label: String): Unit = {
    val key = evaluator.getName
    val values = metrics.getMetric(key).map(_.getValue.toFloat).toArray
    evaluatorMetrics.put(key, values)
  }

  def displayResults(trainer: Trainer, isPretraining: Boolean) = {
    import org.bertspark.implicits._

    val results = trainer.getTrainingResult
    val filteredEvaluations: java.util.Map[String, java.lang.Float] =
      if(isPretraining) results.getEvaluations.filter{ case (key, _) => !key.startsWith("validate") }
      else results.getEvaluations

    val evaluations = map2Scala(filteredEvaluations).toSeq
    evaluations.foreach{
      case (key, value) => {
        val xs = metricsAccumulator.getOrElse(key, List[Float]())
        metricsAccumulator.put(key, value :: xs)
      }
    }
    logInfo(logger,  evaluations.mkString("  "))
  }
}

private[bertspark] final object TrainingSummary {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[TrainingSummary])
}
