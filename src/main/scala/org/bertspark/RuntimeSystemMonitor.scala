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
package org.bertspark

import ai.djl.metric.Metrics
import ai.djl.training.listener.MemoryTrainingListener
import ai.djl.util.cuda.CudaUtils
import ai.djl.Device
import java.lang.management.MemoryUsage
import java.util.concurrent.atomic.AtomicInteger
import org.bertspark.config.MlopsConfiguration
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.util.io.LocalFileUtil
import org.slf4j.Logger
import scala.collection.mutable.ListBuffer


/**
 * Simple memory monitor to be added to loss functions
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] trait RuntimeSystemMonitor {
self =>
  import MlopsConfiguration._
  import RuntimeSystemMonitor._

  private[this] lazy val rssBuffer = ListBuffer[String]()
  private[this] lazy val heapBuffer = ListBuffer[String]()
  private[this] lazy val buffersMap = Map[String, ListBuffer[String]](
    "rss" -> rssBuffer,
    "Heap" -> heapBuffer
  )
  protected[this] val counter = new AtomicInteger(0)

  protected[this] val metrics: Metrics = {
    val _metrics = new Metrics()
    MemoryTrainingListener.collectMemoryInfo(_metrics)
    _metrics
  }

  def rss: Unit = collect("rss", 20)
  def heap: Unit = collect("Heap", 24)

  /**
   * Generate a report on the CPU and memory usage
   * @return String/line of RSS, HEAP, Non-HEAP, CPU and GPU usage
   */
  def allMetrics(descriptor: String): String =
    if(metrics != null && metrics.getMetricNames.contains("rss")){
      MemoryTrainingListener.collectMemoryInfo(metrics)

      val rssMemory: Int = runtimeMemoryItem("rss")
      val heapMemory: Int = runtimeMemoryItem("Heap")
      val nonHeapMemory: Int = runtimeMemoryItem("NonHeap")
      val cpuUsage: Float = metrics.latestMetric("cpu").getValue.toFloat
      val (gpuCount, gpuMem) =
        if(mlopsConfiguration.executorConfig.dlDevice == "gpu") {
          val gpuUsage =
            if (CudaUtils.getGpuCount() > 0)
              (0 until CudaUtils.getGpuCount()).map(
                index => {
                  val gpuIndex = s"GPU-${index}"
                  s"$gpuIndex ${runtimeMemoryItem(gpuIndex)}"
                }
              ).mkString(", ")
            else
              "NO GPU"
          val gpuMemory: MemoryUsage = CudaUtils.getGpuMemory(Device.gpu(0))
          val maxMemoryMb: Double = gpuMemory.getMax*0.001
          val usedMb: Double = gpuMemory.getUsed*0.001
          val usageRate: Double = 100*usedMb/maxMemoryMb
          val gpuMemoryUsage = s"Used mem=${usedMb}MB, Usage=$usageRate%"
          (gpuUsage, gpuMemoryUsage)
        }
        else
          ("", "")
      s"$descriptor RSS=${rssMemory}MB, HEAP=${heapMemory}MB, NON-HEAP=${nonHeapMemory}MB, CPU=${cpuUsage}% GPU=$gpuCount $gpuMem"
    }
    else
      ""

  protected def log(logger: Logger, collectionInterval: Int, marker: String): Unit = {
    val count = counter.incrementAndGet()

    if(count % collectionInterval == 0)
      logDebug(
        logger, {
          val metricsSummary = allMetrics(marker)
          if(metricsSummary.nonEmpty) metricsSummary else ""
        }
      )
  }

  // ----------------------------- Supporting method ------------------------------------

  private def runtimeMemoryItem(metricName: String): Int = {
    val value: Float = metrics.latestMetric(metricName).getValue.toFloat
    (value*invMbytes).floor.toInt
  }

  private def collect(metricName: String, batchCountInterval: Int): Unit =
    if(metrics != null && metrics.getMetricNames.contains(metricName) &&  mlopsConfiguration.isLogTrace){
      MemoryTrainingListener.collectMemoryInfo(metrics)
      val buffer = buffersMap.getOrElse(metricName, ListBuffer[String]())
      buffer.append(runtimeMemoryItem(metricName).toString)
      if(buffer.size == 0)
        buffer.append("\n")

      if(buffer.size % batchCountInterval == 0)
        LocalFileUtil.Save.local(s"output/$metricName-${buffer.size}.csv", s"\n${buffer.mkString("\n")}")
    }
}

private[bertspark] final object RuntimeSystemMonitor {
  final private val invMbytes = 1.0/(1024*1024)

  private val rssMemoryCollector = ListBuffer[Int]()

  def rssHistory: Seq[Int] = rssMemoryCollector
}
