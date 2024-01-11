package org.bertspark.modeling

import org.bertspark.util.plot.MPlotConfiguration
import org.scalatest.flatspec.AnyFlatSpec

private[bertspark] final class TrainingSummaryTest extends AnyFlatSpec {

  it should "Succeed in display text for metrics output" in {
    val metricsOutput = new TrainingSummary {
    }
    val function1 = (x: Float) => Math.sin(x*2)*x*0.05F
    val function2 = (x: Float) => Math.exp(x*0.1F)*Math.cos(x)*0.2F
    metricsOutput += ("key1", Array.tabulate(1000)(n => function1(n.toFloat).toFloat))
    metricsOutput += ("key2", Array.tabulate(1000)(n => function2(n.toFloat).toFloat))

    val textSummary = metricsOutput.toRawText
    println(textSummary)
  }

  it should "Succeed in display plots for metrics output" in {
    val metricsOutput = new TrainingSummary {
    }
    val function1 = (x: Float) => Math.sin(x*0.05F)
    val function2 = (x: Float) => Math.cos(x*0.02F)+ 0.2F
    metricsOutput += ("key1", Array.tabulate(1000)(n => function1(n.toFloat).toFloat))
    metricsOutput += ("key2", Array.tabulate(1000)(n => function2(n.toFloat).toFloat))

    val plotConfig = MPlotConfiguration("Eval", "x", "Distribution", "params")
    metricsOutput.toPlot(plotConfig)
  }
}
