package org.bertspark.util.plot

import org.apache.commons.math3.distribution.NormalDistribution
import org.bertspark.nlp.trainingset.KeyedValues
import org.scalatest.flatspec.AnyFlatSpec

private[plot] final class MLinePlotTest extends AnyFlatSpec {

  ignore should "Succeed plot two variables" in {
    val plotConfig = MPlotConfiguration("Test", "x-label", "y-label", "params")
    val linePlot = new MLinePlot(plotConfig)
    val x = Array.tabulate(150)(n => n*0.1F - 2.5F)
    val normalDist = new NormalDistribution(0.0, 0.4)
    val y = ("Values", x.map(normalDist.density(_).toFloat))

    linePlot(x, y)
  }


  ignore should "Succeed plot three variables" in {
    val plotConfig = MPlotConfiguration("Test", "x", "Distribution", "params")
    val linePlot = new MLinePlot(plotConfig)
    val x = Array.tabulate(200)(n => ((100 -n)*0.1F))
    val normalDist = new NormalDistribution(0.0, 0.4)
    val normalDist2 = new NormalDistribution(0.0, 0.9)
    val y = ("Normal(0, 0.4)", x.map(normalDist.density(_).toFloat))
    val z = ("Normal(0, 0.9)", x.map(normalDist2.density(_).toFloat))
    linePlot(x, y, z)
  }

  it should "Succeed plot four variables" in {
    val plotConfig = MPlotConfiguration("Test", "x", "Distribution", "params")
    val linePlot = new MLinePlot(plotConfig)
    val x = Array.tabulate(200)(n => ((100 -n)*0.1F))
    val normalDist = new NormalDistribution(0.0, 0.4)
    val normalDist2 = new NormalDistribution(0.0, 0.9)
    val function = (x: Double) => Math.sin(x*2)*x*0.05
    val function2 = (x: Double) => Math.exp(x*0.1)*Math.cos(x)*0.2
    val labeledValues = Seq[KeyedValues](
      ("Normal(0, 0.4)", x.map(normalDist.density(_).toFloat)),
      ("Normal(0, 0.9)", x.map(normalDist2.density(_).toFloat)),
      ("x.sin(x)", x.map(function(_).toFloat)),
      ("e.cos(x)",  x.map(function2(_).toFloat))
    )

    linePlot(x, labeledValues)
  }
}
