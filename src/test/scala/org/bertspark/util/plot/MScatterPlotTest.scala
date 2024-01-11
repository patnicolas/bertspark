package org.bertspark.util.plot

import org.apache.commons.math3.distribution.NormalDistribution
import org.scalatest.flatspec.AnyFlatSpec

private[plot] final class MScatterPlotTest extends AnyFlatSpec {

  it should "Succeed plot two variables" in {
    val plotConfig = MPlotConfiguration("Test", "x-label", "y-label", "params")
    val scatterPlot = new MScatterPlot(plotConfig)
    val x = Array.tabulate(150)(n => n*0.1F - 2.5F)
    val normalDist = new NormalDistribution(0.0, 0.4)
    val y = ("obs", x.map(normalDist.density(_).toFloat))

    scatterPlot(x, y)
  }



  it should "Succeed plot three variables" in {
    val plotConfig = MPlotConfiguration("Test", "x-label", "y-label", "params")
    val scatterPlot = new MScatterPlot(plotConfig)
    val x = Array.tabulate(150)(n => n*0.1F - 2.5F)
    val normalDist = new NormalDistribution(0.0, 0.4)
    val normalDist2 = new NormalDistribution(0.0, 0.9)
    val y = ("Normal(0. 0.4)", x.map(normalDist.density(_).toFloat))
    val z = ("Normal(0. 0.9)", x.map(normalDist2.density(_).toFloat))
    scatterPlot(x, y, z)
  }
}
