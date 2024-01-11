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
package org.bertspark.util.plot

import org.apache.commons.lang3.ArrayUtils
import org.bertspark.nlp.trainingset.KeyedValues
import org.bertspark.util.plot.MPlot.buildTable
import tech.tablesaw.api.Table
import tech.tablesaw.plotly.api.LinePlot
import tech.tablesaw.plotly.Plot


private[bertspark] case class MPlotConfiguration(
  title: String,
  xLabel: String,
  yLabel: String,
  groupCol: String)  {
  override def toString: String = s"Title: $title, XLabel: $xLabel, YLabel: $yLabel, Columns: $groupCol"
}


/**
 * Generic Line plot
 * @param plotConfiguration Configuration of the plot
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class MLinePlot(plotConfiguration: MPlotConfiguration) extends MPlot[MLinePlot] {
  override def apply(table: Table): Unit = {
    val figure = LinePlot.create(
      plotConfiguration.title,
      table,
      plotConfiguration.xLabel,
      plotConfiguration.yLabel,
      plotConfiguration.groupCol)
    Plot.show(figure)
  }

  def apply(x: Array[Float], y: KeyedValues): Unit = {
    val table = buildTable(plotConfiguration, x, y)
    apply(table)
  }


  def apply(x: Array[Float], y: KeyedValues, z: KeyedValues): Unit = {
    val table = buildTable(plotConfiguration, x, y, z)
    apply(table)
  }

  def apply(x: Array[Float], y: Seq[KeyedValues]): Unit = {
    val table = buildTable(plotConfiguration, x, y)
    apply(table)
  }
}


private[bertspark] final object MLinePlot {
  def combine(x: Array[Float], y: Float): Array[Float] = ArrayUtils.addFirst(x, y)
}