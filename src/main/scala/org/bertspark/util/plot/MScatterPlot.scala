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

import org.bertspark.nlp.trainingset.KeyedValues
import org.bertspark.util.plot.MPlot.buildTable
import tech.tablesaw.api.Table
import tech.tablesaw.plotly.api.ScatterPlot
import tech.tablesaw.plotly.Plot


/**
 * Generic scatter plot to be display on a web page
 * @param plotConfiguration Configuration for this plot
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class MScatterPlot(plotConfiguration: MPlotConfiguration) extends MPlot[MLinePlot] {

  override def apply(table: Table): Unit = {
    val figure = ScatterPlot.create(
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
