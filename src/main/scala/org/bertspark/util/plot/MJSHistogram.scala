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

import tech.tablesaw.api.{IntColumn, StringColumn, Table}
import tech.tablesaw.plotly.components.Layout
import tech.tablesaw.plotly.traces.HistogramTrace
import tech.tablesaw.plotly.traces.HistogramTrace.HistFunc


/**
 * Generic Histogram plot. The various constructors are defined in the companion object
 * @param layout Layout for the histogram
 * @param table Table
 * @param extractColumns extraction of Histogram from Table
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class MJSHistogram(
  layout: Layout,
  table: Table,
  override val extractColumns: Table => HistogramTrace) extends MJSPlot[MJSHistogram, HistogramTrace](layout, table) {
}


private[bertspark] final object MJSHistogram {
  def apply(
    layout: Layout,
    table: Table,
    extractColumns: Table => HistogramTrace): MJSHistogram = new MJSHistogram(layout, table, extractColumns)


  def apply(
    layout: Layout,
    yCol: String,
    xCol: String,
    distribution: Seq[(String, Int)],
    extractColumns: Table => HistogramTrace
  ): MJSHistogram = {
    val xColumn = distribution.foldLeft(StringColumn.create(yCol)) { case (typeVar, (varType, _)) => typeVar.append(varType)}
    val yColumn = distribution.foldLeft(IntColumn.create(xCol)) { case (typeVar, (_, varValue)) => typeVar.append(varValue)}

    val table = Table.create(xColumn, yColumn)
    apply(layout, table, extractColumns)
  }


  def apply(
    yCol: String,
    xCol: String,
    distribution: Seq[(String, Int)],
    extractColumns: Table => HistogramTrace
  ): MJSHistogram = {
    val layout = Layout.builder.title("Histogram").build
    apply(layout, yCol, xCol, distribution, extractColumns)
  }

  def apply(
    yCol: String,
    xCol: String,
    distribution: Seq[(String, Int)],
    histFunction: HistFunc = HistFunc.SUM
  ): MJSHistogram = {
    val layout = Layout.builder.title("Histogram").build
    val extractColumns = (table: Table) => HistogramTrace.builder(table.stringColumn(yCol), table.intColumn(xCol))
        .histFunc(histFunction)
        .build
    apply(layout, yCol, xCol, distribution, extractColumns)
  }

  def apply(yCol: String, distribution: Seq[String]): MJSHistogram = {

    val xColumn = distribution.foldLeft(StringColumn.create(yCol)) { case (typeVar, varType) => typeVar.append(varType)}
    val table = Table.create(xColumn)
    val layout = Layout.builder.title("Histogram").build
    val extractColumns = (table: Table) => HistogramTrace.builder(table.stringColumn(yCol), table.intColumn("count"))
        .histFunc(HistFunc.COUNT)
        .build
    new MJSHistogram(layout, table, extractColumns)
  }
}
