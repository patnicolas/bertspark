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

import tech.tablesaw.api.Table
import tech.tablesaw.plotly.components._
import tech.tablesaw.plotly.traces.AbstractTrace
import tech.tablesaw.plotly.Plot

/**
 * Base class for all plots
 * @param layout Layout for the plot
 * @param table Tabular structure for the generation of plots
 * @tparam T Type of the plot
 * @tparam U Type of the trace
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] abstract class MJSPlot[T <: MJSPlot[T,U], U <: AbstractTrace](layout: Layout, table: Table) {

  val extractColumns: Table => U

  def apply(): Unit = {
    val trace = extractColumns(table)
    Plot.show(new Figure(layout, trace))
  }
}



private[bertspark] final object MJSPlot  {
}