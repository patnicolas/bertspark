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

import tech.tablesaw.api.{FloatColumn, StringColumn, Table}

/**
 * Generic plot without trace
 * @tparam T Type of the plot
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
trait MPlot[T <: MPlot[T]] {
  self =>
  def apply(table: Table): Unit
}


private[bertspark] final object MPlot {

  def buildTable(
    plotConfiguration: MPlotConfiguration,
    x: Array[Float],
    y: (String, Array[Float])): Table = {
    import org.bertspark.implicits._

    val paramsY = Array.fill(x.length)(y._1)
    Table.create(plotConfiguration.title).addColumns(
      FloatColumn.create(plotConfiguration.xLabel, x :_*),
      FloatColumn.create(plotConfiguration.yLabel, y._2 :_*),
      StringColumn.create(plotConfiguration.groupCol, paramsY)
    )
  }

  def buildTable(
    plotConfiguration: MPlotConfiguration,
    x: Array[Float],
    y: (String, Array[Float]),
    z: (String, Array[Float])): Table = {
    import org.bertspark.implicits._

    val paramsY = Array.fill(x.length)(y._1)
    val paramsZ = Array.fill(x.length)(z._1)
    val params = paramsY ++ paramsZ
    val yzValues = y._2 ++ z._2

    Table.create(plotConfiguration.title).addColumns(
      FloatColumn.create(plotConfiguration.xLabel, x ++ x :_*),
      FloatColumn.create(plotConfiguration.yLabel, yzValues :_*),
      StringColumn.create(plotConfiguration.groupCol, params)
    )
  }

  def buildTable(
    plotConfiguration: MPlotConfiguration,
    x: Array[Float],
    y: Seq[(String, Array[Float])]): Table = {
    import org.bertspark.implicits._

    val (params, values) = y.map{ case (str, ar) => (Array.fill(ar.size)(str), ar)}.unzip
    val v = values.flatten.toArray
    val xValues = Array.fill(y.size)(x).flatten
    Table.create(plotConfiguration.title).addColumns(
      FloatColumn.create(plotConfiguration.xLabel, xValues :_*),
      FloatColumn.create(plotConfiguration.yLabel, v :_*),
      StringColumn.create(plotConfiguration.groupCol, params.flatten)
    )
  }


  def buildTable(
    plotConfiguration: MPlotConfiguration,
    x: Array[Float],
    y: Array[Float],
    z: Array[Float],
    t: Array[Float]): Table = {
    import org.bertspark.implicits._

    val paramsY = Array.fill(x.length)("y")
    val paramsZ = Array.fill(x.length)("z")
    val paramsT = Array.fill(x.length)("t")
    val params = paramsY ++ paramsZ ++ paramsT
    val yzValues = y ++ z ++ t

    Table.create(plotConfiguration.title).addColumns(
      FloatColumn.create(plotConfiguration.xLabel, x ++ x ++ x :_*),
      FloatColumn.create(plotConfiguration.yLabel, yzValues :_*),
      StringColumn.create(plotConfiguration.groupCol, params)
    )
  }

  def buildTable(
    plotConfiguration: MPlotConfiguration,
    x: Array[Float],
    y: Array[Float],
    z: Array[Float],
    t: Array[Float],
    u: Array[Float]): Table = {
    import org.bertspark.implicits._

    val yName = x.getClass.getDeclaredFields.head.getName

    val paramsY = Array.fill(x.length)("y")
    val paramsZ = Array.fill(x.length)("z")
    val paramsT = Array.fill(x.length)("t")
    val paramsU = Array.fill(x.length)("u")
    val params = paramsY ++ paramsZ ++ paramsT ++ paramsU
    val yzValues = y ++ z ++ t ++ u

    Table.create(plotConfiguration.title).addColumns(
      FloatColumn.create(plotConfiguration.xLabel, x ++ x ++ x ++ x:_*),
      FloatColumn.create(plotConfiguration.yLabel, yzValues :_*),
      StringColumn.create(plotConfiguration.groupCol, params)
    )
  }
}
