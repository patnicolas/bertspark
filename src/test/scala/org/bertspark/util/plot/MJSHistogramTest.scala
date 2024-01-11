package org.bertspark.util.plot

import org.scalatest.flatspec.AnyFlatSpec
import tech.tablesaw.api.{IntColumn, StringColumn, Table}
import tech.tablesaw.plotly.components.Layout
import tech.tablesaw.plotly.traces.HistogramTrace
import tech.tablesaw.plotly.traces.HistogramTrace.HistFunc

private[plot] final class MJSHistogramTest extends AnyFlatSpec {
/*
  it should "Succeed drawing histogram first constructor" in {
    val distribution = Seq.tabulate(30)(n => {
      if(n%3 == 0x0) "A"
      else if(n%5 == 0x0) "B"
      else if(n%7 == 0x0) "C"
      else if(n%2 == 0x0) "D"
      else "E"
    })
    val typeVar = StringColumn.create("type")
    val xColumn = distribution./:(typeVar) { case (typeVar, (varType, _)) => typeVar.append(varType)}
    val yColumn = distribution./:(IntColumn.create("count")) { case (typeVar, (_, varValue)) => typeVar.append(varValue)}

    val table = Table.create(xColumn, yColumn)
    val layout = Layout.builder.title("Histogram").build
    val traceFunc = (table: Table) => HistogramTrace.builder(table.stringColumn("type"), table.intColumn("count"))
        .histFunc(HistFunc.COUNT)
        .build

    val histogramPlot = MHistogramPlot(layout, table, traceFunc)
    histogramPlot()
  }

 */

  ignore should "Succeed drawing histogram second constructor" in {
    val distribution = Seq[(String, Int)](
      ("A", 3), ("B", 1), ("C", 7), ("A",2), ("D", 4), ("C",10),  ("C", 11), ("A",9), ("D", 2), ("C",7)
    )

    val histogramPlot = MJSHistogram("categories", "count", distribution)
    histogramPlot()
  }


  it should "Succeed drawing histogram first constructor and distribution" in {
    val distribution = Seq[(String, Int)](
    ("A", 3), ("B", 1), ("C", 7), ("A",2), ("D", 4), ("C",10)
    )
    val typeVar = StringColumn.create("type")
    val xColumn = distribution.foldLeft(typeVar) { case (typeVar, (varType, _)) => typeVar.append(varType)}
    val yColumn = distribution.foldLeft(IntColumn.create("count")) { case (typeVar, (_, varValue)) => typeVar.append(varValue)}

    val table = Table.create(xColumn, yColumn)
    val layout = Layout.builder.title("Histogram").build
    val traceFunc = (table: Table) => HistogramTrace.builder(table.stringColumn("type"), table.intColumn("count"))
        .histFunc(HistFunc.SUM)
        .build

    val histogramPlot = MJSHistogram(layout, table, traceFunc)
    histogramPlot()
  }

  ignore should "Succeed drawing histogram second constructor and distribution" in {
    val distribution = Seq[(String, Int)](
      ("A", 3), ("B", 1), ("C", 7), ("A",2), ("D", 4), ("C",10),  ("C", 11), ("A",9), ("D", 2), ("C",7)
    )

    val histogramPlot = MJSHistogram("categories", "count", distribution)
    histogramPlot()
  }
}
