package org.bertspark

import org.apache.spark.sql.SparkSession

package object modeling {

  trait InputValidation {
    @throws(clazz = classOf[InvalidParamsException])
    protected def validate(args: Seq[String]): Unit
  }


  trait ModelOutput[T] {
    protected def output(t: T)(implicit sparkSession: SparkSession): Unit
  }

  final val labelIndicesSeparator = "||"
}
