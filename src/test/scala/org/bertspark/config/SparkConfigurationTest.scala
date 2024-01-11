package org.bertspark.config

import org.scalatest.flatspec.AnyFlatSpec

private[bertspark] final class SparkConfigurationTest extends AnyFlatSpec{

  it should "Succeed stripping margin" in {
    val s1 =
      s"""field1
         |field2
         |""".stripMargin
    println(s1)
  }

  it should "Succeed initializing the spark context" in {
    val sparkConfig = SparkConfiguration.mlSparkConfig
    println(sparkConfig.toString)
  }
}
