package org.bertspark.kafka.simulator

import org.scalatest.flatspec.AnyFlatSpec

class KafkaSimulatorTest extends AnyFlatSpec {

  it should "Succeed preselecting request for testing" in {
    val s3SrcFolder = "requests/Cornerstone"
    val s3DestFolder = "requestsEval/Cornerstone"
    val limit = 20000
  }
}
