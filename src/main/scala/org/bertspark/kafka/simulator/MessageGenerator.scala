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
  */
package org.bertspark.kafka.simulator

import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.kafka.prodcons.TypedKafkaProducer
import org.slf4j.{Logger, LoggerFactory}



/**
  * Predicting message generator for testing purpose
  * @param requestProducer       Producer/generator for prediction request
  * @param requests              Set of requests
  * @param produceIntervalMs Time out (sleep) in milliseconds
  * @param numRepeats            Number of batches of requests used for the generator
  *
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] abstract class MessageGenerator[T](
  requestProducer: TypedKafkaProducer[T],
  protected val requests: Seq[T],
  produceIntervalMs: Long,
  numRepeats: Int) extends Thread {
  import MessageGenerator._

  protected[this] val topic: String
  var repeatCount = 0

  /**
    * Thread that manage the simulator
    */
  override def run: Unit =
    while (repeatCount < numRepeats) {
      proceed
      repeatCount += 1
    }


  def proceed: Unit = {
    logDebug(logger,s"Started producing ${requests.size} requests for $topic")
    var requestCounter = repeatCount
    val requestIterator = requests.toIterator
    while(requestIterator.hasNext) {
      val request = requestIterator.next()
      produce(((requestCounter.toString, request)))
      requestCounter += 1
    }
    if(requestCounter == 0)
      logger.warn(s"No requests iterator")
  }

  def produce(req: (String, T)): Unit = {
    requestProducer.send(req)
    logDebug(logger, s"Message #${req._1.toInt} sent to ${topic}")
  }
}

private[bertspark] final object MessageGenerator {
  final val logger: Logger = LoggerFactory.getLogger("MessageGenerator")
}

