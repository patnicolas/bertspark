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

import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.delay
import org.bertspark.kafka.prodcons.TypedKafkaProducer
import org.bertspark.kafka.serde.RequestSerDe.RequestMessage
import org.bertspark.kafka.simulator.RequestMessageGenerator.convertToRequestMessages
import org.bertspark.modeling.SubModelsTaxonomy.subModelTaxonomy
import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalRequest, PRequest}
import org.bertspark.util.io.S3Util
import org.slf4j.{Logger, LoggerFactory}


/**
  * Versatile generator for request messages
  *
  * @param topic Producer topic - Response
  * @param pRequests Iterator for requests
  * @param produceIntervalMs Time out (sleep) in milliseconds
  * @param numRepeats  Number of batches of requests used for the generator
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] final class RequestMessageGenerator private (
  override protected val topic: String,
  pRequests: Seq[PRequest],
  produceIntervalMs: Long = 0L,
  numRepeats: Int = 1)
    extends MessageGenerator[RequestMessage](
      new TypedKafkaProducer[RequestMessage](org.bertspark.kafka.serde.RequestSerDe.serializingClass, topic),
      convertToRequestMessages(pRequests),
      produceIntervalMs,
      numRepeats) {
  import MessageGenerator._

  def execute: Boolean = {
    this.start()
    logDebug(logger, s"Starts generating of ${pRequests.size} prediction requests")
    val expectedDuration = requests.size * 10000L
    delay(expectedDuration)
    true
  }
}



/**
  * Singleton for constructor
  */
private[bertspark] final object RequestMessageGenerator {
  final val logger: Logger = LoggerFactory.getLogger("RequestMessageGenerator")


  /**
    * Constructor using S3 file a source
    * @param s3RequestsPath Path to S3 folder file
    * @param produceTopic Kafka topic for the producer
    * @param ingestIntervalMs Time interval between batches
    * @param sampleSize Maximum number of request per batch
    * @param numRepeats Number of batches
    * @return Instance of FeedbackMessageGenerator
    */
  def apply(
    s3RequestsPath: String,
    produceTopic: String,
    ingestIntervalMs: Long = 0L,
    sampleSize: Int = -1,
    numRepeats: Int = 1)(implicit sparkSession: SparkSession): RequestMessageGenerator = {
    import org.bertspark.config.MlopsConfiguration._
    import sparkSession.implicits._

    val rawRequestDS: Dataset[InternalRequest] = try {
      val requestDS = S3Util.s3ToDataset[InternalRequest](
        mlopsConfiguration.storageConfig.s3Bucket,
        s3RequestsPath,
        header = false,
        fileFormat = "json"
      )   .dropDuplicates("id")

      if(sampleSize > 0) requestDS.limit(sampleSize).persist() else requestDS.persist()
    }
    catch {
      case e: IllegalArgumentException =>
        MessageGenerator.logger.error(s"FeedbackMessageGenerator: ${e.getMessage}")
        sparkSession.emptyDataset[InternalRequest]
    }

    val numRawRequests = rawRequestDS.count()
    logDebug(MessageGenerator.logger, s"Request gen 1: loaded $numRawRequests requests")
    val validRequestDS = rawRequestDS.filter(
      internalRequest => subModelTaxonomy.isSupported(internalRequest.context.emrLabel)
    )
    val numValidRequests = validRequestDS.count()

    if(numValidRequests > 0) {
      logDebug(MessageGenerator.logger, s"Request gen 3: validated $numValidRequests requests")

      val requestDS =
        if (numValidRequests > 512) {
          val fraction = sampleSize.toDouble / numValidRequests
          if (fraction >= 1.0) validRequestDS
          else validRequestDS.sample(fraction)
        }
        else
          validRequestDS
      logDebug(MessageGenerator.logger, s"Request gen 3: selected ${requestDS.count()} requests")
      val pRequests = requestDS.map(PRequest(_)).collect()

      MessageGenerator.logger.info(s"${pRequests.length} were simulated through Kafka")
      rawRequestDS.unpersist()
      new RequestMessageGenerator(produceTopic, pRequests, ingestIntervalMs, numRepeats)
    }
    else {
      rawRequestDS.unpersist()
      throw new IllegalStateException(s"Not enough generated valid request from Kafka generator")
    }
  }


  def generate(requests: Seq[PRequest], numRepeats: Int): Unit = {
    import org.bertspark.config.MlopsConfiguration._

    val validRequests = requests.filter(
      internalRequest => subModelTaxonomy.isSupported(internalRequest.context.emrLabel)
    )
    if(validRequests.isEmpty)
      logger.warn(s"WARN: request with label ${requests.map(_.context.emrLabel).mkString("  ")} not supported")

    val produceTopic = mlopsConfiguration.runtimeConfig.requestTopic
    val generator =  new RequestMessageGenerator(produceTopic, requests, 200, numRepeats)
    generator.start()
    delay(8000L)
  }


  private def convertToRequestMessages(pRequests: Seq[PRequest]): Seq[RequestMessage] =
    if(pRequests.nonEmpty) {
      pRequests.map(
        req =>
          if(req.notes.nonEmpty) {
            val correctedNote = req.notes.head.replace("'", "''")
            req.copy(notes = Seq[String](correctedNote))
          }
          else
            PRequest()
      )   .filter(_.id.nonEmpty)
          .map(RequestMessage(_))
    }
    else {
      MessageGenerator.logger.warn("Message generator has no requests")
      Seq.empty[RequestMessage]
    }

}
