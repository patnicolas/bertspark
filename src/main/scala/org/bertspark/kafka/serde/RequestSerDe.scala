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
package org.bertspark.kafka.serde

import org.apache.kafka.common.serialization.{Deserializer, Serializer}
import org.bertspark.kafka.serde.RequestSerDe.RequestMessage
import java.util
import org.bertspark.kafka.serde.SerDe.serDePrefix
import org.bertspark.nlp.medical.MedicalCodingTypes.PRequest



/**
  * Serializer for the prediction request
  * @see org.mlops.nlp.medical.MedicalCodeTypes
  * @author Patrick Nicolas
  * @version 0.5
  */
final class RequestSerializer extends Serializer[RequestMessage] {
  override def serialize(topic: String, request: RequestMessage): Array[Byte] = {
    val content = SerDe.write[RequestMessage](request)
    content
  }

  protected override def close(): Unit = { }

  protected override def configure(configs: util.Map[String, _], isKey: Boolean): Unit = { }
}


/**
  * Deserializer for the prediction request
  * @see org.mlops.nlp.medical.MedicalCodeTypes
  * @author Patrick Nicolas
  * @version 0.5
  */
private[bertspark] final class RequestDeserializer extends Deserializer[RequestMessage] {

  override def deserialize(topic: String, bytes: Array[Byte]): RequestMessage =
    SerDe.read[RequestMessage](bytes, classOf[RequestMessage])

  protected override def close(): Unit = { }

  protected override def configure(configs: util.Map[String, _], isKey: Boolean): Unit = { }
}




/**
  * Singleton for serializing and deserializing classes for prediction response
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] final object RequestSerDe extends SerDe {
  override val serializingClass = s"$serDePrefix.RequestSerializer"
  override val deserializingClass = s"$serDePrefix.RequestDeserializer"


  /**
    * Wrapper for the prediction request
    * @param key Key used in
    * @param timestamp Time stamp the request was created
    * @param mlRequestPayload Prediction request
    */
  case class RequestMessage(
    timestamp: Long,
    mlRequestPayload: PRequest
  )  {
    override def toString: String = s"Timestamp: $timestamp\n${mlRequestPayload.toString}"
  }

  final object RequestMessage {
    def apply(requestPayload: PRequest): RequestMessage = RequestMessage(System.currentTimeMillis(), requestPayload)
  }
}