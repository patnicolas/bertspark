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
import java.util
import org.bertspark.kafka.serde.ResponseSerDe.ResponseMessage
import org.bertspark.kafka.serde.SerDe.serDePrefix
import org.bertspark.nlp.medical.MedicalCodingTypes.PResponse


/**
  * Serializer for the prediction response
  * @see org.mlops.nlp.medical.MedicalCodeTypes
  * @author Patrick Nicolas
  * @version 0.5
  */
private[bertspark] final class ResponseSerializer extends Serializer[ResponseMessage] {
  override def serialize(topic: String, predictResp: ResponseMessage): Array[Byte] =
    SerDe.write[ResponseMessage](predictResp)

  protected override def close(): Unit = {}
  protected override def configure(configs: util.Map[String, _], isKey: Boolean): Unit = { }
}


/**
  * Deserializer for the prediction response
  * @see org.mlops.nlp.medical.MedicalCodeTypes
  * @author Patrick Nicolas
  * @version 0.5
  */
private[bertspark] final class ResponseDeserializer extends Deserializer[ResponseMessage] {
  override def deserialize(topic: String, bytes: Array[Byte]): ResponseMessage =
    SerDe.read[ResponseMessage](bytes, classOf[ResponseMessage])

  protected override def close(): Unit = {}
  protected override def configure(configs: util.Map[String, _], isKey: Boolean): Unit = { }
}


/**
  * Singleton for serializing and deserializing classes for prediction response
  * @author Patrick Nicolas
  * @version 0.5
  */
private[bertspark] final object ResponseSerDe extends SerDe {
  override val serializingClass = s"${serDePrefix}.ResponseSerializer"
  override val deserializingClass = s"${serDePrefix}.ResponseDeserializer"

  /**
    * Message wrapping the prediction response
    * @param timestamp Time stamp the response was created
    * @param status HTTP status
    * @param error HTTP error descripiton
    * @param errorMessage Error message
    * @param mlResponsePayload Prediction response
    */
  case class ResponseMessage(
    timestamp: Long,
    status: Int,
    error: String,
    errorMessage: String,
    mlResponsePayload: PResponse
  )


  final object ResponseMessage {
    def apply(
      status: Int,
      error: String,
      errorMessage: String,
      mlResponsePayload: PResponse): ResponseMessage =
      ResponseMessage(System.currentTimeMillis(), status, error, errorMessage, mlResponsePayload)

    def apply(
      mlResponsePayload: PResponse): ResponseMessage =
      ResponseMessage(System.currentTimeMillis(), 200, "OK", "", mlResponsePayload)
  }
}
