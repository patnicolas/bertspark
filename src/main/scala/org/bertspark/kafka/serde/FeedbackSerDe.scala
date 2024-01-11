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
import org.bertspark.kafka.serde.FeedbackSerDe.FeedbackMessage
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalFeedback
import java.util
import org.bertspark.kafka.serde.SerDe.serDePrefix


/**
  * Serializer for the Feedback request
  * @see org.mlops.nlp.medical.MedicalCodeTypes
  * @author Patrick Nicolas
  * @version 0.6
  */
final class FeedbackSerializer extends Serializer[FeedbackMessage] {
  override def serialize(topic: String, request: FeedbackMessage): Array[Byte] = {
    val content = SerDe.write[FeedbackMessage](request)
    content
  }

  protected override def close(): Unit = { }

  protected override def configure(configs: util.Map[String, _], isKey: Boolean): Unit = { }
}


/**
  * Deserializer for the feedback request
  * @see org.mlops.nlp.medical.MedicalCodeTypes
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] final class FeedbackDeserializer extends Deserializer[FeedbackMessage] {

  override def deserialize(topic: String, bytes: Array[Byte]): FeedbackMessage =
    SerDe.read[FeedbackMessage](bytes, classOf[FeedbackMessage])

  protected override def close(): Unit = { }

  protected override def configure(configs: util.Map[String, _], isKey: Boolean): Unit = { }
}




private[bertspark] final object FeedbackSerDe extends SerDe {
  override val serializingClass = s"$serDePrefix.FeedbackSerializer"
  override val deserializingClass = s"$serDePrefix.FeedbackDeserializer"

  /**
    * Wrapper for the prediction request
    * @param key Key used in
    * @param timestamp Time stamp the request was created
    * @param payload Prediction request
    */
  case class FeedbackMessage(
    timestamp: Long,
    payload: InternalFeedback
  )  {
    override def toString: String = s"Timestamp: $timestamp\n${payload.toString}"
  }

  final object FeedbackMessage {
    def apply(payload: InternalFeedback): FeedbackMessage = FeedbackMessage(System.currentTimeMillis(), payload)
  }
}
