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

import java.util
import org.apache.kafka.common.serialization.{Deserializer, Serializer}
import org.bertspark.kafka.serde.AckSerDe.AckMessage
import org.bertspark.kafka.serde.SerDe.serDePrefix

/**
  * Serializer for acknowledgment message
  * @author Patrick Nicolas
  * @version 0.6
  */
final class AckSerializer extends Serializer[AckMessage] {
  override def serialize(topic: String, request: AckMessage): Array[Byte] = {
    val content = SerDe.write[AckMessage](request)
    content
  }

  protected override def close(): Unit = { }
  protected override def configure(configs: util.Map[String, _], isKey: Boolean): Unit = { }
}


/**
  * Deserializer for the acknowledgment
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] final class AckDeserializer extends Deserializer[AckMessage] {
  override def deserialize(topic: String, bytes: Array[Byte]): AckMessage =
    SerDe.read[AckMessage](bytes, classOf[AckMessage])

  protected override def close(): Unit = { }
  protected override def configure(configs: util.Map[String, _], isKey: Boolean): Unit = { }
}




private[bertspark] final object AckSerDe extends SerDe {
  override val serializingClass = s"$serDePrefix.AckSerializer"
  override val deserializingClass = s"$serDePrefix.AckDeserializer"

  /**
    * Wrapper for the acknowledgment message
    * @param key Key used in
    * @param timestamp Time stamp the request was created
    * @param payload Acknowledgment string
    */
  case class AckMessage(
    timestamp: Long,
    payload: String
  )  {
    override def toString: String = s"Timestamp: $timestamp\n${payload.toString}"
  }

  final object AckMessage {
    def apply(payload: String): AckMessage = AckMessage(System.currentTimeMillis(), payload)
  }
}
