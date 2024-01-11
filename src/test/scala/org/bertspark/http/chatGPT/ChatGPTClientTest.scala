package org.bertspark.http.chatGPT

import org.bertspark.http.chatGPT.ChatGPTClient.ChatGPTUserRequest
import org.bertspark.util.EncryptionUtil
import org.scalatest.flatspec.AnyFlatSpec


private[chatGPT] final class ChatGPTClientTest extends AnyFlatSpec {

  ignore should "Succeed encrypting key" in {
    val key = "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"
    val encryptedKey = EncryptionUtil(key)
    print(encryptedKey)
  }

  ignore should "Succeed generate a JSON representation of ChatGPT request" in {
    val chatCPTUserRequest = ChatGPTUserRequest("What is the purpose of psychology")
    println(chatCPTUserRequest.toJson)
  }

  it should "Succeed invoking ChatGPT" in {
    val prompt = "What is the color of the moon"
    val request = ChatGPTUserRequest(prompt)

    val chatGPTClient = ChatGPTClient()
    chatGPTClient(request).foreach(response => print(response.toString))
  }
}
