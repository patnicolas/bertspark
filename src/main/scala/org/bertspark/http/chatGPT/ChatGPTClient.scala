package org.bertspark.http.chatGPT

import com.fasterxml.jackson.databind.json.JsonMapper
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import java.io.{BufferedReader, InputStreamReader, OutputStream}
import java.net.{HttpURLConnection, URL}
import java.nio.charset.StandardCharsets
import org.bertspark.util.EncryptionUtil
import org.slf4j.{Logger, LoggerFactory}


private[bertspark] final class ChatGPTClient(
  url: String,
  timeout: Int,
  encryptedAPIKey: String
) {
  import ChatGPTClient._


  def apply(chatGPTRequest: ChatGPTRequest): Option[ChatGPTResponse] = {
    var outputStream: Option[OutputStream] = None

    try {
      EncryptionUtil.unapply(encryptedAPIKey).map(
        apiKey => {
          // Create and initialize the HTTP connection
          val connection = new URL(url).openConnection.asInstanceOf[HttpURLConnection]
          connection.setRequestMethod("POST")
          connection.setRequestProperty("Content-Type", "application/json")
          connection.setRequestProperty("Accept", "application/json")
          connection.setRequestProperty("Authorization", s"Bearer $apiKey")
          connection.setConnectTimeout(timeout)
          connection.setDoOutput(true)

          // Write into the connection output stream
          outputStream = Some(connection.getOutputStream)
          outputStream.foreach(_.write(chatGPTRequest.toJsonBytes))

          // If request failed....
          if(connection.getResponseCode != HttpURLConnection.HTTP_OK)
            throw new IllegalStateException(s"Request failed with HTTP code ${connection.getResponseCode}")

          // Retrieve the JSON string from the connection input stream
          val response = new BufferedReader(new InputStreamReader(connection.getInputStream))
                .lines
                .reduce(_ + _)
                .get
            // Instantiate
          ChatGPTResponse.fromJson(response)
        }
      )
    }
    catch {
      case e: java.io.IOException =>
        logger.error(e.getMessage)
        None
      case e: IllegalStateException =>
        logger.error(e.getMessage)
        None
    }
    finally {
      outputStream.foreach(
        os => {
          os.flush
          os.close
        }
      )
    }
  }
}

private[bertspark] final object ChatGPTClient {
  final val logger: Logger = LoggerFactory.getLogger("ChatGPTClient")
  final private val defaultChatGPTUrl = "https://api.openai.com/v1/chat/completions"
  final private val defaultTimeOutMs = 10000
  final private val defaultEncryptedKey =
    "3E15/zCLlPOtLraltp19BKwOVBrpGBshqa03YYRJOeNYHcKvePMqwSijX6IaUv91vQItA9LXKrmRptDSombSPg=="


  def apply(timeout: Int, encryptedAPIKey: String): ChatGPTClient =
    new ChatGPTClient(defaultChatGPTUrl, timeout, encryptedAPIKey)

  def apply(): ChatGPTClient = new ChatGPTClient(defaultChatGPTUrl, defaultTimeOutMs, defaultEncryptedKey)


  trait ChatGPTRequest {
    import ChatGPTRequest._
    def toJson: String = mapper.writeValueAsString(this)
    def toJsonBytes: Array[Byte] = {
      val rawOutput = toJson
      rawOutput.getBytes(StandardCharsets.UTF_8)
    }
  }

  object ChatGPTRequest {
    // Instantiate a singleton for the Jackson serializer/deserializer
    val mapper = JsonMapper.builder().addModule(DefaultScalaModule).build()
    mapper.registerModule(DefaultScalaModule)
    mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
  }

  /**
    * Basic requests
    * @param model
    * @param role
    * @param content
    */
  case class ChatGPTMessage(role: String, content: String) {
    override def toString: String = s"Role: $role\n$content"
  }

  case class ChatGPTUserRequest(model: String, messages: Seq[ChatGPTMessage]) extends ChatGPTRequest

  object ChatGPTUserRequest {
    def apply(content: String): ChatGPTUserRequest =
      ChatGPTUserRequest("gpt-3.5-turbo", Seq[ChatGPTMessage](ChatGPTMessage("user", content)))
  }

  /**
    * Comprehensive request
    * @param model
    * @param user
    * @param prompt
    * @param temperature
    * @param max_tokens
    * @param top_p
    * @param n
    * @param presence_penalty
    * @param frequency_penalty
    */
  case class ChatGPTDevRequest(
    model: String,
    user: String,
    prompt: String,
    temperature: Double,
    max_tokens: Int,
    top_p: Int = 1,
    n: Int = 1,
    presence_penalty: Int = 0,
    frequency_penalty: Int = 1) extends ChatGPTRequest


  case class ChatGPTChoice(message: ChatGPTMessage, index: Int, finish_reason: String) {
    override def toString: String = s"Finish reason:$finish_reason\n${message.toString}"
  }

  case class ChatGPTUsage(prompt_tokens: Int, completion_tokens: Int, total_tokens: Int) {
    override def toString: String = s"$prompt_tokens  prompt tokens, $completion_tokens completion tokens"
  }

  case class ChatGPTResponse(
    id: String,
    `object`: String,
    created: Long,
    model: String,
    choices: Seq[ChatGPTChoice],
    usage: ChatGPTUsage
  ) {
    override def toString: String = s"Usage: ${usage.toString}\nChoice: ${choices.head.toString}"

  }

  object ChatGPTResponse {
    // Instantiate the JSON response
    def fromJson(content: String): ChatGPTResponse = {
      import ChatGPTRequest._
      mapper.readValue[ChatGPTResponse](content, classOf[ChatGPTResponse])
    }
  }
}
