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
package org.bertspark.http


import java.io.{BufferedReader, InputStream, InputStreamReader, IOException}
import java.util.concurrent.Callable
import org.apache.http.StatusLine
import org.apache.http.client.ClientProtocolException
import org.apache.http.client.config.RequestConfig
import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.{CloseableHttpClient, HttpClientBuilder}
import org.slf4j.{Logger, LoggerFactory}
import scala.concurrent.TimeoutException


/**
  * Generic HTTP post client
  * @param content Content of the post
  * @param url Target URL
  * @param headers Header for the HTTP connectior
  *
  * @author Patrick Nicolas
  * @version 0.7
  */
private[bertspark] final class HttpPostClient(
  content: String,
  url: String,
  headers: Seq[(String, String)] = Seq.empty[(String, String)]) extends Callable[Option[(Int, String)]] {
  import HttpPostClient._

  def call(): Option[(Int, String)] = {
    var httpClient: CloseableHttpClient = null
    var httpPost: HttpPost = null

    try {
      // Let's build the request with loos
      val config = RequestConfig.custom()
          .setConnectTimeout(defaultIdleTimeoutMs)
          .setConnectionRequestTimeout(defaultIdleTimeoutMs)
          .setSocketTimeout(defaultIdleTimeoutMs)
          .build

      httpClient = HttpClientBuilder.create.setDefaultRequestConfig(config).build
      httpPost = new HttpPost(url)

      httpPost.setEntity(new StringEntity(content))
      httpPost.setHeader("Content-Type", "application/json")
      httpPost.setHeader("Accept", "application/json")

      // Add optional headers (i.e. Rule engine credentials, authentication,....)
      if (headers.nonEmpty)
        headers.foreach { case (name, value) => httpPost.setHeader(name, value) }

      // Extract the raw response
      val httpResponse = httpClient.execute(httpPost)
      val status: StatusLine = httpResponse.getStatusLine

      // Get status and error code if necessary
      val finalStatus =
        if (status.getStatusCode != OkStatusCode) {
          logger.error(s"Error code ${status.getStatusCode} for $url")
          status.getStatusCode
        }
        else
          OkStatusCode

      val response = getResponseContent(httpResponse.getEntity.getContent)
      response.map((finalStatus, _))
    }
    catch {
      case e: IllegalArgumentException =>
        logger.error(s"Illegal argument ${e.getMessage}")
        None
      case e: ClientProtocolException =>
        logger.error(s"Failed client protocol ${e.getMessage}")
        None
      case e: TimeoutException =>
        logger.error(s"Timed out ${e.getMessage}")
        None
      case e: IOException =>
        logger.info(s"I/O failure ${e.getMessage}")
        None
      case e: Exception =>
        logger.info(s"Undefined failure  ${e.getMessage}")
        None
    }
    finally {
      // Perform clean up with connection and the client code
      if (httpPost != null)
        httpPost.releaseConnection
      if (httpClient != null)
        httpClient.close
    }
  }

  @inline
  def getUrl: String = url
}


private[bertspark] final object HttpPostClient {
  final val logger: Logger = LoggerFactory.getLogger("HttpPostClient")
  final val defaultIdleTimeoutMs = 180000
  final val OkStatusCode: Int = 200

  /**
    * We close both the buffer reader and the input stream from the http connection.
    * @param inputStream
    * @return Optional content of the response
    */
  final private def getResponseContent(inputStream: InputStream): Option[String] = {
    var bufferedReader: BufferedReader = null

    try {
      bufferedReader = new BufferedReader(new InputStreamReader(inputStream))
      val buf = new StringBuilder
      var line: String = null

      do {
        line = bufferedReader.readLine
        if (line != null)
          buf.append(line)
      } while (line != null)
      Some(buf.toString)
    }
    catch {
      case e: IOException =>
        logger.error(e.getMessage)
        None
    }
    finally {
      try {
        inputStream.close
      }
      catch {
        case e: IOException => logger.error(e.getMessage)
      }
      if (bufferedReader != null) {
        try {
          bufferedReader.close
        }
        catch {
          case e: IOException => logger.error(e.getMessage)
        }
      }
    }
  }

}
