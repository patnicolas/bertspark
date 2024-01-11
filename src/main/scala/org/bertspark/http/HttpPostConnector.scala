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

import java.io.{FileNotFoundException, IOException}
import java.util.concurrent.{ExecutionException, TimeUnit}
import org.apache.kafka.common.errors.TimeoutException
import org.slf4j.{Logger, LoggerFactory}


/**
  * Implementation of basic connectivity to a remote service using a REST API
  * @param httpPostClient Reference to the client to the post
  * @param timeout Connectivity time out
  *
  * @author Patrick Nicolas
  * @version 0.7
  */
private[bertspark] final class HttpPostConnector(httpPostClient: HttpPostClient, timeout: Long) {
  import HttpPostConnector._

  /**
    * Connect and format the content of a multi-part Post to a remote service
    * @return Optional response as a pair of {status, response content/payload}
    */
  final def post: Option[(Int, String)] =
    try {
      // Submit to executor queue for processing
      val f = executorServer.submit(httpPostClient)
      // Block on the future return
      Some(f.get(timeout, TimeUnit.SECONDS).getOrElse(500, ""))
    }
    catch {
      case e: FileNotFoundException =>
        logger.error(s"Incorrect configuration file posting to ${httpPostClient.getUrl} ${e.getMessage}")
        None
      case e: IOException =>
        logger.error(s"Could not processed output from ${httpPostClient.getUrl} ${e.getMessage}")
        None
      case e: InterruptedException =>
        logger.error(s"Connectivity to ${httpPostClient.getUrl} is interrupted ${e.getMessage}")
        None
      case e: ExecutionException =>
        logger.error(s"${httpPostClient.getUrl} failed execution ${e.getMessage}")
        None
      case e: TimeoutException =>
        logger.error(s"Connectivity to ${httpPostClient.getUrl} timed out ${e.getMessage}")
        None
      case e: Exception =>
        logger.error(s"Undefined exception for ${httpPostClient.getUrl} with ${e.getMessage}")
        None
    }

  @inline
  final def getUrl: String = httpPostClient.getUrl
}


private[bertspark] final object HttpPostConnector {
  import java.util.concurrent.{ExecutorService, Executors}
  final val logger: Logger = LoggerFactory.getLogger("HttpPostConnector")

  final private val DefaultThreadPoolSize = 64
  final lazy val executorServer: ExecutorService = Executors.newFixedThreadPool(DefaultThreadPoolSize)
}