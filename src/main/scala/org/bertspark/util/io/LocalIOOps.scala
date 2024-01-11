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
 * and limitations under the License.
 */
package org.bertspark.util.io

import java.io.{FileNotFoundException, IOException}
import org.apache.spark.sql.{Dataset, Encoder, Encoders, SparkSession}
import org.slf4j.{Logger, LoggerFactory}
import scala.reflect.ClassTag


/**
 * Managing I/O operations to store collection into a  local file system
 * @param filename Name of file to contains colleciotn
 * @param project Convert a element type to a string (JSON, CSV,...)
 * @param instantiate Instantiate a element from a string representation
 * @param classTag$T$0 Class tag
 * @tparam T Type of element (entry, row, record... ) in the collection to be stored
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class LocalIOOps[T: ClassTag](
  filename: String,
  project: T => String,
  instantiate: String => T) extends IOOps[T]{
  import LocalIOOps._

  /**
   * Save the array into a local file
   * @param data Data collection
   * @return Returns true if successful, false otherwise
   */
 override  def save(data: Array[T]): Boolean = try {
    require(data.nonEmpty, "Cannot save undefined data")

    LocalFileUtil.Save.local(filename, data.map(project(_)).mkString("\n"))
  } catch {
    case e: FileNotFoundException =>
      logger.error(e.getMessage)
      false
    case e: IOException =>
      logger.error(e.getMessage)
      false
    case e: Exception =>
      logger.error(e.getMessage)
      false
  }

  /**
   * load the data from the local storage
   * @return Data collection
   */
  override def load: Array[T] =
    LocalFileUtil.Load.local(filename, (str: String) => str).map(_.map(instantiate(_))).getOrElse(Array.empty[T])

  /**
   * load the data set from local file(s)
   * @return Data set
   */
  override def loadDS: Dataset[T] = {
    import org.bertspark.implicits._
    implicit val encoder: Encoder[T] = null
    val results: Seq[T] = load.toSeq
    sparkSession.createDataset(results)
  }
}


private[bertspark] final object LocalIOOps {
  final private val logger: Logger = LoggerFactory.getLogger("LocalIOOps")
}


