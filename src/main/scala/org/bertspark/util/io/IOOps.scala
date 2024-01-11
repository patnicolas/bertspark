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

import org.apache.spark.sql._

/**
 * Generic operators for I/O operations for storing collections
 * @tparam T Type of element (entry, row, record... ) in the collection to be stored
 * @author Patrick Nicolas
 * @version 0.1
 */
trait IOOps[T] {
self =>
  /**
   * Save data into the appropriate storage
   * @param data data collection
   * @return Returns true if successful, false otherwise
   */
  def save(data: Array[T]): Boolean

  /**
   * load the data from the appropriate storage
   * @return Data collection
   */
  def load: Array[T]

  /**
   * load the data set from the appropriate storage
   * @return Data set
   */
  def loadDS: Dataset[T]
}