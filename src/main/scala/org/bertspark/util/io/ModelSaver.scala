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

import ai.djl.Model

/**
  * @author Patrick Nicolas
  * @version 0.6
  */
trait ModelSaver {
  protected[this] val subModelName: String

  protected def save(model: Model, epochNo: Int, metrics: List[String]): Unit = {}
}
