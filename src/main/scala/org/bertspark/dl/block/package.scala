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
package org.bertspark.dl


/** Variable and type shared across all deep learning blocks
 * - Convolution block (CNN)
 * - Deconvolution block (CNN)
 * - Variational block (AutoEncoder)
 * - Multi-layer perceptron block (MLP)
 * - Restricted Boltzmann model block (RBM)
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
package object block {
  import ai.djl.ndarray.types.Shape

  type BaseHiddenLayer = (Int, String)

  final val no2dShape = new Shape(-1, -1)
  final val no3dShape = new Shape(-1, -1, -1)

  final def isValidShape(shape: Shape): Boolean = shape.head > -1
}
