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
package org.bertspark

import org.bertspark.util.io.LocalFileUtil


/**
 *
 * Classes and methods related to Natural Language Processing such as:
 * - Tokenization
 * - Document segmentation
 * - Cleansing training set
 * - Domain vocabulary
 * - Categorization of contextual variables
 * - Abbreviations
 * - ...
 *
 * @author Patrick Nicolas
 * @version 0.3
 */
package object nlp {
  final val tokenSeparator = "\\s+"
/*
  final private val fsStemsFilename = s"conf/codes/stems.csv"
  lazy val stemsMap: Map[String, String] = LocalFileUtil
      .Load
      .local(fsStemsFilename, (s: String) => s)
      .map(
        lines => {
          lines.map(
            line => {
              val fields = line.split(",")
              (fields.head, fields(1))
            }
          ).toMap
        }
      ).getOrElse({
        println(s"ERROR failed to load stems map from $fsStemsFilename")
        Map.empty[String, String]
      })

 */


}
