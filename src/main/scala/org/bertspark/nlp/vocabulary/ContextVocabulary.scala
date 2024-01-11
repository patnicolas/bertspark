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
package org.bertspark.nlp.vocabulary

import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.util.io.LocalFileUtil
import org.bertspark.util.io.LocalFileUtil.CSV_SEPARATOR
import org.slf4j.{Logger, LoggerFactory}


/**
  * *
  */
private[vocabulary] final class ContextVocabulary extends VocabularyComponent {
  import ContextVocabulary._

  override val vocabularyName: String = "ContextVocabulary"

  override def build(tokens: Array[String], requestDS: Dataset[InternalRequest]): Array[String] = {
    val newTokens = build.getOrElse({
      logger.error(s"Failed to load CPT and modifiers file")
      Array.empty[String]
    })
    tokens ++ newTokens
  }

  private def build: Option[Array[String]] = load
}

/**
  * Vocabulary context for modifiers
  */
private[bertspark] final object ContextVocabulary {
  final private val logger: Logger = LoggerFactory.getLogger("ContextVocabulary")
  final private val indexedCSVFileName = "conf/codes/indexed.csv"

  lazy val modifiers: Set[String] =
    LocalFileUtil.Load.local(indexedCSVFileName).map(
      content => {
        import scala.collection.mutable.ListBuffer

        val lines = content.split("\n")
        lines.foldLeft(ListBuffer[String]())(
          (buf, line) => {
            val ar = line.split(CSV_SEPARATOR)
            val index = ar(1).toInt
            if(index >= 200000)
              buf.append(ar.head.trim)
            buf
          }
        ).toArray
      }
    )   .map(_.toSet)
        .getOrElse({
          logger.error(s"Failed to load modifiers from $indexedCSVFileName")
          Set.empty[String]
        })

  def apply(): ContextVocabulary = new ContextVocabulary

  def load: Option[Array[String]] =  {
    import org.bertspark.util.io.LocalFileUtil
    LocalFileUtil.Load.local(indexedCSVFileName).map(
      content => {
        import scala.collection.mutable.ListBuffer
        val lines = content.split("\n")
        lines.foldLeft(ListBuffer[String]())(
          (buf, line) => {
            val ar = line.split(CSV_SEPARATOR)
            val index = ar(1).toInt
            if(index >= 200000) {

              buf.append(s"${ar.head.trim}_mod")
            }
            else if(index >= 100000) buf.append(s"${ar.head.trim}_cpt")
            buf
          }
        ).toArray
      }
    )
  }
}
