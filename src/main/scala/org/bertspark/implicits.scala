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

import ai.djl.util.{Pair, PairList}
import org.apache.spark.sql.SparkSession
import org.bertspark.config.SparkConfiguration.confToSessionFromFile
import org.slf4j.{Logger, LoggerFactory}
import scala.jdk.CollectionConverters._


/**
 * Singleton wrapper for implicit conversion
 * @author Patrick Nicolas
 * @version 0.2
 */
private[bertspark] final object implicits {
  import scala.language.implicitConversions
  final private val logger: Logger = LoggerFactory.getLogger("implicits")

  // implicit val system = ActorSystem("MLOps-Service")
  implicit val sparkSession: SparkSession = confToSessionFromFile
  def close: Unit = {
    logger.info("Closing Spark session!")
    sparkSession.close()
  //  system.terminate()
  }

  // ----------- Java type to Scala type collection conversion ---------------

  implicit def arrayOfArray(input: java.util.ArrayList[java.util.ArrayList[java.lang.Double]]): Seq[Array[Double]] =
    input.asScala.map(_.asScala.toArray.map(_.toDouble))

  implicit def arrayOfDouble(input: java.util.ArrayList[java.lang.Double]): Seq[Double] =
    input.asScala.map(_.toDouble).toSeq

  implicit def collectionOf[T](input: java.util.Collection[T]): Seq[T]= input.asScala.toSeq

  implicit def listOf[T](input: java.util.List[T]): List[T]= input.asScala.toList

  implicit def arrayOf[T](input: java.util.ArrayList[T]): Seq[T] = input.asScala.toSeq

  implicit def map2Scala[T](input: java.util.Map[String, T]): scala.collection.mutable.Map[String, T] =
    input.asScala.map { case (key, value) => (key, value) }

  implicit def arrayOfString(input: java.util.ArrayList[java.lang.String]): Seq[String] = input.asScala

  implicit def collection2ScalaStr(input: java.util.Collection[java.lang.String]): Seq[String] = input.asScala.toSeq

  implicit def collection2Scala[T](input: java.util.Collection[T]): Iterable[T] = input.asScala

  implicit def set2Scala[T](input: java.util.Set[T]): scala.collection.Set[T] = input.asScala


  // ----------- Scala collections type to Java type collection conversion -----------

  implicit def scalaMap2JavaKey[K, T](input: scala.collection.Map[K, T]): java.util.Map[K, T] = input.asJava

  implicit def scalaMap2Java[T](input: scala.collection.Map[String, T]): java.util.Map[String, T] = input.asJava

  implicit def scalaArray2Java[T](input: Array[T]): java.util.List[T] = input.toList.asJava

  implicit def scalaSeq2JavaList[T](input: Seq[T]): java.util.List[T] = input.toList.asJava

  implicit def scalaSeq2JavaList[T](input: scala.collection.mutable.ListBuffer[T]): java.util.List[T] = input.asJava

  implicit def scalaSeq2JavaList[T](input: scala.List[T]): java.util.List[T] = input.asJava

  implicit def seqToPairs[T](input: Seq[(T, AnyRef)]): PairList[T, java.lang.Object] = {
    val converted: java.util.List[Pair[T, java.lang.Object]] =
      scalaSeq2JavaList[Pair[T, java.lang.Object]](input.map{ case (t, anyRef) => new Pair(t, anyRef)})
    new PairList[T, java.lang.Object](converted)
  }
}
