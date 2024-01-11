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

import java.io.File
import org.apache.spark.sql.{Dataset, Encoder, SparkSession}
import org.apache.spark.SparkException
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.delay
import org.bertspark.nlp.medical.MedicalCodingTypes.{FeedbackLineItem, InternalContext, InternalFeedback, MlEMRCodes}
import org.bertspark.nlp.tokenSeparator
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer

/**
 *
 * @param s3Bucket
 * @param s3Folder
 * @param numPartitions
 * @param sparkSession
 * @param encoder
 * @tparam T Type of element (entry, row, record... ) in the collection to be stored
 */
private[bertspark] final class S3IOOps[T](
  s3Bucket: String,
  s3Folder: String,
  numPartitions: Int = 1)(implicit sparkSession: SparkSession, encoder: Encoder[T]) extends IOOps[T] {
  import S3IOOps._

  override def save(data: Array[T]): Boolean = try {
    require(data.nonEmpty, s"Cannot save undefined data")
    import sparkSession.implicits._

    S3Util.datasetToS3[T](
      s3Bucket,
      data.toSeq.toDS(),
      s3Folder,
      header = false,
      fileFormat = "json",
      toAppend = false,
      numPartitions
    )
    true
  }
  catch {
    case e: SparkException =>
      println(s"Spark exception saving s3://${s3Bucket}/${s3Folder}")
      false
    case e: Exception =>
      logger.error(s"Failed saving to s3://${s3Bucket}/${s3Folder}")
      false
  }

  /**
   * load the data set from S3
   * @return Data set
   */
  override def loadDS: Dataset[T] =
    S3Util.s3ToDataset[T](s3Bucket, s3Folder, header = false, fileFormat = "json")

  /**
   * load the data from S3
   * @return Data collection
   */
  override def load: Array[T] = loadDS.collect()
}



private[bertspark] final object S3IOOps {
  final private val logger: Logger = LoggerFactory.getLogger("S3IOOps")
  import org.bertspark.implicits._
  import sparkSession.implicits._

  val clientIdentifier = 0
  val practiceIdentifier = 1
  val patientId = 2
  val locationId = 3
  val orderNumber = 4
  val visitNumber = 5
  val procedureCode = 6
  val modifiers = 7
  val units = 8
  val placeOfServiceCode = 9
  val icd10Diagnosis1 = 10
  val icd10Diagnosis2 = 11
  val icd10Diagnosis3 = 12
  val icd10Diagnosis4 = 13
  val icd10Diagnosis5 = 14

  def s3ToS3Feedback(args: Seq[String]): Unit = {
    require(
      args.size == 6,
      s"Cmd line, ${args.mkString(" ")} should be 's3ToS3Feedback srcBucket prefix destFolder suffix limit'"
    )
    val s3SrcBucket = args(1)
    val prefix = args(2)
    val s3DestFolder = args(3)
    val suffix = args(4)
    val limit = args(5).toInt

    s3ToS3Feedback(s3SrcBucket, prefix, s3DestFolder, suffix, limit)
  }

  def s3ToS3Feedback(
    s3SrcBucket: String,
    prefix: String,
    s3DestFolder: String,
    suffix: String,
    limit: Int): Unit = {
    val keys = S3Util.getS3Keys(s3SrcBucket, prefix).filter(_.endsWith(suffix)).take(limit)
    val step = 16
    logDebug(logger, s"${keys.size} S3 keys with step: $step")

    (0 until keys.size by step).foreach(
      index => {
        val lastIndex = if(index+step > keys.size) keys.size else index+step
        val keySlice = keys.slice(index, lastIndex).toSeq
        processSlice(s3SrcBucket, s3DestFolder, keySlice)
      }
    )
  }


  def s3ToFs[T](
    bucketName: String,
    s3Folder: String,
    destination: String
  )(implicit sparkSession: SparkSession, encoder: Encoder[T]): Int = {

    val ds = S3Util.s3ToDataset[T](
      bucketName,
      s3Folder,
      false,
      "json"
    )
    val absoluteDirName = s"mlops/input/$destination"
    val newFile = new File(absoluteDirName)
    if(!newFile.exists())
      newFile.mkdir()

    val iterator = ds.toLocalIterator()
    val collector = ListBuffer[String]()
    var count = 0
    while(iterator.hasNext) {
      val field = iterator.next()
      val fieldJson = LocalFileUtil.Json.mapper.writeValueAsString(field)
      collector.append(fieldJson)
      if(collector.size > 512) {
        LocalFileUtil.Save.local(s"/Users/patricknicolas/$absoluteDirName/part-$count.json", collector.mkString("\n"))
        count += 1
        collector.clear
        delay(1000L)
      }
    }
    if(collector.nonEmpty) {
      LocalFileUtil.Save.local(s"/Users/patricknicolas/$absoluteDirName/part-$count.json", collector.mkString("\n"))
      count += 1
    }

    count*10
  }


  // ------------------  Supporting methods --------------------------------------

  private def processSlice(s3SrcBucket: String, s3DestFolder: String, keys: Seq[String]): Unit = {
    val internalFeedbacks: Seq[InternalFeedback] = keys.flatMap(
      key => {
        S3Util.download(s3SrcBucket, key).map(
          content => {
            val rows = content.split("\n")
            val pairs = rows.tail.map(
              row => {
                val fields = row.split(",")
                val key = s"${fields(patientId)}_${fields(visitNumber)}_${fields(orderNumber)}".replace("\"", "")
                val cpt = fields(procedureCode).trim
                val mods = fields(modifiers).trim
                val lineItemStr = s"${cpt},${mods},${fields(icd10Diagnosis1)},${fields(icd10Diagnosis2)},${fields(icd10Diagnosis3)}"
                val medicalCodes: Array[String] = {
                  try {
                    lineItemStr
                        .split(",")
                        .map(lineItem => lineItem.substring(1, lineItem.length - 1))
                  }
                  catch {
                    case e: StringIndexOutOfBoundsException =>
                      logger.error(s"S3ToS3Feedback ${e.getMessage}")
                      Array.empty[String]
                  }
                }

                val cptCode = if(cpt.nonEmpty) cpt.substring(1, cpt.length-1) else cpt
                val modCodes = if(mods.nonEmpty) mods.substring(1, mods.length-1) else mods
                (key, cptCode, modCodes, medicalCodes)
              }
            ).filter(_._4.nonEmpty)

            pairs.groupBy(_._1).map{
              case (key, ar) => {
                val lineItems = ar.map{
                  case (_, _, _, codes) => {
                    if(codes(4).nonEmpty)
                      FeedbackLineItem(0, codes.head, codes(1).split(tokenSeparator), Seq[String](codes(2), codes(3), codes(4)), 1, "UN", 0.0)
                    else if(codes(3).nonEmpty)
                      FeedbackLineItem(0, codes.head, codes(1).split(tokenSeparator), Seq[String](codes(2), codes(3)), 1, "UN", 0.0)
                    else if(codes(2).nonEmpty)
                      FeedbackLineItem(0, codes.head, codes(1).split(tokenSeparator), Seq[String](codes(2)), 1, "UN", 0.0)
                    else
                      FeedbackLineItem(0, codes.head, codes(1).split(tokenSeparator), Seq.empty[String], 1, "UN", 0.0)
                  }
                }

                val emrCode = MlEMRCodes(0, ar.head._2, ar.head._3.split(tokenSeparator), Seq.empty[String], 1, "UN")
                InternalFeedback(key, Seq[MlEMRCodes](emrCode), lineItems)
              }
            }.toSeq
          }
        ).getOrElse(throw new IllegalStateException(s"Failed to download from $s3SrcBucket"))
      }
    )

    try {
      S3Util.datasetToS3[InternalFeedback](
        internalFeedbacks.toDS(),
        s3DestFolder,
        header = false,
        fileFormat = "json",
        toAppend = true,
        numPartitions = 2
      )
    }
    catch {
      case e: IllegalStateException => logger.error(e.getMessage)
    }
  }
}

