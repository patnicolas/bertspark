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
package org.bertspark.util.rdbms

import java.sql.SQLException
import org.apache.spark.sql.SparkSession
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.delay
import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalContext, InternalRequest, MlEMRCodes}
import org.bertspark.nlp.tokenSeparator
import org.bertspark.util.io.S3Util
import org.bertspark.util.rdbms.PredictionsTbl.orderedAllQuery
import org.slf4j.{Logger, LoggerFactory}

/**
 *
 * @param postgreSql
 * @param tableName
 * @author Patrick Nicolas
 * @version 0.5
 */
case class PredictionsTbl(postgreSql: PostgreSql, override val tableName: String) extends BaseTbl[PredictionsTbl] {

  override def count: Int = BaseTbl.queryCount(tableName, postgreSql)

  /**
   * The connection to the RDS data base has to be explicitly clodes
   */
  override  def close: Unit = postgreSql.close


  /**
   * Default query for the current table
   * @param limit Maximum number of fields to be retrieved
   * @return List of rows with each row is defined as a sequence of fields with String type
   */
  override def defaultQuery(limit: Int): Seq[Seq[String]] =
    PredictionsTbl.query(postgreSql, tableName, limit, orderedAllQuery, "")

  def defaultQuery(limit: Int, condition: String): Seq[Seq[String]] =
    PredictionsTbl.query(postgreSql, tableName, limit, orderedAllQuery, condition)

  def queryMaxId(limit: Int): Seq[Seq[String]] =
    PredictionsTbl.query(postgreSql, tableName, limit, orderedAllQuery, "")
}


/**
 * Singleton for flatten request and response tables
 * {{{
 *   THe default prediction table `defaultS3PredictionTable` should be overridden for testing...
 *   * defaultS3Prefix: Default folder in aideo-tech-outbound-encounters-prod bucket. The file for a given data source
 *       is defaultS3Prefix/$dataSource
 *   * defaultS3PredictionTable Name of the table to be created on RDS service
 *   * defaultNumPartitions Number of partitions to be used in processing
 *   * defaultSegmentSize  Size of segments to extract
 * }}}
 * @note Changes to be made for improving performance of insertion in the data base are marked with @todo
 */
private[bertspark] final object PredictionsTbl {
  private val logger: Logger = LoggerFactory.getLogger("PredictionTbl")

  final val orderedAllQuery = "id,encounter_id,date,version,age,gender,taxonomy,customer,client,modality,place_of_service,date_of_service,emr,provider,patient,metadata,auto_code_state,auto_coded,predicted_line_1,predicted_line_2,predicted_line_3,predicted_line_4,predicted_line_5,predicted_line_6,predicted_codes,note,timeStamp"
  final val orderedRequestQuery = "id,encounter_id,date,version,age,gender,taxonomy,customer,client,modality,place_of_service,date_of_service,emr,provider,patient,note"

  final val defaultTransform: Seq[String] => InternalRequest = (fields: Seq[String]) => {
    val id = {
      val encounter_fields = fields(1).split("_")
      if (encounter_fields.size == 4) s"${encounter_fields.head}_${encounter_fields(1)}_${encounter_fields(3)}" else ""
    }
    if (id.nonEmpty) {
      val age = fields(4).toInt
      val gender = fields(5)
      val taxonomy = fields(6)
      val customer = fields(7)
      val client = fields(8)
      val modality = fields(9)
      val pos = fields(10)
      val dos = fields(11)
      val emr = fields(12).trim
      val provider = fields(13)
      val patient = fields(14)
      val note = fields(25)
      val emrFields = emr.split(tokenSeparator)
      val emrCode = MlEMRCodes(0, emrFields.head, emrFields.tail, Seq.empty[String], 1, "UN")
      val internalContext = InternalContext(
        "",
        age,
        gender,
        taxonomy,
        customer,
        client,
        modality,
        pos,
        dos,
        Seq[MlEMRCodes](emrCode),
        provider,
        patient,
        "",
        "",
        "",
        ""
      )
      InternalRequest(id, internalContext, Seq[String](note))
    }
    else {
      logger.warn(s"Incorrect encounter id format ${fields(1)}")
      InternalRequest()
    }
  }


  def tblToS3Request(args: Seq[String]): Unit = {
    require(args.size == 5, s"Cmd line '${args.mkString(" ")}' should be 'rdbmsToS3 s3DestFolder customer date limit'")
    import org.bertspark.implicits._

    val postgreSql = PostgreSql()
    val s3DestFolder = args(1)
    val condition = s"customer='${args(2)}' AND DATE >'${args(3)}'"
    val limit = args(4).toInt

    PredictionsTbl.tblToS3Request(
      postgreSql,
      orderedAllQuery,
      condition,
      s3DestFolder,
      defaultTransform,
      limit
    )
    delay(5000L)
    postgreSql.close
  }


  def tblToS3Request(
    postgreSql: PostgreSql,
    fields: String,
    condition: String,
    s3Folder: String,
    transform: Seq[String] => InternalRequest)(implicit sparkSession: SparkSession): Unit =
    tblToS3Request(postgreSql, fields, condition, s3Folder, transform)

  def tblToS3Request(
    postgreSql: PostgreSql,
    fields: String,
    condition: String,
    s3Folder: String)(implicit sparkSession: SparkSession): Unit =
    tblToS3Request(postgreSql, fields, condition, s3Folder, defaultTransform)

  def tblToS3Request(
    postgreSql: PostgreSql,
    condition: String,
    s3Folder: String)(implicit sparkSession: SparkSession): Unit =
    tblToS3Request(postgreSql, orderedRequestQuery, condition, s3Folder, defaultTransform)

  def tblToS3Request(
    postgreSql: PostgreSql,
    fields: String,
    condition: String,
    s3Folder: String,
    transform: Seq[String] => InternalRequest,
    limit: Int = -1)(implicit sparkSession: SparkSession): Unit = try {
    import sparkSession.implicits._
/*
    val step = 80000
    var nextId = 4249629
    var prevId = nextId - step

    while(prevId > 1000000) {

 */
       val queryResults = query(postgreSql, "orchestrator_predictions", limit, fields, condition)
       val requestDS = queryResults.map(transform(_)).filter(_.id.nonEmpty).toDS()
      logDebug(logger, s"Number of requests: ${requestDS.count()}")

       S3Util.datasetToS3[InternalRequest](
         requestDS,
         s3Folder,
         false,
         "json",
         true,
         2
       )
   //   nextId = prevId
    //  prevId = nextId - step
  //  }
  } catch {
    case e: IllegalArgumentException => logger.error(e.getMessage)
  }

  /**
    * Generic query statement
    *
    * @param tableName Name of the table
    * @param limit     Limit number of records to extract
    * @param fields    List of fields, comma separated
    * @param condition WHERE condition statement
    * @return Query string
    */
  def queryStatement(tableName: String, limit: Int, fields: String, condition: String): String =
    if (condition.isEmpty)
      if (limit > 0) s"SELECT $fields FROM $tableName limit $limit"
      else s"SELECT $fields FROM $tableName"
    else if (limit > 0) s"SELECT $fields FROM $tableName WHERE $condition limit $limit"
    else s"SELECT $fields FROM $tableName WHERE $condition"


  /**
    * Query all the fields of this given tables
    * @param postgreSql          reference to the current PostgreSQL client
    * @param predictionTableName Name of the  prediction table
    * @param limit               Limit of the query
    * @param fields              List of column or fields to be retrieved
    * @param condition           WHERE condition for this query as 'WHERE $condition'
    * @return Sequence of rows as list of string
    */
  @throws(clazz = classOf[IllegalArgumentException])
  def query(
    postgreSql: PostgreSql,
    predictionTableName: String,
    limit: Int,
    fields: String = orderedAllQuery,
    condition: String = ""): Seq[Seq[String]] = try {
    import scala.collection.mutable.ListBuffer

    val defaultQueryStmt = queryStatement(predictionTableName, limit, fields, condition)
    logDebug(logger, defaultQueryStmt)

    postgreSql.executeQuery(defaultQueryStmt) match {
      case Right(rs) =>
        val predictionsResults = ListBuffer[Seq[String]]()
        while (rs.next) {
          val record = Seq[String](
            rs.getString(1),
            rs.getString(2),
            rs.getString(3),
            rs.getString(4),
            rs.getInt(5).toString,
            rs.getString(6),
            rs.getString(7),
            rs.getString(8),
            rs.getString(9),
            rs.getString(10),
            rs.getString(11),
            rs.getString(12),
            rs.getString(13),
            rs.getString(14),
            rs.getString(15),
            rs.getString(16),
            rs.getString(17),
            rs.getBoolean(18).toString,
            rs.getString(19),
            rs.getString(20),
            rs.getString(21),
            rs.getString(22),
            rs.getString(23),
            rs.getString(24),
            rs.getString(25),
            rs.getString(26),
            rs.getLong(27).toString
          )
          predictionsResults.append(record)
        }
        predictionsResults
      case Left(errorMsg) =>
        throw new IllegalStateException(s"Query error: $errorMsg")
    }
  }
  catch {
    case e: SQLException =>
      throw new IllegalStateException(s"SQL exception: ${e.getMessage}")
    case e: Exception =>
      throw new IllegalStateException(s"Undefined exception: ${e.getMessage}")
  }

  /**
    * Query all the fields of this given tables
    *
    * @param postgreSql          reference to the current PostgreSQL client
    * @param predictionTableName Name of the  prediction table
    * @return Sequence of rows as list of string
    */
  def query(postgreSql: PostgreSql, predictionTableName: String): Seq[Seq[String]] =
    query(postgreSql, predictionTableName, -1, orderedAllQuery, "")
}


