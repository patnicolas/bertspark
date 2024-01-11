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

import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer


/**
 *
 * @tparam T Actual type of the table
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] trait BaseTbl[T <: BaseTbl[T]] {
  protected[this] val tableName: String

  def count: Int
  def defaultQuery(limit: Int): Seq[Seq[String]]
  def close: Unit
}



private[bertspark] final object BaseTbl {
  private val logger: Logger = LoggerFactory.getLogger("BaseTbl")


  def queryCount(tableName: String, postgreSql: PostgreSql, distinctColumn: String = ""): Int = {
    val sqlQuery =
      if (distinctColumn.nonEmpty)
        s"SELECT count(DISTINCT $distinctColumn) from $tableName"
      else
        s"SELECT count(encounter_id) from $tableName"

    postgreSql.executeQuery(sqlQuery) match {
      case Right(rs) =>
        var count = 0
        while (rs.next) {
          count += rs.getInt(1)
        }
        count
      case Left(errorMsg) =>
        logger.error(errorMsg)
        -1
    }
  }

  /**
   * Test if the connectivity to the RDS/PostgreSQL data base has been established and the targeted
   * table has been created
   * @param postgreSql Reference to the wrapper for the PostgreSQL database
   * @param createElement Script to create the target table
   * @return true if connection and table creation succeeds
   */
  def isReady(postgreSql: PostgreSql, createElement: String): Boolean = {
    val connected = postgreSql.isConnected
    if (connected) postgreSql.executeUpdate(createElement).isRight else false
  }

  def createIndexStmt(colName: String, tableName: String, prefix: String): String = {
    val indexName = s"${colName}_${prefix}_index"
    s"""CREATE INDEX $indexName ON $tableName($colName)"""
  }

  def drop(postgreSql: PostgreSql, tableName: String): Int =
    postgreSql.executeUpdate(s"DROP TABLE IF EXISTS $tableName") match {
      case Right(cnt) => cnt
      case Left(errorMsg) =>
        logger.error(s"Failed to drop $tableName $errorMsg")
        -1
    }


  def truncate(postgreSql: PostgreSql, tableName: String): Int =
    postgreSql.executeUpdate(s"DELETE FROM $tableName") match {
      case Right(cnt) => cnt
      case Left(errorMsg) =>
        logger.error(s"Failed to delete $tableName $errorMsg")
        -1
    }


  def duplicate(postgreSql: PostgreSql, srcTableName: String, dupTableName: String): Boolean = {
    // drop(postgreSQL, duplicate_name)
    val result = postgreSql.executeUpdate(s"CREATE TABLE $dupTableName AS (SELECT * FROM $srcTableName)")
    result.isRight
  }


  def getIndices(postgreSql: PostgreSql): Seq[Seq[String]] = {
    val stmt = s"""SELECT tablename,indexname FROM pg_indexes""" // WHERE tablename='$tableName'"""
    postgreSql.executeQuery(stmt) match {
      case Right(rs) =>
        val indicesBuf = ListBuffer[Seq[String]]()
        while (rs.next) {
          val record = Seq[String](
            rs.getString(1),
            rs.getString(2)
          )
          indicesBuf.append(record)
        }
        indicesBuf.toSeq

      case Left(errorMsg) =>
        logger.error(errorMsg)
        Seq.empty[Seq[String]]
    }
  }
}



