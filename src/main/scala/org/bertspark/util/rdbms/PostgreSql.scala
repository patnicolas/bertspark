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

import com.amazonaws.auth.{AWSStaticCredentialsProvider, BasicAWSCredentials}
import java.io.IOException
import java.sql.{Connection, DriverManager, PreparedStatement, ResultSet, SQLException, Statement}
import org.bertspark.config.MlopsConfiguration.dataConfigMap
import org.bertspark.config.DatabaseConfig
import org.slf4j.{Logger, LoggerFactory}


/**
 * Wrapper for PostgreSQL RDBMS as implemented in the RDS service
 * @param databaseConfig Configuration for accessing PostgreSQL RDBMS
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class PostgreSql private (databaseConfig: DatabaseConfig) {
  import PostgreSql._

  /**
   * Lazy instantiation of a connection to RDS, using the configuration for the PostgreSQL data base
   */
  private[this] val connection: Option[Connection] = {
    var _connection: Connection = null
    try {
      val url = s"${connectionPrefix}${databaseConfig.host}:${databaseConfig.port.toString}/${databaseConfig.dbName}"
      logger.info(s"Connect to RDS")

      Class.forName(driverName)
      _connection = DriverManager.getConnection(url, databaseConfig.user, databaseConfig.encryptedPwd)
      _connection.setAutoCommit(true)
      Some(_connection)
    }
    catch {
      case e: IOException =>
        logger.error(s"Connection IO error: ${e.getMessage}")
        None
      case e: SQLException =>
        logger.error(s"Connection SQL error: ${e.getMessage}")
        None
      case e: Exception =>
        logger.error(s"Connection: ${e.getMessage}")
        None
      case e: Throwable =>
        logger.error(s"Connection: ${e.getMessage}")
        None
    }
  }

  @inline
  final def isConnected: Boolean = connection.isDefined

  def close: Unit = connection.foreach(_.close)

  /**
   * Commit the current transaction
   */
  def commit: Unit = connection.foreach(_.commit)


  /**
   * Execute query for a given statement 'SELECT a,b,.. FROM table ...'
   * @param sqlStatement SQL statement for query
   * @return Either an error message or a JDBC Result set
   */
  final def executeQuery(sqlStatement: String): Either[String, ResultSet] =
    if (connection.isDefined) {
      val c = connection.get
      var stmt: Statement = null
      try {
        stmt = c.createStatement
        val result = stmt.executeQuery(sqlStatement)
        Right(result)
      }
      catch {
        case e: IOException =>
          errorMsg[ResultSet](s"IO failure for ${sqlStatement}: ${e.getMessage}")
        case e: SQLException =>
          errorMsg[ResultSet](s"SQL error for ${sqlStatement}: ${e.getMessage}")
        case e: Exception =>
          errorMsg[ResultSet](s"Undefined error for ${sqlStatement}: ${e.getMessage}")
      }
    }
    else
      errorMsg[ResultSet](s"Connection for ${databaseConfig.host}:${databaseConfig.port}/${databaseConfig.dbName} undefined")


  /**
   * Generic update to a table. The update may depends on a previous query, to extract the current id
   * @param updateStmt Update statement
   * @param queryStmt Query statement to extract the id (SERIAL)
   * @return Either an error message or the latest primary id
   */
  def executeUpdate(updateStmt: String, queryStmt: String = ""): Either[String, Int] =
    if (connection.isDefined) {
      val c = connection.get
      var stmt: Statement = null
      try {
        stmt = c.createStatement
        stmt.executeUpdate(updateStmt)
        val id =
          if(queryStmt.nonEmpty) {
            val rs = stmt.executeQuery(queryStmt)
            var primaryId = 0
            while(rs.next)
              primaryId = rs.getInt(1)
            primaryId
          }
          else
            0
        Right(id)
      }
      catch {
        case e: IOException =>
          errorMsg[Int](s"IO failure for ${updateStmt}: ${e.getMessage}")
        case e: SQLException =>
          errorMsg[Int](s"SQL error for ${updateStmt}:  ${e.getMessage}")
        case e: Exception =>
          errorMsg[Int](s"Undefined error for ${updateStmt}:  ${e.getMessage}")
      }
      finally {
        if (stmt != null)
          stmt.close
      }
    }
    else
      errorMsg[Int](s"Connection for ${databaseConfig.host}:${databaseConfig.port}/${databaseConfig.dbName} failed")


  def executePreparedStatement(tblName: String, anyVals: Seq[AnyVal], numValues: Int): Either[String, Boolean] =
    if (connection.isDefined) {
      val c = connection.get
      var stmt: PreparedStatement = null
      try {
        stmt = c.prepareStatement(s"INSERT INTO $tblName VALUES (${questionMark(numValues)});")
        setValues(anyVals, stmt)
        stmt.executeUpdate()
        Right(true)
      }
      catch {
        case e: IOException =>
          Left(s"IO failure for ${tblName}: ${e.getMessage}")
        case e: SQLException =>
          Left(s"SQL error for ${tblName}:  ${e.getMessage}")
        case e: Exception =>
          Left(s"Undefined error for ${tblName}:  ${e.getMessage}")
      }
      finally {
        if (stmt != null)
          stmt.close
      }
    }
    else
      Left(s"Connection for ${databaseConfig.host}:${databaseConfig.port}/${databaseConfig.dbName} failed")




  private def errorMsg[T](err: String): Either[String, T] = {
    PostgreSql.logger.error(err)
    Left(err)
  }
}

/**
 * Singleton for generic parameterized method to generate statements
 */
private[bertspark] final object PostgreSql  {
  private val logger: Logger = LoggerFactory.getLogger("PostgreSql")

  final private val driverName = "org.postgresql.Driver"
  final private val connectionPrefix = "jdbc:postgresql://"


  /**
   * Generic constructor for default configuration (loaded from configuration file dataConfig)
   * @return Instance of Postgre SQL RDBMS
   */
  def apply(): PostgreSql = apply(dataConfigMap.get("ai-ml").get)

  /**
   * Generic constructor for a given database configuration
   * @param databaseConfig Configuration for the remote PostgreSQL database engine
   * @return Instance of Postgre SQL RDBMS
   */
  def apply(databaseConfig: DatabaseConfig): PostgreSql = new PostgreSql(databaseConfig)

  /**
   * Detailed constructor ....
   * @param host Host (IP, URL or localhost)
   * @param port Listening port
   * @param dbName Name of the data base
   * @param user User name
   * @param password Password
   * @return Instance of Postgre SQL RDBMS
   */
  def apply(name: String,
            host: String,
            port: Int,
            dbName: String,
            user: String,
            password: String,
            region: String,
            autoCommit: Boolean = true): PostgreSql = {
    val config = DatabaseConfig(name, host, port, dbName, user, password, region)
    new PostgreSql(config)
  }


  final def questionMark(numValues: Int): String = numValues match {
    case 1 => "?"
    case 2 => "?,?"
    case 3 => "?,?,?"
    case 4 => "?,?,?,?"
    case 5 => "?,?,?,?,?"
    case 6 => "?,?,?,?,?,?"
    case _ =>
      logger.error(s"Number insert value: ${numValues} is out of range")
      ""
  }

  val accessKey = "AKIAJLMBUGQN2Q2FWANQ"
  val secretKey = "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"
  private lazy val credentials = new BasicAWSCredentials(accessKey, secretKey)

  import com.amazonaws.services.rds.AmazonRDSClientBuilder

  private def rdsClient(name: String) = {
    val region = dataConfigMap.get(name).map(_.region).getOrElse("us-east-2")
    AmazonRDSClientBuilder
        .standard()
        .withCredentials(new AWSStaticCredentialsProvider(credentials))
        .withRegion(region)
        .build
  }



  def setValues(anyValues: Seq[AnyVal], stmt: PreparedStatement): Unit =
    anyValues.indices.foreach(
      index =>
        if (anyValues(index).isInstanceOf[Double])
          stmt.setDouble(index + 1, anyValues(index).asInstanceOf[Double])
        else if (anyValues(index).isInstanceOf[Int])
          stmt.setInt(index + 1, anyValues(index).asInstanceOf[Int])
        else {
          val inputStr = anyValues(index).asInstanceOf[String]
          stmt.setString(index + 1, s"\'${inputStr}\'")
        }
    )

  def getTypedValue(anyValues: Seq[AnyVal]): String = {
    import scala.collection.mutable.ListBuffer

    val buf = new ListBuffer[String]
    anyValues.foreach(anyVal => buf.append(anyVal.toString))
    buf.mkString(", ")
  }
}
