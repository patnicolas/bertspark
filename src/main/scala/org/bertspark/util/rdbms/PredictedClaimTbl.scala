package org.bertspark.util.rdbms

import java.sql.SQLException
import org.bertspark.nlp.medical.MedicalCodingTypes.lineItemSeparator
import org.bertspark.predictor.TPredictor.PredictedClaim
import org.bertspark.util.DateUtil.{longToDate, requestDateFormat}
import org.slf4j.{Logger, LoggerFactory}


private[bertspark] case class PredictedClaimTbl(
  postgreSql: PostgreSql,
  tableName: String
) extends BaseTbl[PredictedClaimTbl]  {

  /**
    * String to create a table with the appropriate fields.
    */
  protected[this] val createTable: String =
    s"""CREATE TABLE IF NOT EXISTS $tableName (
       |id SERIAL,
       |date VARCHAR(16) NOT NULL,
       |vocabulary VARCHAR(12) NOT NULL,
       |tokenizer VARCHAR(32) NOT NULL,
       |min_label_freq INT NOT NULL,
       |max_label_freq INT NOT NULL,
       |transformer VARCHAR(24) NOT NULL,
       |segmentation VARCHAR(32) NOT NULL,
       |num_segments INT NOT NULL,
       |sub_model_sample_size INT NOT NULL,
       |max_masking_size INT NOT NULL,
       |classifier VARCHAR(32) NOT NULL,
       |min_sample_sub_model_size INT NOT NULL,
       |max_sample_sub_model_size INT NOT NULL,
       |encounter_id VARCHAR(96) NOT NULL,
       |age INT NOT NULL,
       |gender CHAR(2) NOT NULL,
       |customer VARCHAR(32) NOT NULL,
       |client VARCHAR(32) NOT NULL,
       |procedure_category VARCHAR(32) NOT NULL,
       |place_of_service VARCHAR(32) NOT NULL,
       |date_of_service VARCHAR(32) NOT NULL,
       |emr VARCHAR(256) NOT NULL,
       |note TEXT NOT NULL,
       |auto_code_state INT NOT NULL,
       |claim VARCHAR(512) NOT NULL,
       |latency INT NOT NULL)""".stripMargin.replaceAll("\n", " ")



  def isReady: Boolean = {
    val connected = postgreSql.isConnected
    if (connected) postgreSql.executeUpdate(createTable).isRight else false
  }

  /**
    * The connection to the RDS data base has to be explicitly clodes
    */
  override  def close: Unit = postgreSql.close
  override def count: Int = ???
  override   def defaultQuery(limit: Int): Seq[Seq[String]] = ???
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
private[bertspark] final object PredictedClaimTbl {
  private val logger: Logger = LoggerFactory.getLogger("PredictedClaimTable")
  import org.bertspark.config.MlopsConfiguration._

  final private val predictedClaimTableName = mlopsConfiguration.runtimeConfig.table

  def apply(postgreSql: PostgreSql): PredictedClaimTbl = new PredictedClaimTbl(postgreSql, predictedClaimTableName)


  /**
    * Insertion mechanism
    * @param prediction Prediction request and response
    * @param postgreSql Reference to Postgre SQL instance
    * @param tableName Name of the table to be inserted into
    * @return True if insertion succeeds, False otherwise
    */
  final def insertPrediction(prediction: PredictedClaim, postgreSql: PostgreSql, tableName: String): Boolean = try {
    val date = longToDate(-1L, requestDateFormat)
    val claim = prediction.lineItems.map(_.toCodesComma).mkString(lineItemSeparator)
    val stmt =
      s"""INSERT INTO $tableName (
         |date,
         |vocabulary,
         |tokenizer,
         |min_label_freq,
         |max_label_freq,
         |transformer,
         |segmentation,
         |num_segments,
         |sub_model_sample_size,
         |max_masking_size,
         |classifier,
         |min_sample_sub_model_size,
         |max_sample_sub_model_size,
         |encounter_id,
         |age,
         |gender,
         |customer,
         |client,
         |procedure_category,
         |place_of_service,
         |date_of_service,
         |emr,
         |note,
         |auto_code_state,
         |claim,
         |latency)
         | VALUES(
         |'$date',
         |'${mlopsConfiguration.preProcessConfig.vocabularyType}',
         |'${mlopsConfiguration.preTrainConfig.tokenizer}',
         |${mlopsConfiguration.preProcessConfig.minLabelFreq},
         |${mlopsConfiguration.preProcessConfig.maxLabelFreq},
         |'${mlopsConfiguration.preTrainConfig.transformer}-${mlopsConfiguration.runId}',
         |'${mlopsConfiguration.preTrainConfig.sentenceBuilder}',
         |${mlopsConfiguration.preTrainConfig.numSentencesPerDoc},
         |${mlopsConfiguration.preTrainConfig.maxNumRecords},
         |${mlopsConfiguration.preTrainConfig.maxMaskingSize},
         |'FFNN-${mlopsConfiguration.classifyConfig.modelId}',
         |${mlopsConfiguration.classifyConfig.minNumRecordsPerLabel},
         |${mlopsConfiguration.classifyConfig.maxNumRecordsPerLabel},
         |'${prediction.id}',
         |${prediction.age},
         |'${prediction.gender}',
         |'${prediction.customer}',
         |'${prediction.client}',
         |'${prediction.procedureCategory}',
         |'${prediction.pos}',
         |'${prediction.dos}',
         |'${prediction.emrCodes}',
         |'${prediction.note.replaceAll("'", "")}',
         |'${claim}'
         |${prediction.latency})""".stripMargin.replaceAll("\n", " ")

    // Execution the insert statement on Postgre SQL
    postgreSql.executeUpdate(stmt) match {
      case Right(_) =>
        logger.info(s"Succeed inserting into $tableName")
        true
      case Left(errorMsg) =>
        logger.error(s"\n *** $errorMsg")
        false
    }
  }
  catch {
    case e: SQLException =>
      logger.error(s"SQL exception: ${e.toString}")
      false
    case e: Exception =>
      logger.error(s"Undefined exception ${e.toString}")
      false
  }

  /**
    * Insertion mechanism
    * @param prediction Prediction request and response
    * @param postgreSQL Reference to Postgre SQL instance
    * @param tableName Name of the table to be inserted into
    * @return True if insertion succeeds, False otherwise
    */
  final def insertPrediction(predictions: Seq[PredictedClaim], postgreSQL: PostgreSql, tableName: String): Boolean = try {
    val date = longToDate(-1L, requestDateFormat)
    val values = predictions.map(
      prediction => {
        val classifierName = s"FeedForwardNeural-${mlopsConfiguration.classifyConfig.modelId}"
        val claim = prediction.lineItems.map(_.toCodesComma).mkString(lineItemSeparator)
        val note = prediction.note.replaceAll("\\)", "").replaceAll("'", "")

        s"('$date','${mlopsConfiguration.preProcessConfig.vocabularyType}','${mlopsConfiguration.preTrainConfig.tokenizer}',${mlopsConfiguration.preProcessConfig.minLabelFreq}," +
        s"${mlopsConfiguration.preProcessConfig.maxLabelFreq},'${mlopsConfiguration.preTrainConfig.transformer}-${mlopsConfiguration.runId}'," +
        s"'${mlopsConfiguration.preTrainConfig.sentenceBuilder}', ${mlopsConfiguration.preTrainConfig.numSentencesPerDoc}, ${mlopsConfiguration.preTrainConfig.maxNumRecords}," +
        s"${mlopsConfiguration.preTrainConfig.maxMaskingSize},'$classifierName',${mlopsConfiguration.classifyConfig.minNumRecordsPerLabel}," +
        s"${mlopsConfiguration.classifyConfig.maxNumRecordsPerLabel}," +
        s"'${prediction.id}',${prediction.age},'${prediction.gender}'," +
            s"'${prediction.customer}','${prediction.client}','${prediction.procedureCategory}','${prediction.pos}'," +
            s"'${prediction.dos}','${prediction.emrCodes}','$note',${prediction.autoCodeState},'$claim',${prediction.latency})"
      }
    ).mkString(",")

    val stmt =
      s"""INSERT INTO $tableName (
         |date,
         |vocabulary,
         |tokenizer,
         |min_label_freq,
         |max_label_freq,
         |transformer,
         |segmentation,
         |num_segments,
         |sub_model_sample_size,
         |max_masking_size,
         |classifier,
         |min_sample_sub_model_size,
         |max_sample_sub_model_size,
         |encounter_id,
         |age,
         |gender,
         |customer,
         |client,
         |procedure_category,
         |place_of_service,
         |date_of_service,
         |emr,
         |note,
         |auto_code_state,
         |claim,
         |latency)
          VALUES $values""".stripMargin.replaceAll("\n", " ")

    postgreSQL.executeUpdate(stmt) match {
      case Right(_) =>
        logger.info(s"Inserted ${predictions.size} predictions into $tableName")
        true
      case Left(errorMsg) =>
        logger.error(errorMsg)
        false
    }
  }
  catch {
    case e: SQLException =>
      logger.error(s"SQL exception: ${e.toString}")
      false
    case e: Exception =>
      logger.error(s"Undefined exception ${e.toString}")
      false
  }

  final def insertPrediction(predictions: Seq[PredictedClaim], postgreSql: PostgreSql): Boolean =
    insertPrediction(predictions, postgreSql, predictedClaimTableName)


    def queryStatement(tableName: String, limit: Int, fields: String, condition: String): String =
    if(condition.isEmpty)
      if(limit > 0) s"SELECT $fields FROM $tableName limit $limit" else  s"SELECT $fields FROM $tableName"
    else
      if(limit > 0) s"SELECT $fields FROM $tableName WHERE $condition limit $limit" else  s"SELECT $fields FROM $tableName WHERE $condition "


  /**
    * Query all the fields of this given tables
    * @param postgreSQL reference to the current PostgreSQL client
    * @param predictionTableName Name of the  prediction table
    * @param limit Limit of the query
    * @param fields List of column or fields to be retrieved
    * @param condition WHERE condition for this query as 'WHERE $condition'
    * @return Sequence of rows as list of string
    */
  def query(
    postgreSQL: PostgreSql,
    predictionTableName: String,
    limit: Int,
    fields: String,
    condition: String = ""): Seq[Seq[String]] = try {
    import scala.collection.mutable.ListBuffer

    val defaultQueryStmt = queryStatement(predictionTableName, limit, fields, condition)
    postgreSQL.executeQuery(defaultQueryStmt) match {
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
        logger.error(errorMsg)
        Seq.empty[Seq[String]]
    }
  }
  catch {
    case e: SQLException =>
      logger.error(e.getMessage)
      Seq.empty[Seq[String]]
    case e: Exception =>
      logger.error(e.getMessage)
      Seq.empty[Seq[String]]
  }
}

