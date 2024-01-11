package org.bertspark.util.rdbms

import org.apache.spark.sql.SparkSession
import org.bertspark.config.MlopsConfiguration
import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalContext, MlEMRCodes, InternalRequest, Pipe}
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.bertspark.util.rdbms.BaseTbl.logger
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer

class SNTbl {

}



private[bertspark] final object SNTbl {
  private val logger: Logger = LoggerFactory.getLogger("SNTbl")

  case class SNMetaData(
    location_id: String,
    location_name: String,
    patient_id: String,
    id: String,
    patient_first_name: String,
    patient_middle_name: String,
    patient_last_name: String,
    sex: String,
    date_of_birth: String,
    date_of_service: String,
    date_of_service_to: String,
    state_seen_in: String,
    place_of_service_code: Int,
    department: String,
    provider_id: String,
    provider_name: String,
    payer_id: String,
    payer_name: String,
    procedure_category: String,
    provider_npi: String,
    box24: String,
    box19: String,
    mrn: String,
    accession: String,
    is_feedback: String,
    coder_id: Int
  )

  def query(postgreSql: PostgreSql)(implicit sparkSession: SparkSession): Unit = {
    import sparkSession.implicits._

    val stmt = s"SELECT * FROM sn_data"
    postgreSql.executeQuery(stmt).foreach(
      resSet => {

        val predictReqBuf = ListBuffer[InternalRequest]()
        while (resSet.next) {
          val id = resSet.getString(2)
          val client = resSet.getString(4)
          val placeOfService = resSet.getString(5)
          val dateOfService = resSet.getString(6)
          val note = resSet.getString(9)
          val metaData = resSet.getString(7)
          val snMetaData = LocalFileUtil.Json.mapper.readValue(metaData, classOf[SNMetaData])
          val customer = "SN"
          val modality = "ER"
          val date = snMetaData.date_of_birth.substring(0, 4).toInt
          val age = 2022 - date
          val gender = snMetaData.sex
          val patientId = snMetaData.patient_id

          val contextReq = InternalContext(
            "",
            age,
            gender: String,
            "",
            customer: String,
            client: String,
            modality: String,
            placeOfService: String,
            dateOfService: String,
            Seq.empty[MlEMRCodes],
            snMetaData.provider_id,
            patientId: String,
            "",
            "",
            "",
            "")

          predictReqBuf.append(InternalRequest(id, contextReq, Seq[String](note)))

          if(predictReqBuf.size > 8192) {
            val ds = predictReqBuf.toSeq.toDS()
            S3Util.datasetToS3[InternalRequest](
              MlopsConfiguration.mlopsConfiguration.storageConfig.s3Bucket,
              ds,
              s3OutputPath = "requests/SN",
              header = false,
              fileFormat = "json",
              toAppend = true,
              numPartitions = 1
            )
            predictReqBuf.clear()
          }
        }
      }
    )
  }


  def queryStatement(tableName: String, limit: Int, fields: String, condition: String): String =
    if(condition.isEmpty)
      if(limit > 0)
        s"SELECT $fields FROM $tableName limit $limit"
      else
        s"SELECT $fields FROM $tableName"
    else
      if(limit > 0)
        s"SELECT $fields FROM $tableName WHERE $condition limit $limit"
      else
        s"SELECT $fields FROM $tableName WHERE $condition "
}
