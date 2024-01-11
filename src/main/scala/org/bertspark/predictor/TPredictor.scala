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
package org.bertspark.predictor

import org.bertspark.config.{ExecutionMode, S3PathNames}
import org.bertspark.nlp.medical.MedicalCodingTypes.{lineItemSeparator, InternalRequest, MlLineItem, PRequest, PResponse}
import org.bertspark.util.io.S3Util
import org.bertspark.util.rdbms.{PostgreSql, PredictedClaimTbl}
import org.bertspark.{delay, InvalidParamsException}
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.modeling.InputValidation
import org.bertspark.modeling.SubModelsTaxonomy.subModelTaxonomy
import org.bertspark.predictor.model.PResponses
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer


sealed trait PredictionStorage
case object NoPredictionStorage extends PredictionStorage
case object DatabasePredictionStorage extends PredictionStorage
case object S3PredictionStorage extends PredictionStorage


/**
  * Request handler for requests for prediction
  * {{{
  * This class generate a prediction from a request with the following output
  * - PResponse back to Kafka
  * - Predicted claim to S3
  * }}}
  *
  * @author Patrick Nicolas
  * @version 0.6
  */
private[bertspark] final class TPredictor private (predictionStorage: PredictionStorage) extends InputValidation {
  import TPredictor._

  private[this] val collector = ListBuffer[PredictedClaim]()
  private[this] val postgreSql = PostgreSql()
  ExecutionMode.setEvaluation

  validate(Seq.empty[String])
  delay(timeInMillis = 1000L)

  @throws(clazz = classOf[InvalidParamsException])
  override protected def validate(args: Seq[String]): Unit = {
    if(!postgreSql.isConnected)
      throw new InvalidParamsException(s"Output prediction database table undefined")

    if(!subModelTaxonomy.isValid)
      throw new InvalidParamsException("Classifier parameters are undefined")
  }

  /**
    * Generic generation of prediction of type PResponses
    * @param requests Set of requests
    * @return Set of responses
    */
  def apiProcess(requests: Seq[PRequest]): Seq[PResponse] = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val start = System.currentTimeMillis()
    val responseGroup: PResponses = PResponses(requests)
    val responses = responseGroup.allResponses
    val requestsMap = requests.map(req => (req.id, req)).toMap
    val duration = (System.currentTimeMillis() - start)/requests.size

    logDebug(logger, msg = s"Auto-coding stats: ${responseGroup.getStats}")

    val predictions = responses.map(
      response => {
        val request = requestsMap(response.id)
        PredictedClaim(request, response, duration)
      }
    ).filter(_.id.nonEmpty)

    try {
      predictionStorage match {
        case DatabasePredictionStorage => updateDatabase(predictions)

        case S3PredictionStorage =>
          S3Util.datasetToS3[PredictedClaim](
            predictions.toDS(),
            S3PathNames.s3PredictedClaimPath,
            header = false,
            fileFormat = "json",
            toAppend = true,
            numPartitions = 4
          )

        case NoPredictionStorage =>
          logger.warn(s"Prediction are not stored")
      }
    }
    catch {
      case e: IllegalArgumentException =>
        TPredictor.logger.error(s"TPredictor.apiProcess ${e.getMessage}")
    }
    responses
  }

  private def updateDatabase(predictions: Seq[PredictedClaim]): Unit = {
    collector ++= predictions

    if(collector.size > flushingSize) {
      (collector.indices by flushingSize).foreach(
          index => {
            val limit = if(index + flushingSize < collector.size) index + flushingSize else collector.size
            PredictedClaimTbl.insertPrediction(collector.slice(index, limit), postgreSql)
            delay(timeInMillis = 200L)
          }
      )
      collector.clear
    }
  }

  def close(): Unit = postgreSql.close

  override def toString: String = subModelTaxonomy
      .predictorSubModelsMap
      .map{ case (subModel, predictor) => s"$subModel:\n${predictor.toString}"}
      .mkString("\n")
}


/**
  * Implement API process to invoke the prediction workflow.
  */
private[bertspark] final object TPredictor {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[TPredictor])

  final private val flushingSize = 4
  final val unsupportedState = 0
  final val oracleState = 1
  final val predictedState = 2
 // final val stateDescriptors = Array[String]("Unsupported", "Oracle", "Predicted")

  final private val noPredictionStorageLbl = "none"
  final private val databasePredictionStorageLbl = "database"
  final private val s3PredictionStorageLbl = "s3"

  def apply(): TPredictor = apply(mlopsConfiguration.runtimeConfig.predictionStorage)

  def apply(predictionStorage: String): TPredictor = predictionStorage match {
    case `databasePredictionStorageLbl` => new TPredictor(DatabasePredictionStorage)
    case `s3PredictionStorageLbl` => new TPredictor(S3PredictionStorage)
    case `noPredictionStorageLbl` => new TPredictor(NoPredictionStorage)
    case _ =>
      logger.warn(s"Prediction storage $predictionStorage is not supported, used none")
      new TPredictor(NoPredictionStorage)
  }

  def apply(predictionStorage: PredictionStorage): TPredictor = new TPredictor(predictionStorage)



  case class PredictedClaim(
    id: String,
    age: Int,
    gender: String,
    customer: String,
    client: String,
    procedureCategory: String,
    emrCodes: String,
    pos: String,
    dos: String,
    note: String,
    autoCodeState: Int,
    lineItems: Seq[MlLineItem],
    latency: Long) {
    override def toString: String =
      s"Id:$id,State:$autoCodeState,claim:${lineItems.map(_.toCodesComma).mkString(lineItemSeparator)}"
  }

  final object PredictedClaim {
    def apply(): PredictedClaim = PredictedClaim("", -1, "", "", "", "", "", "", "", "", 0, Seq.empty[MlLineItem], 0L)

    def apply(request: PRequest, response: PResponse, latency: Long): PredictedClaim = {
      PredictedClaim(
        response.id,
        request.context.age,
        request.context.gender,
        request.context.customer,
        request.context.client,
        request.context.procedureCategory,
        request.context.emrLabel,
        request.context.placeOfService,
        request.context.dateOfService,
        request.notes.head,
        response.autoCodingState,
        response.claim.lineItems,
        latency
      )
    }
  }


  def unsupported(unsupportedRequests: Seq[InternalRequest]): Seq[PResponse] =
    if(unsupportedRequests.nonEmpty) unsupportedRequests.map(req => PResponse(req.id)) else Seq.empty[PResponse]
}


