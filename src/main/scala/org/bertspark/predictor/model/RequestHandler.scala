package org.bertspark.predictor.model

import ai.djl.ndarray.NDManager
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.modeling.SubModelsTaxonomy
import org.bertspark.nlp.medical.MedicalCodingTypes.{FeedbackLineItem, InternalRequest, MlClaimResp, PResponse}
import org.bertspark.nlp.trainingset.{ContextualDocument, TokenizedTrainingSet}
import org.bertspark.predictor.TPredictor.predictedState
import org.slf4j.{Logger, LoggerFactory}

trait RequestHandler {
  def apply(inputRequests: Seq[InternalRequest]): Seq[PResponse]
}


object RequestHandler {
  final val logger: Logger = LoggerFactory.getLogger("RequestHandler")

}



private[bertspark] final class OracleHandler(classifierSubModelConfig: SubModelsTaxonomy) extends RequestHandler {
  import RequestHandler._

  /**
    * Extract Oracle requests from input requests
    * @param inputRequests Input requests
    * @return Sequence of predictions (responses)
    */
  override def apply(inputRequests: Seq[InternalRequest]): Seq[PResponse] =
    if (inputRequests.nonEmpty) {
      val oraclePredictions = inputRequests.map(
        request => {
          val label = classifierSubModelConfig.getOracleLabel(request.context.emrLabel)
          if(label.isDefined) {
            logDebug(logger, msg = s"Find Oracle model for ${label.get}")
            PResponse(request.id, label.get)
          }
          else {
            logger.warn(s"Failed to find label for oracle ${request.context.emrLabel}")
            PResponse("")
          }
        }
      ).filter(_.id.nonEmpty)
      oraclePredictions
    }
    else
      Seq.empty[PResponse]
}


private[bertspark] final class PredictionHandler(classifierSubModelConfig: SubModelsTaxonomy) extends RequestHandler {
  import RequestHandler._

  override def apply(inputRequests: Seq[InternalRequest]): Seq[PResponse] =
    if (inputRequests.nonEmpty) {
      import org.bertspark.implicits._
      import sparkSession.implicits._

      val groupedByEmr = inputRequests.groupBy[String](_.context.emrLabel)
      logDebug(logger, msg = s"${groupedByEmr.size} groups to predict")


      // There is no guarantee that we will be able to find a sub model
      val pResponses = if (groupedByEmr.nonEmpty) {
        var count = 0
        val totalNumRecords: Int = groupedByEmr.map(_._2.size).reduce(_ + _)

        val docIdPredictions: Seq[DocIdPrediction] = groupedByEmr.map {
          case (subModel, req) =>
            if (req.nonEmpty) {
              val tokenizedTrainingDS = req.toDS().map(
                request => {
                  import org.bertspark.nlp.trainingset.implicits._
                  val ctxDocument: ContextualDocument = request
                  TokenizedTrainingSet(ctxDocument)
                }
              )

              val ndManager = NDManager.newBaseManager()
              val validPrediction = classifierSubModelConfig.getPredictorModel(subModel).map(
                predictor => {
                  val predicted = predictor.predict(ndManager, tokenizedTrainingDS)
                  if (predicted.nonEmpty) {
                    count += req.size
                    val ratio = count.toDouble / totalNumRecords
                    logger.info(
                      s"Find Predictive ${predicted.head._2} for $subModel $count/$totalNumRecords $ratio%"
                    )
                    predicted
                  }
                  else {
                    logger.error(s"Empty prediction for sub model $subModel")
                    Seq.empty[DocIdPrediction]
                  }
                }
              ).getOrElse({
                logger.error(s"Failed to match with predictive sub model $subModel")
                Seq.empty[(String, String)]
              })

              ndManager.close()
              validPrediction
            }
            else {
              logger.error(s"No request associated with the sub model $subModel")
              Seq.empty[(String, String)]
            }
        }.filter(_.nonEmpty).flatten.toSeq
        formatPredictionResponses(docIdPredictions)
      }
      else {
        logger.error("No trained sub-model found!")
        Seq.empty[PResponse]
      }
      pResponses
    }
    else {
      logger.error("None of requests have a predictive sub-models")
      Seq.empty[PResponse]
    }


  private def formatPredictionResponses(docIdPredictions: Seq[DocIdPrediction]): Seq[PResponse] = {
    logger.info(s"Succeed predicting ${docIdPredictions.size} claims")

    docIdPredictions.map {
      case (id, prediction) =>
        val predictionFields = prediction.split(":")
        if (predictionFields.size == 2) {
          val lineItems: Seq[FeedbackLineItem] = FeedbackLineItem.toLineItems(predictionFields(1))
          val claimResp = MlClaimResp(lineItems)
          PResponse(id, true, predictedState, claimResp, "", "OK")
        }
        else {
          logger.warn(s"Failed to generate a prediction $prediction for $id")
          PResponse("", true, -1, MlClaimResp(Seq.empty[FeedbackLineItem]), "", "Failed")
        }
    }.filter(_.autoCodingState >= 0)
  }
}


private[bertspark] final class UnsupportedHandler() extends RequestHandler {

  override def apply(inputRequests: Seq[InternalRequest]): Seq[PResponse] =
    inputRequests.map(internalRequest => PResponse(internalRequest.id))
}

