package org.bertspark.nlp.trainingset


import org.bertspark.util.rdbms.{PostgreSql, PredictionsTbl}
import org.scalatest.flatspec.AnyFlatSpec

private[trainingset] final class TrainingSetTest extends AnyFlatSpec {

  ignore should "Succeed generating contextual document from a prediction" in {
    import org.bertspark.nlp.trainingset.implicits._

    val condition = "customer='Cornerstone'"
    val postgreSql = PostgreSql()
    val prediction = new PredictionsTbl(postgreSql, "predictions")
    val predictions = prediction.defaultQuery(10000, condition)
    val firstPrediction = predictions.head
    val contextualDocument: ContextualDocument = firstPrediction
    println(contextualDocument.toString)
    postgreSql.close
  }
}
