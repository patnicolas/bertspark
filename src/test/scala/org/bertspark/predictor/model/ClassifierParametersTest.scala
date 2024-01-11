package org.bertspark.predictor.model

import org.apache.spark.sql.Dataset
import org.bertspark.config.S3PathNames
import org.bertspark.delay
import org.bertspark.modeling.SubModelsTaxonomy.subModelTaxonomy
import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalContext, InternalFeedback, InternalRequest, MlEMRCodes}
import org.bertspark.util.io.S3Util
import org.bertspark.util.SparkUtil
import org.scalatest.flatspec.AnyFlatSpec

private[model] final class ClassifierParametersTest extends AnyFlatSpec {

  ignore should "Succeed loading classifier parameters" in {
    val classifiersParams = subModelTaxonomy
    println(classifiersParams.toString)
  }


  ignore should "Succeed extracting and testing few notes" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "requestsProd/CMBS"
    val requestDS = S3Util.s3ToDataset[InternalRequest](s3Folder)
    val targets = Set[String]("73130 LT", "73130 LT 26")
    val targetedRequestDS = requestDS.filter(req => targets.contains(req.context.emrLabel)).limit(10)
    val targetedRequests = targetedRequestDS.collect()
    val targetedIds = targetedRequests.map(_.id)
    val s3FeedbackFolder = "feedbacksProd/CMBS"
    val feedbackDS = S3Util.s3ToDataset[InternalFeedback](s3FeedbackFolder)
        .filter(feedback => targetedIds.contains(feedback.id))

    val feedbacks = feedbackDS.collect()
    println(s"Found ${targetedRequests.mkString("\n")}")
    val predictionHandler = new PredictionHandler(subModelTaxonomy)
    val responses = predictionHandler(targetedRequests)
    println(responses.mkString("\n"))
  }

  ignore should "Succeed parsing set of labels" in {
    val input = "aaaa bbbb||cccc||ddd eeee"
    val fields = input.split("\\|\\|")
    println(fields.mkString("\n"))
  }


  ignore should "Succeed extracting break down from feedbacks" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3FeedbackFolder = "feedbacksProd/CMBS"
    val ds = S3Util.s3ToDataset[InternalFeedback](
      s3FeedbackFolder, false, "json"
    )
    val emrs = ds.map(_.context.emrLabel).distinct().collect()
    val oracles = emrs.filter(subModelTaxonomy.oracleMap.contains(_))
    val predictive = emrs.filter(subModelTaxonomy.predictiveMap.contains(_))
    println(s"Num oracles: ${oracles.size} predictions: ${predictive.size}")
  }

  ignore should "Succeed computing the distribution per sub-models, labels and requests" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    import org.bertspark.config.MlopsConfiguration._

    val minNumRequestsPerLabels = 32
    val s3FeedbackFolder = "feedbacksProd/CMBS"
    val ds = S3Util.s3ToDataset[InternalFeedback](
      s3FeedbackFolder, false, "json"
    ).map(List[InternalFeedback](_)).persist()

    val groupedByLabelRDD = SparkUtil.groupBy[List[InternalFeedback], String](
      (feedback: List[InternalFeedback])=> feedback.head.toFinalizedSpace,
      (f1: List[InternalFeedback], f2: List[InternalFeedback]) => f1 ::: f2,
      ds
    ).map(grouped => (grouped.head.toFinalizedSpace, grouped.size))
    val filterByLabelRDD = groupedByLabelRDD.filter(_._2 > minNumRequestsPerLabels)
    val numFilteredRequests = filterByLabelRDD.map(_._2).reduce(_ + _)
    val groupedByLabels = filterByLabelRDD.collect().sortWith(_._2 > _._2)

    println(s"GroupedByEmr:\nTotal number of requests: ${ds.count} Num filtered requests: $numFilteredRequests\n${groupedByLabels.map{ case (label, cnt) => s"$label: $cnt"}.mkString("\n")}")
    val validLabelSet = groupedByLabels.map(_._1).distinct.toSet
    val s3SubModelPath = "mlops/CMBS/models/447/subModels.csv"

    S3Util.download(
      mlopsConfiguration.storageConfig.s3Bucket,
      s3SubModelPath
    ).map(
      subModels => {
        val lines = subModels.split("\n")
        val newLines = lines.map(
          line => {
            val fields = line.split(",")
            if(fields(1) == "1")
              line
            else {
              val labels = fields(2).split("\\|\\|")
              val extractedLabels = labels.filter(validLabelSet.contains(_))
              if(extractedLabels.nonEmpty) {
                println(s"Original labels: ${labels.size} new labels: ${extractedLabels.size}")
                val newLine = s"${fields.head},${extractedLabels.size},${extractedLabels.mkString("||")},none"
                newLine
              }
              else
                ""
            }
          }
        ).filter(_.nonEmpty)

        val newContent = newLines.sortWith(_ < _).mkString("\n")
        S3Util.upload(
          mlopsConfiguration.storageConfig.s3Bucket,
          s"mlops/CMBS/models/447/subModels-$minNumRequestsPerLabels.csv",
          newContent
        )
      }
    )
  }


  ignore should "Succeed generating response from an Oracle" in {
    val emrCodes1 =  Seq[MlEMRCodes](MlEMRCodes(0, "74178", Seq[String]("26", "GC"), Seq.empty[String], 1, "UN"))
    val emrCodes2 =  Seq[MlEMRCodes](MlEMRCodes(0, "74450", Seq[String]("26", "52"), Seq.empty[String], 1, "UN"))

    val internalRequest1 = InternalRequest("", InternalContext(emrCodes1), Seq.empty[String])
    val internalRequest2 = InternalRequest("", InternalContext(emrCodes2), Seq.empty[String])

    val oracleRequests = Seq[InternalRequest](internalRequest1, internalRequest2)
    val predictionHandler = new OracleHandler(subModelTaxonomy)
    val responses = predictionHandler(oracleRequests)
    println(responses.map(_.toCodes).mkString("\n"))
  }


  ignore should "Succeed generate a simple test bench" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    import org.bertspark.config.MlopsConfiguration._

    val requestDS = S3Util.s3ToDataset[InternalRequest](S3PathNames.s3RequestsPath, false, "json").persist()

    val trainedRequests = S3Util.download(mlopsConfiguration.storageConfig.s3Bucket, s"mlops/ALL/models/450/subModels-16.csv")
        .map(
          content => {
            val lines = content.split("\n")
            lines.map(
              line => {
                val fields = line.split(",")
                fields.head.trim
              }
            )
          }
        ).getOrElse(Array.empty[String]).toSet

    val testRequestDS: Dataset[InternalRequest] = requestDS
        .filter(req =>
          trainedRequests.contains(req.context.emrLabel)
        )
        .limit(128)

    testRequestDS.show()
    S3Util.datasetToS3[InternalRequest](
      testRequestDS,
      "requestsProd/ALL-Test",
      false,
      "json",
      false,
      1
    )
    delay(8000L)
  }


  it should "Succeed running a tests" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val requests = S3Util.s3ToDataset[InternalRequest](
      "requestsProd/ALL-Test", false, "json"
    ).collect
    val predictionHandler = new PredictionHandler(subModelTaxonomy)
    val responses = predictionHandler(requests)
    println(responses.mkString("\n"))
  }
}
