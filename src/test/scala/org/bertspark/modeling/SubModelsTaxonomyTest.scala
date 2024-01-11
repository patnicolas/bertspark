package org.bertspark.modeling


import org.apache.spark.sql.Dataset
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames
import org.bertspark.delay
import org.bertspark.modeling.SubModelsTaxonomy.subModelTaxonomy
import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalContext, InternalFeedback, InternalRequest, MlEMRCodes}
import org.bertspark.predictor.model.{OracleHandler, PredictionHandler}
import org.bertspark.util.io.S3Util
import org.bertspark.util.SparkUtil
import org.scalatest.flatspec.AnyFlatSpec

private[modeling] final class SubModelsTaxonomyTest extends AnyFlatSpec{

  it should "Succeed evaluating equality of maps" in {
    val map1 = Map[String, Int]("A"->1, "B" -> 2)
    val map2 = Map[String, Int]("A"->1, "B" -> 3)
    val map3 = Map[String, Int]("A"->1)
    val map4 = Map[String, Int]("A"->1, "B" -> 2)
    assert(map1 == map1)
    assert(map1 != map2)
    assert(map1 == map4)
  }

  ignore should "Succeed extracting sub models by categories" in {
    val subModelTaxonomy = SubModelsTaxonomy.load

    val labelIndicesMap: Map[Int, String] = subModelTaxonomy.indexedLabels
    println(s"Num Oracle sub models: ${subModelTaxonomy.oracleMap.size} - Num of predictive models ${subModelTaxonomy.predictiveMap.size}")
    println(s"\n\n\nOracle sub models map\n${subModelTaxonomy.oracleMap.map{ case (k, v) => s"$k ${labelIndicesMap.getOrElse(v, "")}"}.mkString("\n")}")
    println(s"Predictive sub models map\n${subModelTaxonomy.predictiveMap.map {
      case (k, indices) =>
        val labels = indices.map(idx => labelIndicesMap.getOrElse(idx, "")).filter(_.nonEmpty)
        s"$k $labels"
    }.mkString("\n")}")
  }


  ignore should "Succeed loading classifier parameters" in {
    println(subModelTaxonomy.toString)
  }


  it should "Succeed filter existing taxonomy" in {
    val filteredSubModelTaxonomy = SubModelsTaxonomy.filter(subModelTaxonomy)
    if(mlopsConfiguration.evaluationConfig.subModelFilterThreshold > 0.0F)
      println(filteredSubModelTaxonomy.toString)
    else
      assert(subModelTaxonomy.equals(filteredSubModelTaxonomy))
  }


  ignore should "Succeed extracting and testing few notes" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "requestsProd/CMBS"
    val requestDS = S3Util.s3ToDataset[InternalRequest](s3Folder, false, "json")
    val targets = Set[String]("73130 LT", "73130 LT 26")
    val targetedRequestDS = requestDS.filter(req => targets.contains(req.context.emrLabel)).limit(10)
    val targetedRequests = targetedRequestDS.collect()
    val targetedIds = targetedRequests.map(_.id)
    val s3FeedbackFolder = "feedbacksProd/CMBS"
    val feedbackDS = S3Util.s3ToDataset[InternalFeedback](s3FeedbackFolder, false, "json")
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
}
