package org.bertspark.analytics

import org.bertspark.analytics.MetricsCollector.{fnLbl, fpLbl, metrics, tpLbl, MetricCollectorRecord}
import org.bertspark.nlp.medical.MedicalCodingTypes.{FeedbackLineItem, InternalFeedback, MlEMRCodes}
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable.{HashMap, ListBuffer}

private[analytics] final class MetricsCollectorTest extends AnyFlatSpec {

  it should "Succeed parsing output metrics for each sub models" in {
    import org.bertspark.config.MlopsConfiguration._

    val s3Folder = "mlops/XLARGE5/metrics/480/A-471"
    val s3Bucket = mlopsConfiguration.storageConfig.s3Bucket
    val accumulator = ListBuffer[(String, Float, Float)]()

    S3Util.getS3Keys(s3Bucket, s3Folder).foreach(
      s3Path => {
        S3Util.download(s3Bucket, s3Path).map(
          content => {
            val lines = content.split("\n")
            var subModelName: String = ""
            var coreAccuracy = -1.0F
            var strictAccuracy = -1.0F

            lines.foreach(
              line =>
                if(line.contains("*"))
                  subModelName = line.substring(0, line.indexOf(","))
                else if (line.contains("coreAccuracy"))
                  coreAccuracy = line.substring(line.indexOf(",")+1).toFloat
                else if (line.contains("strictAccuracy"))
                  strictAccuracy = line.substring(line.indexOf(",")+1).toFloat
            )

            if(subModelName.nonEmpty && coreAccuracy >= 0.0F && strictAccuracy >= 0.0F)
              accumulator.append((subModelName, coreAccuracy, strictAccuracy))
            else {
              println(s"ERROR subModel $subModelName has not accuracy components")
            }
            println(s"Sub model $subModelName completed ${accumulator.size} collected!")
          }
        )
      }
    )
    val collectedData = accumulator.sortWith(_._3 > _._3).map{
      case (subModel, coreAccuracy, strictAccuracy) => s"$subModel,$coreAccuracy,$strictAccuracy"
    }.mkString("\n")

    LocalFileUtil.Save.local("output/subModelStats.csv", s"SubModel,Core Match,StrictMatch\n$collectedData")
  }


  ignore should "Succeed evaluating sub models accuracy" in {
    import org.bertspark.config.MlopsConfiguration._

    val s3Folder = "mlops/XLARGE5/metrics/480/A-471/prediction-05-24-2023-04.05.470"
    val collectorMap = HashMap[String, List[(Int, Int)]]()

    // Status,Type,SubModel,Predicted,Actual,NumClaims
    val collected = S3Util.download(mlopsConfiguration.storageConfig.s3Bucket, s3Folder)
        .map(
          content => {
            content.split("\n").map(
              line => {
                if(line.size > 8) {
                  val fields = line.split(",")
                  if(fields.size > 3) {
                    val xs = collectorMap.getOrElse(fields(2), List.empty[(Int, Int)])
                    val statusValue = if (fields.head == 'S') 1 else 0
                    val subModelType = if (fields(1) == "Oracle") 0 else 1
                    collectorMap.put(fields(2), (statusValue, subModelType) :: xs)
                  }
                }
              }
            )
            collectorMap
          }
        ).getOrElse(HashMap.empty[String, List[(Int, Int)]])

    if(collected.nonEmpty) {
      val (trainedMap, oracleMap) = collected.toSeq.partition{ case (_, xs) => xs.head._2 == 1}
      val subModelsMap: Seq[(String, Float)] = trainedMap.map{
        case (subModel, xs) => (subModel, xs.map(_._1).sum.toFloat/xs.length)
      }.sortWith(_._2 > _._2)

      LocalFileUtil.Save.local(
        "output/predictionstats.csv",
        subModelsMap.map{ case (subModel, rate) => s"$subModel,$rate"}.mkString("\n")
      )
    }

  }

  ignore should "Succeed extracting label accuracy distribution" in {
    val fsNohupFilename = "output/nohup-461.txt"
    val subModelDistribution = MetricsCollector.collectTrainingLabelDistribution(fsNohupFilename)
    val distributionMapStr = subModelDistribution.map{ case (label, acc) => s"$label:$acc"}
    println(s"Distribution of ${distributionMapStr.size}\n${distributionMapStr.mkString("\n")}")
  }

  ignore should "Succeed updating metrics collector Oracle match 1" in {
    import MetricsCollectorTest._
    val predictedFeedback = FeedbackLineItem(0, "78014", Seq[String]("26", "GC"),  Seq[String]("E21.3"), 0, "UN", 0.0)
    val labeledFeedback = FeedbackLineItem(0, "78014", Seq[String]("26", "GC"),  Seq[String]("E21.3"), 0, "UN", 0.0)
    val emrCodes = Seq[MlEMRCodes](MlEMRCodes(0, "78014", Seq[String]("26", "GC"), Seq.empty[String], 0, "UN"))

    val internalFeedback =  InternalFeedback(
      "1",
      emrCodes,
      Seq[FeedbackLineItem](predictedFeedback),
      Seq[FeedbackLineItem](labeledFeedback)
    )

    val metricsCollector = new MyMetricsCollector
    metricsCollector.updateMetrics(internalFeedback)
    val (allMetrics, coreMetrics) = metricsCollector.getMetrics
    println(s"\nOracle match 1 --------\nAll metrics:\n${allMetrics.toString}\nCore metrics:\n${coreMetrics.toString}")
  }


  ignore should "Succeed updating metrics collector Oracle match 2" in {
    import MetricsCollectorTest._

    val predictedFeedback = FeedbackLineItem(0, "78014", Seq[String]("26", "GC"),  Seq[String]("E21.3"), 0, "UN", 0.0)
    val labeledFeedback = FeedbackLineItem(0, "78014", Seq[String]("26", "GC"),  Seq[String]("Z91.08"), 0, "UN", 0.0)
    val emrCodes = Seq[MlEMRCodes](MlEMRCodes(0, "78014", Seq[String]("26", "GC"), Seq.empty[String], 0, "UN"))
    val internalFeedback =  InternalFeedback(
      "1",
      emrCodes,
      Seq[FeedbackLineItem](predictedFeedback),
      Seq[FeedbackLineItem](labeledFeedback)
    )

    val metricsCollector = new MyMetricsCollector
    metricsCollector.updateMetrics(internalFeedback)
    val (allMetrics, coreMetrics) = metricsCollector.getMetrics
    println(s"\nOracle match 2 --------\nAll metrics:\n${allMetrics.toString}\nCore metrics:\n${coreMetrics.toString}")
  }


  ignore should "Succeed updating metrics collector trained model match" in {
    import MetricsCollectorTest._
    // 73070 26 RT
    val predictedFeedback = FeedbackLineItem(0, "72100", Seq[String]("26"),  Seq[String]("K40.90"), 0, "UN", 0.0)
    val labeledFeedback = FeedbackLineItem(0, "72100", Seq[String]("26"),  Seq[String]("K40.90"), 0, "UN", 0.0)
    val emrCodes = Seq[MlEMRCodes](MlEMRCodes(0, "72100", Seq[String]("26"), Seq.empty[String], 0, "UN"))
    val internalFeedback =  InternalFeedback(
      "1",
      emrCodes,
      Seq[FeedbackLineItem](predictedFeedback),
      Seq[FeedbackLineItem](labeledFeedback)
    )

    val metricsCollector = new MyMetricsCollector
    metricsCollector.updateMetrics(internalFeedback)
    val (allMetrics, coreMetrics) = metricsCollector.getMetrics
    println(s"\nTrained model match --------\nAll metrics:\n${allMetrics.toString}\nCore metrics:\n${coreMetrics.toString}")
  }

  ignore should "Succeed updating metrics collector trained model unmatched" in {
    import MetricsCollectorTest._
    // 73070 26 RT

    val predictedFeedback = FeedbackLineItem(0, "72100", Seq[String]("26"),  Seq[String]("K40.90"), 0, "UN", 0.0)
    val labeledFeedback = FeedbackLineItem(0, "72100", Seq[String]("26"),  Seq[String]("J81.90"), 0, "UN", 0.0)
    val emrCodes = Seq[MlEMRCodes](MlEMRCodes(0, "72100", Seq[String]("26"), Seq.empty[String], 0, "UN"))

    val internalFeedback =  InternalFeedback(
      "1",
      emrCodes,
      Seq[FeedbackLineItem](predictedFeedback),
      Seq[FeedbackLineItem](labeledFeedback)
    )

    val metricsCollector = new MyMetricsCollector
    metricsCollector.updateMetrics(internalFeedback)
    val (allMetrics, coreMetrics) = metricsCollector.getMetrics
    println(s"\nTrained unmatched --------\nAll metrics:\n${allMetrics.toString}\nCore metrics:\n${coreMetrics.toString}")
  }

  ignore should "Succeed updating metrics collector not supported" in {
    import MetricsCollectorTest._
    val predictedFeedback = FeedbackLineItem(0, "9999", Seq[String]("26", "RT"),  Seq[String]("K40.90"), 0, "UN", 0.0)
    val labeledFeedback = FeedbackLineItem(0, "9999", Seq[String]("26", "RT"),  Seq[String]("Z12.37"), 0, "UN", 0.0)
    val emrCodes = Seq[MlEMRCodes](MlEMRCodes(0, "9999", Seq[String]("26", "RT"), Seq.empty[String], 0, "UN"))
    val internalFeedback =  InternalFeedback(
      "1",
      emrCodes,
      Seq[FeedbackLineItem](predictedFeedback),
      Seq[FeedbackLineItem](labeledFeedback)
    )

    val metricsCollector = new MyMetricsCollector
    metricsCollector.updateMetrics(internalFeedback)
    val (allMetrics, coreMetrics) = metricsCollector.getMetrics
    println(s"\nNot supported model --------\nAll metrics:\n${allMetrics.toString}\nCore metrics:\n${coreMetrics.toString}")
  }

  ignore should "Succeed updating a metrics records" in {
    val metricsCollector = new MetricCollectorRecord
    metricsCollector += tpLbl
    metricsCollector += tpLbl
    metricsCollector += fpLbl
    metricsCollector += tpLbl
    metricsCollector += fnLbl

    val labelAccuracy = 3.0/5
    assert(Math.abs(metricsCollector.accuracy - labelAccuracy) < 0.01 == true)
    val precisionLabel = 0.75
    assert(Math.abs(metricsCollector.precision - precisionLabel) < 0.01 == true)

    val metrics = metricsCollector.getMetrics
    println(metrics.toString)
  }

  ignore should "Succeed updating the metrics collector" in {
    val metricsCollector = new MetricsCollector {
      override protected[this] val lossName: String = "MyLoss"
      def update(predictedSelectedIndices: Seq[Int], labeledSelectedIndices: Seq[Int]): Unit = {
        super.updateBatchMetrics(predictedSelectedIndices, labeledSelectedIndices)
      }
    }

    val predictedSelectedIndices = Seq[Int](45, 991, 8, 45, 11, 23, 45, 981)
    val labeledSelectedIndices = Seq[Int](45, 991, 19, 45, 12, 23, 50, 981)

    metricsCollector.update(predictedSelectedIndices, labeledSelectedIndices)
    println(metricsCollector.toString)
  }


  ignore should "Succeed extracting match from feedback core match" in {
    val feedbackLineItem1 = FeedbackLineItem(0, "77123",  Seq[String]("26"), Seq[String]("Z12.31"), 0, "UN", 0.0)
    val feedbackLineItem2 = FeedbackLineItem(0, "77123",  Seq[String]("26", "LT"), Seq[String]("M78.919", "Z12.31"), 0, "UN", 0.0)

    val predictedLineItems = Seq[FeedbackLineItem](feedbackLineItem1)
    val labeledLineItems = Seq[FeedbackLineItem](feedbackLineItem2)
    val mlEmrCodes =  Seq[MlEMRCodes](MlEMRCodes(0, "77123", Seq[String]("26"), Seq.empty[String], 1, "UN"))

    val feedback: InternalFeedback = InternalFeedback.apply("id", mlEmrCodes, predictedLineItems, labeledLineItems)
    val succeeded = MetricsCollector.metrics(feedback).map {
      case (res, _) =>
        !res.isStrict && res.isCore && !res.isFirstLine
    }.getOrElse(false)
    assert(succeeded)
  }


  ignore should "Succeed extracting match from feedback record- strict match" in {
    val feedbackLineItem1 = FeedbackLineItem(0, "77123",  Seq[String]("26"), Seq[String]("Z12.31"), 0, "UN", 0.0)
    val feedbackLineItem2 = FeedbackLineItem(0, "G3761",  Seq[String]("26", "LT"), Seq[String]("M78.919", "Z12.31"), 0, "UN", 0.0)

    val predictedLineItems = Seq[FeedbackLineItem](feedbackLineItem1, feedbackLineItem2)
    val labeledLineItems = Seq[FeedbackLineItem](feedbackLineItem1, feedbackLineItem2)
    val mlEmrCodes =  Seq[MlEMRCodes](MlEMRCodes(0, "90012", Seq.empty[String], Seq.empty[String], 1, "UN"))

    val feedback: InternalFeedback = InternalFeedback.apply("id", mlEmrCodes, predictedLineItems, labeledLineItems)
    val succeeded = MetricsCollector.metrics(feedback).map { case (res, _) =>
      res.isStrict && res.isCore && res.isFirstLine
    }.getOrElse(false)
    assert(succeeded)
  }


  ignore should "Succeed extracting match from feedback record- strict match not core" in {
    val feedbackLineItem1 = FeedbackLineItem(0, "77123",  Seq[String]("26"), Seq[String]("Z12.31"), 0, "UN", 0.0)
    val feedbackLineItem2 = FeedbackLineItem(0, "G3761",  Seq[String]("26", "LT"), Seq[String]("M78.919", "Z12.31"), 0, "UN", 0.0)

    val predictedLineItems = Seq[FeedbackLineItem](feedbackLineItem1, feedbackLineItem2)
    val labeledLineItems = Seq[FeedbackLineItem](feedbackLineItem2, feedbackLineItem1)
    val mlEmrCodes =  Seq[MlEMRCodes](MlEMRCodes(0, "90012", Seq.empty[String], Seq.empty[String], 1, "UN"))

    val feedback: InternalFeedback = InternalFeedback.apply("id", mlEmrCodes, predictedLineItems, labeledLineItems)
    val succeeded = metrics(feedback).map { case (res, _)  =>
      res.isStrict && !res.isCore && res.isFirstLine
    }.getOrElse(false)
    assert(succeeded)
  }

  ignore should "Succeed extracting match from feedback record- first line match" in {
    val feedbackLineItem1 = FeedbackLineItem(0, "77123",  Seq[String]("26"), Seq[String]("Z12.31"), 0, "UN", 0.0)
    val feedbackLineItem2 = FeedbackLineItem(0, "77881",  Seq[String]("26", "LT"), Seq[String]("M78.919"), 0, "UN", 0.0)
    val feedbackLineItem3 = FeedbackLineItem(0, "77881",  Seq[String]("26", "LT"), Seq[String]("Z12.3"), 0, "UN", 0.0)

    val predictedLineItems = Seq[FeedbackLineItem](feedbackLineItem1, feedbackLineItem2)
    val labeledLineItems = Seq[FeedbackLineItem](feedbackLineItem3, feedbackLineItem1)
    val mlEmrCodes =  Seq[MlEMRCodes](MlEMRCodes(0, "90012", Seq.empty[String], Seq.empty[String], 1, "UN"))

    val feedback: InternalFeedback = InternalFeedback.apply("id", mlEmrCodes, predictedLineItems, labeledLineItems)
    val succeeded = metrics(feedback).map { case (res, _) =>
      !res.isStrict && !res.isCore && res.isFirstLine
    }.getOrElse(false)
    assert(succeeded)
  }


  ignore should "Succeed extracting match from feedback record- core match" in {
    val feedbackLineItem1 = FeedbackLineItem(0, "77123",  Seq[String]("26"), Seq[String]("Z12.31", "M89.11"), 0, "UN", 0.0)
    val feedbackLineItem2 = FeedbackLineItem(0, "77881",  Seq[String]("26", "LT"), Seq[String]("M78.919"), 0, "UN", 0.0)
    val feedbackLineItem3 = FeedbackLineItem(0, "77123",  Seq[String]("26"), Seq[String]("Z12.3"), 0, "UN", 0.0)

    val predictedLineItems = Seq[FeedbackLineItem](feedbackLineItem1, feedbackLineItem2)
    val labeledLineItems = Seq[FeedbackLineItem](feedbackLineItem2, feedbackLineItem3)
    val mlEmrCodes =  Seq[MlEMRCodes](MlEMRCodes(0, "90012", Seq.empty[String], Seq.empty[String], 1, "UN"))

    val feedback: InternalFeedback = InternalFeedback.apply("id", mlEmrCodes, predictedLineItems, labeledLineItems)
    val succeeded = metrics(feedback).map {
      case (res, _) => !res.isStrict && !res.isCore && !res.isFirstLine
    }.getOrElse(false)
    assert(succeeded)
  }

}



final object MetricsCollectorTest {

  class MyMetricsCollector extends MetricsCollector {
    override val lossName = ""
    override def updateMetrics(internalFeedback: InternalFeedback): Boolean = super.updateMetrics(internalFeedback)
  }

}

