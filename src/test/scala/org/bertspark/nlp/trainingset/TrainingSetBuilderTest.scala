package org.bertspark.nlp.trainingset

import org.apache.spark.sql.{Dataset, SaveMode}
import org.bertspark.config.{MlopsConfiguration, S3PathNames}
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.modeling.{SubModelsTaxonomy, TrainingLabelIndexing}
import org.bertspark.nlp.medical.MedicalCodingTypes.{FeedbackLineItem, InternalContext, InternalFeedback, InternalRequest, MlClaimEntriesWithCodes, MlEMRCodes}
import org.bertspark.nlp.trainingset.TrainingSetBuilder.groupLabeledRequestsByEmr
import org.bertspark.nlp.trainingset.TrainingSetBuilderTest.createRequestTest
import org.bertspark.util.io.S3Util.{accessKey, datasetToS3, s3ToDataset, secretKey}
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.bertspark.util.SparkUtil
import org.scalatest.flatspec.AnyFlatSpec


private[trainingset] final class TrainingSetBuilderTest extends AnyFlatSpec {

  it should "Succeed getting distribution of label - numNotes" in {
    LocalFileUtil.Load.local(s"output/SubModelClaimNotesDistribution.txt").foreach(
      content => {
        val lines = content.split("\n")
        val rankedLabels = lines.filter(line => line.nonEmpty & line.contains(":"))
            .map(
              line => {
                val fields = line.split(":")
                try {
                  (fields.head, fields(1).toInt)
                }
                catch{
                  case e: NumberFormatException =>
                    println(s"ERROR: ${e.getMessage}")
                    (fields.head, -1)
                }
              }
            ).sortWith(_._2 > _._2)
        println(rankedLabels.map{ case (label, cnt) => s"$label: $cnt"}.mkString("\n"))
      }
    )
  }


  ignore should "Succeed estimating auto coding rates" in {
    println(s"Estimate coding rate for 1 note/label: ${TrainingSetBuilder.estimateAutoCodingRate(1)}")
  }


  ignore should "Succeed extracting unique emr from requests" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = s"mlops/XLARGE2/training/TF92"
    val requestDS = S3Util.s3ToDataset[SubModelsTrainingSet](
      s3Folder, false, "json"
    ).dropDuplicates("id")

    println(s"Num requests for ${s3Folder}: ${requestDS.count}")
    val emrLabelsDS = requestDS.map(_.subModel.replace("  ", " ")).distinct
    println(s"Number distinct emrLabels: ${emrLabelsDS.count()}")
  }

  ignore should "Succeed creating sub models list from training set" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val subModelsTrainingDS = S3Util.s3ToDataset[SubModelsTrainingSet](
      mlopsConfiguration.storageConfig.s3Bucket,
      S3PathNames.s3ModelTrainingPath,
      header = false,
      fileFormat = "json"
    )
    val labelIndices = TrainingLabelIndexing.save(subModelsTrainingDS)
    SubModelsTaxonomy.save(subModelsTrainingDS, labelIndices)
  }


  ignore should "Succeed listing unique emr code from various folders" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3InternalRequest = "requestsProd/CMBS"

    val internalRequestDS = S3Util.s3ToDataset[InternalRequest](
      mlopsConfiguration.storageConfig.s3Bucket,
        s3InternalRequest,
      header = false,
      fileFormat = "json"
    ).dropDuplicates("id").map(_.context.emrLabel).distinct()
    val internalRequests = internalRequestDS.collect().sortWith(_ < _)
    LocalFileUtil.Save.local("output/CMBS-EMR-Requests.txt", internalRequests.mkString("\n"), false)
    println(internalRequests.mkString("\n"))
  }

  ignore should "Succeed listing unique id for contextual document" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3InternalRequest = "mlops/CMBS/contextDocument/AMA"

    val internalRequestDS = S3Util.s3ToDataset[ContextualDocument](
      mlopsConfiguration.storageConfig.s3Bucket,
      s3InternalRequest,
      header = false,
      fileFormat = "json"
    ).map(_.id).distinct()
    val internalRequests = internalRequestDS.collect().sortWith(_ < _)
    LocalFileUtil.Save.local("output/CMBS-ID-ContextDocument.txt", internalRequests.mkString("\n"), false)
    println(internalRequests.mkString("\n"))
  }


  ignore should "Succeed listing unique emr code from requests/feedbacks" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3InternalRequest = "requestsProd/CMBS"

    val internalRequestDS = S3Util.s3ToDataset[InternalRequest](
      mlopsConfiguration.storageConfig.s3Bucket,
      s3InternalRequest,
      header = false,
      fileFormat = "json"
    ).dropDuplicates("id")


    val s3Feedbacks = "feedbacksProd/CMBS"
    val internalFeedbackDS = S3Util.s3ToDataset[InternalRequest](
      mlopsConfiguration.storageConfig.s3Bucket,
      s3Feedbacks,
      header = false,
      fileFormat = "json"
    ).dropDuplicates("id")

    val subModelDS = SparkUtil.sortingJoin[InternalRequest, InternalRequest](
      internalRequestDS, "id", internalFeedbackDS, "id"
    ).map{
      case (req, _) => req.id
    }.distinct()

    val subModels = subModelDS.collect().sortWith(_ < _)
    LocalFileUtil.Save.local("output/CMBS-ID-Requests-Feedbacks.txt", subModels.mkString("\n"), false)
  }

  ignore should "Succeed listing unique emr code from training" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3InternalRequest = "mlops/CMBS/training/AMA"

    val internalRequestDS = S3Util.s3ToDataset[SubModelsTrainingSet](
      mlopsConfiguration.storageConfig.s3Bucket,
      s3InternalRequest,
      header = false,
      fileFormat = "json"
    ).map(_.subModel).distinct()
    val internalRequests = internalRequestDS.collect().sortWith(_ < _)
    LocalFileUtil.Save.local("output/CMBS-EMR-training.txt", internalRequests.mkString("\n"), false)

  }

  ignore should "Succeed found intersection between requests and feedbacks" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val requestDS = try {
      val requestPath = S3PathNames.s3RequestsPath
      S3Util.s3ToDataset[InternalRequest](
        mlopsConfiguration.storageConfig.s3Bucket,
        requestPath,
        header = false,
        fileFormat = "json"
      ).map(_.id).distinct
    }
    catch {
      case e: IllegalStateException =>
        println(s"Error Contextual document: ${e.getMessage}")
        sparkSession.emptyDataset[String]
    }

    val s3FeedbackFolder = S3PathNames.s3FeedbacksPath
    val rawFeedbackDS = S3Util.s3ToDataset[InternalFeedback](
      s3FeedbackFolder,
      false,
      "json"
    ).map(_.id).distinct

    val requestIds = requestDS.collect().sortWith(_ < _)
    val feedbackIds = rawFeedbackDS.collect().sortWith(_ < _)
    val intersection = requestIds.intersect(feedbackIds)
    LocalFileUtil.Save.local(
      "output/compare.txt",
      s"RequestIds:\n${requestIds.mkString("\n")}\nFeedbackIds:\n${feedbackIds.mkString("\n")}\nIntersection:\n${intersection.mkString("\n")}"
    )
  }

  ignore should "Succeed converting labeled claim into label - 1" in {
    val emrCpts = Seq[MlEMRCodes](
      MlEMRCodes(0, "77889", Seq[String]("26", "LT"), Seq.empty[String])
    )
    val contextualDocument = ContextualDocument("id", Array.empty[String], "")
    val lineItems = Seq[FeedbackLineItem](
      FeedbackLineItem(0, "G6738", Seq.empty[String], Seq[String]("I25.1"), 0, "UN", 0.1),
      FeedbackLineItem(0, "77000", Seq[String]("GC"), Seq[String]("Z12.31"), 0, "UN", 0.1),
      FeedbackLineItem(0, "77889", Seq[String]("26", "LT"), Seq[String]("I10"), 0, "UN", 0.1)
    )
    val labeledRequest = LabeledRequest(contextualDocument, emrCpts, lineItems)
    val hierarchicalLabels = TrainingLabel(labeledRequest)
    println(s"Conversion1: ${hierarchicalLabels.toString}")
  }

  ignore should "Succeed converting labeled claim into label - 2" in {
    val emrCpts = Seq[MlEMRCodes](
      MlEMRCodes(0, "77889", Seq[String]("26", "LT"), Seq.empty[String])
    )
    val contextualDocument = ContextualDocument("id", Array.empty[String], "")
    val lineItems = Seq[FeedbackLineItem](
      FeedbackLineItem(0, "77889", Seq[String]("26", "LT"), Seq[String]("I10"), 0, "UN", 0.1),
      FeedbackLineItem(0, "G6738", Seq.empty[String], Seq[String]("I25.1"), 0, "UN", 0.1),
      FeedbackLineItem(0, "77000", Seq[String]("GC"), Seq[String]("Z12.31"), 0, "UN", 0.1),
    )
    val labeledRequest = LabeledRequest(contextualDocument, emrCpts, lineItems)
    val hierarchicalLabels = TrainingLabel(labeledRequest)
    println(s"Conversion2: ${hierarchicalLabels.toString}")
  }


  ignore should "Succeed joining context document and labels" in {
    val emrCpts1 = Seq[MlEMRCodes](
      MlEMRCodes(0, "77889", Seq[String]("26", "LT"), Seq.empty[String])
    )
    val lineItems1 = Seq[FeedbackLineItem](
      FeedbackLineItem(0, "77889", Seq[String]("26", "LT"), Seq[String]("B89.11", "M78.12"), 0, "UN", 0.1),
      FeedbackLineItem(0, "G6738", Seq.empty[String], Seq[String]("I25.1"), 0, "UN", 0.1),
      FeedbackLineItem(0, "77000", Seq[String]("GC"), Seq[String]("Z12.31"), 0, "UN", 0.1),
    )
    val context1 = InternalContext("", 11, "M", "T", "Cu", "Cl", "M", "Pos", "Dos", emrCpts1, "", "", "", "", "", "")

    val feedback1 = InternalFeedback(
      "1",
      true,
      context1,
      MlClaimEntriesWithCodes(),
      MlClaimEntriesWithCodes(lineItems1),
      MlClaimEntriesWithCodes()
    )
    val contextualDocument1 = ContextualDocument("1")

    val emrCpts2 = Seq[MlEMRCodes](
      MlEMRCodes(0, "66001", Seq[String]("GC"), Seq.empty[String])
    )
    val lineItems2 = Seq[FeedbackLineItem](
      FeedbackLineItem(0, "G6738", Seq.empty[String], Seq[String]("I25.1"), 0, "UN", 0.1),
      FeedbackLineItem(0, "66001", Seq[String]("GC"), Seq[String]("I10", "M78.99"), 0, "UN", 0.1)
    )
    val context2 = InternalContext("", 11, "M", "T", "Cu", "Cl", "M", "Pos", "Dos", emrCpts2, "", "", "", "", "", "")

    val feedback2 = InternalFeedback(
      "2",
      true,
      context2,
      MlClaimEntriesWithCodes(),
      MlClaimEntriesWithCodes(lineItems2),
      MlClaimEntriesWithCodes()
    )
    val contextualDocument2 = ContextualDocument("2")

    import org.bertspark.implicits._
    import sparkSession.implicits._
    val feedbackDS = Seq[InternalFeedback](feedback1, feedback2).toDS()
    val contextualDocumentDS = Seq[ContextualDocument](contextualDocument1, contextualDocument2).toDS()

    val labeledRequestDS = TrainingSetBuilder.joinContextDocumentAndLabels(contextualDocumentDS, feedbackDS)
    println(labeledRequestDS.collect.mkString("\n"))
  }

  ignore should "Succeed group notes by labels" in {
    val contextualDocument1 = ContextualDocument("1")
    val emrCpts1 = Seq[MlEMRCodes](
      MlEMRCodes(0, "77889", Seq[String]("26", "LT"), Seq.empty[String])
    )
    val lineItems1 = Seq[FeedbackLineItem](
      FeedbackLineItem(0, "77889", Seq[String]("26", "LT"), Seq[String]("B89.11", "M78.12"), 0, "UN", 0.1),
      FeedbackLineItem(0, "G6738", Seq.empty[String], Seq[String]("I25.1"), 0, "UN", 0.1),
      FeedbackLineItem(0, "77000", Seq[String]("GC"), Seq[String]("Z12.31"), 0, "UN", 0.1),
    )
    val labeledRequest1 = LabeledRequest(contextualDocument1, emrCpts1, lineItems1)

    val contextualDocument2 = ContextualDocument("2")
    val emrCpts2 = Seq[MlEMRCodes](
      MlEMRCodes(0, "66001", Seq[String]("GC"), Seq.empty[String])
    )
    val lineItems2 = Seq[FeedbackLineItem](
      FeedbackLineItem(0, "G6738", Seq.empty[String], Seq[String]("I25.1"), 0, "UN", 0.1),
      FeedbackLineItem(0, "66001", Seq[String]("GC"), Seq[String]("I10", "M78.99"), 0, "UN", 0.1)
    )
    val labeledRequest2 = LabeledRequest(contextualDocument2, emrCpts2, lineItems2)


    val contextualDocument3 = ContextualDocument("3")
    val emrCpts3 = Seq[MlEMRCodes](
      MlEMRCodes(0, "66001", Seq[String]("GC"), Seq.empty[String])
    )
    val lineItems3 = Seq[FeedbackLineItem](
      FeedbackLineItem(0, "77182", Seq.empty[String], Seq[String]("Z12.31"), 0, "UN", 0.1),
      FeedbackLineItem(0, "66001", Seq[String]("GC"), Seq[String]("I10", "M78.99"), 0, "UN", 0.1)
    )
    val labeledRequest3 = LabeledRequest(contextualDocument3, emrCpts3, lineItems3)

    import org.bertspark.implicits._
    import sparkSession.implicits._
    val labeledRequestDS = Seq[LabeledRequest](labeledRequest1, labeledRequest2, labeledRequest3).toDS()
    val groupContextDocumentAndLabelDS: Dataset[(String, Seq[List[TrainingLabel]])] = groupLabeledRequestsByEmr(labeledRequestDS)

    val groupContextDocumentAndLabelStr = groupContextDocumentAndLabelDS.map{
      case (emr, seqXs) => {
        val seqXsStr = seqXs.map(_.map(_.getLabel).mkString(" || ")).mkString("\n*")
        s"------$emr------:\n*$seqXsStr"
      }
    }.collect
    println(groupContextDocumentAndLabelStr.mkString("\n\n"))
  }

  ignore should "Create a small training set" in {
    createRequestTest
  }



  ignore should "Succeed writing into S3" in {
    val input = Seq[String]("aa", "bb", "cc", "dd", "ee", "ff")
    import org.bertspark.implicits._
    import sparkSession.implicits._
    val inputDS = input.toDS

    val accessConfig = inputDS.sparkSession.sparkContext.hadoopConfiguration

    accessConfig.set("fs.s3a.access.key", accessKey)
    accessConfig.set("fs.s3a.secret.key", secretKey)
    val s3Bucket = "aideo-tech-autocoding-v1"
    val s3OutputPath = "temp/tt.json"

    inputDS
        .write
        .format("json")
        .option("header", false)
        .mode(SaveMode.Append)
        .save(path = s"s3a://$s3Bucket/$s3OutputPath")
  }

  ignore should "Succeed extracting a given training set" in  {
    val s3Bucket = "aideo-tech-autocoding-v1"
    val s3RequestFolder = "requests/Cornerstone"
    val s3FeedbackFolder = "feedbacks/40/7/utrad"
    val s3OutputFolder = "mlops/CornerstoneA"
    val args = Seq[String](
      s3Bucket,
      s3RequestFolder,
      s3FeedbackFolder,
      s3OutputFolder,
      "16"
    )
    import org.bertspark.implicits._
    TrainingSetBuilder(args)
  }

  ignore should "Succeed loading Training data" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3TrainingDataFolder = "mlops/CornerstoneTest"
    val numRecords = 50

    val trainingSetProcessor = TrainingSet(s3TrainingDataFolder, numRecords, Map.empty[String, Int])
    val requests = trainingSetProcessor.getLabeledRequests(5)
    println(requests.mkString("\n"))
  }
}


private[trainingset] final object TrainingSetBuilderTest {

  private def createRequestTest: Unit ={
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Bucket = MlopsConfiguration.mlopsConfiguration.storageConfig.s3Bucket
    val s3RequestFolder = S3PathNames.s3RequestsPath
    val s3FeedbackFolder = "feedbacks/CornerstoneTest"

    val uniqueFeedbackDS = s3ToDataset[InternalFeedback](
      s3Bucket,
      s3FeedbackFolder,
      header = false,
      fileFormat = "json").dropDuplicates("id")

    val feedbackIds = uniqueFeedbackDS.map(_.id).collect()
    println(s"Number of test feedback ids ${feedbackIds.size}")

    val testRequestDS = s3ToDataset[InternalRequest](
      s3Bucket,
      s3RequestFolder,
      header = false,
      fileFormat = "json")
        .dropDuplicates("id")
        .filter(request => feedbackIds.contains(request.id))

    println(s"Number of test requests ${testRequestDS.count()}")
    datasetToS3[InternalRequest](
      testRequestDS,
      "requests/CornerstoneTest",
      false,
      "json",
      false,
      4
    )
  }


}