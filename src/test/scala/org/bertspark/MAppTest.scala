package org.bertspark

import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalFeedback, InternalRequest}
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.ContextualDocument
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable.ListBuffer

class MAppTest extends AnyFlatSpec{
  import org.bertspark.config.MlopsConfiguration._

  it should "Succeed collecting and sorting key words" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3ContextualDocumentFolder = s"mlops/XLARGE4/contextDocument"
    val textTokenDS = S3Util.s3ToDataset[ContextualDocument](s3ContextualDocumentFolder)
        .flatMap(_.text.split(tokenSeparator))
        .distinct()

  }


  ignore should "Succeed converting and parsing abbreviations map" in {
    val fsAbbreviationsMap = s"conf/abbreviationsMap"
    LocalFileUtil.Load.local(fsAbbreviationsMap).foreach(
      content => {
        val lines = content.split("\n")
        val filteredLines = lines.filter(line => !(line.contains(".") || line.contains("/")))
        val converted = filteredLines.map(_.toLowerCase)
        LocalFileUtil.Save.local("temp/newAbbreviationsMap", converted.mkString("\n"))
      }
    )
  }

  ignore should "Succeed creating a test request set" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3FeedbackFolder = "feedbacksProd/TEST"
    val feedbackIdDS = S3Util.s3ToDataset[InternalFeedback](s3FeedbackFolder).map(_.id).persist().cache()
    val feedbackIds = feedbackIdDS.collect()

    val s3RequestFolder = "requestsProd/XLARGE2"
    val requestsDS = S3Util.s3ToDataset[InternalRequest](s3RequestFolder)

    val feedbackIds_brdCast = sparkSession.sparkContext.broadcast[Seq[String]](feedbackIds)

    val testInternalRequestDS = requestsDS.mapPartitions(
      (iter: Iterator[InternalRequest]) => {
        val feedbackIdsValue = feedbackIds_brdCast.value.toSet

        val collector = ListBuffer[InternalRequest]()
        while(iter.hasNext) {
          val internalRequest = iter.next()
          if(feedbackIdsValue.contains(internalRequest.id))
            collector.append(internalRequest)
        }
        collector.iterator
      }
    )
    println(s"There are ${testInternalRequestDS.count()} test internal requests")

    S3Util.datasetToS3[InternalRequest](
      mlopsConfiguration.storageConfig.s3Bucket,
      testInternalRequestDS,
      s3OutputPath = "requestsProd/TEST",
      header = false,
      fileFormat = "json",
      toAppend = false,
      numPartitions = 6
    )
  }
}
