package org.bertspark.util.io

import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalFeedback, InternalRequest}
import org.bertspark.nlp.trainingset.{ContextualDocument, SubModelsTrainingSet}
import org.bertspark.util.SparkUtil
import org.scalatest.flatspec.AnyFlatSpec


private[io] final class S3IOOpsTest extends AnyFlatSpec {

  it should "Succeed generating a CMBS-test set of requests from CMBS-feedbacks" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    import org.bertspark.config.MlopsConfiguration._

    val s3RequestsInput = "requestsProd/CMBS"
    val s3FeedbackInput = "feedbacksProd/CMBS-Test"
    val testFeedbackDS = S3Util.s3ToDataset[InternalFeedback](s3FeedbackInput, false, "json").persist()
    println(s"Number of feedbacks: ${testFeedbackDS.count()}")
    val testRequestDS = S3Util.s3ToDataset[InternalRequest](s3RequestsInput, false, "json")

    val outputRequestDS = SparkUtil.sortingJoin[InternalRequest, InternalFeedback](
      testRequestDS, "id", testFeedbackDS, "id"
    ).map(_._1)
    println(s"Number of output requests: ${outputRequestDS.count()}")

    S3Util.datasetToS3[InternalRequest](
      mlopsConfiguration.storageConfig.s3Bucket,
      outputRequestDS,
      "requestsProd/CMBS-Test",
      false,
      "json",
      false,
      4
    )
  }

  ignore should "Succeed transferring data from S3 to local file" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    import org.bertspark.config.MlopsConfiguration._

    S3IOOps.s3ToFs[ContextualDocument](
      mlopsConfiguration.storageConfig.s3Bucket,
      "mlops/CMBS/contextDocument/FullNote",
      "CMBS/contextDocument/FullNote"
    )
  }

  ignore should "Succeed transferring data from s3" in {
    val s3SrcBucket = "aideo-tech-data-integration-prod"
    val prefix = "ut-rad/feedback"
    val s3DestFolder = "feedbacksProd/Cornerstone"
    val limit = 20

    S3IOOps.s3ToS3Feedback(s3SrcBucket, prefix, s3DestFolder, "txt", limit)
  }

  ignore should "Succeed in removing duplicate in request and feedback" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    import org.bertspark.config.MlopsConfiguration._

    val s3RequestFolder = "feedbacksProd/CMBS"
    val s3RequestDest = "feedbacksProd/CMBS-X"
    val requestsDS = S3Util.s3ToDataset[InternalFeedback](s3RequestFolder, false, "json").dropDuplicates("id")

    S3Util.datasetToS3[InternalFeedback](
      mlopsConfiguration.storageConfig.s3Bucket,
      requestsDS,
      s3RequestDest,
      false,
      "json",
      true,
      8
    )
  }
}
