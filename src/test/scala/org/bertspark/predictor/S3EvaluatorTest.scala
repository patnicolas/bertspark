package org.bertspark.predictor

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalFeedback
import org.bertspark.predictor.S3Evaluator.logger
import org.bertspark.util.io.S3Util
import org.scalatest.flatspec.AnyFlatSpec

private[predictor] final class S3EvaluatorTest extends AnyFlatSpec {

  it should "Succeed selecting sub models" in  {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = S3PathNames.s3ClassifierModelPath
    val subModelSet = S3Util.getS3Keys(mlopsConfiguration.storageConfig.s3Bucket, s3Folder).map(
      path => {
        val relativePath = path.substring(s3Folder.length+1)
        val separatorIndex = relativePath.indexOf("/")
        val subModelName = if(separatorIndex != -1) relativePath.substring(0, separatorIndex) else ""
        subModelName
      }
    ).filter(_.nonEmpty).toSet
    println(s"Evaluation from models loaded ${subModelSet.size} sub-models from $s3Folder")

    val rawFeedbacks = S3Util.s3ToDataset[InternalFeedback](S3PathNames.s3FeedbacksPath).head(5)
    rawFeedbacks.foreach(feedback => println(feedback.context.emrLabel))

  }

  ignore should "Succeed evaluating classifier using sample requests from S3" in {
    import org.bertspark.implicits._

    val s3Evaluator = S3Evaluator(128)
    s3Evaluator.execute
  }

  ignore should "Succeed comparing subModels.csv and modelTaxonomy" in {
    import org.bertspark.config.MlopsConfiguration._

    val s3SubModelFolder = "mlops/CMBS/models/447/subModels.csv"
    val s3SubModelTaxonomyFolder = "mlops/CMBS/models/447/D-447/modelTaxonomy.csv"

    S3Util.downloadCSVFields(mlopsConfiguration.storageConfig.s3Bucket, s3SubModelFolder)
        .map(_.partition(_(1) == "1"))
        .foreach{ case (oracle, preTrained) => println(s"subModels: ${oracle.size},${preTrained.size}")}

    S3Util.downloadCSVFields(mlopsConfiguration.storageConfig.s3Bucket, s3SubModelTaxonomyFolder)
        .map(_.partition(_(1) == "1"))
        .foreach{ case (oracle, preTrained) => println(s"Taxonomy: ${oracle.size},${preTrained.size}")}
  }
}
