package org.bertspark.analytics

import org.bertspark.analytics.PredictionAnalysis.analyzeNoteSection
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.medical.MedicalCodingTypes.{lineItemSeparator, InternalFeedback}
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.bertspark.nlp.vocabulary.ContextVocabulary
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.bertspark.util.SparkUtil
import org.scalatest.flatspec.AnyFlatSpec


private[bertspark] final class PredictionAnalysisTest extends AnyFlatSpec {


  it should "Succeed analyzing unbalanced labels" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "mlops/XLARGE/training/TF92"
    val targetLabel = "72156 26"
    val trainingDS = S3Util.s3ToDataset[SubModelsTrainingSet](s3Folder, false, "json")
        .filter(_.subModel == targetLabel)
    val firstIcd = "M47.892"
    val secondIcd = "G95.89"
    val thirdIcd = "M50.322"

    val firstIcdCount = trainingDS.map(
      trainData =>
        s"$firstIcd: ${trainData.labeledTrainingData.filter(_.label == s"$targetLabel $firstIcd").size}"
    ).collect

    val secondIcdCount = trainingDS.map(
      trainData =>
        s"$secondIcd: ${trainData.labeledTrainingData.filter(_.label == s"$targetLabel $secondIcd").size}"
    ).collect

    val thirdIcdCount = trainingDS.map(
      trainData =>
        s"$thirdIcd: ${trainData.labeledTrainingData.filter(_.label == s"$targetLabel $thirdIcd").size}"
    ).collect

    println(s"Count: ${firstIcdCount.mkString(" ")}\n${secondIcdCount.mkString(" ")}, ${thirdIcdCount.mkString(" ")}")
  }


  ignore should "Succeed analyzing correlation between primary ICD terminating digit with modifier" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "mlops/XLARGE/training/TF92"
    val trainingDS = S3Util.s3ToDataset[SubModelsTrainingSet](s3Folder, false, "json")

    val trainDataSubModel12DS = trainingDS.filter( trainData => {
      val label = trainData.labeledTrainingData.head.label
      val fields = label.split(tokenSeparator)
      (1 until fields.size).exists(
        index => {
          val lastLabelChar: Char = fields(index).toCharArray.last
          lastLabelChar == '1' || lastLabelChar == '2'
        }
      )
    }).map(trainData => s"${trainData.subModel}: ${trainData.labeledTrainingData.map(_.label).mkString(" - ").distinct}")

    LocalFileUtil.Save.local("output/subModel12.txt", trainDataSubModel12DS.collect.mkString("\n"))
  }



  ignore should "Succeed extracting sections" in {
    val s3Folder = "requestsProd/CMBS"
    val maxNumRequests = 128
    analyzeNoteSection(s3Folder, maxNumRequests)
  }

  ignore should "Succeed extracting Oracle for a given dataset" in {
    import org.bertspark.config.MlopsConfiguration._
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3FeedbackFolder = S3PathNames.s3FeedbacksPath
    val rawFeedbackDS = S3Util.s3ToDataset[InternalFeedback](
      s3FeedbackFolder,
      false,
      "json"
      ).dropDuplicates("id").persist()

    val distinctEMRDS = rawFeedbackDS.map(_.context.emrLabel).distinct
    println(s"${distinctEMRDS.count} distinct EMRs")

    val internalFeedbackDS = rawFeedbackDS.map(
          feedback => (
              feedback.context.emrLabel,
              List[String](feedback.finalized.lineItems.map(_.lineItemSpace).mkString(lineItemSeparator))
          )
        ).persist()
    internalFeedbackDS.show()

    val groupInternalFeedbackDS = SparkUtil.groupByKey[(String, List[String]), String](
      (s: (String, List[String])) => s._1,
      (s1: (String, List[String]), s2:(String, List[String])) => (s1._1, s1._2 ::: s2._2),
      internalFeedbackDS
    )
    val oracleFeedbackRDD = groupInternalFeedbackDS
        .map(_._2)
        .filter{ case (_, xs) => xs.size == 1}
    println(s"Num of oracle feedback groups: ${oracleFeedbackRDD.count}")

    val oracleRDD = oracleFeedbackRDD.map{ case (emr, xs) => s"$emr,${xs.head}"}
    println(s"${oracleRDD.count()} Oracles for ${groupInternalFeedbackDS.count()} sub-models and ${internalFeedbackDS.count()} unique notes")
    oracleRDD.toDS().show()

    S3Util.upload(
      mlopsConfiguration.storageConfig.s3Bucket,
      s"${mlopsConfiguration.storageConfig.s3RootFolder}/${mlopsConfiguration.target}/oracle.csv",
      oracleRDD.collect.sortWith(_ < _).mkString("\n")
    )
  }
}
