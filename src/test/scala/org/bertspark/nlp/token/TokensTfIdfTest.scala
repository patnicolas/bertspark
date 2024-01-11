package org.bertspark.nlp.token

import org.apache.spark.ml.feature.CountVectorizer
import org.bertspark.nlp.token.TfIdf.{rawFeaturesCol, wordsCol, WeightedToken}
import org.bertspark.nlp.token.TokensTfIdf.processTokens
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.{ContextualDocument, SubModelsTrainingSet}
import org.bertspark.nlp.vocabulary.MedicalTerms.buildFromAMA
import org.bertspark.util.io.{LocalFileUtil, S3IOOps, S3Util}
import org.scalatest.flatspec.AnyFlatSpec

private[token] final class TokensTfIdfTest extends AnyFlatSpec {

  ignore should "Succeed applying TFIDF to content" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    import org.bertspark.config.MlopsConfiguration._

    val target = "CMBS"

    val s3WeightedTokensFolder = s"mlops/$target/models/weightedTokens.csv"
    val s3IOOps = new S3IOOps[WeightedToken](
      mlopsConfiguration.storageConfig.s3Bucket,
      s3WeightedTokensFolder,
      1
    )

    val s3ContextDocumentFolder = s"mlops/$target/contextDocument/AMA"
    val contextDocumentDS = S3Util.s3ToDataset[ContextualDocument](
      mlopsConfiguration.storageConfig.s3Bucket,
      s3ContextDocumentFolder,
      false, "json"
    ).map(contextDoc => (contextDoc.contextVariables ++ contextDoc.text.split(tokenSeparator))).limit(20000)

    val tokensTfIdf = new TokensTfIdf(contextDocumentDS)
    tokensTfIdf(s3IOOps)
  }


  it should "Succeed finding terms distribution per labels" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val countVectorizer = new CountVectorizer().setInputCol(wordsCol).setOutputCol(rawFeaturesCol)
    val rawMedicalTerms = buildFromAMA().filter(_.size > 1).sortWith(_ < _)
    val medicalTerms: Set[String] = rawMedicalTerms.toSet

    val isXnum = medicalTerms.contains("xnum")
    val target = "CMBS"
    val s3TrainingFolder = s"mlops/$target/training/AMA"
    val groupedSubModelTS = S3Util.s3ToDataset[SubModelsTrainingSet](
      s3TrainingFolder,
      false,
      "json")
        .limit(1024)
        .flatMap(_.labeledTrainingData.groupBy(_.label).toSeq)
        .map {
          case (label, ctxDocs) => {
            val tokens = ctxDocs.flatMap(
              ctxDoc => {
                (ctxDoc.contextualDocument.contextVariables ++ ctxDoc.contextualDocument.text.split(tokenSeparator))
                    .filter(token => token.length > 1 && medicalTerms.contains(token))
              }
            )
            (label, tokens)
          }
        }

    val tokenDS = groupedSubModelTS.map(_._2.toArray)
    println(s"Ready to TF_IDF ${tokenDS.count()} tokens")
    val weightedTokens = processTokens(tokenDS, countVectorizer)
    val weightedTokenStr = weightedTokens
        .sortWith(_.weight > _.weight)
        .map{ wt => s"${wt.token}:${wt.weight}"}

    LocalFileUtil.Save.local("output/labelTfIdf.csv", weightedTokenStr.mkString("\n"))
    /*
    val groupedSubModel = groupedSubModelTS.collect()

    val weightedTokens = groupedSubModel.map {
      case (label, seq) =>
        val tokenDS = seq.toDS()
        val weightedTokens = processTokens(tokenDS, countVectorizer)
        val weightedTokenStr = weightedTokens
            .sortWith(_.weight > _.weight)
            .map{ wt => s"${wt.token}:${wt.weight}"}.mkString(" ")
        s"$label,$weightedTokenStr"
    }

     */

  }
}
