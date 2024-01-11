package org.bertspark.nlp.trainingset

import org.bertspark.config.S3PathNames
import org.bertspark.delay
import org.bertspark.nlp.tokenSeparator
import org.bertspark.util.io.S3Util
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

private[trainingset] final class TrainingSetAugmentationTest extends AnyFlatSpec{

  ignore should "Succeed generating table for expected number of permutation"   in {
    val inv_log_2 = 1.0F/Math.log(2)
    val values = (1 until 128).map(n => (n, (Math.log(n)*inv_log_2).ceil.toInt)).toMap
    println(values.mkString(", "))
  }

  it should "Succeed augmenting an contextual document" in {
    val rand = new Random(42L)
    val id = "aaa"
    val text = "kearney regional medical marie cho ##quette birth fracture date xnum xnum xnum referring was nov doctor allen john"
    val ctxVariables = Array[String]("4_age","f_gender","amb_cust","no_client","tempcodes_modality","22_pos","G9500_cpt","no_mod")

    val textTokens = text.split(tokenSeparator)

    var n = rand.nextInt(ctxVariables.size + textTokens.size-1)
    if(n < ctxVariables.size)
      ctxVariables(n) = "[UNK]"
    else
      textTokens(n-ctxVariables.size) = "[UNK]"
    println(ctxVariables.mkString(" "))
    println(textTokens.mkString(" "))

    n = rand.nextInt(ctxVariables.size + textTokens.size-1)
    if(n < ctxVariables.size)
      ctxVariables(n) = "[UNK]"
    else
      textTokens(n-ctxVariables.size) = "[UNK]"
    println(ctxVariables.mkString(" "))
    println(textTokens.mkString(" "))
  }




  ignore should "Succeed extending a set of sub model taxonomy records" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val initialSubModelTrainingSetDS = S3Util.s3ToDataset[SubModelsTrainingSet](
      S3PathNames.s3ModelTrainingPath,
      header = false,
      fileFormat = "json")
        .map(subModelsTS => {
          val correctedLabeledTrainingSet = subModelsTS
              .labeledTrainingData
              .map(ts => ts.copy(label = ts.label.replace("  ", " ")))
          subModelsTS.copy(subModel = subModelsTS.subModel.trim, labeledTrainingData = correctedLabeledTrainingSet)
        })
        .limit(10)



    S3Util.datasetToS3[SubModelsTrainingSet](
      initialSubModelTrainingSetDS,
      s3OutputPath = "mlops/XLARGE3/training/TF93-A",
      header = false,
      fileFormat = "json",
      toAppend = false,
      numPartitions = 4
    )
    delay(timeInMillis = 2000L)
  }

  ignore should "Succeed updating an array" in {
    val init = Array[Int](1, 0, 0, 1)
    val output = init.mkString
    println(output)
    println(output.toCharArray.map(_.toString).mkString)
  }
}
