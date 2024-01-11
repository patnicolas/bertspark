package org.bertspark.nlp.vocabulary

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.util.io.S3Util.s3ToDataset
import org.scalatest.flatspec.AnyFlatSpec

private[vocabulary] final class CodingTermsTfIdfTest extends AnyFlatSpec {

  it should "Succeed extracting TF-IDF scored and ranked coding terms" in {
    import org.bertspark.implicits._, sparkSession.implicits._

    val requestDS = s3ToDataset[InternalRequest](
      mlopsConfiguration.storageConfig.s3Bucket,
      S3PathNames.s3RequestsPath,
      header = false,
      fileFormat = "json").limit(512).dropDuplicates("id")

    val codeTermsTfIdf = CodingTermsTfIdf(0.6, "emr",  30)
    val output = codeTermsTfIdf.build(Array.empty[String], requestDS)
    assert(output.size > 0)
    println(output.take(10).mkString(" "))
  }
}
