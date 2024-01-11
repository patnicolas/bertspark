package org.bertspark.util.rdbms

import org.bertspark.util.rdbms.PredictionsTbl.{defaultTransform, orderedAllQuery, orderedRequestQuery}
import org.scalatest.flatspec.AnyFlatSpec


private[rdbms] final class PredictionsTblTest extends AnyFlatSpec {

  it should "Succeed generating S3 folder from prediction table" in {
    import org.bertspark.implicits._

    val postgreSql = PostgreSql()
    val s3Folder = "requestsDB/Cornerstone"
    val condition = "customer='Cornerstone'"

    PredictionsTbl.tblToS3Request(
      postgreSql,
      orderedAllQuery,
      condition,
      s3Folder,
      defaultTransform,
      30
      )
    postgreSql.close
  }
}
