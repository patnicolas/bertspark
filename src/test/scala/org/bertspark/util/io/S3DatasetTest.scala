package org.bertspark.util.io

import org.bertspark.nlp.medical.encodePredictReq
import org.scalatest.flatspec.AnyFlatSpec


private[io] final class S3DatasetTest extends AnyFlatSpec {

  it should "Succeed loading content from S3" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "requests/Cornerstone"
    val maxNumRecords = 32

    val storageInfo = SingleS3Dataset(s3Folder, encodePredictReq, maxNumRecords)
    val contextDocumentIterator = storageInfo.getContentIterator

    var count = 1
    while(contextDocumentIterator.hasNext) {
      val contextDocument = contextDocumentIterator.next()
      println(s"#$count ${contextDocument.contextVariables.mkString("\n")}\n${contextDocument.text.substring(0, 48)} ......")
      count += 1
    }
  }
}
