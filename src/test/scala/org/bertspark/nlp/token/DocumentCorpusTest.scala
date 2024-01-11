package org.bertspark.nlp.token

import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.nlp.medical.encodePredictReq
import org.bertspark.util.io.SingleS3Dataset
import org.scalatest.flatspec.AnyFlatSpec

private[bertspark] final class DocumentCorpusTest extends AnyFlatSpec {

  it should "Succeed Extracting corpus vocabulary from S3 storage" in  {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "requests/Cornerstone-mini"
    val maxNumRecords = 32

    val storage = SingleS3Dataset[InternalRequest](s3Folder, encodePredictReq, maxNumRecords)
    val domainCorpus = DocumentCorpus[ExtBERTTokenizer, CtxTxtNSentencesBuilder](
      storage,
      ExtBERTTokenizer(),
      new CtxTxtNSentencesBuilder)
    println(domainCorpus.toString)
  }


  it should "Succeed Extracting corpus vocabulary from storage" in  {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    val s3Folder = "requests/Cornerstone-mini"
    val maxNumRecords = 32

    val storage = SingleS3Dataset[InternalRequest](s3Folder, encodePredictReq, maxNumRecords)

    val domainCorpus = DocumentCorpus[ExtBERTTokenizer, CtxTxtNSentencesBuilder](
      storage,
      ExtBERTTokenizer(),
      new CtxTxtNSentencesBuilder)
    println(domainCorpus.toString)
  }
}
