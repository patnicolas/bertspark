package org.bertspark.nlp.vocabulary

import ai.djl.modality.nlp.bert.WordpieceTokenizer
import org.bertspark.config.MlopsConfiguration.{mlopsConfiguration, vocabulary}
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.util.io.S3Util.s3ToDataset
import org.scalatest.flatspec.AnyFlatSpec


private[vocabulary] final class WordPieceExtractorTest extends AnyFlatSpec {

  ignore should "Succeed extracting word pieces from a sentence" in {
    import org.bertspark.implicits._

    println((0 until 20).map(vocabulary.getToken(_)).mkString(" "))
    val maxNumChars = 2
    val sentence = "DOB: 05/20/1997\\\\r\\\\nHistory Number:970548\\\\r\\\\nEXAM: CT Cervical Spine without IV contrast. Sagittal and coronal reformats were created.\\\\r\\\\nINDICATION: Rollover MVC\\\\r\\\\nCOMPARISON: None available\\\\r\\\\n\\\\r\\\\nAll CT scans at UTMCK utilize one or more of the following dose-reduction techniques: automated exposure control, iterative reconstruction, and/or manual adjustment of tube current and voltage for size.\\\\r\\\\n\\\\r\\\\nFINDINGS:\\\\r\\\\nCervical spine alignment is normal. \\\\r\\\\nThe cervical spine vertebral bodies are of normal height and attenuation. \\\\r\\\\nPrevertebral soft tissues are unremarkable. \\\\r\\\\nNo asymmetric facet joint widening. \\\\r\\\\nNo acute cervical spine fracture. \\\\r\\\\nThe central canal and neural foramina are grossly patent at all levels. \\\\r\\\\nLung apices are unremarkable.\\\\r\\\\n\\\\r\\\\nIMPRESSION:\\\\r\\\\n\\\\r\\\\nNo acute findings.\\\\r\\\\n\\\\r\\\\n-- \\\\r\\\\n \\\\r\\\\n \\\\r\\\\n  I have personally reviewed the image(s) and the resident interpretation and agree with the findings.\\\\r\\\\n\\\\r\\\\nPLAT-50-148\\\\r\\\\n\\\\r\\\\n\\\\r\\\\n.\\\\r\\\\nAuthenticated By: FOX MD, DANIEL R                            06/01/2019 02:04\\\\r\\\\nResident: Zalis MD RES, Adam R\\\\r\\\\n\\\\r\\\\n**FINAL REPORT**"
    val wordPiecesTokenizer = new WordpieceTokenizer(vocabulary, "[UNK]", maxNumChars)
    val tokens: java.util.List[String] = wordPiecesTokenizer.tokenize(sentence.toLowerCase)
    val subTokens: scala.List[String] = tokens
    println(subTokens.mkString(" "))
  }

  it should "Succeed extracting word pieces from a set of tokens" in {
    import org.bertspark.implicits._, sparkSession.implicits._

    val requestDS = s3ToDataset[InternalRequest](
      mlopsConfiguration.storageConfig.s3Bucket,
      S3PathNames.s3RequestsPath,
      header = false,
      fileFormat = "json").limit(512).dropDuplicates("id")

    val wordPieceExtractor = WordPieceExtractor(60, 10)
    val wordPieces = wordPieceExtractor.build(Array.empty[String], requestDS)
    assert(wordPieces.nonEmpty)
    println(s"Word pieces with max char 10:\n${wordPieces.mkString(" ")}")
  }
}
