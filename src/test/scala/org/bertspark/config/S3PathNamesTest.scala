package org.bertspark.config

import org.bertspark.config.S3PathNames._
import org.scalatest.flatspec.AnyFlatSpec

private[config] final class S3PathNamesTest extends AnyFlatSpec {

  it should "Succeed generating paths for S3 folders" in {
    import MlopsConfiguration._
    val vocabularyType1 = mlopsConfiguration.preProcessConfig.vocabularyType
    val target = mlopsConfiguration.target
    val folder1 = s"mlops/$target/vocabulary/$vocabularyType1"
    println(s3VocabularyPath)
    assert(s3VocabularyPath == folder1)

    val vocabularyType2 = "xyz"
    val folder2 = s"mlops/$target/vocabulary/xyz"
    println(getVocabularyS3Path(vocabularyType2))
    assert(getVocabularyS3Path(vocabularyType2) == folder2)

    val folder3 = s"mlops/${mlopsConfiguration.target}/training/$vocabularyType1"
    println(s3ModelTrainingPath)
    assert(s3ModelTrainingPath == folder3)

    val folder4 = s"mlops/${mlopsConfiguration.target}/training/$vocabularyType2"
    println(getS3ModelTrainingPath(vocabularyType2))
    assert(getS3ModelTrainingPath(vocabularyType2) == folder4)

    val folder5 = s"mlops/${mlopsConfiguration.target}/contextDocument/$vocabularyType1"
    println(s3ContextualDocumentPath)
    assert(s3ContextualDocumentPath == folder5)

    val folder6 = s"mlops/${mlopsConfiguration.target}/contextDocument/$vocabularyType2"
    println(getS3ContextualDocumentPath(vocabularyType2))
    assert(getS3ContextualDocumentPath(vocabularyType2) == folder6)

    val folder7 = s"mlops/${mlopsConfiguration.target}/contextDocument/cluster$vocabularyType1"
    println(s3ContextualDocumentGroupPath)
    assert(s3ContextualDocumentGroupPath == folder7)

    val folder8 = s"mlops/${mlopsConfiguration.target}/contextDocument/cluster$vocabularyType2"
    println(getS3ContextualDocumentGroupPath(vocabularyType2))
    assert(getS3ContextualDocumentGroupPath(vocabularyType2) == folder8)
  }

}
