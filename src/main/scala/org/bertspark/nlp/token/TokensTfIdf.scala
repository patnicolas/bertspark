/**
 * Copyright 2022,2023 Patrick R. Nicolas. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package org.bertspark.nlp.token

import org.apache.spark.ml.feature._
import org.apache.spark.sql._
import org.bertspark.nlp.token.TfIdf.{featuresCol, rawFeaturesCol, wordsCol, WeightedToken}
import org.bertspark.util.io._
import org.bertspark.config.MlopsConfiguration.DebugLog.{logDebug, logInfo}
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.medical.MedicalCodingTypes.InternalRequest
import org.bertspark.nlp.medical.NoteProcessors.{cleanse, specialCharCleanserRegex}
import org.bertspark.nlp.vocabulary.CodingTermsTfIdf.abbrStopTokens
import org.bertspark.nlp.vocabulary.MedicalVocabulary
import org.slf4j._
import scala.collection.mutable.HashMap


/**
 * Generic TF-IDF computation of terms extracted from a Corpus
 * @param corpusTermsDS Dataset of terms extracted from a Corpus
 * @param sparkSession Implicit reference to the current Spark context
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class TokensTfIdf(
  corpusTermsDS: Dataset[Array[String]]
)(implicit sparkSession: SparkSession) extends TfIdf[TokensTfIdf] {
  import TokensTfIdf._

  override def apply(ioOps: IOOps[WeightedToken]): Array[WeightedToken] = {
    val weightedTokens = processTokens(corpusTermsDS, countVectorizer).sortWith(_.token < _.token)
    ioOps.save(weightedTokens)
    weightedTokens
  }
}




private[bertspark] final object TokensTfIdf {
  final private val logger: Logger = LoggerFactory.getLogger("TokensTfIdf")

  /**
   * Constructor for TfIdf algorithm extraction of terms loaded using S3 data set and compliant with
   * the current custom vocabulary
   * @param s3IO Operator on S3 folder
   * @param mVocabulary Current vocabulary
   * @param sparkSession Implicit reference to the current Spark context
   * @return Instance of TfIdf
   */
  def apply(
    s3IO: S3IOOps[String],
    mVocabulary: MedicalVocabulary)(implicit sparkSession: SparkSession): TokensTfIdf = {
    import sparkSession.implicits._

    val tokens = s3IO.loadDS.map(getTerms(_, mVocabulary))
    new TokensTfIdf(tokens)
  }


  /**
   * Constructor for TfIdf algorithm extraction of terms loaded using S3 data set and compliant with
   * the current custom vocabulary
   * @param s3IO Operator on S3 folder
   * @param extractContent Extract content from the type T
   * @param sparkSession Implicit reference to the current Spark context
   * @return Instance of TfIdf
   */
  def apply[T](
    s3IO: S3IOOps[T],
    extractContent: T => String,
    maxNumRecords: Int = -1)(implicit sparkSession: SparkSession): TokensTfIdf = {
    import sparkSession.implicits._
    val tokens =
      if(maxNumRecords > 0) s3IO.loadDS.limit(maxNumRecords).map(t => getTerms(extractContent(t)))
      else s3IO.loadDS.map(t => getTerms(extractContent(t)))

    logDebug(logger, s"TF-IDF ${tokens.count()} tokens")
    new TokensTfIdf(tokens)
  }

  /**
   * Extract the weighted token from a corpus of tokens complying with a given vocabulary
   * @param corpusTermsDS Date set of terms extracted from a Corpus
   * @param countVectorizer Count vectorizer
   * @param sparkSession Implicit reference to the current Spark context
   * @return Sequence of weighted token from the vocabulary
   */
  def processTokens(
    corpusTermsDS: Dataset[Array[String]],
    countVectorizer: CountVectorizer
  )(implicit sparkSession: SparkSession): Array[WeightedToken] = {
    // Step 3: Compute TF-IDF for all the medical terms found in the corpus or training set
    val (tfIdfDataFrame, weightedEntities) = computeTfIdf(corpusTermsDS, countVectorizer)
    logInfo(logger,  "MO: Completed computation Tf-Idf weights")

    // Step 4: Reduce the vocabulary to numFeatures by ordering the words by decreasing order of TF-IDF weights
    val weightedVocabulary = reduceVocabulary(tfIdfDataFrame, weightedEntities)
    logInfo(logger,  s"MO: Completed reduction ${weightedVocabulary.size} weighted entities")
    weightedVocabulary
  }


  def extractTermsFrequencies(args: Seq[String]): Unit = {
    require(args.size == 4, s"${args.mkString(" ")} should be 'createTfIdf numRecords outputFileName type'")

    args(3) match {
      case "Tf" => extractTermsTf(args)
      case "TfIdf" => extractTermsTfIdf(args)
      case _ => extractTermsTf(args)
    }
  }

  /*
    // $example on$
    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat")
    )).toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    val featurizedData = hashingTF.transform(wordsData)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show()
    // $example off$

   */
  case class LabelTokens(label: String, tokens: Array[String])

  def getTfIdf(documentEntitiesDS: Dataset[LabelTokens]): DataFrame = {
    val hashingTF = new HashingTF().setInputCol("tokens").setOutputCol("rawFeatures")
    val featurizedData = hashingTF.transform(documentEntitiesDS)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledDataDF = idfModel.transform(featurizedData)
    rescaledDataDF.show()
    val resultDF = rescaledDataDF.select("tokens", "features")
    resultDF
  }

  /**
   * Extract a data frame of terms with tf-idf weigths and associated vocabulary
   * @param documentEntitiesDS Dataset of documents as collection of relevant medical terms
   * @param countVectorizer Term count vectorizer
   * @return Pair of data frame of terms with tf-idf weights and associated vocabulary
   */
  private def computeTfIdf(
    documentEntitiesDS: Dataset[Array[String]],
    countVectorizer: CountVectorizer
  ): (DataFrame, Array[String]) = {
    require(documentEntitiesDS.count() > 0L, "Failed to compute TF-IDF for undefined document")

    // Add words column ot the data set
    val documentDF = documentEntitiesDS.toDF(wordsCol)
    // Generate a data frame for TF
    val countVectorizerModel = countVectorizer.fit(documentDF)
    val tfDataFrame = countVectorizerModel.transform((documentDF))
    tfDataFrame.cache()

    // Initialize the Inverse Document frequency model
    val idf = new IDF().setInputCol(rawFeaturesCol).setOutputCol(featuresCol)
    val idfModel = idf.fit(tfDataFrame)
    // Compute the IDF factors
    val tfIdfTermsDataFrame = idfModel.transform(tfDataFrame)
    (tfIdfTermsDataFrame, countVectorizerModel.vocabulary)
  }



  /**
   * Reduce the initial vocabulary of relevant medical terms using their TF-IDF weights. Only the top numFeatures
   * terms with the highest TF-IDF weights are selected as the final set of relevant medical terms
   * @param tfIdfTermsDataFrame Medical terms with TfIdf weights
   * @param vocabulary Initial vocabulary of medical terms
   * @param sparkSession Implicit reference to the current Spark context
   * @return Sequence of Weighted relevant medical terms
   */
  private def reduceVocabulary(
    tfIdfTermsDataFrame: DataFrame,
    vocabulary: Array[String]
  )(implicit sparkSession: SparkSession): Array[WeightedToken] = {
    import org.apache.spark.ml.linalg.Vector

    require(vocabulary.nonEmpty, "Cannot reduce empty vocabulary")
    import sparkSession.implicits._

    val featureDF = tfIdfTermsDataFrame.select(featuresCol)
    val docWeightDS: Dataset[Array[Float]] = featureDF.map(
      feature => {
        val vec = feature.getAs[Vector](0)
        vec.toArray.map(_.toFloat)
      }
    )

    val docWeightsIterator = docWeightDS.toLocalIterator()
    val acc = new HashMap[Int, Float]()
    var index = 0
    while( docWeightsIterator.hasNext) {
      val docWeight = docWeightsIterator.next()
      if(index < docWeight.length)
        acc.put(index, docWeight(index))
      else
        logger.error(s"Index $index is out of bounds")
      index += 1
    }
    logDebug(logger, s"Document weighted completed")
    acc.map { case (idx, w) => WeightedToken(vocabulary(idx), w) }.toArray
  }


  private def getTerms(text: String, mVocabulary: MedicalVocabulary): Array[String] =
    text.split("\\W+").map(_.toLowerCase).filter(word => word.length > 2 && mVocabulary.contains(word))

  private def getTerms(text: String, minTermsLength: Int = 2): Array[String] =
    cleanse(text,  specialCharCleanserRegex)
        .filter(!abbrStopTokens.contains(_))
        .map(_.toLowerCase)
        .filter(_.length > minTermsLength)


  def extractTermsTf(args: Seq[String]): Unit = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val maxNumRecords = args(1).toInt
    val outputFileName = args(2)
    val ds = S3Util.s3ToDataset[InternalRequest](
      S3PathNames.s3RequestsPath,
      false,
      "json"
    ).limit(maxNumRecords).map(_.notes.head)

    val collectedTermsDS = ds.flatMap(note => getTerms(note, 2).map((_, 1)))
    val weightedTermsDS = collectedTermsDS.groupByKey(_._1).reduceGroups(((kn1: (String, Int), kn2: (String, Int)) => (kn1._1, kn1._2 + kn2._2))).map(_._2)
    val rankedWeightedTerms = weightedTermsDS.collect.sortWith(_._2 > _._2).map{
      case (term, w) => s"$term,$w"
    }

    LocalFileUtil.Save.local(outputFileName, rankedWeightedTerms.mkString("\n"))
  }


  def extractTermsTfIdf(args: Seq[String]): Unit = {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val numOfRequests = args(1).toInt
    val outputFileName = args(2)
    val s3IO = new S3IOOps[InternalRequest](
      mlopsConfiguration.storageConfig.s3Bucket,
      s"${mlopsConfiguration.storageConfig.s3RequestFolder}/${mlopsConfiguration.target}"
    )

    val tokenTfIdf = TokensTfIdf(s3IO, (req: InternalRequest) => req.notes.head, numOfRequests)
    val outputS3IO = new S3IOOps[WeightedToken](
      mlopsConfiguration.storageConfig.s3Bucket,
      s"${mlopsConfiguration.storageConfig.s3RootFolder}/${mlopsConfiguration.target}/tfidf2"
    )
    val weightedTokens = tokenTfIdf(outputS3IO)
    val topWeightedTokens = weightedTokens.sortWith(_.weight > _.weight)
    logDebug(
      logger,
      s"Number of top weighted tokens: ${topWeightedTokens.size}\n${topWeightedTokens.take(20).mkString(" ")}"
    )
    LocalFileUtil.Save.local(outputFileName, topWeightedTokens.mkString("\n"))
  }
}
