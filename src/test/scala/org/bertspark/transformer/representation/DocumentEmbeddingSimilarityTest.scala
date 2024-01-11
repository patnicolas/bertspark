package org.bertspark.transformer.representation

import ai.djl.ndarray._
import org.bertspark._
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.{ContextualDocument, TokenizedTrainingSet}
import org.bertspark.transformer.representation
import org.bertspark.util.NDUtil
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable.ListBuffer
import scala.util.Random



private[representation] final class DocumentEmbeddingSimilarityTest extends AnyFlatSpec {
  import org.bertspark.transformer.representation.DocumentEmbeddingSimilarity._
  import DocumentSimilarityTest._

  ignore should "Succeed extracting CLS embedding" in {
    val ndManager = NDManager.newBaseManager()

    val embeddings1 = createWordEmbedding(0.6F)
    val embeddings2 = createWordEmbedding(0.1F)

    val ndEmbeddings1: NDArray = ndManager.create(embeddings1)
    val ndEmbeddings2 = ndManager.create(embeddings2)
    val expanded1 = ndEmbeddings1.expandDims(0)
    val expanded2 = ndEmbeddings2.expandDims(0)
    val ndEmbeddingsGroups: NDArray = NDUtil.concat(Array[NDArray](expanded1, expanded2))
    println(s"Input tensor:\n${NDUtil.display(ndEmbeddingsGroups)}")
    val ndClsTarget: NDArray = ndEmbeddingsGroups.get("0:,0,:")
    val shape = ndClsTarget.getShape()

    println(ndEmbeddingsGroups.getShape)
    println(s"CLS target shape:${shape.toString}")
    println(s"Target:\n${NDUtil.display(ndClsTarget)}")
    ndManager.close()
  }

  it should "Succeed computing similarity for identical notes - random negative sampling" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val sameEntries: Seq[(String, Seq[TokenizedTrainingSet])] =
      generateTokenizedIndexedSet(
        "birth of date history number exam tomography head without intravenous contrast indication dizziness ataxia transient visual technique helical tomography of the head was obtained without intravenous contrast axial coronal and sagittal images were created dose reduction technique was used including one or more of the following automated exposure control adjustment of ma and kv according to patient size and or iterative reconstruction comparison april findings there are changes of advancing chronologic age including mild ventriculomegaly and moderate sulcal and cisternal prominence the sulcal prominence is generalized without specific regional localization there are mild white matter changes consistent with chronic ischemic demyelination",
        "randomNeg")

    val ndManager = NDManager.newBaseManager()
    val modelSimilarity = computeSimilarity(ndManager, sameEntries.toDS(), -1)

    println(s"Similarity within labels: ${modelSimilarity.meanWithinLabels}")
    println(s"Similarity across labels: ${modelSimilarity.meanAcrossLabels}")
    ndManager.close()
  }

  it should "Succeed computing similarity for identical notes - default negative sampling" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val sameEntries: Seq[(String, Seq[TokenizedTrainingSet])] =
      generateTokenizedIndexedSet(
        "birth of date history number exam tomography head without intravenous contrast indication dizziness ataxia transient visual technique helical tomography of the head was obtained without intravenous contrast axial coronal and sagittal images were created dose reduction technique was used including one or more of the following automated exposure control adjustment of ma and kv according to patient size and or iterative reconstruction comparison april findings there are changes of advancing chronologic age including mild ventriculomegaly and moderate sulcal and cisternal prominence the sulcal prominence is generalized without specific regional localization there are mild white matter changes consistent with chronic ischemic demyelination",
        "unkNeg")

    val ndManager = NDManager.newBaseManager()
    val modelSimilarity = computeSimilarity(ndManager, sameEntries.toDS(), -1)

    println(s"Similarity within labels: ${modelSimilarity.meanWithinLabels}")
    println(s"Similarity across labels: ${modelSimilarity.meanAcrossLabels}")
    ndManager.close()
  }


  ignore should "Succeed compute the cosine of two vectors" in {
    import DocumentSimilarityTest._

    val ndManager = NDManager.newBaseManager()

    val vectorSize = mlopsConfiguration.getEmbeddingsSize.toInt
    val x = Array.tabulate(vectorSize)(n => Math.sqrt(n+1).toFloat)
    val ndX = ndManager.create(x)
    val selfSimilarity = NDUtil.cosine(ndX, ndX)
    val _selfSimilarity = representation.cosine(x, x)
    println(s"Self cosine: $selfSimilarity, ${_selfSimilarity}, ${NDUtil.computeSimilarity(ndX, ndX, "cosine")}")

    val y = x.map(_ * -2.0F)
    val ndY = ndManager.create(y)
    val similarity2 = NDUtil.cosine(ndX, ndY)
    val _similarity2 = representation.cosine(x, y)
    println(s"Scaled cosine: $similarity2, ${_similarity2}, ${NDUtil.computeSimilarity(ndX, ndY, "cosine")}")

    val z = x.reverse
    val similarity3 = NDUtil.cosine(ndX, ndManager.create(z))
    val _similarity3 = representation.cosine(x, z)
    println(s"Reversed cosine $similarity3, ${_similarity3}")

    val t = x.map(- _)
    val similarity5 = NDUtil.cosine(ndManager.create(x), ndManager.create(t))
    val _similarity5 = representation.cosine(x, t)
    println(s"Negative cosine: $similarity5, ${_similarity5}")
    ndManager.close()
  }

  ignore should "Succeed computing similarity in CLS predictions" in {
    import org.bertspark.config.MlopsConfiguration._
    import org.bertspark.implicits._

    val ndManager = NDManager.newBaseManager()
    val s3TrainingSetFolder = s"${mlopsConfiguration.storageConfig.s3RootFolder}/${mlopsConfiguration.target}/training1"
    val modelSimilarity = computeSimilarity(ndManager, s3TrainingSetFolder, -1)
    println(modelSimilarity.toString)
    ndManager.close()
  }
}



private[representation] final object DocumentSimilarityTest {

  def generateTokenizedIndexedSet(
    text: String,
    negSampleType: String): Seq[(String, Seq[TokenizedTrainingSet])] = {
    val (tokenizedTraining1, tokenizedTraining2) = generateSimilarTokenizedIndexedSet(text)
    val (tokenizedTraining3, tokenizedTraining4) = negSampleType match {
      case "randomNeg" => createNegRandomTokenizedIndexedSet(tokenizedTraining1)
      case "defaultNeg" =>  generateNegFixedTokenizedIndexedSet(tokenizedTraining1)
      case _ => generateNegUnkTokenizedIndexedSet(tokenizedTraining1)
    }

    Seq[(String, Seq[TokenizedTrainingSet])](
      ("76700 26 GC", Seq[TokenizedTrainingSet](
        tokenizedTraining1,
        tokenizedTraining2,
        tokenizedTraining3,
        tokenizedTraining4))
    )
  }

  private def generateSimilarTokenizedIndexedSet(text: String): (TokenizedTrainingSet, TokenizedTrainingSet) = {
    val contextualDocument1 = ContextualDocument(
      "id1",
      Array[String]("3_age","m_gender","cornerstone_cust","no_client","ct_modality","23_pos","70450_cpt","26_mod"),
      text
    )
    val contextualDocument2 = contextualDocument1.copy(id = "id2")
    val tokenizedTrainingSet1 = TokenizedTrainingSet(
      contextualDocument1,
      "R10.13",
      Array.empty[Float]
    )
    val tokenizedTrainingSet2 = TokenizedTrainingSet(
      contextualDocument2,
      "R10.13",
      Array.empty[Float]
    )
    (tokenizedTrainingSet1, tokenizedTrainingSet2)
  }


  private def createNegRandomTokenizedIndexedSet(
    positiveTokenizedTrainingSet: TokenizedTrainingSet
  ): (TokenizedTrainingSet, TokenizedTrainingSet)= {
    val vocabularySampleSize = 2048

    val textTokens = positiveTokenizedTrainingSet.contextualDocument.text.split(tokenSeparator)
    val negativeText = {
      var count = 0
      val negTokens = ListBuffer[String]()
      do {
        val token = vocabulary.getToken(Random.nextInt(vocabularySampleSize))
        if(!textTokens.contains(token)) {
          count += 1
          negTokens.append(token)
        }
      } while (count < textTokens.length+12)
      negTokens.mkString(" ")
    }

    val contextTokens = positiveTokenizedTrainingSet.contextualDocument.contextVariables
    val negativeContext =  {
      var count = 0
      val negTokens = ListBuffer[String]()
      do {
        val token = vocabulary.getToken(Random.nextInt(vocabularySampleSize))
        if(!contextTokens.contains(token)) {
          count += 1
          negTokens.append(token)
        }
      } while (count < contextTokens.length)
      negTokens.toArray
    }

    val contextualDocument1 = ContextualDocument("id1", negativeContext, negativeText)
    val contextualDocument2 = ContextualDocument("id2", negativeContext, negativeText)

    val tokenizedTrainingSet1 = TokenizedTrainingSet(
      contextualDocument1,
      "R11.02",
      Array.empty[Float]
    )
    val tokenizedTrainingSet2 = TokenizedTrainingSet(
      contextualDocument2,
      "R11.02",
      Array.empty[Float]
    )
    (tokenizedTrainingSet1, tokenizedTrainingSet2)
  }

  private def generateNegFixedTokenizedIndexedSet(
    positiveTokenizedTrainindSet: TokenizedTrainingSet
  ): (TokenizedTrainingSet, TokenizedTrainingSet)= {
    val vocabularySampleSize = 2048

    val textTokens = positiveTokenizedTrainindSet.contextualDocument.text.split(tokenSeparator)
    val negativeText = {
      var fixedNewToken: String = ""
      do {
        val token = vocabulary.getToken(Random.nextInt(vocabularySampleSize))
        if(!textTokens.contains(token))
          fixedNewToken = token
      } while (fixedNewToken.isEmpty)
      Seq.fill(64)(fixedNewToken).mkString(" ")
    }

    val contextTokens = positiveTokenizedTrainindSet.contextualDocument.contextVariables
    val negativeContext =  {
      var fixedContextualToken: String = ""
      do {
        val token = vocabulary.getToken(Random.nextInt(vocabularySampleSize))
        if(!contextTokens.contains(token))
          fixedContextualToken = token
      } while (fixedContextualToken.isEmpty)
      Array.fill(64)(fixedContextualToken)
    }

    val contextualDocument1 = ContextualDocument("id1", negativeContext, negativeText)
    val contextualDocument2 = ContextualDocument("id2", negativeContext, negativeText)

    val tokenizedTrainingSet1 = TokenizedTrainingSet(
      contextualDocument1,
      "R11.02",
      Array.empty[Float]
    )
    val tokenizedTrainingSet2 = TokenizedTrainingSet(
      contextualDocument2,
      "R11.02",
      Array.empty[Float]
    )
    (tokenizedTrainingSet1, tokenizedTrainingSet2)
  }


  private def generateNegUnkTokenizedIndexedSet(
    positiveTokenizedTrainindSet: TokenizedTrainingSet
  ): (TokenizedTrainingSet, TokenizedTrainingSet)= {
    val fixedNewToken = "AAAAAA"
    val textTokens = positiveTokenizedTrainindSet.contextualDocument.text.split(tokenSeparator)
    val negativeText = Seq.fill(64)(fixedNewToken).mkString(" ")
    val negativeContext = Array.fill(8)(fixedNewToken)

    val contextualDocument1 = ContextualDocument("id1", negativeContext, negativeText)
    val contextualDocument2 = ContextualDocument("id2", negativeContext, negativeText)

    val tokenizedTrainingSet1 = TokenizedTrainingSet(
      contextualDocument1,
      "R11.02",
      Array.empty[Float]
    )
    val tokenizedTrainingSet2 = TokenizedTrainingSet(
      contextualDocument2,
      "R11.02",
      Array.empty[Float]
    )
    (tokenizedTrainingSet1, tokenizedTrainingSet2)
  }

  def createWordEmbedding(bias: Float): Array[Array[Float]] = {
    val embedding1 = Array.fill(8)(1.0F+bias)
    val embedding2 = Array.fill(8)(1.7F+bias)
    val embedding3 = Array.fill(8)(0.3F+bias)
    Array[Array[Float]](
      embedding1, embedding2, embedding3
    )
  }
}
