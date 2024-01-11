package org.bertspark.nlp.trainingset

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.ContextualDocumentGroup.segmentSeparator
import org.bertspark.nlp.trainingset.ContextualDocumentClusterTest.{getContextDocumentText, getTextFromTokens}
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

private[trainingset] final class ContextualDocumentClusterTest extends AnyFlatSpec {

  it should "Succeed extracting contextual documents from a document cluster - 1" in {
    import ContextualDocumentGroup._

    val texts = Array.fill(3)(getTextFromTokens)
    val contextualDocument = ContextualDocument("id", Array.empty[String], texts.mkString(s" $segmentSeparator "))
    val contextualDocuments: Array[ContextualDocument] = contextualDocument
    assert(contextualDocuments.size == 3)
    println(contextualDocuments.map(_.text).mkString("\n\n"))
  }

  it should "Succeed extracting contextual documents from a document cluster - 2" in {
    import ContextualDocumentGroup._

    val texts = getContextDocumentText(2)
    val contextualDocument = ContextualDocument("id", Array.empty[String], texts.mkString(s" $segmentSeparator "))
    val contextualDocuments: Array[ContextualDocument] = contextualDocument
    assert(contextualDocuments.size == 42)
    println(contextualDocuments.map(_.text).mkString("\n\n"))
  }
}


private[trainingset] final object ContextualDocumentClusterTest {

  final private val textInput =
    "The primary goal of the project is to implement low latency distributed inference of deep learning models using Spark, Kafka and Amazon open source deep java library. The use case for the project is the prediction of entire medical insurance claim given a set of clinical notes/charts and EMR contextual data."
  private val textInputs = textInput.split(tokenSeparator)

  def getTextTokens: Array[String] = textInputs.slice(Random.nextInt(8), textInputs.size - Random.nextInt(8))
  def getTextFromTokens: String = getTextTokens.mkString(" ")

  def getContextDocumentText(numGroups:Int): Array[String] = {
    val limit = numGroups*mlopsConfiguration.preTrainConfig.numSentencesPerDoc
    (0 until limit).map(_ => getTextFromTokens).toArray ++ Array[String](getTextFromTokens, getTextFromTokens)
  }

}
