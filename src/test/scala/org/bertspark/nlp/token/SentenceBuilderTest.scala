package org.bertspark.nlp.token

import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.nlp.token.SentencesBuilder.{countTokens, str}
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.ContextualDocument
import org.bertspark.nlp.trainingset.ContextualDocumentGroup.segmentSeparator
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

private[token] final class SentenceBuilderTest extends AnyFlatSpec {

  it should "Succeed generate an arbitrary set of segments - CtxTxtNSentencesBuilder" in {
    val contextVariables = Array[String](
      "3_age","f_gender","cornerstone_cust","no_client","mri_modality","22_pos","73221_cpt","26_mod"
    )
    val text = "dob phillip clark ginger withers diag edith jarnigan medicare humana procedure ordered mr mri upper ext joint contrast right rt admitting dx complete rotator cuff tear or rupture of right shoulder not specified as traumatic working dx complete rotator cuff tear or rupture of right shoulder not specified as traumatic mri shoulder history rotator cuff tear technique routine mri of the right shoulder without contrast comparison none result tendons supraspinatus complete full thickness tear with retraction medial to the glenohumeral joint infraspinatus complete full thickness tear with retraction medial to the glenohumeral joint subscapularis complete full thickness tear teres minor intact biceps tendon torn and retracted muscle the severe muscle atrophy of the subscapularis supraspinatus and infraspinatus muscles bones and marrow no fracture or bone marrow replacing process glenohumeral joint including cartilage large areas of full thickness cartilage loss on both sides of the joint with subchondral cyst formation moderate joint effusion with synovitis degeneration and tearing of the labrum acromioclavicular joint mild degenerative changes other no other significant findings impression massive rotator cuff tear with severe tendon retraction and severe muscle atrophy severe rotator cuff arthropathy transcriptionist phillip clark physician radiologist read by phillip clark physician radiologist reviewed and signed by phillip clark physician radiologist released date time"

    val contextualDocument = ContextualDocument("id", contextVariables, text)
    val sentencesBuilder = new CtxTxtNSentencesBuilder
    val segments = sentencesBuilder(contextualDocument)
    println(s"CtxTxtNSentencesBuilder:${countTokens(segments).mkString(" ")}\n${str(segments)}")
    assert(countTokens(segments).sum == contextVariables.size + text.split(tokenSeparator).size)
  }

  ignore should "Succeed generate an arbitrary set of segments - CtxNSentencesBuilder" in {
    val contextVariables = Array[String](
      "3_age","f_gender","cornerstone_cust","no_client","mri_modality","22_pos","73221_cpt","26_mod"
    )
    val text = "dob phillip clark ginger withers diag edith jarnigan medicare humana procedure ordered mr mri upper ext joint contrast right rt admitting dx sectionfindings complete rotator cuff tear or rupture of right shoulder not specified as traumatic working dx complete rotator cuff tear or rupture of right shoulder not specified as traumatic mri shoulder sectionimpression history rotator cuff tear technique routine mri of the right shoulder without contrast comparison none result tendons supraspinatus complete full thickness tear with retraction medial to the glenohumeral joint infraspinatus complete full thickness tear with retraction medial to the glenohumeral joint subscapularis complete full thickness tear teres minor intact biceps tendon torn and retracted muscle the severe muscle atrophy of the subscapularis supraspinatus and infraspinatus muscles bones and marrow no fracture or bone marrow replacing process glenohumeral joint including cartilage large areas of full thickness cartilage loss on both sides of the joint with subchondral cyst formation moderate joint effusion with synovitis degeneration and tearing of the labrum acromioclavicular joint mild degenerative changes other no other significant findings impression massive rotator cuff tear with severe tendon retraction and severe muscle atrophy severe rotator cuff arthropathy transcriptionist phillip clark physician radiologist read by phillip clark physician radiologist reviewed and signed by phillip clark physician radiologist released date time"

    val contextualDocument = ContextualDocument("id", contextVariables, text)
    val sentencesBuilder = new CtxNSentencesBuilder
    val segments = sentencesBuilder(contextualDocument)
    val textLength = text.split(tokenSeparator).length
    val inputSize = contextVariables.length + textLength
    val predictedSize = segments.tail.map(_._2.split(tokenSeparator).length).reduce(_ + _) + segments.head._1.split(tokenSeparator).length + segments.head._2.split(tokenSeparator).length
    println(s"CtxNSentencesBuilder:\n${str(segments)}")
  }

  it should "Succeed generate an arbitrary set of segments - SectionsSentencesBuilder" in {
    val contextVariables = Array[String](
      "3_age","f_gender","cornerstone_cust","no_client","mri_modality","22_pos","73221_cpt","26_mod"
    )
    val text = "dob phillip clark ginger withers diag edith jarnigan medicare humana procedure ordered mr mri upper ext joint contrast right rt admitting dx sectionfindings complete rotator cuff tear or rupture of right shoulder not specified as traumatic working dx complete rotator cuff tear or rupture of right shoulder not specified as traumatic mri shoulder sectionimpression history rotator cuff tear technique routine mri of the right shoulder without contrast comparison none result tendons supraspinatus complete full thickness tear with retraction medial to the glenohumeral joint infraspinatus complete full thickness tear with retraction medial to the glenohumeral joint subscapularis complete full thickness tear teres minor intact biceps tendon torn and retracted muscle the severe muscle atrophy of the subscapularis supraspinatus and infraspinatus muscles bones and marrow no fracture or bone marrow replacing process glenohumeral joint including cartilage large areas of full thickness cartilage loss on both sides of the joint with subchondral cyst formation moderate joint effusion with synovitis degeneration and tearing of the labrum acromioclavicular joint mild degenerative changes other no other significant findings impression massive rotator cuff tear with severe tendon retraction and severe muscle atrophy severe rotator cuff arthropathy transcriptionist phillip clark physician radiologist read by phillip clark physician radiologist reviewed and signed by phillip clark physician radiologist released date time"

    val contextualDocument = ContextualDocument("id", contextVariables, text)
    val sentencesBuilder = new SectionsSentencesBuilder
    val segments = sentencesBuilder(contextualDocument)
    val textLength = text.split(tokenSeparator).length
    val inputSize = contextVariables.length + textLength
    val predictedSize = segments.tail.map(_._2.split(tokenSeparator).length).reduce(_ + _) + segments.head._1.split(tokenSeparator).length + segments.head._2.split(tokenSeparator).length
    println(s"SectionsSentencesBuilder:\n${str(segments)}")
  }

  ignore should "Succeed processing labeled Sentences model" in {
    val contextualDocument = ContextualDocument(
      "id", Array.empty[String],
      SentenceBuilderTest.getContextDocumentText.mkString(s" $segmentSeparator ")
    )
    val sentenceBuilder: SentencesBuilder = new LabeledSentencesBuilder
    val contextSentences = sentenceBuilder(contextualDocument)
    val dump = contextSentences.map{ case (ctx, txt) => s"$ctx - $txt"}.mkString("\n")
    println(dump)
  }

  ignore should "Succeed processing CtxTxtNSentencesBuilder sentences model" in {
    val sentenceBuilder: SentencesBuilder = new CtxTxtNSentencesBuilder
    println(s"CTXTXT-TXT:\n${SentenceBuilderTest.process(sentenceBuilder).mkString("\n")}")
  }
}


private[token] final object SentenceBuilderTest {

  final private val textInput =
    "The primary goal of the project is to implement low latency distributed inference of deep learning models using Spark, Kafka and Amazon open source deep java library. The use case for the project is the prediction of entire medical insurance claim given a set of clinical notes/charts and EMR contextual data."
  private val textInputs = textInput.split(tokenSeparator)

  def getTextTokens: Array[String] = textInputs.slice(Random.nextInt(8), textInputs.size - Random.nextInt(8))
  def getTextFromTokens: String = getTextTokens.mkString(" ")

  def getContextDocumentText: Array[String] =
    (0 until mlopsConfiguration.preTrainConfig.numSentencesPerDoc).map(_ => getTextFromTokens).toArray

  def process(sentencesBuilder: SentencesBuilder): Array[(String, String)] = {
    val contextualDocument = ContextualDocument(
      "id",
      Array[String]("50_age", "f_gender", "amb_cust", "no_client", "78123_cpt", "22_mod", "no_modality"),
      "first second third fourth fifth sixth seventh eigth nineth tenth eleventh"
    )
    sentencesBuilder(contextualDocument)
  }
}
