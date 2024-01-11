package org.bertspark.nlp.token

import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.ContextualDocument
import org.scalatest.flatspec.AnyFlatSpec

private[bertspark] final class DomainTokenizerTest extends AnyFlatSpec{

  ignore should "Succeed applying a simple Tokenizing a given text without vocabulary" in {
    val input = "This is not the correct answer. Paris is actually the capital of France but not Italy"

    val domainTokenizer = DomainTokenizer[ExtBERTTokenizer, CtxTxtNSentencesBuilder](
      ExtBERTTokenizer(),
      new CtxTxtNSentencesBuilder
    )
    val contextualDocument = ContextualDocument("id", Array[String]("no_cust", "mri"), input)
    val documentComponents = domainTokenizer(contextualDocument)
    println(s"Tokens from simple domainTokenizer: ${documentComponents.getTokens.mkString(" ")}")
  }

  ignore should "Succeed applying a Bert a given text" in {
    val input = "This is not the correct answer. Paris is actually the capital of France but not Italy"

    val tokenizer2 = DomainTokenizer[ExtBERTTokenizer, CtxTxtNSentencesBuilder](
      ExtBERTTokenizer(),
      new CtxTxtNSentencesBuilder)
    val contextualDocument = ContextualDocument("id", Array[String]("no_cust", "mri"), input)
    val documentComponents = tokenizer2(contextualDocument)
    println(s"Tokens from Bert domainTokenizer2: ${documentComponents.getTokens.mkString(" ")}")
  }

  ignore should "Succeed in creating the default vocabulary" in {
    import DomainTokenizer._, org.bertspark.config.MlopsConfiguration._
    val minFrequency = 2
    val vocabBuilder = vocabularyBuilder(minFrequency)
    val defaultVocab = vocabBuilder.build()
    assert(defaultVocab.size > 1)
    val token = defaultVocab.getToken(4)
    println(s"Token: $token")
  }

  it should "Succeed instantiate a domain tokenizer with a minimalist constructor" in {
    val input = "tomography abdomen pelvis contrast aortic ankylosing provided quadrant lower left abdominal pain diverticulitis suspected pain additional history none no comparison tomography examination of the abdomen and pelvis performed using the tomography scanner during and following the bolus infusion contrast solution thin were obtained from above the dome of diaphragm through the iliac crest and then from the iliac crest through the tuberosities chest none abdomen abdominal organs normal normal wall normal normal normal none pelvis normal normal normal normal none normal tomography of the abdomen and pelvis tomography exams are performed using one or more of following dose reduction automated exposure control adjustment of the according the patients size or use of iterative reconstruction technique signed by dystrophy muscular on"
    val contextualDocument = ContextualDocument("id", Array[String]("no_cust", "mri", "abc_Client", "modality_CT"), input)

    val domainTokenizer = DomainTokenizer[CtxTxtNSentencesBuilder](new CtxTxtNSentencesBuilder)
    val textTokens = input.split(tokenSeparator)

    val domainComponents = domainTokenizer(contextualDocument)
    println(s"Initial number of tokens: ${textTokens.length+4}")
    assert(domainComponents.getTokens.size == 2, "Failed to generate proper tokens")
    assert(domainComponents.getTokens(1).contains("mri"))
    assert(domainComponents.getTokens(1).contains("no_cust"))
    println(domainComponents.toString)
  }
}
