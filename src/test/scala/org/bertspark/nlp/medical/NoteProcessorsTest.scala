package org.bertspark.nlp.medical

import org.bertspark.nlp.medical.NoteProcessors._
import org.scalatest.flatspec.AnyFlatSpec


private[medical] final class NoteProcessorsTest extends AnyFlatSpec {
  final private val input = "This is not. \\r\\n\\r\\nvalue\n to be done, 8 once (for C8): all value's 6% with no\r\n \\n!! 0.7 values\\r\\n It (is) done with 23.1% cc."

  ignore should "Succeed replacing digit only content" in {
    val input = "this is C6 with 34 and 4.13% values"
    val output = input.split("\\s+").map(token => if(token.matches(numRegex)) "xnum" else token)
    println(output.mkString(" "))
  }

  ignore should "Succeed extract tokens from sentence" in {
    val text = "This is not you a typical value? or sentence <. It is( Tuesday"
    val tokens = NoteProcessors.extractTokens(text)
    println(tokens.mkString(" "))
  }

  ignore should "Succeed replacing eol characters" in {
    val cleansed = input.replaceAll(numRegexPerCent, numLabelPerCent).replaceAll(numRegex, numLabel)
    println(cleansed)
  }

  it should "Succeed cleansing a content" in {
    val regex = specialCharCleanserRegex
    val cleansed = cleanse(input, regex)
    println(s"$input\n${cleansed.mkString(" ")}")
  }

  ignore should "Succeed cleansing individual token" in {
    val tokens = Seq[String]("hello", "%why", "(ct)", "34c")
    val cleansedTokens = tokens.map(_.replaceAll(specialCharCleanserRegex, ""))
    println(cleansedTokens.mkString(" "))
  }
}
