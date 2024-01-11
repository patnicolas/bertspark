package org.bertspark.analytics

import org.bertspark.config.MlopsConfiguration.vocabulary
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.vocabulary.MedicalCodeDescriptors.{fsCptCodesPath, fsIcdCodesPath, getIcdDescriptors, logger}
import org.bertspark.util.io.LocalFileUtil
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable.ListBuffer

class VocabularyAnalyzerTest extends AnyFlatSpec {

  ignore should "Succeed updating a vocabulary with CPT descriptors" in {

    val cptDescriptors = LocalFileUtil.Load.local(fsCptCodesPath, (s: String) => {
      val ar: Array[String] = s.split("\t")
      if (ar.size > 3) {
        val cptCode = ar(1)
        val descriptor = ar(3).toLowerCase
        (cptCode, descriptor)
      } else
        ("", "")
    }, header = true)
        .getOrElse(throw new IllegalStateException("Failed loading CPT descriptors"))
        .filter(_._1.nonEmpty)
        .flatMap {
          case (cp, description) => description
              .split(tokenSeparator)
              .filter(_.size > 1)
              .map(_.toLowerCase)
              .filter(_.forall(ch => ch > 96 && ch < 127))
        }.distinct

    val AMAterms = LocalFileUtil.Load.local("input/AMA").map(_.split("\n")).get

    val newAMATerms = LocalFileUtil.Load.local("input/AMA").map(
      content => {
        val lines = content.split("\n")
        (lines ++ cptDescriptors).distinct.sortWith(_ < _)
      }
    ).get
    newAMATerms.map(
      content => LocalFileUtil.Save.local("conf/AMA", content.mkString("\n"))
    )
  }


  it should "Succeed extracting abbreviations" in {
    val abbreviationFile = "input/abbreviations.txt"

    val abbreviationMap = ListBuffer[String]()
    LocalFileUtil.Load.local(abbreviationFile).foreach(
      content => {
        val lines = content.split("\n")
        lines.filter(_.size > 2).map(
          line => {
            val fields = line.trim.split("\t")
            if(fields.size == 2)
              abbreviationMap.append(s"${fields.head.trim},${fields(1).trim.toLowerCase}")
            else if(fields.size == 4) {
              abbreviationMap.append(s"${fields.head.trim},${fields(1).trim.toLowerCase}")
              abbreviationMap.append(s"${fields(2).trim},${fields(3).trim.toLowerCase}")
            }
          }
        )
      }
    )

    val abbrWords = abbreviationMap.flatMap(
      line => {
        val ar = line.split(",")
        ar(1).trim.split(tokenSeparator)
      }
    ).distinct.filter(_.length > 1)

    LocalFileUtil.Save.local("conf/abbreviationsMap", abbreviationMap.mkString("\n"))

    val newAMATerms = LocalFileUtil.Load.local("input/AMA").map(
      content => {
        val lines = content.split("\n")
        (lines ++ abbrWords).distinct.sortWith(_ < _)
      }
    )
    newAMATerms.map(
      content => LocalFileUtil.Save.local("conf/AMA", content.mkString("\n"))
    )
  }
}
