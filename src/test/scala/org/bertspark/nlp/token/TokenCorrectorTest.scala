package org.bertspark.nlp.token

import org.apache.commons.text.similarity.LevenshteinDistance
import org.bertspark.config.MlopsConfiguration.ConstantParameters
import org.bertspark.nlp.token.TokenCorrector.{extract, extractAliases}
import org.bertspark.util.io.LocalFileUtil
import org.scalatest.flatspec.AnyFlatSpec

private[token] final class TokenCorrectorTest extends AnyFlatSpec {

  ignore should "Succeed evaluating" in {
    val levenshteinDistance = new LevenshteinDistance()
    val distance = levenshteinDistance.apply("peritonitic", "peritonitis")
    println(s"LevenshteinDistance: $distance")
  }

  ignore should "Succeed finding match for token" in {
    val fsVocabulary = "conf/vocabularyTF95/XLARGE4.csv"
    val tokenCorrector = TokenCorrector(fsVocabulary, 1)
    val word = "peritonitis"
    val similarToken = tokenCorrector(word)
    assert(similarToken.nonEmpty)
  }

  ignore should "Succeed generate equivalent for existing dictionary" in {
    import org.bertspark.implicits._

    val fsVocabulary = "conf/vocabularyTF95/XLARGE4.csv"
    val tokenCorrector = TokenCorrector(fsVocabulary, 1)
    tokenCorrector.extractSimilarTokens
  }


  ignore should "Succeed in reducing similar tokens" in {
    import org.bertspark.implicits._

    val fsVocabulary = "conf/vocabularyTF95/XLARGE4.csv"
    val tokenCorrector = TokenCorrector(fsVocabulary, 1)
    tokenCorrector.reduceSimilarTokens
  }

  ignore should "Succeed extracting a short list of medical terms" in {
    TokenCorrector.generateMedicalTermsShortList
  }

  ignore should "Succeed reduce dictionary" in {
    import org.bertspark.implicits._

    val fsVocabulary = "conf/vocabularyTF95/XLARGE4.csv"
    val tokenCorrector = TokenCorrector(fsVocabulary, 1)

    tokenCorrector.reduceDictionary
  }

  ignore should "Succeed creating aliases" in {
    import org.bertspark.implicits._

    extractAliases(0.01)
  }

  ignore should "Succeed extracting stems" in {
    val numChars = 7
    val initialTerms = LocalFileUtil
        .Load
        .local(ConstantParameters.termsSetFile, (s: String) => s)
        .map(_.map(_.toLowerCase))
        .getOrElse({
          println(s"ERROR Vocabulary: Medical terms ${ConstantParameters.termsSetFile} is undefined")
          Array.empty[String]
        }).filter(_.length > 4)
    extract(initialTerms, numChars)
  }

  ignore should "Succeed reorganizing stems" in {
    val fsStemsFilename = "output/stems.csv"
    val fsFinalStemsFilename = "conf/codes/stems.csv"
    val distinctStems = LocalFileUtil.Load.local(fsStemsFilename, (s: String) => s).map(_.distinct.sortWith(_ < _))
    distinctStems.foreach(
      stems => LocalFileUtil.Save.local(fsFinalStemsFilename, stems.mkString("\n"))
    )

    val uniqueVariantStemsO = distinctStems.map(
      lines => {
        val variantStemPairs = lines.map(
          line => {
            val fields = line.split(",")
            if (fields.length == 2) (fields.head, fields(1)) else ("", "")
          }
        ).filter(_._1.nonEmpty)

        val uniqueVariants = variantStemPairs.map(_._1).distinct
        val uniqueStems = variantStemPairs.map(_._2).distinct
        (uniqueVariants, uniqueStems)
      }
    ).getOrElse((Array.empty[String],  Array.empty[String]))

    val dictionary = LocalFileUtil
        .Load
        .local("conf/vocabularyTF95/TF95", (s: String) => s)
        .getOrElse(Array.empty[String])


    val uniqueVariants = uniqueVariantStemsO._1.toSet
    val updateDictionary1 = dictionary.filter(!uniqueVariants.contains(_))
    println(s"Original dictionary: ${dictionary.length} update dictionary1 ${updateDictionary1.length}")

    val updateDictionary2 = (updateDictionary1 .toSeq ++ uniqueVariantStemsO._2).distinct
    println(s"Original dictionary: ${dictionary.length} update dictionary2 ${updateDictionary2.length}")

    LocalFileUtil.Save.local("conf/vocabularyTF95/TF96", updateDictionary2.mkString("\n"))
  }


  it should "Succeed updating aliases map" in {
    val abbreviationsMapFilename = "conf/abbreviationsMap"
    val aliasesMap = "conf/codes/aliasesMap.csv"
    val abbreviationsMap = LocalFileUtil
        .Load
        .local(abbreviationsMapFilename, (s: String) => s.toLowerCase)
        .map(
          lines =>  {
            lines.map(
              line => {
                val fields = line.split(",")
                if(fields.size == 2) (fields.head.trim, fields(1).trim) else ("", "")
              }
            ).filter(_._1.nonEmpty)
          }
        )
        .getOrElse({
          println(s"ERROR failed to load $abbreviationsMapFilename")
          Array.empty[(String, String)]
      })

    val existingAliasesMap = LocalFileUtil
        .Load
        .local(aliasesMap, (s: String) => s)
        .map(
          lines =>  {
            lines.map(
              line => {
                val fields = line.split(",")
                if(fields.size == 2) (fields.head.trim, fields(1).trim) else ("", "")
              }
            ).filter(_._1.nonEmpty)
          }
        )
        .getOrElse({
          println(s"ERROR failed to load $aliasesMap")
          Array.empty[(String, String)]
        })

    val updatedAliasesMap = (abbreviationsMap ++ existingAliasesMap).distinct.sortWith(_._1 < _._1)
    LocalFileUtil.Save.local(
      fsFileName = "conf/aliasesMap2.csv",
      updatedAliasesMap.map{ case (k,v) => s"$k,$v"}.mkString("\n")
    )

    val descriptors = abbreviationsMap.map(_._2).distinct.flatMap(_.split("\\s+"))

    val dictionary = LocalFileUtil
        .Load
        .local("conf/vocabularyTF95/TF96", (s: String) => s)
        .getOrElse(Array.empty[String])



    val updateDictionary = (dictionary .toSeq ++ descriptors).distinct
    println(s"Original dictionary: ${dictionary.size} update dictionary2 ${updateDictionary.size}")

    LocalFileUtil.Save.local("conf/vocabularyTF95/TF97", updateDictionary.mkString("\n"))
  }
}
