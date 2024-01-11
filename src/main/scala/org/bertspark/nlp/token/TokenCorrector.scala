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

import org.apache.commons.text.similarity.LevenshteinDistance
import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.config.MlopsConfiguration.{mlopsConfiguration, ConstantParameters}
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.ContextualDocument
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.{HashMap, ListBuffer}
import scala.compat.java8.collectionImpl.Accumulator
import scala.util.Random


/**
  * Class to correct a token (typo) using the Levenshtein distance
  * @param dictionary Current dictionary
  * @param threshold Threshold of number of incorrect characters
  *
  * @author Patrick Nicolas
  * @version 0.8
  */
private[bertspark] case class TokenCorrector(dictionary: Set[String], threshold: Int) {
  require(dictionary.nonEmpty, "TokenCorrector had undefined dictionary")
  require(threshold > 0 && threshold < 4, s"TokenCorrector $threshold is out of range [1,3]")

  import TokenCorrector._



  /**
    * Find a token which is similar to an existing token in a dictionary
    * @param token Candidate token
    * @return Similar token if found or empty otherwise
    */
  def apply(token: String): String =
    if(dictionary.contains(token)) {
      val levenshteinDistance = new LevenshteinDistance()
      var minDistance = Integer.MAX_VALUE
      var closestMatch = ""

      dictionary.foreach(
        entry => {
          if (token != entry) {
            val currentDistance = levenshteinDistance.apply(entry, token)
            if (currentDistance < minDistance) {
              minDistance = currentDistance
              closestMatch = entry
            }
          }
        }
      )
      if(minDistance <= threshold) closestMatch else ""
    }
    else
      ""

  def extractSimilarTokens(implicit sparkSession: SparkSession): Unit = {
    import sparkSession.implicits._

    val dictionaryDS = dictionary.toSeq.toDS()
    val variantDS = dictionaryDS.mapPartitions(
      (iter: Iterator[String]) => {
        val variantMap = HashMap[String, List[String]]()

        while (iter.hasNext) {
          val entry = iter.next()
          val similarToken = apply(entry)

          // If we found a similar token in the dictionary
          if (similarToken.nonEmpty && similarToken != entry) {
            //  Update the map entry -> similar tokens
            val existingSimilarTokens = variantMap.getOrElse(entry, List.empty[String])
            variantMap.put(entry, (similarToken :: existingSimilarTokens).distinct)
            // and update the map similar token -> entry and other similar tokens
            val existingEntries = variantMap.getOrElse(similarToken, List.empty[String])
            variantMap.put(similarToken, (entry :: existingEntries).distinct)
            logDebug(logger, s"Found similarity $entry -> $similarToken")
          }
        }
        variantMap.toSeq.toIterator
      }
    )

    LocalFileUtil.Save.local(
      fsFileName = "output/similartokens.csv",
      variantDS.map { case (k, xs) => s"$k,${xs.mkString("|")}" }.collect().mkString("\n")
    )
  }



  def reduceDictionary(implicit sparkSession: SparkSession): Unit = {
    import sparkSession.implicits._

    // Load the medical terms that have more than 2 characters for similar tokens
    val medicalTermsSet = LocalFileUtil
        .Load
        .local(ConstantParameters.medicalTermsSetFile, (s: String) => s)
        .map(_.map(_.toLowerCase))
        .filter(_.size > 1)
        .getOrElse({
          logger.error(s"Vocabulary: medical terms ${ConstantParameters.medicalTermsSetFile} is undefined")
          Array.empty[String]
        }).toSet

    val medicalTermsSet_brdCast = sparkSession.sparkContext.broadcast[Set[String]](medicalTermsSet)

    val dictionaryDS = dictionary.toSeq.toDS()
    val similarTokenFromDictionaryDS: Dataset[(String, String)] = dictionaryDS.mapPartitions(
      (iter: Iterator[String]) => {

        val medicalTermsSetValue = medicalTermsSet_brdCast.value
        val similarTokenMap = HashMap[String, String]()
        while(iter.hasNext) {
          val dictionaryEntry = iter.next()
          if(medicalTermsSetValue.contains(dictionaryEntry)) {
            similarTokenMap.put(dictionaryEntry, dictionaryEntry)
          }
          else {
            val similarToken = apply(dictionaryEntry)
            if(medicalTermsSetValue.contains(similarToken)) {
              similarTokenMap.put(dictionaryEntry, similarToken)
            }
          }
        }
        similarTokenMap.toIterator
      }
    )

    val similarTokenFromDictionaryPairs = similarTokenFromDictionaryDS.collect
    val medicalTermAliasesMap = similarTokenFromDictionaryPairs.map{ case (aliasTerm, term) => s"$aliasTerm,$term"}.sortWith(_ < _)
    LocalFileUtil.Save.local(fsFileName = "output/medicalTermAliases.csv", medicalTermAliasesMap.mkString("\n"))

    val similarTokenFromDictionarySet =  similarTokenFromDictionaryPairs.map(_._1).toSet
    val reducedDictionary = dictionary.filter(!similarTokenFromDictionarySet.contains(_)).toSeq
    LocalFileUtil.Save.local(fsFileName = "output/reducedDictionary.txt", reducedDictionary.sortWith(_ < _).mkString("\n"))
  }



        /**
    * Extract a map of dictionary similar tokens associated with a valid medical term in a dictionary
    * @param sparkSession Implicit reference to the current spark context
    */
  def reduceSimilarTokens(implicit sparkSession: SparkSession): Unit = {
    import sparkSession.implicits._

    val dictionaryDS = dictionary.toSeq.toDS()
    // Generate the dataset of dictionary token -> Similar tokens
    val similarTokenFromDictionaryDS: Dataset[(String, List[String])] = dictionaryDS.mapPartitions(
      (iter: Iterator[String]) => {

        val variantMap = HashMap[String, List[String]]()
        while(iter.hasNext) {
          val entry = iter.next()
          val similarToken = apply(entry)
          // If we found a similar token in the dictionary
          if(similarToken.nonEmpty && similarToken != entry) {
            val existingSimilarTokens = variantMap.getOrElse(entry, List.empty[String])
            variantMap.put(entry, (similarToken :: existingSimilarTokens).distinct)
            logDebug(logger, msg = s"Found similarity $entry -> $similarToken")
          }
        }
        variantMap.toSeq.toIterator
      }
    ).persist.cache

    // Load the medical terms that have more than 2 characters for similar tokens
    val medicalTermsSet = LocalFileUtil
        .Load
        .local(ConstantParameters.medicalTermsSetFile, (s: String) => s)
        .map(_.map(_.toLowerCase))
        .filter(_.size > 2)
        .getOrElse({
          logger.error(s"Vocabulary: medical terms ${ConstantParameters.medicalTermsSetFile} is undefined")
          Array.empty[String]
        }).toSet


    // Extract the map of similar token to valid 'core' token
    val medicalTermsSet_brdCast = sparkSession.sparkContext.broadcast[Set[String]](medicalTermsSet)
    val validSimilarTokensDS: Dataset[(String, String)] = similarTokenFromDictionaryDS.mapPartitions(
      (iter: Iterator[(String, List[String])]) => {
        val medicalTermsSetValue = medicalTermsSet_brdCast.value

        val collector = HashMap[String, String]()
        while(iter.hasNext) {
          val (key, otherKeys) = iter.next()
          val allSimilarTokens = key :: otherKeys
          val filtered = otherKeys.filter(medicalTermsSetValue.contains(_))
          if(filtered.nonEmpty)
            filtered.foreach(found => allSimilarTokens.foreach(collector.put(_, found)))
          else
            logger.error(s"Failed to found a valid similar token for $key")
        }
        collector.toIterator
      }
    )

    LocalFileUtil.Save.local(
      fsFileName = "output/tokenSimilarity.csv",
      validSimilarTokensDS.map{ case (k, v) => s"$k,$v"}.distinct.collect().sortWith(_ < _).mkString("\n")
    )
  }

  def apply(): Unit = {
    val variantMap = HashMap[String, List[String]]()
    var count = 0
    dictionary.foreach(
      entry => {
        val similarToken = apply(entry)
        // If we found a similar token in the dictionary
        if(similarToken.nonEmpty && similarToken != entry) {
          val existingSimilarTokens = variantMap.getOrElse(entry, List.empty[String])
          variantMap.put(entry, similarToken :: existingSimilarTokens)
          val existingEntries = variantMap.getOrElse(similarToken, List.empty[String])
          variantMap.put(similarToken, entry :: existingEntries)
        }
        count += 1
        logDebug(logger, msg = s"Count $count for similarity of vocabulary")
      }
    )

    LocalFileUtil.Save.local(
      fsFileName = "output/similartokens.csv",
      variantMap.map{ case (k, xs) => s"$k,${xs.mkString("|")}"}.mkString("\n")
    )
  }

  /**
    *
    * @param contextualDocument
    * @return
    */
  def apply(contextualDocument: ContextualDocument, visited: HashMap[String, String]): ContextualDocument = {
    val textTokens = contextualDocument.text.split(tokenSeparator)

    val newTextTokens = textTokens.map(
      token => {
        if(visited.contains(token))
          visited.get(token).get
        else {
          val newToken = apply(token)
          if (newToken.nonEmpty) {
            visited.put(token, newToken)
            newToken
          }
          else token
        }
      }
    )
    contextualDocument.copy(text = newTextTokens.mkString(" "))
  }
}




private[bertspark] object TokenCorrector {
  final private val logger: Logger = LoggerFactory.getLogger("TokenCorrector")

  def apply(vocabularyFile: String, threshold: Int):  TokenCorrector = {
    val vocabularyEntries = LocalFileUtil
        .Load
        .local(vocabularyFile)
        .map(_.split("\n").toSet)
        .getOrElse(Set.empty[String])
    new TokenCorrector(vocabularyEntries, threshold)
  }

  def generateMedicalTermsShortList: Unit = {
    val laMedicalTerms = "conf/codes/laMedicalTerms.txt"
    val correctedTermsSet = LocalFileUtil
        .Load
        .local(laMedicalTerms, (s: String) => s)
        .map(
          lines => {
            lines.map(
              line => {
                val fields = line.split("-")
                if(fields.size == 2) fields.head.toLowerCase.split(tokenSeparator) else Array.empty[String]
              }
            ).filter(_.nonEmpty).flatten.distinct
          }
        )
        .getOrElse({
          logger.error(s"Vocabulary: Medical terms $laMedicalTerms is undefined")
          Array.empty[String]
        }).sortWith(_ < _)

    LocalFileUtil.Save.local(ConstantParameters.medicalTermsSetFile, correctedTermsSet.mkString("\n"))
  }

  private def computeLevenshtein(vocabulary: Set[String], token: String, threshold:Int): String = {
    val levenshteinDistance = new LevenshteinDistance()
    var minDistance = Integer.MAX_VALUE
    var closestMatch = ""

    vocabulary.foreach(
      entry => {
        if (token != entry) {
          val currentDistance = levenshteinDistance.apply(entry, token)
          if (currentDistance < minDistance) {
            minDistance = currentDistance
            closestMatch = entry
          }
        }
      }
    )
    if(minDistance <= threshold) closestMatch else ""
  }


  def extractAliases(fraction: Double)(implicit sparkSession: SparkSession): Unit = {
    import sparkSession.implicits._

    // Load the medical terms that have more than 2 characters for similar tokens
    val inputDS = S3Util.s3ToDataset[ContextualDocument](S3PathNames.s3ContextualDocumentPath)
        .sample(fraction, System.currentTimeMillis())
        .flatMap(_.text.split(tokenSeparator))
        .persist()
        .cache()

    val aliasProcessingCount = sparkSession.sparkContext.longAccumulator("aliasProcessingCount")


    logDebug(logger, s"loaded ${inputDS.count()} contextual documents")
    val similarTokenFromDictionaryDS: Dataset[(String, String)] = inputDS.mapPartitions(
      (iter: Iterator[String]) => {
        val medicalTermsSetValue =  LocalFileUtil
            .Load
            .local(ConstantParameters.medicalTermsSetFile, (s: String) => s)
            .map(_.map(_.toLowerCase))
            .filter(_.size > 2)
            .getOrElse({
              logger.error(s"Vocabulary: medical terms ${ConstantParameters.medicalTermsSetFile} is undefined")
              Array.empty[String]
            }).toSet

        val similarTokenMap = HashMap[String, String]()
        while(iter.hasNext) {
          val entry = iter.next()
          aliasProcessingCount.add(1L)

          if(!medicalTermsSetValue.contains(entry)) {
            val similarToken = computeLevenshtein(medicalTermsSetValue, entry, threshold = 1)
            if(similarToken.nonEmpty && similarToken != entry && medicalTermsSetValue.contains(similarToken)) {
              similarTokenMap.put(entry, similarToken)
            }
          }
        }
        logDebug(logger, s"${aliasProcessingCount.name.get}: ${aliasProcessingCount.value}")
        similarTokenMap.toIterator
      }
    )

    val similarTokenFromDictionaryPairs = similarTokenFromDictionaryDS.distinct.collect
    val medicalTermAliasesMap = similarTokenFromDictionaryPairs.map{ case (aliasTerm, term) => s"$aliasTerm,$term"}.sortWith(_ < _)
    LocalFileUtil.Save.local(fsFileName = "output/medicalTermAliases.csv", medicalTermAliasesMap.mkString("\n"))
  }

  def correct(
    contextualDocumentDS: Dataset[ContextualDocument],
    dictionary: Set[String],
    threshold: Int)(implicit sparkSession: SparkSession): Dataset[ContextualDocument] = {
    import sparkSession.implicits._

    val dictionary_brdCast = sparkSession.sparkContext.broadcast[Set[String]](dictionary)
    val threshold_brdCast = sparkSession.sparkContext.broadcast[Int](threshold)

    contextualDocumentDS.mapPartitions(
      (iter: Iterator[ContextualDocument]) => {
        val visitedTokensMap = HashMap[String, String]()
        val dictionaryValue = dictionary_brdCast.value
        val thresholdValue = threshold_brdCast.value
        val tokenCorrector = new TokenCorrector(dictionaryValue, thresholdValue)

        val newTokensCollector = ListBuffer[ContextualDocument]()
        while(iter.hasNext) {
          val ctxDocument = iter.next()
          newTokensCollector.append(tokenCorrector(ctxDocument, visitedTokensMap))
        }
        newTokensCollector.toIterator
      }
    )
  }


  def correct(threshold: Int)(implicit sparkSession: SparkSession): Unit = {
    import sparkSession.implicits._

    val contextualDocumentDS = S3Util.s3ToDataset[ContextualDocument](S3PathNames.s3ContextualDocumentPath)
    val medicalTermsSet = LocalFileUtil
        .Load
        .local(ConstantParameters.termsSetFile, (s: String) => s)
        .map(_.map(_.toLowerCase))
        .getOrElse({
          logger.error(s"Vocabulary: Medical terms ${ConstantParameters.termsSetFile} is undefined")
          Array.empty[String]
        }).toSet

    val correctedContextualDocumentDS = correct(contextualDocumentDS, medicalTermsSet, threshold)
    S3Util.datasetToS3[ContextualDocument](
      mlopsConfiguration.storageConfig.s3Bucket,
      correctedContextualDocumentDS,
      s3OutputPath = s"${S3PathNames.s3ContextualDocumentPath}-T",
      header = false,
      fileFormat = "json",
      toAppend = false,
      numPartitions = 16
    )
  }



  def extract(terms: Seq[String], numChars: Int): Unit = {
    val collector = HashMap[String, String]()
    var index = 0
    do {
      val baseTerm = terms(index)
      val newIndex = extract(index, terms, numChars)
      if (newIndex - index > 1) {
        val variants = (index+1 until newIndex).map(terms(_))
        variants.foreach(collector.put(_, baseTerm))
      }
      index = newIndex
    } while(index < terms.length-1)

    val dump = collector.toSeq.sortWith(_._2 < _._2).map{ case (variant, stem) => s"$variant,$stem"}.mkString("\n")
    LocalFileUtil.Save.local("output/stems.csv", dump)
  }

  private def extract(cursor: Int, terms: Seq[String], numChars: Int): Int = {
    var index = cursor
    val baseTerm = terms(cursor)
    var commonStem = true
    do {
      index += 1
      commonStem = compare(baseTerm, terms(index), numChars)
    } while(commonStem)
    index
  }

  private def compare(from: String, to: String, numChars: Int): Boolean =
    from.size >= numChars && to.size >= numChars && from.substring(0 , numChars) == to.substring(0, numChars)
}
