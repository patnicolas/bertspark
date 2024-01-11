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
 */
package org.bertspark.nlp.trainingset

import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.medical.MedicalCodingTypes.{InternalContext, InternalRequest, MlEMRCodes}
import org.bertspark.nlp.token.{PretrainingInput, SentencesBuilder, TokenizerPreProcessor}
import org.bertspark.Labels
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.S3PathNames.{s3ContextualDocumentPath, s3TransformerModelPath}
import org.bertspark.nlp.medical.ContextEncoder.encodeContext
import org.bertspark.nlp.token.SentencesBuilder.{concatenate, ContextSentence}
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.ContextualDocument.maxDisplayChars
import org.bertspark.util.io.S3Util
import org.bertspark.util.SparkUtil
import org.slf4j.{Logger, LoggerFactory}


/**
 * Raw records including document id, context values and the text document
 *
 * @param id Identifier for the document
 * @param contextVariables Context associated with the variables
 * @param text Text or content of the document defined as a aggregated (mkString) of tokens
 *             extracted from the text.
 *
 * @author Patrick Nicolas
 * @version 0.2
 */
private[bertspark] case class ContextualDocument(
  id: String,
  contextVariables: scala.Array[String],
  text: String) extends PretrainingInput {
  override def getId: String = id

  override def toString: String =
      s"""Id: $id, Context: ${contextVariables.mkString("\n")}
       |Document: ${if(text.length > maxDisplayChars) text.substring(0, maxDisplayChars) else text} ...."""
          .stripMargin

  def summary: String = s"$id: ${if(text.length > maxDisplayChars) text.substring(0, maxDisplayChars) else text} ...."

  def toContextSentence: ContextSentence = (contextVariables.mkString(" "), text)
}


/**
  * Singleton or namespace for the contextual document builder
  * @author Patrick Nicolas
  * @version 0.2
  */
private[bertspark] final object ContextualDocument {
  final private def maxDisplayChars = 48

  def apply(id: String): ContextualDocument = ContextualDocument(id, Array.empty[String], "")

  final lazy val nullContextualDocument = ContextualDocument("", Array.empty[String], "")
  /**
   * Data set pre-processor for notes and requests (un-labeled)
   * {{{
   *   Input: - s3Path for requests (i.e. requests/Cornerstone/)
   *          - numRecords to sample
   *   Output: Contextual Document ContextualDocument(id: String, contextVariables: Array[String], text: String)
   *           in S3 (i.e. mlops/Cornerstone/context-document/)
   * }}}
   * @param sparkSession Implicit reference to the current Spark context
   *
   * @author Patrick Nicolas
   * @version 0.2
   */
  final class ContextualDocumentBuilder(implicit sparkSession: SparkSession) {
    import sparkSession.implicits._
    import ContextualDocumentBuilder._

    /**
      * Extract Tokens from the request (document and context) to produce a contextual document
      */
    def apply(sampledRequestDS: Dataset[InternalRequest], numSubModels: Int): Unit =
      execute(sampledRequestDS)

    /**
     * Extract Tokens from the request (document and context) to produce a contextual document
     */
    def apply(vocabularyType: String, numRecords: Int): Unit = {
      // Step 1: Load the internal requests
      val requestDS = try {
        val rawRequestDS = S3Util.s3ToDataset[InternalRequest](S3PathNames.s3RequestsPath).dropDuplicates("id")
        if(numRecords > 0) {
          val totalNumRecords = rawRequestDS.count()
          val ratio = numRecords.toFloat/totalNumRecords
          // we apply random sampling only if the size of sample is smaller than the dataset...
          if(ratio < 0.95F) {
            logDebug(logger, msg = s"Ratio for $numRecords over $totalNumRecords ratio=$ratio")
            rawRequestDS.sample(ratio)
          }
          else
            rawRequestDS
        }
        else {
          logger.error(s"No request were found for ${S3PathNames.s3RequestsPath}")
          rawRequestDS
        }
      }
      catch {
        case e: IllegalStateException =>
          logger.error(s"Contextual document: ${e.getMessage}")
          sparkSession.emptyDataset[InternalRequest]
      }

      if(requestDS.isEmpty)
        throw new IllegalStateException(s"Request dataset for ${S3PathNames.s3RequestsPath} is empty")
      logDebug(logger, msg = s"Load ${requestDS.count()} requests for contextual document")

      // Step 3: Pre-select the internal requests according to a predefined number of sub models.
      execute(requestDS)
    }
  }


  final object ContextualDocumentBuilder {
    final private val logger: Logger = LoggerFactory.getLogger("ContextualDocumentBuilder")

    def apply()(implicit sparkSession: SparkSession): ContextualDocumentBuilder = new ContextualDocumentBuilder

    private def execute(sampledRequestDS: Dataset[InternalRequest])(implicit sparkSession: SparkSession): Unit =  {
      import sparkSession.implicits._
      import org.bertspark.config.MlopsConfiguration._

      // Restrict the size of sample if needed -1 if entire
      val numSplits = mlopsConfiguration.preProcessConfig.numSplits
      val dsSplits = sampledRequestDS.randomSplit(Array.fill(numSplits)(1.0/numSplits))

      // Generate the output file path (HDFS format)
      val outputFile = s3ContextualDocumentPath
      var counter = 0
      val step = (sampledRequestDS.count().toDouble/numSplits).floor.toInt
      val numPartitions = 4
      logDebug(logger, msg = s"Contextual document builder with $numPartitions partitions and $step step")
      val totalNumberSamples = sampledRequestDS.count()

      // Split the data set to avoid overflowing memory
      dsSplits.foreach(
        splitDS => {
          val start = System.currentTimeMillis()

          // Token extraction for this split
          val contextualDocumentDS = splitDS.map(extractContextualDocument(_)).filter(_.id.nonEmpty)
          // Save into S3
          val isToAppend = counter != 1
          S3Util.datasetToS3[ContextualDocument](
            mlopsConfiguration.storageConfig.s3Bucket,
            contextualDocumentDS,
            outputFile,
            header = false,
            fileFormat = "json",
            toAppend = isToAppend,
            numPartitions
          )
          counter += step
          logDebug(logger,  {
            val duration = (System.currentTimeMillis() - start)*0.001
            s"Contextual document built $counter out of $totalNumberSamples notes ${100.0F*counter/totalNumberSamples}%" +
            s"$counter records processed in $duration secs. average: ${duration/step}"
          })
        }
      )
    }

    /**
     * Convert a prediction request (stored in S3) to contextual document
     * @param internalRequest Prediction request
     * @return contextual document
     */
    final def extractContextualDocument(internalRequest: InternalRequest): ContextualDocument = {
      import org.bertspark.implicits._

      if(internalRequest.context.EMRCpts.nonEmpty) {
        val contextualEmbedding = encodeContext(internalRequest.context)
        val tokenizer = TokenizerPreProcessor()

        // Implicit conversion between Java and Scala List
        val textTokens: scala.List[String] = tokenizer.tokenize(internalRequest.notes.head)
        logger.info(s"- Num tokens: ${textTokens.size + contextualEmbedding.size}")
        ContextualDocument(internalRequest.id, contextualEmbedding, textTokens.mkString(" "))
      }
      else
        ContextualDocument("", Array.empty[String], "")
    }


    def preSelectRequest(
      sampledRequestDS: Dataset[InternalRequest],
      numSubModels: Int)(implicit sparkSession: SparkSession): Dataset[InternalRequest] = {
      import sparkSession.implicits._

      val selectedInternalRequestDS =
        if(numSubModels > 0) {
          val sampleRequestXsDS = sampledRequestDS.map(List[InternalRequest](_))
          val groupedByEmr = SparkUtil.groupByKey[List[InternalRequest], String](
            (req: List[InternalRequest]) =>
                // @todo *** Need to organize the EMR with number first.??
              req.head.context.emrLabel,
            (req1: List[InternalRequest], req2: List[InternalRequest]) => req1 ::: req2,
            sampleRequestXsDS
          )   .toDS()
              .limit(numSubModels)

          val emrLabels = groupedByEmr.map(_._1).collect()
          logDebug(logger, msg = s"EMR labels ${emrLabels.mkString("\n")}")

          try {
          //  S3Util.upload(s"$s3TransformerModelPath/subModels.txt", emrLabels.mkString("\n"))
            groupedByEmr.flatMap(_._2)
          }
          catch {
            case e: IllegalArgumentException =>
              logger.error(s"preSelectRequest ${e.getMessage}")
              sparkSession.emptyDataset[InternalRequest]
          }
        }
        else
          sampledRequestDS
      selectedInternalRequestDS
    }


    /**
     * Convert a prediction record from a table to contextual document
     * @param prediction Prediction records from RDS table
     * @return contextual document
     */
    final def extractContextualDocument(prediction: Seq[String]): ContextualDocument = {
      if(prediction(12).nonEmpty) {
        def extractContext(prediction: Seq[String]): InternalContext = {
          val emrCodes =
            if (prediction(12).nonEmpty) Seq[MlEMRCodes](MlEMRCodes(prediction(12)))
            else Seq.empty[MlEMRCodes]

          InternalContext(
            "x",
            prediction(4).toInt,
            prediction(5),
            prediction(6),
            prediction(7),
            prediction(8),
            prediction(9),
            prediction(10),
            prediction(11),
            emrCodes,
            prediction(13),
            prediction(14),
            "",
            "",
            "",
            ""
          )
        }

        val context = extractContext(prediction)
        import org.bertspark.implicits._
        val contextualEmbedding = encodeContext(context)
        val tokenizer = TokenizerPreProcessor()

        // Implicit conversion between Java and Scala List
        val termsText: scala.List[String] = tokenizer.tokenize(prediction(25))
        ContextualDocument(prediction(1), contextualEmbedding, termsText.mkString(" "))
      }
      else
        ContextualDocument("", Array.empty[String], "")
    }


    /**
     * Evaluate the extraction of tokens per sentences
     * @param numRequests Number of requests
     * @param sparkSession Implicit reference to the current Spark context
     */
    def evaluate(numRequests: Int)(implicit sparkSession: SparkSession): Unit = {
      import org.bertspark.config.MlopsConfiguration._
      import sparkSession.implicits._, Labels._

      // Load the requests to generated tokens
      val start1 = System.currentTimeMillis()
      val s3RequestsFolder = s"${mlopsConfiguration.storageConfig.s3RequestFolder}/${mlopsConfiguration.target}"
      val requestDS = try {
        S3Util.s3ToDataset[InternalRequest](
          mlopsConfiguration.storageConfig.s3Bucket,
          s3RequestsFolder,
          header = false,
          fileFormat = "json"
        )
      }
      catch {
        case e: IllegalStateException =>
          logger.error(s"Contextual document evaluate: ${e.getMessage}")
          sparkSession.emptyDataset[InternalRequest]
      }
      logDebug(logger, msg = s"Loaded $numRequests requests in ${(System.currentTimeMillis() - start1) * 0.001} secs.")

      // Restrict the size of sample if needed (-1 if entire ts
      val sampledRequestDS = if (numRequests > 0) requestDS.limit(numRequests) else requestDS
      val sentenceBuilder = SentencesBuilder()

      // Split the data set to avoid overflowing memory
      val rawTokens = sampledRequestDS.collect.map(
        request => {
          val contextualDocument = extractContextualDocument(request)
          (request.notes.head, sentenceBuilder.apply(contextualDocument))
        }
      )
      // Pads the tokens
      val paddedTokensSentences = rawTokens.map{
        case (text, segments) => (text, segments.map(seg => padding(concatenate(seg))))
      }.toSeq
      // Save into S3
      S3Util.datasetToS3[(String, Array[String])](
        mlopsConfiguration.storageConfig.s3Bucket,
        paddedTokensSentences.toDS(),
        S3PathNames.s3TokensEvaluationPath,
        header = false,
        fileFormat = "json",
        toAppend = false,
        numPartitions = 1
      )
    }

    def padding(str: String): String = {
      val tokens = str.split(tokenSeparator)
      val limit = mlopsConfiguration.getMinSeqLength

      if(tokens.size >= limit) tokens.take(limit).mkString(" ")
      else (tokens ++ Array.fill(limit - tokens.size)("P")).mkString(" ")
    }
  }
}


