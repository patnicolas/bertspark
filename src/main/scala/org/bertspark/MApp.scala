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
package org.bertspark


import ai.djl.ndarray.NDManager
import ai.djl.Device
import ai.djl.Device.gpu
import ai.djl.engine.Engine
import ai.djl.ndarray.types.DataType
import org.slf4j.{Logger, LoggerFactory}
import org.bertspark.Labels.configValidation
import org.bertspark.config.MlopsConfiguration.{initializeProperty, mlopsConfiguration}
import org.bertspark.analytics.{PredictionAnalysis, Reporting, VocabularyAnalyzer}
import org.bertspark.classifier.training.TClassifier
import org.bertspark.implicits.collection2Scala
import org.bertspark.nlp.trainingset.TrainingSetBuilder
import org.bertspark.transformer.training.{TPretraining, TTransferLearning}
import org.bertspark.transformer.representation.DocumentEmbeddingSimilarity
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.hpo.ClassifierTrainHPO
import org.bertspark.kafka.KafkaPipeline
import org.bertspark.kafka.simulator.KafkaSimulator
import org.bertspark.modeling.BaseTrainingListener
import org.bertspark.modeling.ModelExecution.createRequestRouting
import org.bertspark.nlp.token.TokenCorrector.extractAliases
import org.bertspark.nlp.token.TokensTfIdf
import org.bertspark.nlp.trainingset.TrainingSetBuilder.estimateAutoCodingRate
import org.bertspark.nlp.vocabulary.{MedicalAbbreviations, MedicalVocabularyBuilder}
import org.bertspark.util.io.S3IOOps.s3ToS3Feedback
import org.bertspark.util.rdbms.PredictionsTbl.tblToS3Request


/**
 * '''Command lines'''
 * ==config==
 *   ''Display the current configuration and set up for application''
 *   - Output: Standard output
 *
 * ==buildMedicalVocabulary requestFolder sampleSize numSplits/Int==
 *   ''Build the custom medical vocabulary according to configuration''
 *   - Input:
 *   - Output:
 *
 * ==buildTrainingSet numRequests/Int numContextualDocuments/Int==
 *   ''Build training set from contextual embedding and feedback''
 *   - Input:
 *   - Output:
 *
 * ==preTrainBert isTraining/Boolean==
 *  ''Pre-train transformer encoder/BERT''
 *
 * ==classify isTraining/Boolean subModelName/String==
 *   ''Train classifier (Full connected neural network)''
 *   ''Train or predict the BERT classifier for a given sub model (or all sub models if subModelName = LL)''
 * ==similarity==
 *
 *
 * ==predictionAnalysis normalizedAccuracy sourceFolder epochIndex==
 *   - Input: Folder containing the compare data
 *   - Output: Normalized, aggregated accuracy
 *
 * ==preProcessRequests s3RequestFolder numRequests==
 *   - Input: Folder containing the requests
 *   - Output: Folder containing the contextual
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] object MApp {
  final private val logger: Logger = LoggerFactory.getLogger("MApp")

  def main(args: Array[String]): Unit = {
    if (init) {
      import implicits._
      try {
        args(0) match {
          case "testGPUs" => testGPUs(args(1).toInt)
          // Step 0: Extract requests and feedbacks from various sources (S3, Database,....)
          case "updateRequests" => tblToS3Request(args)
          case "updateFeedbacks" => s3ToS3Feedback(args)

          // Step 1: Build vocabulary
          //         - buildVocabulary vocabulary sourceForTFnotes numNotesUsedForWordPiece tfTreshold
          case "buildVocabulary" =>
            logDebug(logger, msg = s"Cmd line: ${args.mkString(" ")}")
            MedicalVocabularyBuilder.vocabulary(args)

          // Step 2: Build training set: Contextual document for transformer/BERT encoder
          //         and training (Request grouped by emr) for neural classifier
          //  - buildTrainingSet                             # Breaks all sub-models into sub group saved on temp/
          //  - buildTrainingSet generateSubModels           # Generates subModels and indexed labels files
          //  - buildTrainingSet Vocabulary numRecords false # Generate contextual and training set
          //  - buildTrainingSet Vocabulary numRecords true  # Generate training set from contextual documents and feedbacks
          case "buildTrainingSet" =>
            logDebug(logger, msg = s"""Cmd line: ${args.mkString(" ")}""")
            TrainingSetBuilder(args)
            // Close Spark context
            org.bertspark.implicits.close

          // Step 4: Pre-train the BERT encoder
          //   - preTrainBert         # To launch the pre-training
          case "preTrainBert" =>
            logDebug(logger, msg = s"Cmd line: ${args.mkString(" ")}")
            TPretraining.execute(args)

          // Step 5: Train the MLP classifier
          //  - trainClassifier file.txt  # To training a list of sub-models defined in file.txt
          //  - trainClassifier ALL       # Train all sub models in S3 training folder
          //  - trainClassifier ALL 6     # Train all sub models in S3 training folder for 6 sub models for augmentation
          case "trainClassifier" =>
            logDebug(logger, msg = s"Cmd line: ${args.mkString(" ")}")
            TClassifier(args).apply()

          // Step 6: Evaluate the prediction from a random sampling of requests and feedbacks
          //     - evaluate s3 fromRequest numRequests   # Evaluate using random sample from requests and feedbacks
          //     - evaluate s3 fromTraining numRequests  # Evaluate using random sample from training set and feedbacks
          //     - evaluate s3 fromModels numRequests    # Evaluate from models loaded on S3
          //     - evaluate Kafka numRandomRequests      # Evaluate using Kafka messages
          case "evaluate" =>
            logDebug(logger, msg = s"Cmd line: ${args.mkString(" ")}")
            TEvaluator(args).execute

          // Step 7: Launch predictor through Kafka
          //         - launch
          case "launch" => TEvaluator().execute

          // --------------  Utilities/Support ------------------------------

          case "createAliases" =>
            import org.bertspark.implicits._
            val fraction = args(1).toDouble
            extractAliases(fraction)

          case "estimateCodingRate" => println(estimateAutoCodingRate)

          case "countDataset" =>
            logDebug(logger, msg = s"Cmd line: ${args.mkString(" ")}")
            Reporting.datasetCounts(args)
          case "repartition" => Reporting.repartition(args)
          case "kafkaPipeline" => KafkaPipeline(args)
          case "extractTermsFrequencies" => TokensTfIdf.extractTermsFrequencies(args)
          case "kafkaSimulator" => KafkaSimulator(args)
          case "similarity" => logger.info(DocumentEmbeddingSimilarity().similarity(args(1).toInt).toString)
          case "plot" => plot(args)
          case "medicalAbbreviations" => MedicalAbbreviations.create
          case "transferLearning" =>
            logDebug(logger, msg = s"Cmd line: ${args.mkString(" ")}")
            (new TTransferLearning)()
          case "createRequestRouting" => createRequestRouting
          case "numTokensPerDoc" =>
            logDebug(logger, msg = s"Cmd line: ${args.mkString(" ")}")
            Reporting.getNumTokensPerDoc

          case "analyze" => logger.info(PredictionAnalysis.newInstance(args).execute)
          case "analyzeVocabulary" => logger.info(new VocabularyAnalyzer(args(1).toInt).analyze(args(2).toBoolean))
          case "classifyHPO" =>
            ClassifierTrainHPO(args).map(_.execute)
                .getOrElse(throw new IllegalStateException("Classifier train HPO not defined"))
          case _ =>
            logger.error(s"Command line argument incorrect: ${args.mkString(", ")}")
        }
      } catch {
        case e: RunOutGPUMemoryException =>
          e.printStackTrace()
          logger.error(e.getMessage)
        case e: DLException =>
          e.printStackTrace()
          logger.error(s"DL exception ${e.getMessage}")
        case e: IllegalStateException =>
          e.printStackTrace()
          logger.error(s"MApp undefined state ${e.getMessage}")
        case e: Exception =>
          e.printStackTrace()
          logger.error(s"MApp undefined exception: ${e.getMessage}")
      }
    }
  }

  def plot(args: Seq[String]): Unit = {
    val filename = args(1)
    BaseTrainingListener.createPlots(filename)
  }


  private def testGPUs(index: Int): Boolean =
    try {
      logger.info(s"List of engines: ${Engine.getAllEngines.mkString(" ")}")
      val instance = Engine.getInstance()
      val defaultEngineName = instance.getEngineName
      logger.info(s"Default engine name $defaultEngineName")
      // Optional test
      logDebug(logger, testGPU(index))
      true
    }
    catch {
      case e: ExceptionInInitializerError =>
        println(s"ERROR: ${e.getMessage}")
        false
    }

  private def testGPU(index: Int): String = {
    val executor = mlopsConfiguration.executorConfig
    if (executor.dlDevice == "gpu" || executor.dlDevice == "any") {
      val ndManager = if (index == -1) NDManager.newBaseManager() else NDManager.newBaseManager(gpu(index))
      val ndArray = ndManager.arange(1.0F, 20.0F, 0.5F, DataType.FLOAT32, Device.gpu(index))
      val ndResult = ndArray.mul(3.5F)
      val result = ndResult.toFloatArray
      println(s"GPU($index): ${result.mkString(" ")}")

      if (executor.numDevices > 1) {
        val ndArray2 = ndManager.arange(1.0F, 4.0F, 0.5F, DataType.FLOAT32, Device.gpu(1))
        val ndResult2 = ndArray2.add(3.5F)
        val result2 = ndResult2.toFloatArray
        println(s"GPU(1): ${result2.mkString(" ")}")
      }
      ndManager.close()
    }
    else {
      val ndManager = NDManager.newBaseManager()
      val ndArray = ndManager.arange(1.0F, 8.0F, 0.5F, DataType.FLOAT32, Device.cpu())
      val ndResult = ndArray.mul(3.5F)
      val result = ndResult.toFloatArray
      println(s"CPU: ${result.mkString(" ")}")
    }
    "CPU test completed"
  }


  private def init: Boolean = {
    initializeProperty
    configValidation
    true
  }

  override def toString: String = {
    s"""
       |
       |====================================== Configuration =========================================
       |${mlopsConfiguration.toString}
       |===============================================================================================
       |""".stripMargin
  }
}
