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
package org.bertspark.transformer.training

import ai.djl.nn.transformer._
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.loss.Loss
import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark._
import org.bertspark.config.{ExecutionMode, FsPathNames, S3PathNames}
import org.bertspark.util.io.S3Util
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.dl.model.NeuralModel
import org.bertspark.nlp.trainingset.{ContextualDocument, ContextualDocumentGroup, SubModelsTrainingSet}
import org.bertspark.transformer.block.PretrainingModule
import org.bertspark.transformer.config.BERTConfig
import org.bertspark.transformer.dataset.{PretrainingDataset, TDatasetConfig}
import org.bertspark.transformer.model.TPretrainingModel
import org.bertspark.config.MlopsConfiguration.DebugLog.logInfo
import org.bertspark.modeling.ModelExecution
import org.bertspark.nlp.tokenSeparator
import org.slf4j.{Logger, LoggerFactory}


/**
 * {{{
 * Implements the pre-training of a given BERT model for which the configuration is defined in the
 * first experiment in the configuration file conf/bertSparkConfig.json
 * This class invokes the pre-training model
 * }}}
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class TPretraining extends ModelExecution with modeling.InputValidation {
  import TPretraining._
  import org.bertspark.implicits._

  validate(Seq.empty[String])
  private[this] val preTrainedBertModel = BaseTrainingSet.initTrainingModel
  private[this] val pretrainingDataset: PretrainingDataset[ContextualDocument] = BaseTrainingSet.initTrainingSet


  override protected def validate(args: Seq[String]): Unit = {
    if(mlopsConfiguration.executorConfig.batchSize < 2)
      throw new InvalidParamsException(
        s"Pre-training batch size ${mlopsConfiguration.executorConfig.batchSize} should be > 1"
      )
    if(!S3Util.exists(mlopsConfiguration.storageConfig.s3Bucket, S3PathNames.s3ContextualDocumentPath))
      throw new InvalidParamsException(
        s"Contextual documents do not exists for pre-training in ${S3PathNames.s3ContextualDocumentPath}"
      )
  }

  /**
   * Train the pre-training defined by the
    * BERTPreTrainingBlock and BERTPreTrainingModel
   */
  override protected def train(): Float =
    try {
      logInfo(logger, msg = s"Training set is ready!!")
      val loss: Loss = new BertPretrainingLoss()
      val trainingContext = NeuralModel.buildTrainingContext(new NormalInitializer, loss, "Pretraining")

      logInfo(logger,  msg = s"Training context is ready with ${ExecutionMode.toString}")
      val outputPath = preTrainedBertModel.train(trainingContext, pretrainingDataset, testDataset = null)
      logInfo(logger,  msg = s"Training completed with model path: $outputPath")
      1.0F
    }
    catch {
      case e: DLException =>
        org.bertspark.printStackTrace(e)
        logger.error(s"DL failure ${e.getMessage}")
        0.0F
      case e: Exception =>
        org.bertspark.printStackTrace(e)
        logger.error(s"Unknown exception ${e.getMessage}")
        0.0F
    }
    finally {
      import implicits._
      close
    }
}


/**
 * {{{
 *   Singleton for
 *   - Constructors
 *   - Building data set for pre-training
 * }}}
 */
private[bertspark] final object TPretraining {
  final private val logger: Logger = LoggerFactory.getLogger("TPretraining")

  /**
    * Default constructor for which similarity is not computed
    * @return Instance of TPretraining
    */
  def apply(): TPretraining = {
    ExecutionMode.setPretraining
    new TPretraining
  }


  def execute(args: Seq[String]): Unit = {
    ExecutionMode.setPretraining
    if(args.size == 1)
      (new TPretraining)()
    else
      ContextualDocumentTrainingSet.contextualDocumentSizes
  }


  /**
    * Define bast training set
    */
  trait BaseTrainingSet {
    def initTrainingSet(implicit sparkSession: SparkSession): PretrainingDataset[ContextualDocument]

    def initTrainingModel: TPretrainingModel = {
      import FsPathNames._
      val bertConfig = BERTConfig()

      // Select default pre-training block  Use TPretrainingBlock as adapter/interface to both
      val preTrainedBertBlock =
        if(mlopsConfiguration.preTrainConfig.isCustomerPretrainBlock) PretrainingModule(bertConfig)
        else PretrainingModule()

      // Build the training model
      new TPretrainingModel(
        preTrainedBertBlock,
        BERTConfig.getEmbeddingsSize(mlopsConfiguration.preTrainConfig.transformer),
        getPreTrainModelOutput
      )
    }
  }

  final object BaseTrainingSet {
    def initTrainingModel: TPretrainingModel =
      if(mlopsConfiguration.isLabeledSentencesBuilder)
        (new ContextualDocumentGroupTrainingSet).initTrainingModel
      else
        (new ContextualDocumentTrainingSet)initTrainingModel

    def initTrainingSet(implicit sparkSession: SparkSession): PretrainingDataset[ContextualDocument] =
      if(mlopsConfiguration.isLabeledSentencesBuilder)
        (new ContextualDocumentGroupTrainingSet).initTrainingSet
      else
        (new ContextualDocumentTrainingSet).initTrainingSet
  }



  final class ContextualDocumentGroupTrainingSet extends BaseTrainingSet {
    import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug

    override def initTrainingSet(implicit sparkSession: SparkSession): PretrainingDataset[ContextualDocument] = {
      import sparkSession.implicits._

      val s3Folder = S3PathNames.s3ContextualDocumentGroupPath
      logDebug(logger, msg = s"Initialize contextual document clusters from $s3Folder")

      val contextualDocumentClusterDS = try {
        S3Util.s3ToDataset[ContextualDocumentGroup](
          mlopsConfiguration.storageConfig.s3Bucket,
          s3Folder,
          false,
          "json"
        )
      } catch {
        case e: IllegalArgumentException =>
          logger.error(s"Init training set: ${e.getMessage}")
          sparkSession.emptyDataset[ContextualDocumentGroup]
      }

      val contextualDocumentDS: Dataset[ContextualDocument] = contextualDocumentClusterDS.flatMap(
        contextualDocumentCluster => {
          // Implicit conversion from ContextualDocumentCluster to ContextualDocument
          import ContextualDocumentGroup._
          val contextualDocuments: Array[ContextualDocument] = contextualDocumentCluster
          contextualDocuments
        }
      )

      val datasetConfig: TDatasetConfig = TDatasetConfig(true)

      val pretrainingDataset = PretrainingDataset(contextualDocumentDS, datasetConfig)
      pretrainingDataset.prepare()
      logDebug(logger, msg = s"${pretrainingDataset.size} contextual document clusters loaded")
      pretrainingDataset
    }
  }


  /**
   * Build the Pre-training training set
   */
  final class ContextualDocumentTrainingSet extends BaseTrainingSet {
    import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug

    /**
      * Initialize and select the contextual document as input to the training of the transformer
      *
      * @param sparkSession Implicit reference to the current Spark context
      * @return Pre-training data set
      */
    override def initTrainingSet(implicit sparkSession: SparkSession): PretrainingDataset[ContextualDocument] = {

      val datasetConfig = TDatasetConfig(true)
      val contextualDocDS = getContextualDocumentDS
      logDebug(logger, s"${contextualDocDS.count()} contextual documents loaded")
      val pretrainingDataset = PretrainingDataset(contextualDocDS, datasetConfig)
      pretrainingDataset.prepare()
      pretrainingDataset
    }

    /**
      * Extract the contextual documents that are used in pre-training
      *
      * @param sparkSession Implicit reference to the current spark context
      * @return The data set of Contextual document used in pre-training, empty data set if failed to load and
      *         extract training data set from S3
      */
    def getContextualDocumentDS(implicit sparkSession: SparkSession): Dataset[ContextualDocument] = try {
      import sparkSession.implicits._

      val allContextualDocumentDS = S3Util.s3ToDataset[ContextualDocument](
        S3PathNames.s3ContextualDocumentPath,
        header = false,
        fileFormat = "json"
      )

      val contextualDocumentDS =
        if(mlopsConfiguration.preTrainConfig.maxNumRecords == -1) allContextualDocumentDS
        else {
          val sampleFraction = mlopsConfiguration.preTrainConfig.maxNumRecords.toFloat/allContextualDocumentDS.count()
          allContextualDocumentDS.sample(sampleFraction)
        }

      ContextualDocumentTrainingSet.contextualDocumentSizes(contextualDocumentDS)
      contextualDocumentDS
    }
    catch {
      case e: IllegalArgumentException =>
        import sparkSession.implicits._

        logger.error(s"Failed getContextualDocumentDS: ${e.getMessage}")
        sparkSession.emptyDataset[ContextualDocument]
    }
  }


      // ----------------------   Supporting methods -----------------------------------

  final object ContextualDocumentTrainingSet {
    /**
      * Compute the various statistics on the number of tokens relative to the
      * current configuration of the transformer encoder
      */
    final def contextualDocumentSizes: Unit = {
      import org.bertspark.implicits._
      import sparkSession.implicits._

      val contextualDocumentDS = S3Util.s3ToDataset[ContextualDocument](
        S3PathNames.s3ContextualDocumentPath, false, "json"
      )
      contextualDocumentSizes(contextualDocumentDS)
    }


    private def getMaxNumTokens: Int = mlopsConfiguration.preTrainConfig.transformer match {
      case "Bert-micro" => mlopsConfiguration.preTrainConfig.numSentencesPerDoc * 128
      case "Bert-base" => mlopsConfiguration.preTrainConfig.numSentencesPerDoc * 256
      case "Bert-large" => mlopsConfiguration.preTrainConfig.numSentencesPerDoc * 512
    }

    private def contextualDocumentSizes(
      contextualDocDS: Dataset[ContextualDocument]
    )(implicit sparkSession: SparkSession): Unit = {
        import sparkSession.implicits._

      val maxNumTokens = getMaxNumTokens
      val ctxDocSizes = contextualDocDS.map(
          ctxDoc => ctxDoc.contextVariables.size + ctxDoc.text.split(tokenSeparator).size
      ).collect()

      val average = ctxDocSizes.sum.toFloat / ctxDocSizes.size
      val sorted = ctxDocSizes.sortWith(_ > _)
      val percentile20 = sorted((ctxDocSizes.size * 0.2F).toInt)
      val percentile15 = sorted((ctxDocSizes.size * 0.15F).toInt)
      val percentile10 = sorted((ctxDocSizes.size * 0.1F).toInt)
      val percentile5 = sorted((ctxDocSizes.size * 0.05F).toInt)
      val truncatedCtxDocs = ctxDocSizes.filter(_ >= maxNumTokens)

      val summary =
          s"""
             |Training set:       ${mlopsConfiguration.target}
             |Run:                ${mlopsConfiguration.runId}
             |Vocabulary:         ${mlopsConfiguration.preProcessConfig.vocabularyType}
             |Transformer:        ${mlopsConfiguration.preTrainConfig.transformer}
             |Sentences layout:   ${mlopsConfiguration.preTrainConfig.sentenceBuilder}
             |Sentences per doc:  ${mlopsConfiguration.preTrainConfig.numSentencesPerDoc}
             |Smallest doc:       ${ctxDocSizes.min}
             |Largest doc:        ${ctxDocSizes.max}
             |Average doc:        $average
             |Num allowed tokens: $maxNumTokens
             |Percentile 80:      $percentile20
             |Percentile 85:      $percentile15
             |Percentile 90:      $percentile10
             |Percentile 95:      $percentile5
             |Num docs:           ${ctxDocSizes.size}
             |Num truncated docs: ${truncatedCtxDocs.size}
             |Truncation ratio:   ${truncatedCtxDocs.size.toFloat / ctxDocSizes.size}
             |""".stripMargin

      S3Util.upload(
        s"${mlopsConfiguration.storageConfig.s3RootFolder}/${mlopsConfiguration.target}/models/${mlopsConfiguration.runId}/sizes.txt",
        summary
      )
    }
  }
}
