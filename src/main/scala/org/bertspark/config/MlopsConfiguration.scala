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
package org.bertspark.config

import ai.djl.engine._
import ai.djl.training.initializer._
import ai.djl.util.cuda.CudaUtils
import java.util.concurrent.atomic.AtomicInteger
import org.bertspark.util.io.{LocalFileUtil, S3Util}
import org.bertspark.Labels._
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.implicits.sparkSession
import org.bertspark.nlp.vocabulary.MedicalCodeDescriptors.CodeDescriptorMap
import org.bertspark.transformer.config.BERTConfig
import org.bertspark.transformer.dataset._
import org.bertspark.util.{DateUtil, EncryptionUtil, TProgress}
import org.bertspark.DLException
import org.bertspark.config.ExecutionMode.isPretraining
import org.bertspark.nlp.TokenizerLoader
import org.slf4j.{Logger, LoggerFactory}


/**
 *
 * @param s3Bucket
 * @param s3RootFolder
 * @param s3RequestFolder
 * @param s3FeedbackFolder
 * @param s3ContextDocumentFolder
 * @param encryptedS3AccessKey
 * @param encryptedS3SecretKey
 */
case class StorageConfig(
  s3Bucket: String,
  s3RootFolder:String,
  s3RequestFolder:String,
  s3FeedbackFolder:String,
  s3ContextDocumentFolder: String,
  encryptedS3AccessKey: String,
  encryptedS3SecretKey: String
) {
  override def toString: String =
    s"""S3 bucket:           $s3Bucket
    |S3 root:             $s3RootFolder
    |S3 request folder:   $s3RequestFolder
    |S3 feedback folder:  $s3FeedbackFolder
    |S3 Context document: $s3ContextDocumentFolder
    |Encrypted access key xxxxxxx
    |Encrypted secret key xxxxxxx""".stripMargin
}

/**
 * Wrapper for the configuration of a RDS database
 * @param host Name of the host of the database
 * @param port Port the database is listening to
 * @param dbName Name of the database
 * @param user User of the database
 * @param encryptedPwd Password for the database
 * @param region Region the RDS service is located
 */
case class DatabaseConfig(
  name: String,
  host: String,
  port: Int,
  dbName: String,
  user: String,
  encryptedPwd: String,
  region: String) {
  override def toString: String =
    s"""Name:          $name
       |Host:          $host
       |Port           $port
       |Database:      $dbName
       |User           $user
       |Password:      XXXX
       |Region:        $region""".stripMargin

  final def parameters: (String, String, Int, String, String) = (dbName, host, port, user, encryptedPwd)
}

/**
 * Wrapper for the configuration of real-time database
 * @param databaseConfig Set of database configuration in RDS
 */
case class DatabasesConfig(databaseConfigs: Seq[DatabaseConfig]) {
  override def toString: String = s"\nDatabases ------------\n${databaseConfigs.mkString("\n\n")}"

  final def parameters: Seq[(String, String, Int, String, String)] =
    databaseConfigs.map(
      config =>
        (config.dbName, config.host, config.port, config.user, EncryptionUtil.unapply(config.encryptedPwd).getOrElse(""))
    )
}





case class ExecutorConfig(
  dlDevice: String,
  dlEngine: String,
  saveModelMode: String,
  batchSize: Int,
  numDevices: Int,
  numThreads: Int,
  maxRssMemMB: Int,
  maxHeapMemMB: Int,
  maxNonHeapMemMB: Int
) {

  final def isMxNetGPU: Boolean = dlDevice == "gpu" && dlEngine == "MXNet" && numDevices > 1

  final def saveOnEachEpoch: Boolean = saveModelMode == "epoch"
  final def saveEndOfTraining: Boolean = saveModelMode == "training"
  final def saveNone: Boolean = saveModelMode == "none"

  override def toString: String =
    s"""Device config.        $dlDevice
    |Device:               ${if(dlDevice == "any") Engine.getInstance.getDevices.map(_.getDeviceType).mkString(" ") else dlDevice}
    |Engine:               $dlEngine
    |Save model mode:      $saveModelMode
    |batchSize             $batchSize
    |Num devices:          $numDevices
    |Num threads:          $numThreads
    |max RSS memory:       $maxRssMemMB
    |max heap memory:      $maxHeapMemMB
    |max non heap memory:  $maxNonHeapMemMB""".stripMargin
}





/**
 *  Configuration for the pre-processor
 * @param minLabelFreq Minimum number of occurrences for a label to be valid for training
 * @param maxLabelFreq Maximum number of occurrences for a label for cut-off
 * @param numSplits Number of split for the various Spark driven pre-processing tasks
 * @param contextualEnabled Are context data included in the tokens
 * @param vocabularyType Path for vocabulary file
 * @param codeSemanticEnabled Is the code -> Associated definition terms enabled?
 */
case class PreProcessConfig(
  minLabelFreq: Int,
  maxLabelFreq: Int,
  customers: Seq[String],
  subModels: Seq[String],
  numSplits: Int,
  sampleSize: Int,
  vocabularyType: String,
  contextualEnabled: Boolean
) {
  require(minLabelFreq > 0, s"Minimum label frequency, $minLabelFreq should be > 0")
  require(maxLabelFreq > minLabelFreq, s"Maximum label frequency, $minLabelFreq should be > $minLabelFreq")
  require(numSplits > 0, s"Number of splits for pre processing should be > 0")

  override def toString: String =
    s"""Min label freq:      $minLabelFreq
    |Max label freq:      $maxLabelFreq
    |Num. splits:         $numSplits
    |Sample size:         $sampleSize
    |Context enabled:     $contextualEnabled
    |Vocabulary type:     $vocabularyType""".stripMargin
}

/**
  *
  * @param modelPrefix
  * @param transformer
  * @param isCustomerPretrainBlock
  * @param sentenceBuilder
  * @param numSentencesPerDoc
  * @param tokenizer
  * @param predictor
  * @param clsAggregation
  * @param epochs
  * @param numSplits
  * @param maxNumRecords
  * @param maxMaskingSize
  * @param optimizer
  */
case class PreTrainConfig(
  modelPrefix: String,
  transformer: String,
  isCustomerPretrainBlock: Boolean,
  sentenceBuilder: String,
  numSentencesPerDoc: Int,
  tokenizer: String,
  predictor: String,
  clsAggregation: String,
  epochs: Int,
  numSplits: Int,
  maxNumRecords: Int,
  maxMaskingSize: Int,
  optimizer: OptimizerConfig
) {
  require(epochs > -1 && epochs < 49, s"Number of pre-training epochs $epochs should be [0, 48]")
  require(numSplits > 0 && numSplits <= 64, s"Number of pre-training splits, $numSplits should be [2, 64]")
  require(maxNumRecords == -1 || maxNumRecords > 2,
    s"Number of sub-models $maxNumRecords to be pre-trained should be > 2 or -1")
  require(numSentencesPerDoc > 0 && numSentencesPerDoc < 64,
    s"Num sentences per doc $numSentencesPerDoc should be [1, 63]")
  import MlopsConfiguration._

  override def toString: String =
    s"""Model prefix:          $modelPrefix
       |Transformer model:     $transformer
       |Has custom pretraining:$isCustomerPretrainBlock
       |Sentences builder:     $sentenceBuilder
       |Num sentences per doc: $numSentencesPerDoc
       |Tokenizer:             $tokenizer
       |Predictor:             $predictor
       |CLS aggregation:       $clsAggregation
       |Num. epochs:           $epochs
       |Num. splits:           $numSplits
       |Max. num. records:     $maxNumRecords
       |Max. masking size:     $maxMaskingSize
       |Optimizer:             ${optimizer.optType}
       |Base learning rate:    ${optimizer.baseLr}
       |Number of steps        ${optimizer.numSteps}
       |Epsilon                ${optimizer.epsilon}
       |""".stripMargin

  final def getSummary: String = s"${transformer}-$sentenceBuilder-$tokenizer"

  @throws(clazz = classOf[IllegalArgumentException])
  def isValid: Boolean = {
    import Validation._

    require(validate(transformerLbl, transformer), s"$transformerLbl $transformer not supported")
    require(validate(sentencesBuilderLbl, sentenceBuilder), s"$sentencesBuilderLbl $sentenceBuilder not supported")
    require(validate(tokenizeLbl, tokenizer), s"$tokenizeLbl $tokenizer not supported")
    require(epochs < 65, s"$epochsLbl $epochs should <= 64")
    require(numSplits > 0 && numSplits < 65, s"$numSplitsLbl $numSplits is out of bounds [1, 64]")
    true
  }

  final def isEmbeddingConcatenate: Boolean = clsAggregation == "concatenate"

  final def isEmbeddingSum: Boolean = clsAggregation == "sum"

  final def isPredictorClsEmbedding: Boolean = predictor == "clsEmbedding"

  final def isPredictionPooledEmbedding: Boolean = predictor == "pooledEmbedding"

  final def isBertMacro: Boolean = transformer == "Bert-micro"

  final def isBertNano: Boolean = transformer == "Bert-nano"

  final def isBertBase: Boolean = transformer == "Bert-base"
}




/**
 * Configuration of the optimizer
 * @param baseLr Base learning rate
 * @param numSteps Number step in the computation of the partial derivatives (backward)
 * @param epsilon Epsilon value is neede
 */
case class OptimizerConfig(optType: String, baseLr: Float, numSteps: Int, epsilon: Float, convergenceLossRatio: Float) {
  override def toString: String =
    s"""Optimizer:            $optType
       |Base learning rate:   $baseLr
       |Num. gradient steps:  $numSteps
       |Epsilon:              $epsilon
       |Convergence loss rate $convergenceLossRatio""".stripMargin

}


/**
  * @param modelId
  * @param modelPrefix
  * @param lossFunction
  * @param weightInitialization
  * @param dlModel
  * @param dlLayout
  * @param augmentation
  * @param trainValidateRatio
  * @param epochs
  * @param numSplits
  * @param minNumRecordsPerLabel
  * @param maxNumRecordsPerLabel
  * @param optimizer
  */
case class ClassifyConfig(
  modelId: String,
  modelPrefix: String,
  lossFunction: String,
  weightInitialization: String,
  dlModel: String,
  dlLayout: Array[Int],
  augmentation: String,
  trainValidateRatio: Float,
  epochs: Int,
  numSplits: Int,
  minNumRecordsPerLabel: Int,
  maxNumRecordsPerLabel: Int,
  optimizer: OptimizerConfig) {
  import MlopsConfiguration._
  isValid

  def isValid: Boolean = {
    require(trainValidateRatio > 0.1F && trainValidateRatio < 1.0F,
      s"$trainValidateRatioLbl $trainValidateRatio is out of bounds [0.1, 1.0[")
    require(epochs > 0 && epochs < 65, s"$epochsLbl $epochs is out of bounds [2, 64]")
    require(numSplits > 0 && numSplits < 65, s"$numSplitsLbl $numSplits is out of bounds [1, 64]")
    require(minNumRecordsPerLabel > 0 && minNumRecordsPerLabel < 8192,
      s"$minSampleSizeLbl ${minNumRecordsPerLabel} is out of bounds [1, 8192]")
    require(maxNumRecordsPerLabel > minNumRecordsPerLabel, s"$minSampleSizeLbl order in sample size window is incorrect")
    true
  }

  def isAugmentation: Boolean = augmentation != "filter"

  @inline
  final def isSuccessRateLoss: Boolean = lossFunction == "successRate"

  @inline
  final def isMlopsLoss: Boolean = lossFunction == "mlopsLoss"

  @inline
  final def isDefaultLoss: Boolean = lossFunction == "default"

  override def toString: String =
    s"""Model prefix:          $modelPrefix
       |Model type:            $dlModel
       |FFNN layout:           ${dlLayout.mkString(" ")}
       |Weight initialization  $weightInitialization
       |Augmentation:          $augmentation
       |Train-validate ratio:  $trainValidateRatio
       |Num. epochs:           $epochs
       |Num. splits:           $numSplits
       |Min size model/class:  $minNumRecordsPerLabel
       |Max size model/class:  $maxNumRecordsPerLabel
       |Loss function:         $lossFunction
       |Weight initialization  $weightInitialization
       |Optimizer:             ${optimizer.optType}
       |Base learning rate:    ${optimizer.baseLr}
       |Number of steps        ${optimizer.numSteps}
       |Epsilon                ${optimizer.epsilon}""".stripMargin
}


/**
  * *
  * @param s3RequestPath
  * @param s3FeedbackPath
  * @param requestTopic
  * @param responseTopic
  * @param feedbackTopic
  * @param ingestIntervalMs
  * @param ackTopic
  * @param compareEnabled
  * @param preloadedSubModels
  * @param numRequestPerSubModel
  */
case class EvaluationConfig(
  classifierOnly: Boolean,
  s3RequestPath: String,
  s3FeedbackPath: String,
  subModelFilterThreshold: Float,
  ingestIntervalMs: Int,
  numRequestPerSubModel: Int,
  compareEnabled: Boolean,
  preloadedSubModels: Boolean
) {
  override def toString: String =
    s"""
       |Classifier only:          $classifierOnly
       |s3 Request Path:          $s3RequestPath
       |s3 Feedback Path:         $s3FeedbackPath
       |Ingest intervalMs         $ingestIntervalMs
       |subModel filter threshold $subModelFilterThreshold
       |Comparison enabled        $compareEnabled
       |Sub models preloaded      $preloadedSubModels
       |Requests per sub model    $numRequestPerSubModel
       |""".stripMargin
}


case class RuntimeConfig(
  predictionStorage: String,
  table: String,
  requestTopic: String,
  responseTopic: String,
  feedbackTopic: String,
  ackTopic: String,
) {
  override def toString: String =
    s"""
       |Prediction Storage:    $predictionStorage
       |Database table:        $table
       |Request topic:         $requestTopic
       |Response topic         $responseTopic
       |Feedback topic         $feedbackTopic
       |Ack topic              $ackTopic
       |""".stripMargin
}


/**
 * Generic configuration for the DL service
 * @param version Version of release
 * @param date Date of release
 * @param debugLogLevel Log level (info, error, debug,..)
 * @param storageConfig Configuration for the storage
 * @param executorConfig Configuration for the executors
 * @param preProcessConfig Configuration for the pre-processor
 * @param preTrainConfig Configuration for the pre-training model
 * @param classifyConfig Configuration for the classification model
 */
case class MlopsConfiguration(
  version: String,
  date: String,
  runId: String,
  debugLogLevel: String,
  target: String,
  storageConfig: StorageConfig,
  databasesConfig: Seq[DatabaseConfig],
  executorConfig: ExecutorConfig,
  preProcessConfig: PreProcessConfig,
  preTrainConfig: PreTrainConfig,
  classifyConfig: ClassifyConfig,
  evaluationConfig: EvaluationConfig,
  runtimeConfig: RuntimeConfig
) {
  override def toString: String = try {
    val instance = Engine.getInstance()
    val isCuda = instance.hasCapability(StandardCapabilities.CUDA)
    s"""
       |Version:            $version
       |Log:                $debugLogLevel
       |Pre-training mode:  $isPretraining
       |Target:             $target
       |Transformer model:  $runId
       |
       |Storage ------------------------------------------------
       |${storageConfig.toString}
       |
       |RDMBS --------------------------------------------------
       |${databasesConfig.mkString("\n\n")}
       |
       |Executor ------------------------------------------------
       |${executorConfig.toString}
       |
       |Pre-processor -------------------------------------------
       |${preProcessConfig.toString}
       |
       |Encoder transformer -------------------------------------
       |${preTrainConfig.toString}
       |Classifier ----------------------------------------------
       |${classifyConfig.toString}
       |Evaluation ----------------------------------------------
       |${evaluationConfig.toString}
       |Runtime ----------------------------------------------
       |${runtimeConfig.toString}
       |
       |Properties ----------------------------------------------
       |Java version:          ${System.getProperty("java.version")}
       |java.library.path:     ${System.getProperty("java.library.path")}
       |Scala version:         ${scala.util.Properties.versionString}
       |Hadoop version:        ${org.apache.hadoop.util.VersionInfo.getVersion}
       |Spark version:         ${sparkSession.version}
       |DJL version:           0.21.0
       |DL engine:             ${Engine.getInstance().getEngineName()}
       |DL engine version:     ${Engine.getInstance().getVersion()}
       |DL device:             ${Engine.getInstance().defaultDevice()}
       |GPU count:             ${if(isCuda) CudaUtils.getGpuCount() else "0"}
       |CUDA version:          ${if(isCuda) CudaUtils.getCudaVersionString else "NA"}
       |cuDNN available:       ${Engine.getInstance().hasCapability(StandardCapabilities.CUDNN)}
       |Collect memory:        ${System.getProperty("collect-memory")}
       |""".stripMargin
  }
  catch {
    case e: ExceptionInInitializerError =>
      println(s"ERROR: ${e.getMessage}")
      "Configuration failed."
  }

  @inline
  final def isLabeledSentencesBuilder: Boolean = preTrainConfig.sentenceBuilder == labeledSentencesBuilderLbl

  /**
   * Retrieve some key training parameters
   * {{{
   *   Optimizer configuration
   *   Max number of epochs
   *   Batch size
   * }}}
   *
   * @return Tuple  (Optimizer config, num epochs, batch Size)
   */
    @throws(clazz = classOf[UnsupportedOperationException])
  final def getTrainingParams: (OptimizerConfig, Int, Int) = {
    if(ExecutionMode.isPretraining || ExecutionMode.isTransferLearning)
      (preTrainConfig.optimizer, preTrainConfig.epochs, executorConfig.batchSize)
    else if(ExecutionMode.isClassifier || ExecutionMode.isHpo)
      (classifyConfig.optimizer, classifyConfig.epochs, executorConfig.batchSize)
    else if(ExecutionMode.isTest)
      (null, -1, -1)
    else
      throw new UnsupportedOperationException(s"Execution model ${ExecutionMode.toString} for training parameters is not supported")
  }

  final def isGPU: Boolean = executorConfig.dlDevice == "any" || executorConfig.dlDevice == "gpu"

  final def saveOnEachEpoch = executorConfig.saveModelMode == "epoch" && ExecutionMode.isPretraining

  @inline
  final def isSingleSegmentDocument: Boolean = numSegmentsPerDocument == 1


  /**
   * Extracts the number of segments per document (used in the Next Sentence predictor)
   *
   * @return Number of segments or 'sentences' for each document
   */
  @throws(clazz = classOf[UnsupportedOperationException])
  final def numSegmentsPerDocument: Int = preTrainConfig.sentenceBuilder match {
    case `ctxTxtNSentencesBuilderLbl` |
         `ctxNSentencesBuilderLbl` |
         `labeledSentencesBuilderLbl` |
          `sectionsSentencesBuilderLbl` => mlopsConfiguration.preTrainConfig.numSentencesPerDoc
    case _ =>
      throw new UnsupportedOperationException(s"Sentence builder ${preTrainConfig.sentenceBuilder} is not supported")
  }

  @throws(clazz = classOf[UnsupportedOperationException])
  final def getInitializer: Initializer = classifyConfig.weightInitialization match {
    case "normal" => new NormalInitializer()
    case "truncatedNormal" => new TruncatedNormalInitializer(0.02F)
    case "xavier" => new XavierInitializer()
    case _ =>
      println(s"Initializer ${classifyConfig.weightInitialization} not supported, revert to normal initializer")
      new NormalInitializer()
  }
  /**
   * {{{
   *  Compute the size of the document embedding vector:
   *  IF concatenate the segment then the output is size_embedding (BERT) * numSegments
   *   ELSE size_embedding
   * }}}
   */
  final def getPredictionOutputSize: Int =
    if(preTrainConfig.isEmbeddingConcatenate)
      (getEmbeddingsSize*numSegmentsPerDocument).toInt
    else
      getEmbeddingsSize.toInt

  @inline
  def isLogLevelDebug: Boolean = debugLogLevel == "debug"

  @inline
  def isLogLevelInfo: Boolean = debugLogLevel == "info"

  /**
   * Test assertion along the data flow if        ogLevel == debug
   *
   * @param assertion Assertion (value == true)
   * @param msg       Message
   */
  def check(assertion: Boolean, msg: String): Unit =
    if(debugLogLevel == "debug") assert(assertion, msg)

  @inline
  def isLogInfo: Boolean = debugLogLevel == "info"

  @inline
  def isLogDebug: Boolean = debugLogLevel == "debug"

  @inline
  def isLogTrace: Boolean = debugLogLevel == "trace"

  def isValid: Boolean = preTrainConfig.isValid



  @inline
  final def getTokenizer: String = preTrainConfig.tokenizer


  @inline
  final def getTransformer: String = preTrainConfig.transformer

  @inline
  final def getBatchSize: Int =  executorConfig.batchSize

  @inline
  final def getEpochs: Int = preTrainConfig.epochs

  @inline
  final def getMaxMaskingSize: Int = preTrainConfig.maxMaskingSize

  @inline
  final def getSentenceBuilder: String = preTrainConfig.sentenceBuilder

  final def getEmbeddingsSize: Long = BERTConfig.getEmbeddingsSize(preTrainConfig.transformer)

  final def getMinSeqLength: Int = BERTConfig.getMinSeqLength(preTrainConfig.transformer)

  def isClsEmbedding: Boolean = preTrainConfig.predictor == "clsEmbedding"

  def isNSP: Boolean = preTrainConfig.sentenceBuilder != "ctxTxtSentencesBuilder"

  final def getParameters: Seq[String] = Seq[String](
    s"date,${DateUtil.longToDate}",
    s"target,$target",
    s"tokenizer,${preProcessConfig.vocabularyType}",
    s"transformerModelNo,$runId",
    s"transformer,${preTrainConfig.transformer}",
    s"customerPretraining,${preTrainConfig.isCustomerPretrainBlock}",
    s"transformerSentencesModel,${preTrainConfig.sentenceBuilder}",
    s"transformerPredictor,${preTrainConfig.predictor}",
    s"transformerClsAggregator,${preTrainConfig.clsAggregation}",
    s"transformerLR,${preTrainConfig.optimizer.baseLr}",
    s"classifierModelNo,${classifyConfig.modelId}",
    s"classifierMode,${classifyConfig.dlModel}",
    s"classifierModelLayout,${classifyConfig.dlLayout}",
    s"ClassifierConvergence,${classifyConfig.optimizer.convergenceLossRatio}",
    s"ClassifierLR,${classifyConfig.optimizer.baseLr}"
  )
}




private[org] final object MlopsConfiguration {
  import scala.collection.mutable.HashMap

  final private val mlopsConfigFilename = "conf/bertSparkConfig.json"

  final val transformerLbl = "Transformer"
  final val sentencesBuilderLbl = "SentencesBuilder"
  final val tokenizeLbl = "Tokenizer"
  final val epochsLbl = "Epochs"
  final val numSplitsLbl = "Splits"
  final val numBatchesLbl = "NumBatches"
  final val minSampleSizeLbl = "minSampleSize"
  final val targetSampleSizeLbl = "targetSampleSize"
  final val trainValidateRatioLbl = "TrainValidateRatio"
  final val batchSizeLbl = "BatchSize"
  final val miniBatchEnabledLbl = "MiniBatchEnabled"
  final val compareLossUsedLbl = "Compare loss used"

  final object ConstantParameters {
    final val termsSetFile = "conf/codes/terms.txt"
    final val medicalTermsSetFile = "conf/codes/medicalTerms.txt"
    final val simFactor = "cosine"
    final val numSubModelsToTrain = 10000000
  }


  /**
   * Wrapper for validation of configuration parameters
   */
  final object Validation {
    final val validationMap = HashMap[String, Set[String]]()

    def register(validationVariable: String, validationSet: Set[String]): Unit =
      validationMap.put(validationVariable, validationSet)

    def validate(validationVariable: String, value: String): Boolean =
      validationMap.getOrElse(validationVariable, Set.empty[String]).contains(value)
  }


  var mlopsConfiguration = {
    configValidation
    LocalFileUtil.Json.load[MlopsConfiguration](mlopsConfigFilename, classOf[MlopsConfiguration])
        .getOrElse(throw new DLException(s"Could not load service configuration from $mlopsConfigFilename"))
  }

  def getJsonMlopsConfiguration: String = LocalFileUtil.Json.mapper.writeValueAsString(mlopsConfiguration)

  lazy val dataConfigMap: Map[String, DatabaseConfig] = {
    val dbaseConfigs = mlopsConfiguration.databasesConfig
    dbaseConfigs.map(
      dbConfig => {
        val _pwd = dbConfig.encryptedPwd
        val decryptedDatabaseConfig = dbConfig.copy(encryptedPwd = EncryptionUtil.unapply(_pwd).getOrElse(""))
        (dbConfig.name, decryptedDatabaseConfig)
      }
    ).toMap
  }

  // Load/Instantiate the vocabulary
  lazy val vocabulary = {
    val vocabularyLoader = new TokenizerLoader {
      override val minFrequency = TDatasetConfig(false).getMinTermFrequency
    }
    vocabularyLoader.model
  }.getOrElse(throw new IllegalStateException("Vocabulary cannot be initialized!"))

  lazy val padIndex = vocabulary.getIndex(padLabel)
  lazy val clsIndex = vocabulary.getIndex(clsLabel)
  lazy val mskIndex = vocabulary.getIndex(mskLabel)
  lazy val unkIndex = vocabulary.getIndex(unkLabel)
  lazy val sepIndex = vocabulary.getIndex(sepLabel)



  /**
   * Singleton wrapper to compute and report latency, and progress in computation
   */
  trait LatencyLog extends TProgress[Int, AtomicInteger] {
    final private val logger: Logger = LoggerFactory.getLogger("LatencyLog")

    private var start = -1L
    final val sampleInterval = 64
    private[this] val recordCount = new AtomicInteger(0)
    private[this] val epochCount = new AtomicInteger(0)

    override protected[this] val progress = (cnt: AtomicInteger) => (cnt.get() * 100.0 / maxValue).floor.toInt
    logger.info(
      s"Expected $maxValue batches, ${maxValue * mlopsConfiguration.executorConfig.batchSize} predictions"
    )

    def log(logger: Logger): Unit = {
      val recordCnt = recordCount.getAndIncrement()
      if(recordCnt == 1)
        epochCount.getAndIncrement()
      if (recordCnt % sampleInterval == 0x00)
        DebugLog.logDebug(logger, estimateDuration)
    }

    // -----------------------   Supporting methods ---------------------------
    private def estimateDuration: String =
      if (ExecutionMode.isPretraining) {
        val totalCount = recordCount.get*epochCount.get
        show(new AtomicInteger(totalCount), "Transformer pretraining")

        if (start == -1L) {
          start = System.currentTimeMillis()
          ""
        }
        else {
          val numRecords = recordCount.get()
          val numPredictions = numRecords * mlopsConfiguration.executorConfig.batchSize
          val duration = (System.currentTimeMillis() - start) * 0.001

          val remainingDuration = (maxValue.toFloat / numRecords - 1.0) * duration
          val remainingMins = (remainingDuration / 60).floor.toInt
          val remainingHours = (remainingMins / 60).floor.toInt
          s"Records: $numRecords, Predictions: $numPredictions, Remaining: $remainingHours Hr ${remainingMins - remainingHours * 60} Min"
        }
      }
      else
        ""
  }

    // -------------------  DEBUGGER ------------------------------

  final object DebugLog {
    final private val infoLevel = 0
    final private val debugLevel = 1
    final private val traceLevel = 2

    final val debugLoggerLevelInt = mlopsConfiguration.debugLogLevel match {
      case "info" => infoLevel
      case "debug" => debugLevel
      case "trace" => traceLevel
      case _ => -1
    }

    @inline
    final def isInfoLevel: Boolean = debugLoggerLevelInt >= infoLevel

    @inline
    final def isDebugLevel: Boolean = debugLoggerLevelInt >= debugLevel

    @inline
    final def isTraceLevel: Boolean = debugLoggerLevelInt >= traceLevel

    def logInfo(logger: Logger, msg: String): Unit = if(isInfoLevel) logger.info(s"INFO: $msg")

    def logDebug(logger: Logger, msg: String): Unit = if(isDebugLevel) logger.info(s"DEBUG: $msg")

    def logTrace(logger: Logger, msg: String): Unit = if(isTraceLevel) logger.info(s"TRACE: $msg")
  }


  /**
   * Initialize the environment variables
   */
  def initializeProperty: Unit = {
    System.setProperty("ai.djl.default_engine", mlopsConfiguration.executorConfig.dlEngine)
    System.setProperty("collect-memory", "true")

    // For MXNet running on GPU... set up the thread pool accordingly
    if(mlopsConfiguration.executorConfig.isMxNetGPU) {
      System.setProperty("MXNET_ENGINE_TYPE", "NaiveEngine")
      System.setProperty("OMP_NUM_THREADS", "1")
      System.setProperty("MXNET_PROFILER_AUTOSTART", "1")
    }
  }


  /**
    * Save the current configuration into the models directory for transformer or classification models
    * @param isTransformerConfig Flag to specify this is a transformer model
    */
    @throws(clazz = classOf[IllegalStateException])
  def saveConfiguration(isTransformerConfig: Boolean): Unit = {
    import S3PathNames._
    val s3Folder =
      if(isTransformerConfig)  s"$s3TransformerModelPath/bertSparkConfig.json"
      else s"${S3PathNames.s3ClassifierModelPath}/bertSparkConfig.json"

    LocalFileUtil.Load.local(mlopsConfigFilename).map(
      S3Util.upload(mlopsConfiguration.storageConfig.s3Bucket, s3Folder, _)
    ).getOrElse(
      throw new IllegalStateException(s"Failed to save configuration for $s3Folder")
    )
  }

}
