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
package org.bertspark.classifier.training

import ai.djl.ndarray.NDManager
import java.util.concurrent.atomic.AtomicInteger
import org.apache.spark.sql._
import org.bertspark._
import org.bertspark.classifier.dataset.ClassifierDatasetLoader
import org.bertspark.classifier.model.TClassifierModel
import org.bertspark.config._
import org.bertspark.config.MlopsConfiguration.DebugLog._
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.modeling.{ModelExecution, SubModelOperations}
import org.bertspark.modeling.SubModelsTaxonomy.subModelTaxonomy
import org.bertspark.nlp.medical._
import org.bertspark.nlp.trainingset._
import org.bertspark.transformer.representation.PretrainingInference
import org.bertspark.util.io._
import org.bertspark.util.TProgress
import org.slf4j._
import scala.collection.mutable.ListBuffer


/**
  * {{{
  * Implements the training of a classifier of a given BERT model for which the configuration is defined
  * in the first experiment in the configuration file conf/bertSparkConfig.json
  * Training: Train only with sub-model which has more than one labels (not oracle)
  * }}}
  * @param subModelNames Target sub models 'All' for all the targets
  * @param numSubModelsForAugmentation Number of sub models used in the evaluation of augmentation techniques
  *
  * @author Patrick Nicolas
  * @version 0.2
 */
private[bertspark] final class TClassifier private (
  subModelNames: Set[String],
  numSubModelsForAugmentation: Int
)(implicit sparkSession: SparkSession)
    extends ModelExecution
        with TProgress[Int, Int]
        with modeling.InputValidation {
  import TClassifier._

  validate(Seq.empty[String])
  ExecutionMode.setClassifier
  private[this] lazy val pretrainingInference = PretrainingInference()


  /**
    * Validate the command line arguments or throw a InvalidParamsException if validation fails
    * @param args Command line arguments
    */
  override protected def validate(args: Seq[String]): Unit = {
    import S3Util._

    if(numSubModelsForAugmentation == 0 || numSubModelsForAugmentation < -1)
      throw new InvalidParamsException(
        s"Number of sub models ${numSubModelsForAugmentation} is out of range"
      )
    if(!pretrainingInference.isPredictorReady)
      throw new InvalidParamsException(
        s"Transformer encoder predictor is not available"
      )
    if(!exists(mlopsConfiguration.storageConfig.s3Bucket, S3PathNames.s3ModelTrainingPath))
      throw new InvalidParamsException(
        s"Training data does not exist for classifier in ${S3PathNames.s3ModelTrainingPath}"
      )
    if(!exists(mlopsConfiguration.storageConfig.s3Bucket, S3PathNames.s3SubModelsStructure))
      throw new InvalidParamsException(
        s"Sub models distribution does not exist for classifier in ${S3PathNames.s3SubModelsStructure}"
      )
    if(!exists(mlopsConfiguration.storageConfig.s3Bucket, S3PathNames.s3LabelIndexMapPath))
      throw new InvalidParamsException(
        s"Label index map does not exist for classifier in ${S3PathNames.s3LabelIndexMapPath}"
      )
  }


  /**
    * Outbound for the progress bar
    */
  override protected[this] val maxValue: Int = subModelNames.size

  /**
    * Compute the current progress in execution as percentage
    */
  override protected[this] val progress: Int => Int =
    (numSubModelsProcessed: Int) => (numSubModelsProcessed*100.0/maxValue).floor.toInt

  /**
   * Train the BERT classifier using the CLS prediction from the pre-trained model
   * {{{
   * Steps:
   *   1. Load the labeled training data from the appropriate Storage
   *   2. Extract the keyed CLS token embedding for each segments
   *   3. Concatenate the CLS token embedding of all the segments within a document
   *   4. Build the Labeled DJL formatted data set
   *   5. Initialize the classifier
   *   6. Train the classifier
   * }}}
   */
  @throws(clazz = classOf[DLException])
  override protected def train(): Float = {
    import sparkSession.implicits._

    logDebug(logger, msg = s"Start training classifier for ${subModelNames.size} models")
    val classifierModel = new TClassifierModel

    logDebug(logger, msg = s"Process ${subModelNames.size} sub models")
    val (accuracySum, _) = train(classifierModel, subModelNames)
    accuracySum/subModelNames.size
  }


  private def train(
    classifierModel: TClassifierModel,
    subModelNamesSlice: Set[String]): (Float, Seq[(String, Long)])= {

    val trainingSetLoader = ClassifierDatasetLoader(
      S3PathNames.s3ModelTrainingPath,
      subModelNamesSlice,
      numSubModelsForAugmentation)

    // Step 1: Load training set
    val tokenizedTrainingSetDS = trainingSetLoader().persist()

    // Step 2: Compute the embeddings using pre-trained transformer encoder
    val docEmbeddings = generateEmbeddings(
      tokenizedTrainingSetDS.toLocalIterator(),
      pretrainingInference,
      subModelNamesSlice.size
    )

    // Step 3: Train the classifier using embeddings...
    tokenizedTrainingSetDS.unpersist()
    trainClassifier(classifierModel, docEmbeddings)
  }
}


/**
 * {{{
 *  This singleton implements
 *  - various constructors
 *  - initialization of the training context for classification given the indexed labels
 * }}}
 */
private[bertspark] final object TClassifier {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[TClassifier])

  val predictionCount = new AtomicInteger(0)

  def defaultProgress(maxValue: Int): Int => Int =
  (numSubModelsProcessed: Int) => (numSubModelsProcessed*100.0/maxValue).floor.toInt

  case class ModelSimilarity(
    meanWithinLabels: Double,
    meanAcrossLabels: Double,
    loss: Double,
    subModelSimilarities: Seq[(String, Double)],
    config: String
  )


  /**
    * Command line arguments constructor
    * @param args Arguments as
    *   trainClassifier file.txt                          # Train classifier on a list of sub-models contained into a file
    *   trainClassifier ALL numSubModelsForAugmentation   # Train classifier on all sub models
    *   trainClassifier [SubModel1, subModel2. ]    // Train classifier on a sub set of sub models
    * @return Instance of the classifier of type MlopsClassification
    */
  def apply(args: Seq[String])(implicit sparkSession: SparkSession): TClassifier =
    args.size match {
      case 2 =>
        if(args(1).endsWith(".txt")) apply(args(1))
        else {
          val subModelNames = args(1).replace("_", " ").split(",").toSet
          apply(subModelNames, -1)
        }
      case 3 =>
        val subModelNames = args(1).replace("_", " ").split(",").toSet
        val numSubModelsForAugmentation = args(2).toInt
        apply(subModelNames, numSubModelsForAugmentation)
      case _ =>
        throw new IllegalArgumentException(
          s"Classifier args ${args.mkString(", ")} should be 'trainClassifier subModelNames [numSubModelsForAugmentation]"
        )
    }

  def apply()(implicit sparkSession: SparkSession): TClassifier = apply(Set[String]("ALL"), -1)


  private def apply(subModelNameSliceFilename: String)(implicit sparkSession: SparkSession): TClassifier =
    LocalFileUtil.Load
        .local(subModelNameSliceFilename)
        .map(_.split("\n").map(_.trim))
        .map(subModels => apply(subModels.toSet, -1))
        .getOrElse(
          throw new IllegalArgumentException(s"Failed to extract sub models from ${subModelNameSliceFilename}")
        )


    /**
      * Default constructor for handling a list of actual sub model names. We need to train only the supported
      * sub models that are not oracles
      * @param subModelNames Targeted subModel
      * @return Instance of Mlops classifier
      */
  @throws(clazz = classOf[IllegalArgumentException])
  private def apply(
    subModelNames: Set[String],
    numSubModelsForAugmentation: Int
  )(implicit sparkSession: SparkSession): TClassifier = {
    require(subModelNames.nonEmpty, "No sub-model have been specified")

    val actualSubModelNames =
      if(subModelNames.head == "ALL") subModelTaxonomy.predictiveMap.keySet
      else subModelNames
    logger.info(s"Ready to classify using ${actualSubModelNames.size} sub models")

    new TClassifier(actualSubModelNames, numSubModelsForAugmentation)
 }

  /**
    * Extract the embeddings from the training set Pair {sub-model -> Tokenized training set}
    * @param tokenizedIndexIterator Java iterator for the training set
    * @param pretrainingInference Pre-training transformer encoder
    * @param maxNumSubModels Maximum number of sub models
    * @param sparkSession Implicit reference to the current Spark context
    * @return Sequence of pairs {sub model -> DJL data set}
    */
  def generateEmbeddings(
    tokenizedIndexIterator: java.util.Iterator[(String, Seq[TokenizedTrainingSet])],
    pretrainingInference: PretrainingInference,
    maxNumSubModels: Int
  )(implicit sparkSession: SparkSession): Seq[(String, LabeledDjlDataset)] = {
    import sparkSession.implicits._

    var progressCnt = 0
    logger.info("Started document embedding prediction")
    val ndManager = NDManager.newBaseManager()

    // Collects the pair {sub model key, dataset of records}
    val labeledDatasetBuf = ListBuffer[(String, LabeledDjlDataset)]()

    while(tokenizedIndexIterator.hasNext() && progressCnt < maxNumSubModels) {
      // CLS predictions per each document pair (documentId, embedding vector)
      val (subModelName, tokenizedIndexedSet) = tokenizedIndexIterator.next()
      logDebug(logger, s"Start generating embedding for $subModelName")

      if(tokenizedIndexedSet.isEmpty)
        logger.warn(s"Tokenized training set for $subModelName is undefined")
      else {
        // We may need to duplicate some of the tokenized training records if
        // number of records per labels is too small....
        val tokenizedIndexedDS: Dataset[TokenizedTrainingSet] = sample(tokenizedIndexedSet.toDS())
        val keyedDocEmbeddings: List[KeyedValues] = pretrainingInference.predict(
          ndManager,
          tokenizedIndexedDS.map(_.contextualDocument)
        )

        if (keyedDocEmbeddings == null || keyedDocEmbeddings.isEmpty || keyedDocEmbeddings.head == null)
          logger.error(s"Keyed CLS prediction are undefined for $subModelName")
        else {
          val labeledDjlDataset = new LabeledDjlDataset(tokenizedIndexedDS, keyedDocEmbeddings, subModelName)
          labeledDatasetBuf.append((subModelName, labeledDjlDataset))
          predictionCount.addAndGet(tokenizedIndexedSet.size)
        }

        progressCnt += 1
        TProgress.show(progressCnt, "Transformer prediction", defaultProgress(maxNumSubModels))
        logDebug(
          logger,
          {
            val rate = progressCnt.toFloat/maxNumSubModels
            s"BERT encoder for $subModelName after $progressCnt out of $maxNumSubModels sub models, ${predictionCount.get()} predictions ($rate%)"
          }
        )
      }
    }
    // Verify that at the training set is not empty
    if(progressCnt == 0)
      logger.debug("Training set for embeddings is undefined")

    // We need to close both the manager and the model
    ndManager.close()
    pretrainingInference.close()
    labeledDatasetBuf
  }

  private def sample(tokenizedIndexedDS: Dataset[TokenizedTrainingSet]): Dataset[TokenizedTrainingSet] = {
    val cnt = tokenizedIndexedDS.count()
    // If there are more tokenized sample than the max number of records allowed for a given label
    // Sample it.
    if(cnt > mlopsConfiguration.classifyConfig.maxNumRecordsPerLabel) {
      val samplingRatio = mlopsConfiguration.classifyConfig.maxNumRecordsPerLabel.toFloat/cnt
      tokenizedIndexedDS.sample(samplingRatio)
    }
    else
      tokenizedIndexedDS
  }


  /**
    * Extract the embedding for the docs
    * @param tokenizedIndexIterator Iterator for the Tokenized training set
    * @param pretrainingInference Reference to the Transformer encoder
    * @param maxNumSubModels Maximum number of sub models used for training
    * @param sparkSession Implicit reference to the current Spark context
    * @return List of keyed values pair {Note/Document -> embedding}
    */
  def docEmbeddings(
    tokenizedIndexIterator: java.util.Iterator[(String, Seq[TokenizedTrainingSet])],
    pretrainingInference: PretrainingInference,
    maxNumSubModels: Int
  )(implicit sparkSession: SparkSession): Seq[KeyedValues] = {
    import sparkSession.implicits._

    var progressCnt = 0
    logger.info("Started document embedding prediction")
    val ndManager = NDManager.newBaseManager()

    // Collects the pair {sub model key, dataset of records}
    val keyedValuesCollector = ListBuffer[KeyedValues]()

    while (tokenizedIndexIterator.hasNext() && progressCnt < maxNumSubModels) {
      // CLS predictions per each document pair (documentId, embedding vector)
      val (subModelName, tokenizedIndexedSet) = tokenizedIndexIterator.next()
      logDebug(logger, msg = s"Start generating embedding for $subModelName")

      if (tokenizedIndexedSet.isEmpty)
        logger.warn(s"Tokenized training set for $subModelName is undefined")
      else {
        val tokenizedIndexedDS = tokenizedIndexedSet.toDS()
        val keyedDocEmbeddings: List[KeyedValues] = pretrainingInference.predict(
          ndManager,
          tokenizedIndexedDS.map(_.contextualDocument)
        )
        keyedValuesCollector.appendAll(keyedDocEmbeddings)
      }
      progressCnt += 1
    }
    keyedValuesCollector
  }



  def trainClassifier(
    classifierModel: TClassifierModel,
    docEmbeddings: Seq[(String, LabeledDjlDataset)]
  ): (Float, Seq[(String, Long)]) = {
    var count = 0
    var accuracySum = 0.0F
    val labeledDatasetIterator = docEmbeddings.iterator
    val subModelNumClasses = ListBuffer[(String, Long)]()

    // Iterate through all the various sub-models ...
    while(labeledDatasetIterator.hasNext) {
      val (subModelName, labeledDataset) = labeledDatasetIterator.next
      logDebug(logger, msg = s"Train classifier for $subModelName")
      try {
        val (numClasses, accuracy) = classifierModel.train(labeledDataset, subModelName)

        if (numClasses < 1)
          logger.warn(s"Classifier $subModelName has no associated classes")
        else {
          subModelNumClasses.append((subModelName, numClasses))
          accuracySum += accuracy
          count += 1
          TProgress.show(count, "Classifier training", defaultProgress(count))
          logDebug(
            logger,
            s"Classifier $subModelName count: $count. accuracy=$accuracy Ave=${accuracySum/count}"
          )
        }
      }
      catch {
        case e: DLException => logger.error(e.getMessage)
      }
    }
    (accuracySum, subModelNumClasses)
  }


  def evaluateTrainingSets(s3TrainingSetFolder: String)(implicit sparkSession: SparkSession): Unit = {
    import sparkSession.implicits._

    try {
      val groupedSubModelsTrainingDS = S3Util.s3ToDataset[SubModelsTrainingSet](
        mlopsConfiguration.storageConfig.s3Bucket,
        s3TrainingSetFolder,
        header = false,
        fileFormat = "json"
      ).persist()

      // We need to weed out the sub models for which at least one label has less
      // than the minimum required (mlopsConfiguration.classifyConfig.minNumRecordsPerLabel) of notes
      val subModelLabelsTaxonomy = SubModelOperations(groupedSubModelsTrainingDS)

      val summaries = Array[Int](2, 4, 8, 16, 20, 24).foldLeft(ListBuffer[String]())(
        (buf, minNumNotesPerLabel) => {
          val subModelsWithinMinNotesPerLabelsDS = subModelLabelsTaxonomy.process
          val finalizedTrainingDataDS: Dataset[(String, Seq[TokenizedTrainingSet])] = subModelsWithinMinNotesPerLabelsDS
              .map(grouped => (grouped.subModel, grouped.labeledTrainingData))
          logDebug(
            logger,
            msg = s"Initial number of sub models: ${groupedSubModelsTrainingDS.count()}, filtered: ${subModelsWithinMinNotesPerLabelsDS.count()}"
          )

          val summary =
            s"""
               |Target:                    ${mlopsConfiguration.target}
               |Run:                       ${mlopsConfiguration.runId}
               |Min num notes per labels:  $minNumNotesPerLabel
               |Num of sub models:         ${finalizedTrainingDataDS.count()}
               |""".stripMargin

          buf += summary
        }
      )

      delay(3000L)
      LocalFileUtil.Save.local("output/numSubModelsStats.txt", summaries.mkString("\n\n"))
      delay(3000L)
    } catch {
      case e: IllegalStateException =>
        logger.error(s"DualS3Dataset: ${e.getMessage}")
        sparkSession.emptyDataset[(String, Seq[TokenizedTrainingSet])]
    }
  }

}
