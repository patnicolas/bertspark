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
package org.bertspark.modeling

import org.apache.spark.sql.{Dataset, SparkSession}
import org.bertspark.classifier.block.ClassificationBlock
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.config.{ExecutionMode, S3PathNames}
import org.bertspark.config.S3PathNames.s3SubModelsStructure
import org.bertspark.delay
import org.bertspark.nlp.medical.MedicalCodingTypes._
import org.bertspark.nlp.trainingset.SubModelsTrainingSet
import org.bertspark.predictor.model._
import org.bertspark.transformer.representation.PretrainingInference
import org.bertspark.util.io.LocalFileUtil.CSV_SEPARATOR
import org.bertspark.util.io.S3Util
import org.slf4j._
import scala.collection.mutable.ListBuffer


/**
  * Define the taxonomy for sub-models in relation with labels. A sub model is defined as
  * {subMode, numLabels, list of label indices}
  *
  * @param indexedLabels Map {label index -> label }
  * @param oracleMap Map {Sub model -> Label index }
  * @param predictiveMap Map {Sub model -> label indices }
  * @param unsupported Set of unsupported sub model
  * @param predictorSubModelsMap Map {Sub model -> (num labels, prediction model) }
  *
  * @author Patrick Nicolas
  * @version 0.7
  */
case class SubModelsTaxonomy(
  indexedLabels: Map[Int, String],
  oracleMap: Map[String, Int],
  predictiveMap: Map[String, Set[Int]],
  unsupported: Set[Int],
  predictorSubModelsMap: Map[String, (Int, PredictorModel)]) {
  import SubModelsTaxonomy._


  def equals(subModelsTaxonomy: SubModelsTaxonomy): Boolean =
    indexedLabels == subModelsTaxonomy.indexedLabels &&
    oracleMap == subModelsTaxonomy.oracleMap &&
    predictiveMap == subModelsTaxonomy.predictiveMap


  /**
    * Retrieve the label associated with an Oracle sub model
    * @param subModel Oracle sub model
    * @return Label is valid, empty string otherwise
    */
  final def getOracleLabel(subModel: String): Option[String] = for {
    oracleLabelIndex <- oracleMap.get(subModel)
    label <- indexedLabels.get(oracleLabelIndex)
  } yield label

  /**
    * Retrieve the labels associated with a predictive sub model
    * @param subModel Predictive sub model
    * @return Set of labels
    */
  final def getPredictiveLabels(subModel: String): Option[Set[String]] = for {
    predictiveLabelIndices <- predictiveMap.get(subModel)
  } yield predictiveLabelIndices.map(indexedLabels(_))


  /**
    * Retrieve Oracle labels
    * @return Set of Oracle labels
    */
  final def oracleLabels: Set[String] = oracleMap.values.map(indexedLabels(_)).toSet

  /**
    * Retrieve predictive (to be trained) labels
    * @return Set of prediction labels
    */
  final def predictiveLabels: Set[String] = predictiveMap.values.flatMap(_.map(indexedLabels(_))).toSet


  /**
    * Retrieve supported labels
    * @return Set of Oracle or predictive labels
    */
  final def supportedLabels: Set[String] = oracleLabels ++ predictiveLabels


  /**
    * Retrieve the number of labels associated with a predictive sub models
    * @param subModel Predictive sub model
    * @return Number of labels > 1 associated with this predictive sub model if found, -1 otherwise
    */
  final def getPredictiveNumLabels(subModel: String): Option[Int] = predictiveMap.get(subModel).map(_.size)


  final def getPredictorModel(subModel: String): Option[PredictorModel] =
    if(predictorSubModelsMap.nonEmpty) predictorSubModelsMap.get(subModel).map(_._2) else None

  def isOracle(subModelName: String): Boolean = oracleMap.keySet.contains(subModelName)

  def isTrained(subModelName: String): Boolean = predictorSubModelsMap.keySet.contains(subModelName)

  def getSupportedSubModelNameSet: Set[String] = oracleMap.keySet ++ predictiveMap.keySet

  def isSupported(subModelName: String): Boolean = oracleMap.contains(subModelName) || predictiveMap.contains(subModelName)

  def isValid: Boolean = predictiveMap.nonEmpty || oracleMap.nonEmpty


  /**
    * Process a set of requests and break them down in Oracle vs. trainable requests
    * @param inputRequests Request for prediction
    * @return Responses as Oracle label and predicted labels
    */
  def apply(inputRequests: Seq[PRequest]): PResponses =
    if(inputRequests.nonEmpty) {
      import SubModelsTaxonomy._

      val oracleHandlers = new OracleHandler(classifierSubModelConfig = this)
      val predictiveHandler = new PredictionHandler(classifierSubModelConfig = this)
      val unsupportedHandler = new UnsupportedHandler()
      val internalRequests = inputRequests.map(InternalRequest(_))

      // We need to make sure that we have at least one Oracle or one Predictive sub models
      // if (oracleSubModelsMap.nonEmpty || predictiveSubModelsMap.nonEmpty) {
      val oracleRequests = ListBuffer[InternalRequest]()
      val predictiveRequests = ListBuffer[InternalRequest]()
      val unsupportedRequests = ListBuffer[InternalRequest]()

      internalRequests.foreach(
        internalRequest => {
          val subModel = internalRequest.context.emrLabel
          if (oracleMap.contains(subModel)) {
            logDebug(logger, msg = s"Found Oracle request for $subModel")
            oracleRequests.append(internalRequest)
          }
          else if (predictorSubModelsMap.contains(subModel)) {
            logDebug(logger, msg = s"Found Predictive request for $subModel")
            predictiveRequests.append(internalRequest)
          }
          else {
            logDebug(logger, msg = s"Found Unsupported request for $subModel")
            unsupportedRequests.append(internalRequest)
          }
        }
      )
      PResponses(
        oracleHandlers(oracleRequests),
        predictiveHandler(predictiveRequests),
        unsupportedHandler(unsupportedRequests))
    }
    else {
      logger.warn("Cannot process undefined requests")
      PResponses()
    }

  override def toString: String = {
    // List predictive/trained sub models
    val predictiveMapStr = predictiveMap.map{
      case (subModel, indices) => s"$subModel: ${indices.map(indexedLabels.getOrElse(_, "")).mkString(" ")}"
    }.mkString("\n")
    // List Oracle sub models
    val oracleMapStr = oracleMap.map{
      case (subModel, index) => s"$subModel: ${indexedLabels.getOrElse(index, "")}"
    }.mkString("\n")
    s"Oracle sub-models:\n$oracleMapStr\nPredictive sub-models:\n$predictiveMapStr"
  }
}


/**
  * Singleton for constructor, Save and Load functions
  */
private[bertspark] object SubModelsTaxonomy {
  final private val logger: Logger = LoggerFactory.getLogger("SubModelsTaxonomy")

  lazy val subModelTaxonomy: SubModelsTaxonomy = load()
  final val emptySubModelsTaxonomy = SubModelsTaxonomy(
    Map.empty[Int, String],
    Map.empty[String, Int],
    Map.empty[String, Set[Int]],
    Set.empty[Int],
    Map.empty[String, (Int, PredictorModel)]
  )


  /**
    * Constructor for classifier sub model data
    * @param indexedLabels Indexed label map {Label index -> label }
    * @param oracle Oracle map {sub-model, label index}
    * @param predictive Predictive map {sub-model -> set of label indices}
    * @return Instance of the classifier sub model data
    */
  def apply(
    indexedLabels: Map[Int, String],
    oracle: Map[String, Int],
    predictive: Map[String, Set[Int]]): SubModelsTaxonomy = SubModelsTaxonomy(
      indexedLabels,
      oracle,
      predictive,
      Set.empty[Int], // Unsupported
      Map.empty[String, (Int, PredictorModel)] // Predictive models map
    )


  /**
    * Constructor using the default sub models file name (i.e. subModels.csv')
    * @return Instance of the sub models taxonomy
    */
  def apply(): SubModelsTaxonomy = apply(S3PathNames.s3SubModelsStructure)


  /**
    * Constructor using the sub models file name (i.e. subModels.csv')
    * @param subModelsFilename Explicit name of the sub model file
    * @return Instance of the sub models taxonomy
    */
  def apply(subModelsFilename: String): SubModelsTaxonomy = try {
    // Load the indexed label map from S3
    val indexedLabelMap = TrainingLabelIndexing.load
    if(indexedLabelMap.isEmpty)
      throw new IllegalStateException("Failed to load indexed labels")

    val labeledIndices = indexedLabelMap.map{ case (label, index) => (index, label)}
    val entries = S3Util.downloadCollection(
      mlopsConfiguration.storageConfig.s3Bucket,
      subModelsFilename
    ).map(
      _.filter(
        line => {
          val ar = line.split(CSV_SEPARATOR)
          if (ar.size < 3) {
            logger.error(s"Sub model $subModelsFilename has incorrect line $line")
            false
          }
          else
            true
        }
      ).map(
        line => {
          val ar = line.split(CSV_SEPARATOR)
          (ar.head.trim, ar(1).toInt, ar(2).trim)
        }
      )
    ).getOrElse({
      logger.warn(s"Failed to load sub models from S3 $s3SubModelsStructure")
      Seq.empty[(String, Int, String)]
    })

    val (oracleSeq, predictiveSeq) = entries.map{
      case (subModel, cnt, labelIndicesStr) =>
        val labelsIndices = labelIndicesStr.split("\\|\\|").map(_.toInt)
        (subModel, cnt, labelsIndices.toSet)
    }.partition(_._2 == 1)

    SubModelsTaxonomy(
      labeledIndices,
      oracleSeq.map{ case (subModel, _, idx) => (subModel, idx.head)}.toMap,
      predictiveSeq.map{ case (subModel, _, labelIndices) => (subModel, labelIndices) }.toMap
    )
  }
  catch {
    case e: IllegalArgumentException =>
      logger.warn(s"Failed to load and training sub models ${e.getMessage}")
      emptySubModelsTaxonomy

    case e: IllegalStateException =>
      logger.warn(s"Failed to load oad training sub models ${e.getMessage}")
      emptySubModelsTaxonomy
  }



  /**
    * Apply a filter on the list of sub-models defined in 'subModels.csv' . The filter is applied on 'filter.csv'
    * file sub-model, confidence factor
    * @param subModelsTaxonomy Classifier sub models configuration
    * @return Filtered classifier sub models configuration
    */
  def filter(subModelsTaxonomy: SubModelsTaxonomy): SubModelsTaxonomy = {
    val subModelFilterThreshold = mlopsConfiguration.evaluationConfig.subModelFilterThreshold

    if(subModelFilterThreshold > 0.0F) {
      val predictiveSubModelsMap = subModelsTaxonomy.predictiveMap

      // If there is a filter for the sub classifier models using accuracy computed from
      // either from classifier training run or the
      val validSubModels = S3Util.download(
        mlopsConfiguration.storageConfig.s3Bucket,
        S3PathNames.getS3ModelTrainingFilterPath
      ).map {
        content => {
          val lines = content.split("\n")
          lines.map(_.split(",").head).filter(_ (1).toFloat > subModelFilterThreshold)
        }
      }.getOrElse({
        logger.error("No filter file is defined for classifier model filter")
        Array.empty[String]
      })

      val validPredictiveSubModelMap = predictiveSubModelsMap.filter {
        case (subModels, _) => validSubModels.contains(subModels)
      }
      subModelsTaxonomy.copy(predictiveMap  = validPredictiveSubModelMap)
    }
    else {
      logger.warn(s"Sub models have not filter threshold defined")
      subModelsTaxonomy
    }
  }




  /**
    * Generate and save the taxonomy of sub-models on S3   'subModels.csv'. This method
    * is called at the end of building pre-training data set
    * @see org.mlops.nlp.trainingset.TrainingSetBuilder
    * @param groupedSubModelTrainingDS Labeled training set grouped by sub-models
    * @param indexedLabels Sequence of {label -> label index} pairs
    * @param sparkSession Implicit reference to the current Spark context
    */
  def save(
    groupedSubModelTrainingDS: Dataset[SubModelsTrainingSet],
    indexedLabels: Seq[(String, Int)]
  )(implicit sparkSession: SparkSession): Unit = try {
    import sparkSession.implicits._

    val indexedLabelMap = indexedLabels.toMap
    // Extract the taxonomy and initial state of sub-models
    val subModelLabels = groupedSubModelTrainingDS
        .filter(
          emrLabeledTrainingData => emrLabeledTrainingData != null && emrLabeledTrainingData.subModel != null
        )
        .map(
          emrLabeledTrainingData => {
            // Compute the unique labels that have a corresponding index
            val uniqueLabelIndices = emrLabeledTrainingData
                .labeledTrainingData
                .map(ts => {
                  val label = ts.label.replace(",", " ")
                  indexedLabelMap.getOrElse(label, -1)
                })
                .distinct
                .filter(_ > 0)

            // Format the label indices (Single for Oracle, Multiple, '||' separated for Predictive sub models)
            val recordedLabel =
              if(uniqueLabelIndices.size == 1) uniqueLabelIndices.head
              else uniqueLabelIndices.mkString(labelIndicesSeparator)

            // Finally record (sub-model, number_of_labels, label_indices)
            val sz = if(uniqueLabelIndices != null) uniqueLabelIndices.size else 0
            s"${emrLabeledTrainingData.subModel.trim},$sz,$recordedLabel"
          }
        ).collect()
    // Save the taxonomy and state of the training set
    S3Util.upload(S3PathNames.s3SubModelsStructure, subModelLabels.mkString("\n"))
  }
  catch {
    case e: IllegalArgumentException => logger.error(s"TrainingSubModelTaxonomy.save ${e.getMessage}")
  }

  /**
    * Initialize the classifier sub models for current sub models file 'subModels.csv'
    * - Training (loading from 'subModels.csv' S3 file
    * - Evaluation which include filter and instantiation of sub models
    * @return Sub models taxonomy (Decomposition to Oracle and predictive models)
    */
  def load(): SubModelsTaxonomy =  load(s3SubModelsStructure)


  // ---------------------  Helper methods --------------------------


  /**
    * Load the prediction models
    * @param subModelsTaxonomy Current sub models taxonomy
    * @return Updated sub models taxonomy
    */
  private def loadPredictionModels(subModelsTaxonomy: SubModelsTaxonomy): SubModelsTaxonomy = {
    import org.bertspark.implicits._

    // Step 1: Initialize Pre-training inference
    val preTrainingInference = PretrainingInference()

    // @todo replace by loading directly from
    //  Load the actual sub model predictors
    val subModelPredictor = loadPredictorFromS3

    // Extract the valid {subModel -> label}
    val validLabelsMap = subModelsTaxonomy.predictiveMap.map{
      case (subModel, labelIndices) =>
        val labels = labelIndices.map(subModelsTaxonomy.indexedLabels(_)).filter(_.nonEmpty)
        (subModel, labels)
    }   // Make sure the sub model predictor actually exists
        .filter{ case (subModel, _) => subModelPredictor.contains(subModel) }

    // Instantiate the predictor models...
    var classifierModelCount =  0
    val _predictedSubModelsMap = validLabelsMap.map{
      case (subModel, _) =>
        subModelsTaxonomy.getPredictiveNumLabels(subModel).map(
          numClasses => {
            val classificationModel = new ClassificationBlock(numClasses)
            try {
              // Instantiate the sub model predictor
              val predictor = PredictorModel(preTrainingInference, classificationModel, subModel)
              delay(20L)
              classifierModelCount += 1
              logDebug(
                logger,
                msg = s"Neural classifier $subModel loaded. Count=$classifierModelCount / ${validLabelsMap.size}"
              )
              (subModel.trim, (numClasses, predictor))
            }
            catch {
              case e: IllegalStateException =>
                logger.error(s"Failed ${e.getMessage}")
                ("", (-1, null))
            }
          }
        ).getOrElse({
          logger.error(s"Failed to extract number of classes for $subModel")
          ("", (-1, null))
        })

    }
    subModelsTaxonomy.copy(predictorSubModelsMap  = _predictedSubModelsMap)
  }


  /**
    * Initialize the classifier sub models  for
    * - Training (loading from 'subModels.csv' S3 file
    * - Evaluation which include filter and instantiation of sub models
    * @param subModelsFilename Name of the sub models file 'subModels.csv'
    * @return Sub models taxonomy (Decomposition to Oracle and predictive models)
    */
  private def load(subModelsFilename: String): SubModelsTaxonomy = {
    val subModelsTaxonomy = SubModelsTaxonomy(subModelsFilename)

    // Only for evaluation and production of classifiers
    if(ExecutionMode.isEvaluation) {
      // Apply filter only if a threshold is defined
      val filteredClassifierSubModelConfig = SubModelsTaxonomy.filter(subModelsTaxonomy)
      // Loads the neural classifiers.
      SubModelsTaxonomy.loadPredictionModels(filteredClassifierSubModelConfig)
    }
    // For training of classifier
    else
      subModelsTaxonomy
  }

  private def loadPredictorFromS3: Set[String] =
    S3Util.getS3Keys(
      mlopsConfiguration.storageConfig.s3Bucket,
      S3PathNames.s3ClassifierModelPath
    )
        .filter(_.endsWith(".params"))
        .map(
          path => {
            val endIndex = path.indexOf("-0000")
            val beginIndex = path.lastIndexOf("/")
            val subModel = path.substring(beginIndex + 1, endIndex)
            subModel
          }
        ).toSet
}

