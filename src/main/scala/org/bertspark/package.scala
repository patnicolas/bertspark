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
package org

import org.apache.spark.sql.SparkSession
import org.bertspark.config.{FsPathNames, MlopsConfiguration}
import org.slf4j.Logger
import org.bertspark.config.MlopsConfiguration._
import org.bertspark.kafka.KafkaEvaluator
import org.bertspark.predictor.S3Evaluator


/**
 * Classes and methods for the operations of deep learning models, NLP and transformers and supporting utilities
 * @author Patrick Nicolas
 * @version 0.1
 */
package object bertspark {
  final val modelTypeLabel = "modelType"
  final val pyTorchInferenceEngine = "PyTorch"
  final val optBertPyTorchType = "distilbert"
  final val mxNetInferenceEngine = "MxNet"
  type DjlDataset = ai.djl.training.dataset.Dataset

  final private val modelPath = "models"

  final def getPretrainingModelPath: String = s"$modelPath/${FsPathNames.getPreTrainModelOutput}"
  final def getClassifierModelPath: String = s"$modelPath/${FsPathNames.getClassifyModelOutput}"

  def convertType[T](dataset: DjlDataset): T =
    if(dataset.isInstanceOf[T])
      dataset.asInstanceOf[T]
    else
      throw new DLException(s"Type of dataset ${dataset.getClass.getName} is incorrect")



  final object Labels {
    final val microBertLbl = "Bert-micro"
    final val nanoBertLbl = "Bert-nano"
    final val baseBertLbl = "Bert-base"
    final val largeBertLbl = "Bert-large"
    final val xLargeBertLbl = "Bert-xlarge"

    final val ctxTxtNSentencesBuilderLbl = "ctxTxtNSentencesBuilder"
    final val ctxNSentencesBuilderLbl = "ctxNSentencesBuilder"
    final val labeledSentencesBuilderLbl = "labeledSentencesBuilder"
    final val sectionsSentencesBuilderLbl = "sectionsSentencesBuilder"

    final val bertTokenizerLbl = "BertTokenizer"
    final val wordPiecesTokenizerLbl = "WordPiecesTokenizer"

    /**
     * Initialize the parameters for the validation
     */
    def configValidation: Unit = {
      Validation.register(
        transformerLbl,
        Set[String](microBertLbl, nanoBertLbl, baseBertLbl, largeBertLbl)
      )

      Validation.register(
        sentencesBuilderLbl,
        Set[String](
          labeledSentencesBuilderLbl,
          ctxNSentencesBuilderLbl,
          ctxTxtNSentencesBuilderLbl,
          sectionsSentencesBuilderLbl
        )
      )

      Validation.register(
        tokenizeLbl,
        Set[String](
          bertTokenizerLbl,
          wordPiecesTokenizerLbl
        )
      )
    }
  }



  private var reservedMemory = Array.fill(1024*1024)(0)
  def releaseReserve: Unit = reservedMemory = null

  def printStackTrace(e: Throwable): Unit = {
    if(MlopsConfiguration.mlopsConfiguration.isLogDebug)
      e.printStackTrace()
  }




  final class DLException(msg: String) extends Exception(msg) {
    override def toString: String = s"DLException: $msg"
  }

  final object DLException {
    def apply(e: Throwable): DLException = {
      printStackTrace(e)
      new DLException(e.getMessage)
    }

    def apply(e: Throwable, msg: String): Unit = {
      if(DebugLog.isDebugLevel)
        printStackTrace(e)
      throw new DLException(s"$msg ${e.getMessage}")
    }
  }

  final class InvalidParamsException(msg: String) extends Exception(msg) {
    override def toString: String = s"InvalidParamsException: $msg"
  }

  final class HasConvergedException(msg: String) extends Exception(msg) {
    override def toString: String = s"Has converged exception: $msg"
  }

  final class RunOutGPUMemoryException(msg: String) extends Exception(msg) {
    override def toString: String = s"RunOutGPUMemoryException $msg"
  }

  final class HPOException(msg: String) extends Exception(msg) {
    override def toString: String = s"HPOException $msg"
  }



  def error[E <: Throwable, U](msg: String, e: E): U = {
    if(DebugLog.isTraceLevel)
      printStackTrace(e)
    throw new DLException(s"$msg ${e.getMessage}")
  }

  final class DataBatchException(msg: String) extends Exception(msg) {
    override def toString: String = s"DLException: $msg"
  }

  final object DataBatchException {
    def apply(e: Throwable): DataBatchException = {
      if(DebugLog.isDebugLevel)
        printStackTrace(e)
      new DataBatchException(e.getMessage)
    }

    def apply(e: Throwable, msg: String): Unit = {
      if(DebugLog.isDebugLevel)
        printStackTrace(e)
      throw new DataBatchException(s"$msg ${e.getMessage}")
    }
  }




  def errorMsg[E <: Exception](msg: String, e: E, logger: Logger): Unit = {
    if(DebugLog.isDebugLevel)
      e.printStackTrace()
    logger.error(s"$msg: ${e.getMessage}")
  }

  /**
   * Default entry for architecture (Kafka, Spark) parameters
   * @param key Native name of the parameter
   * @param value Typed value of the parameter
   * @param isDynamic Is parameter tunable
   * @tparam paramtType Type of parameter (Int, String, Double,....)
   */
  case class ParameterDefinition(key: String, value: String, isDynamic: Boolean, paramType: String) {
    override def toString: String = s"$key $value ${if(isDynamic) "dynamic" else "static"}, $paramType"
  }


  /**
   * Define tuning parameters
   */
  trait TuningParameters[T <: TuningParameters[T]] {
    def getTunableParams: Seq[ParameterDefinition]
  }


  def delay(timeInMillis: Long): Boolean = {
    try {
      Thread.sleep(timeInMillis)
      true
    } catch {
      case e: InterruptedException =>
        println(e.getMessage)
        false
    }
  }



  trait TEvaluator {
    def execute(implicit sparkSession: SparkSession): Unit
  }

  final object TEvaluator {

    /**
      * Constructor for evaluator
      * {{{
      *  Command line: evaluate {s3|kafka} [fromRequest|fromTraining} numRecords
      * }}}
      * @param args Command line arguments
      * @return Instance of evaluator
      */
    def apply(args: Seq[String]): TEvaluator = {

      val evaluationType = args(1)  // S3 or Kafka
      evaluationType match {
        case "kafka" => new KafkaEvaluator(args)
        case "s3" => new S3Evaluator(args)
        case _ =>
          throw new UnsupportedOperationException(s"Evaluation type, $evaluationType not supported")
      }
    }

    /**
      * Constructor for deployment to production
      * @return Instance of evaluator for testing and production
      */
    def apply(): TEvaluator = new KafkaEvaluator(Seq[String]("evaluate", "11"))
  }
}
