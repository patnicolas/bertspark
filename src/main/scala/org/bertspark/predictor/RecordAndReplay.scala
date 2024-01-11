package org.bertspark.predictor

import org.bertspark.classifier.dataset.ClassifierDatasetLoader
import org.bertspark.classifier.model.TClassifierModel
import org.bertspark.classifier.training.TClassifier
import org.bertspark.config.S3PathNames
import org.bertspark.nlp.medical.LabeledDjlDataset
import org.bertspark.nlp.trainingset.{ContextualDocument, KeyedValues, TokenizedTrainingSet}
import org.bertspark.transformer.representation.PretrainingInference
import org.bertspark.util.io.S3Util
import org.bertspark.util.SparkUtil
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer


/**
  * *
  * @param maxNumSubModels
  */
private[bertspark] final class RecordAndReplay(maxNumSubModels: Int = -1) {
  import org.bertspark.implicits._
  import org.bertspark.classifier.training.TClassifier._
  import RecordAndReplay._

  private[this] val pretrainingInference = PretrainingInference()


  def getEmbeddings(subModelNames: Seq[String], newTarget: String): Unit = {
    import sparkSession.implicits._

    val subModels = if(maxNumSubModels > 0) subModelNames.take(maxNumSubModels) else subModelNames

    val loadTrainingSets = ClassifierDatasetLoader(S3PathNames.s3ModelTrainingPath, subModels.toSet, -1)
    val tokenizedTrainingSetDS = loadTrainingSets.apply
    val keyedEmbeddings = docEmbeddings(
      tokenizedTrainingSetDS.toLocalIterator,
      pretrainingInference,
      subModels.size
    ).map{
      case (id, vec) => KeyedEmbedding(id, vec)
    }

    val contextualDocumentDS = tokenizedTrainingSetDS.flatMap{
      case (_, tokenizedTraining) => tokenizedTraining.map(_.contextualDocument)
    }

    val keyedDocumentEmbeddingDS = SparkUtil.sortingJoin[ContextualDocument, KeyedEmbedding](
      contextualDocumentDS,
      "id",
      keyedEmbeddings.toDS(),
      "id"
    ).map {
      case (contextualDocument, keyEmbedding) =>
        s"""${contextualDocument.id}
        |${contextualDocument.contextVariables.mkString(" ")}
        |${contextualDocument.text}
        }${keyEmbedding.embedVector.mkString(" ")}""".stripMargin
    }

    val summary = keyedDocumentEmbeddingDS.collect.mkString("\n\n")
    S3Util.upload(s"${S3PathNames.s3TransformerModelPath}/embeddingSummary", summary)
  }


  def records(subModelNames: Seq[String], newTarget: String): Unit = {
    import sparkSession.implicits._

    val classifierModel = new TClassifierModel
    val subModels = if(maxNumSubModels > 0) subModelNames.take(maxNumSubModels) else subModelNames

    val loadTrainingSets = ClassifierDatasetLoader(S3PathNames.s3ModelTrainingPath, subModels.toSet, -1)
    val tokenizedTrainingSetDS = loadTrainingSets()
    val keyedEmbeddings: Seq[(String, LabeledDjlDataset)] = generateEmbeddings(
      tokenizedTrainingSetDS.toLocalIterator,
      pretrainingInference,
      subModels.size
    )

    val embeddingRequestDS = keyedEmbeddings.map{
      case (subModel, labeledDjlDataset) =>
        EmbeddingRequest(
          subModel,
          labeledDjlDataset.getTokenizedIndexedTrainingSet,
          labeledDjlDataset.getKeyedPredictions)

    }.toDS

    S3Util.datasetToS3[EmbeddingRequest](
      embeddingRequestDS,
      s"${S3PathNames.s3TransformerModelPath}/replay",
      false,
      "json",
      false,
      8
    )
  }


  def replay: Unit = {
    import sparkSession.implicits._
    val classifierModel = new TClassifierModel

    val embeddingRequestDS = S3Util.s3ToDataset[EmbeddingRequest](
      s"${S3PathNames.s3TransformerModelPath}/replay",
      false,
      "json"
    )
    val labeledDjlDatasetCollector = ListBuffer[(String, LabeledDjlDataset)]()

    val embeddingRequestIter = embeddingRequestDS.toLocalIterator()
    while(embeddingRequestIter.hasNext) {
      val embeddingRequest = embeddingRequestIter.next()
      labeledDjlDatasetCollector.append(
        (embeddingRequest.subModel, new LabeledDjlDataset(
          embeddingRequest.tokenizedTrainingSet.toDS(),
          embeddingRequest.keyedPredictions,
          embeddingRequest.subModel
        )
      ))
    }

    TClassifier.trainClassifier(classifierModel, labeledDjlDatasetCollector)
  }

}


private[bertspark] final object RecordAndReplay {
  final private val logger: Logger = LoggerFactory.getLogger(classOf[RecordAndReplay])


  case class EmbeddingRequest(
    subModel: String,
    tokenizedTrainingSet: Seq[TokenizedTrainingSet],
    keyedPredictions: List[KeyedValues]
  )

  case class KeyedEmbedding(id: String, embedVector: Array[Float])

  case class KeyedDocumentEmbedding(contextualDocument: ContextualDocument, embedVector: Array[Float]) {
    override def toString: String =
      s"${contextualDocument.id}\n${contextualDocument.contextVariables.mkString(" ")}\n${contextualDocument.text}\n${embedVector.mkString(" ")}"
  }
}
