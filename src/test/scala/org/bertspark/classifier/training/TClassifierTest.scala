package org.bertspark.classifier.training

import ai.djl.ndarray._
import org.bertspark.nlp.trainingset.{ContextualDocument, TokenizedTrainingSet}
import org.scalatest.flatspec.AnyFlatSpec



private[classifier] final class TClassifierTest extends AnyFlatSpec{

  it should "Succeed estimating the number of sub models for a given min freq of notes per labels" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._
    val s3Folder = "mlops/LARGE/training/TFIDF"
    TClassifier.evaluateTrainingSets(s3Folder)

  }


  ignore should "Succeed iterator through slices" in {
    val indices = (0 until 13 by 5).map(
      index => {
        val limit = if(index+5 >= 13) 13 else index+5
        Range(index, limit)
      }
    )
    println(indices.mkString(" "))
  }


  ignore should "Succeed filtering for Oracle models" in {
    val contextualDoc1 = ContextualDocument("1")
    val contextualDoc2 = ContextualDocument("2")
    val contextualDoc3 = ContextualDocument("3")
    val contextualDoc4 = ContextualDocument("4")
    val label1 = "70450 26 77 GC 59"
    val label2 = "70498 TC"
    val label3 = "72141 XU"
    val label4 = "9999"
    val tokenizedIndexedTrainingSet1 = TokenizedTrainingSet(contextualDoc1, label1, Array.empty[Float])
    val tokenizedIndexedTrainingSet2 = TokenizedTrainingSet(contextualDoc2, label2, Array.empty[Float])
    val tokenizedIndexedTrainingSet3 = TokenizedTrainingSet(contextualDoc3, label3, Array.empty[Float])
    val tokenizedIndexedTrainingSet4 = TokenizedTrainingSet(contextualDoc4, label4, Array.empty[Float])

    val dataset1 = ("70498_TC", Seq[TokenizedTrainingSet](tokenizedIndexedTrainingSet2))
    val dataset2 = ("9999", Seq[TokenizedTrainingSet](tokenizedIndexedTrainingSet4))
    val dataset3 = ("70450_26_77_GC_59", Seq[TokenizedTrainingSet](tokenizedIndexedTrainingSet1, tokenizedIndexedTrainingSet4))
    val dataset4 = ("72141_XU", Seq.empty[TokenizedTrainingSet])
    val dataset5 = ("70450_26_77_GC_59", Seq[TokenizedTrainingSet](tokenizedIndexedTrainingSet1))

    import org.bertspark.implicits._
    import sparkSession.implicits._
/*
    val inputDS = Seq[(String, Seq[TokenizedTrainingSet])](dataset1, dataset2, dataset3, dataset4, dataset5).toDS()
    val outputDS = extractDocumentToTrain(inputDS)
    val output = outputDS.collect()
    val outputStr = output.indices.map(index => s"$index: ${output(index)._1}:${output(index)._2.map(_.label).mkString("\n")}")
    println(outputStr.mkString("\n\n"))

 */
  }


  ignore should "Succeed concatenate CLS predictions" in {
    val ndManager = NDManager.newBaseManager()
    val arraySize = 10
    val ndArray1 = ndManager.create(Array.fill(arraySize)(0.5F))
    val clsPrediction1 = ("doc1", new NDList(ndArray1))

    val ndArray2 = ndManager.create(Array.fill(arraySize)(1.0F))
    val clsPrediction2 = ("doc2", new NDList(ndArray2))

    val ndArray3 = ndManager.create(Array.fill(arraySize)(2.0F))
    val clsPrediction3 = ("doc1", new NDList(ndArray3))

    val ndArray4 = ndManager.create(Array.fill(arraySize)(2.5F))
    val clsPrediction4 = ("doc2", new NDList(ndArray4))
  }
}
