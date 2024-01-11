package org.bertspark.nlp.trainingset

import org.bertspark.nlp.trainingset.TrainingSetSampler.filterPerNumNotesLabels
import org.scalatest.flatspec.AnyFlatSpec


private[trainingset] final class TrainingSetSamplerTest extends AnyFlatSpec{
  import org.bertspark.config.MlopsConfiguration._
  import TrainingSetSamplerTest._


  it should "Succeed extracting valid oracle and trained classifiers" in {
    val (oracleSubModels, trainedSubModels) = TrainingSetSampler.selectSupportedSubModels

    println(s"${oracleSubModels.size} oracle sub models:\n ${oracleSubModels.mkString("\n")}")
    println(s"\n\n${trainedSubModels.size} Predictive sub models:\n${trainedSubModels.mkString("\n")}")
  }

  ignore should "Succeed applying down sampling method on training data above threshold" in {
    val maxFreq = mlopsConfiguration.preProcessConfig.maxLabelFreq
    val hierarchicalLabelsSeq = List.tabulate(maxFreq+10)(n => getHierarchicalLabels(n.toString, "C50.911"))
    val listOfHierarchicalLabels = filterPerNumNotesLabels(hierarchicalLabelsSeq)
    assert(listOfHierarchicalLabels.nonEmpty)
    assert(listOfHierarchicalLabels.head.size <= maxFreq)
  }

  ignore should "Succeed applying down sampling method on training data below threshold" in {
    val minFreq = mlopsConfiguration.preProcessConfig.minLabelFreq
    val hierarchicalLabelsSeq = List.tabulate(minFreq-10)(n => getHierarchicalLabels(n.toString, "C50.911"))
    val listOfHierarchicalLabels = filterPerNumNotesLabels(hierarchicalLabelsSeq)
    assert(listOfHierarchicalLabels.isEmpty)
  }

  ignore should "Succeed applying down sampling method on training data within threshold" in {
    val minFreq = mlopsConfiguration.preProcessConfig.minLabelFreq
    val hierarchicalLabelsSeq = List.tabulate(minFreq+3)(n => getHierarchicalLabels(n.toString, "C50.911"))
    val listOfHierarchicalLabels = filterPerNumNotesLabels(hierarchicalLabelsSeq)
    assert(listOfHierarchicalLabels.nonEmpty)
    assert(listOfHierarchicalLabels.head.size == minFreq+3)
  }

  ignore should "Succeed applying down sampling method on training data above threshold 2" in {
    val maxFreq = mlopsConfiguration.preProcessConfig.maxLabelFreq
    val hierarchicalLabelsSeq = List.tabulate(maxFreq+10)(n => getHierarchicalLabels(n.toString, "C50.911")) :::
        List.tabulate(maxFreq-2)(n => getHierarchicalLabels(n.toString, "C50.901"))

    val listOfHierarchicalLabels = filterPerNumNotesLabels(hierarchicalLabelsSeq)
    assert(listOfHierarchicalLabels.nonEmpty)
    assert(listOfHierarchicalLabels.head.size == maxFreq)
    assert(listOfHierarchicalLabels(1).size == maxFreq-2)
  }

  ignore should "Succeed applying down sampling method on training data below threshold 2" in {
    val minFreq = mlopsConfiguration.preProcessConfig.minLabelFreq
    val hierarchicalLabelsSeq = List.tabulate(minFreq+10)(n => getHierarchicalLabels(n.toString, "C50.911")) :::
        List.tabulate(minFreq-2)(n => getHierarchicalLabels(n.toString, "C50.901"))

    val listOfHierarchicalLabels = filterPerNumNotesLabels(hierarchicalLabelsSeq)
    assert(listOfHierarchicalLabels.size == 1)
    assert(listOfHierarchicalLabels.head.size == minFreq+10)
  }
}


private[trainingset] final object TrainingSetSamplerTest {
  def getContextualDocument(id: String): ContextualDocument =
    ContextualDocument(
      id,
      Array[String]("6_age","f_gender","cornerstone_cust","unknown_modality","22_pos","38792_cpt","no_mod"),
      "unit nm injection radiotracer for sentinel node cpt code primary physician duncan dystrophy muscular birth of date attending physician mark dystrophy muscular node biopsy and impression informed and prior to the periareolar right breast and daily the mci of technetium filtered sulfur colloid injected the subdermal periareolar breast daily the and positions the patient tolerated the procedure well with no immediate or no imaging acquired name dystrophy muscular signed morning therapy name nuclear medicine sentinel node injection history right breast cancer",
    )

  def getHierarchicalLabels(id: String, trailingLabel: String): TrainingLabel =
    TrainingLabel(getContextualDocument(id), "38792", trailingLabel)
}
