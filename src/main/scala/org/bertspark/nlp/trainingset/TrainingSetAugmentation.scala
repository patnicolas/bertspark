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
package org.bertspark.nlp.trainingset
/*
import org.apache.spark.sql.{Dataset, SparkSession}
import org.mlops.config.MlopsConfiguration.DebugLog.logDebug
import org.mlops.config.MlopsConfiguration.mlopsConfiguration
import org.mlops.config.S3PathNames
import org.mlops.delay
import org.mlops.nlp.tokenSeparator
import org.mlops.util.io.{LocalFileUtil, S3Util}
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer



/**
  * @author Patrick Nicolas
  * @version 0.7
  */
private[mlops] final object TrainingSetAugmentation {
  final private val logger: Logger = LoggerFactory.getLogger("TrainingSetAugmentation")
  final private val augmentationFilename = "conf/codes/augmentation.csv"

  final private val invNumPermutation = Map[Int, Int](
    69 -> 7, 101 -> 7, 88 -> 7, 115 -> 7, 5 -> 3, 120 -> 7, 10 -> 4, 56 -> 6, 42 -> 6, 24 -> 5, 37 -> 6, 25 -> 5,
    52 -> 6, 14 -> 4, 110 -> 7, 125 -> 7, 20 -> 5, 46 -> 6, 93 -> 7, 57 -> 6, 78 -> 7, 29 -> 5, 106 -> 7, 121 -> 7,
    84 -> 7, 61 -> 6, 89 -> 7, 116 -> 7, 1 -> 0, 74 -> 7, 6 -> 3, 60 -> 6, 117 -> 7, 85 -> 7, 102 -> 7, 28 -> 5, 38 -> 6,
    70 -> 7, 21 -> 5, 33 -> 6, 92 -> 7, 65 -> 7, 97 -> 7, 9 -> 4, 53 -> 6, 109 -> 7, 124 -> 7, 77 -> 7, 96 -> 7, 13 -> 4,
    41 -> 6, 73 -> 7, 105 -> 7, 2 -> 1, 32 -> 5, 34 -> 6, 45 -> 6, 64 -> 6, 17 -> 5, 22 -> 5, 44 -> 6, 59 -> 6, 118 -> 7,
    27 -> 5, 71 -> 7, 12 -> 4, 54 -> 6, 49 -> 6, 86 -> 7, 113 -> 7, 81 -> 7, 76 -> 7, 7 -> 3, 39 -> 6, 98 -> 7, 103 -> 7,
    91 -> 7, 66 -> 7, 108 -> 7, 3 -> 2, 80 -> 7, 35 -> 6, 112 -> 7, 123 -> 7, 48 -> 6, 63 -> 6, 18 -> 5, 95 -> 7, 50 -> 6,
    67 -> 7, 16 -> 4, 127 -> 7, 31 -> 5, 11 -> 4, 72 -> 7, 43 -> 6, 99 -> 7, 87 -> 7, 104 -> 7, 40 -> 6, 26 -> 5, 55 -> 6,
    114 -> 7, 23 -> 5, 8 -> 3, 75 -> 7, 119 -> 7, 58 -> 6, 82 -> 7, 36 -> 6, 30 -> 5, 51 -> 6, 19 -> 5, 107 -> 7, 4 -> 2,
    126 -> 7, 79 -> 7, 94 -> 7, 47 -> 6, 15 -> 4, 68 -> 7, 62 -> 6, 90 -> 7, 111 -> 7, 122 -> 7, 83 -> 7, 100 -> 7
  )


  final val conversionMap: Map[String, String] = {
    val initial = LocalFileUtil.Load.local(augmentationFilename).map(
      content =>
        content
            .split("\n")
            .map(
              line => {
                val fields = line.split(",")
                if (fields.size == 2)
                  (fields.head, fields(1))
                else
                  ("", "")
              }
            ).toMap
    ).getOrElse(Map.empty[String, String])

    initial ++ initial.map { case (k, v) => (v, k) }
  }


  /**
    * Apply augmentation to an existing training set
    * {{{
    * Command lines
    *    - augment            ## initialized the sub models slices
    *    - augment  s3OutputFolder output/subModelsSlice-n.txt  ## Augment from a sub models slivr
    * }}}
    * @param args
    */
  def augment(args: Seq[String]): Unit = {
    if(args.size > 1) {
      val s3OutputFolder = args(1)
      val fsSubModelsSliceFilename = args(2)
      augment(s3OutputFolder, fsSubModelsSliceFilename)
    }
    else
      InvalidLabelsManager.initialize
  }


  /**
    * Augment the current training set, targeting label for which tokenized records < minNumRecordsPerLabel
    *
    * @param s3OutputFolder Output folder on S3
    * @param fsSubModelsSliceFilename Number of sub models for which the number of records associated with any given labe;
    *                                  needs to be augmented
    */
  def augment(s3OutputFolder: String, fsSubModelsSliceFilename: String): Unit = {
    import org.mlops.implicits._
    import sparkSession.implicits._

    val augmentedGroupedSubModelTrainingSetDS = InvalidLabelsManager.execute(fsSubModelsSliceFilename)
    val numSplits = 24
    val augmentedGroupedSubModelSplits = augmentedGroupedSubModelTrainingSetDS.randomSplit(Array.fill(numSplits)(1.0/numSplits))

    augmentedGroupedSubModelSplits.foreach(
      trainingDS => {
        S3Util.datasetToS3[SubModelsTrainingSet](
          mlopsConfiguration.storageConfig.s3Bucket,
          trainingDS,
          s3OutputFolder,
          header = false,
          fileFormat = "json",
          toAppend = true,
          numPartitions = 2)
        delay(1000L)
      }
    )
  }


  def taxonomyCounts(
    originalSubModelsTrainingSetDS: Dataset[SubModelsTrainingSet],
    augmentedSubModelsTrainingSetDS: Dataset[SubModelsTrainingSet]): Unit = {
    import org.mlops.implicits._
    import sparkSession.implicits._

    logDebug(logger, s"${originalSubModelsTrainingSetDS.count()} original subModelsTrainingSetDS elements!")
    logDebug(logger, s"${augmentedSubModelsTrainingSetDS.count()} augmented subModelsTrainingSetDS elements!")
    val originalSortedCounts: Seq[(String, Seq[(String, Int)])] = originalSubModelsTrainingSetDS.map(
      groupedTrainingSet => {
        val tokenizedTrainingSetCount = groupedTrainingSet
            .labeledTrainingData
            .groupBy(_.label)
            .map { case (label, xs) => (label.trim, xs.size) }
            .toSeq
            .sortWith(_._1 < _._1)
        (groupedTrainingSet.subModel.trim, tokenizedTrainingSetCount)
      }
    ).collect().sortWith(_._1 < _._1)

    val augmentedSortedCounts: Seq[(String, Seq[(String, Int)])] = augmentedSubModelsTrainingSetDS.map(
      groupedTrainingSet => {
        val tokenizedTrainingSetCount = groupedTrainingSet
            .labeledTrainingData
            .groupBy(_.label)
            .map { case (label, xs) => (label.trim, xs.size) }
            .toSeq
            .sortWith(_._1 < _._1)
        (groupedTrainingSet.subModel.trim, tokenizedTrainingSetCount)
      }
    ).collect().sortWith(_._1 < _._1)


    val augmentedSortedCountsMap = augmentedSortedCounts.map {
      case (subModel, tokenizedTrainingSetCount) => (subModel, tokenizedTrainingSetCount.toMap)
    }.toMap

    val subModelDiff = originalSortedCounts.flatMap {
      case (subModel, tokenizedTrainingSetCount) =>
        val augmentedLabelCountsMap = augmentedSortedCountsMap.getOrElse(subModel, Map.empty[String, Int])

        if (augmentedLabelCountsMap.nonEmpty) {
          tokenizedTrainingSetCount.map {
            case (label, cnt) =>
              val count = augmentedLabelCountsMap.getOrElse(
              label,
              {
                logger.error(s"$label could not be found in the augmented set for sub model $subModel")
                -1
              }
              )
              val diff = {
                if (count != cnt)
                  s"${count - cnt} ${if (count > cnt) "**" else "@@"}"
                else if (cnt >= mlopsConfiguration.classifyConfig.minNumRecordsPerLabel)
                  "0 *"
                else
                  "0"
              }
              s"$subModel,$label,$cnt,$count,$diff"
          }
        }
        else {
          logger.error(s"No augmentation available for sub model $subModel")
          Seq[String](" , , , , ")
        }
    }
    logDebug(logger, s"${subModelDiff.size} sub model taxonomy count records")
    LocalFileUtil.Save.local(
      s"output/subModelTaxonomyCounts.csv",
      s"SubModel,Label,Initial count,Augmented count,diff\n${subModelDiff.mkString("\n")}"
    )
    delay(4000L)
  }





  /**
    * Generate a list of similar contextual document including the original one
    *
    * @param contextDocument Input or original contextual document
    * @return Augmented list of contextual documents
    */
  def augment(contextDocument: ContextualDocument, count: Int): Seq[ContextualDocument] = {
    val tokens = contextDocument.text.split(tokenSeparator)
    val outputTokensList = ListBuffer[Array[String]]()
    outputTokensList.append(tokens)

    var index = 0
    var remaining = mlopsConfiguration.classifyConfig.minNumRecordsPerLabel - count + 1

    while(index < tokens.size && remaining > 0) {
      val token = tokens(index)

      conversionMap.get(token).foreach(
        converted => {
          val newOutputTokens = outputTokensList.map(
            outputTokens => {
              if(remaining > 0) {
                val newTokens = outputTokens.clone()
                newTokens(index) = converted
                remaining -= 1
                newTokens
              }
              else
                Array.empty[String]
            }
          ).filter(_.nonEmpty)
          outputTokensList ++= newOutputTokens
        }
      )
      index += 1
    }

    val augmentedTextTokens = outputTokensList.map(_.mkString(" "))
    augmentedTextTokens.indices.map(
      idx => ContextualDocument(s"${contextDocument.id}-$idx", contextDocument.contextVariables, augmentedTextTokens(idx))
    )
  }


  final object InvalidLabelsManager {
    final private val fsInvalidLabelsFilename = "output/invalidLabels.txt"
    final private val fsSubModelsSlicePath = "output/subModelsSlice-"

    def initialize: Unit = {
      import org.mlops.implicits._
      import sparkSession.implicits._

      val subModelTrainingSetDS = S3Util.s3ToDataset[SubModelsTrainingSet](
        S3PathNames.s3ModelTrainingPath,
        header = false,
        fileFormat = "json")

      val labelFreqPairs = subModelTrainingSetDS
            .flatMap(_.labeledTrainingData.map(ts => (ts.label.trim, 1)))
            .rdd
            .reduceByKey(_ + _)
            .collect

      val minNumRecordsPerLabel = mlopsConfiguration.classifyConfig.minNumRecordsPerLabel
      val invalidLabels = labelFreqPairs.filter(_._2 < minNumRecordsPerLabel).map(_._1)
      LocalFileUtil.Save.local(fsInvalidLabelsFilename, invalidLabels.mkString("\n"))
      delay(1000L)
      val subModelsList = subModelTrainingSetDS.map(_.subModel.trim).collect()
      val step = (subModelsList.length.toFloat/6).floor.toInt

      var count = 0
      (0 until subModelsList.length by step).foreach(
        index => {
          val end = if(index + step > subModelsList.length) subModelsList.length else index + step
          val slice =  subModelsList.slice(index, end)
          count += 1
          LocalFileUtil.Save.local(s"${fsSubModelsSlicePath}$count.txt", slice.mkString("\n"))
        }
      )
    }

    def execute(
      segmentFilename: String
    ) (implicit sparkSession: SparkSession): Dataset[SubModelsTrainingSet] = {
      import sparkSession.implicits._

      val subModelsSlices = LocalFileUtil
          .Load
          .local(segmentFilename)
          .map(_.split("\n").toSet)
          .getOrElse(Set.empty[String])

      val labelCandidateForAugmentation = LocalFileUtil
          .Load
          .local(fsInvalidLabelsFilename)
          .map(_.split("\n").toSet)
          .getOrElse(Set.empty[String])

      if (subModelsSlices.nonEmpty) {
        val trainingSetDS = S3Util.s3ToDataset[SubModelsTrainingSet](
            S3PathNames.s3ModelTrainingPath,
            header = false,
            fileFormat = "json")

        val sliceSubModelTrainingSetDS = trainingSetDS.filter(
          subModelRecord => subModelsSlices.contains(subModelRecord.subModel.trim)
        ).map(
          subModelsTS => {
            val correctedLabeledTrainingSet = subModelsTS
                      .labeledTrainingData
                      .map(ts => ts.copy(label = ts.label.replace("  ", " ")))
            subModelsTS.copy(
              subModel = subModelsTS.subModel.trim,
              labeledTrainingData = correctedLabeledTrainingSet
            )
          }
        )
        val augmentedTrainingSetDS = augment(sliceSubModelTrainingSetDS, labelCandidateForAugmentation)
        // taxonomyCounts(sliceSubModelTrainingSetDS, augmentedTrainingSetDS)
        augmentedTrainingSetDS
      }
      else
        sparkSession.emptyDataset[SubModelsTrainingSet]
    }
  }





  /**
    * Augment/extends the current list of records associated with a given label.
    * The training set is to stored in S3
    *
    * @param groupedSubModelTrainingDS Training data loaded from S3
    * @param labelCandidateForAugmentation List of labels that need to be augmented..
    * @param sparkSession              Implicit reference to the current Spark context
    * @return Augmented training set
    */
  def augment(
    groupedSubModelTrainingDS: Dataset[SubModelsTrainingSet],
    labelCandidateForAugmentation: Set[String]
  )(implicit sparkSession: SparkSession): Dataset[SubModelsTrainingSet] = {
    import sparkSession.implicits._

    val accuSubModelsCount = sparkSession.sparkContext.longAccumulator("accuSubModelsCount")
    val accuLabelsCount = sparkSession.sparkContext.longAccumulator("accuLabelsCount")

    val augmentedTrainingSetDS = groupedSubModelTrainingDS.map(
      groupedBySubModel => {
        val subModel = groupedBySubModel.subModel
        accuSubModelsCount.add(1L)

        // Map {label -> Sequence of contextual documents}
        val groupedByLabel: Map[String, Seq[TokenizedTrainingSet]] = groupedBySubModel
            .labeledTrainingData
            .groupBy(_.label)
        accuLabelsCount.add(groupedByLabel.size)

        val tokenizedTS = groupedByLabel.flatMap {
          case (label, tokenizedTSet) =>
            // If this
            if (labelCandidateForAugmentation.contains(label)) {
              val augmentedContextualDocList = ListBuffer[ContextualDocument]()
              val iter = tokenizedTSet.iterator
              var count = 0
              while(iter.hasNext && count < mlopsConfiguration.classifyConfig.maxNumRecordsPerLabel) {
                val nextTokenizedSet= iter.next()
                val augmentedContextualDocuments = augment(nextTokenizedSet.contextualDocument, count)
                augmentedContextualDocList ++= augmentedContextualDocuments
                count += augmentedContextualDocuments.size
              }

              logDebug(logger, s"Newly ${augmentedContextualDocList.size} augmented contextual documents")
              augmentedContextualDocList.map(TokenizedTrainingSet(_, label, Array.empty[Float]))
            }
            else
              boundContextualDocuments(label, tokenizedTSet.map(_.contextualDocument))
        }
        SubModelsTrainingSet(subModel, tokenizedTS.toSeq, groupedBySubModel.labelIndices)
      }
    )
    println(s"${accuSubModelsCount.value}  ${accuLabelsCount.value}")
    augmentedTrainingSetDS
  }

  private def boundContextualDocuments(
    label: String,
    contextualDocuments: Seq[ContextualDocument]
  ): Seq[TokenizedTrainingSet] = {
    val maxNumRecordsPerLabel = mlopsConfiguration.classifyConfig.maxNumRecordsPerLabel
    val boundedContextualDocuments =
      if(contextualDocuments.size > maxNumRecordsPerLabel) {
        logDebug(
          logger,
          s"label $label with ${contextualDocuments.size} records is bounded to $maxNumRecordsPerLabel"
        )
        contextualDocuments.take(maxNumRecordsPerLabel)
      }
      else
        contextualDocuments
    boundedContextualDocuments.map(TokenizedTrainingSet(_, label, Array.empty[Float]))
  }

}

 */