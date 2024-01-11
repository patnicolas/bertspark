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
package org.bertspark.hpo.ga


import org.apache.commons.math3.genetics.AbstractListChromosome
import org.apache.spark.sql.SparkSession
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.hpo.ga.GASearch.{logger, HPOBaseParam}
import org.bertspark.hpo.ClassifierTrainHPOConfig.allSubModelNames
import org.bertspark.util.io.LocalFileUtil
import org.bertspark.util.DateUtil
import org.bertspark.RuntimeSystemMonitor
import org.bertspark.classifier.training.TClassifier
import org.slf4j.{Logger, LoggerFactory}


/**
 * Define the basic chromosome representation for the HPO parameters
 * @param numSubModels Number of sub models
 * @param hpoParameters Initial HPO parameters
 * @param sparkSession Implicit reference to the current Spark context
 *
 * @author Patrick Nicolas
 * @version 0.4
 */
private[bertspark] final class HPOChromosome(
  numSubModels: Int,
  hpoParameters: HPOParameters
)(implicit sparkSession: SparkSession)
    extends AbstractListChromosome[HPOBaseParam](hpoParameters.chromosomeRepresentation) with RuntimeSystemMonitor  {
  import HPOChromosome._

  private[this] val mlopsClassification = TClassifier()
  override def checkValidity(chromosomeRepresentation: java.util.List[HPOBaseParam]): Unit = { }

  override def newFixedLengthChromosome(
    chromosomeRepresentation: java.util.List[HPOBaseParam]
  ): AbstractListChromosome[HPOBaseParam] = {
    val params: List[HPOBaseParam] = HPOChromosome.convertChromosomeRepresentation(chromosomeRepresentation)
    new HPOChromosome(numSubModels, new HPOParameters(params))
  }


  final def getHpoParameters: HPOParameters = hpoParameters

  override def fitness: Double = {
    hpoParameters.updateMlopsConfiguration
    GASearch.logger.info(s"HPO parameters value: ${hpoParameters.toString}")
    // Record and display memory and CPU/GPU usage
    logDebug(
      GASearch.logger,
      {
        val metricsSummary = allMetrics("HPChromosome")
        if(metricsSummary.nonEmpty) metricsSummary else ""
      }
    )
    mlopsClassification()
  }

  def savetoFs: Unit = {
    val outputFileName = s"$outputFilePrefix-${DateUtil.simpleLongToDate}"
    LocalFileUtil.Save.local(outputFileName, toString)
  }

  override def toString: String = s"Number sub models: $numSubModels\n${hpoParameters.toString}"
}


private[bertspark] final object HPOChromosome {
  final private val outputFilePrefix = "output/hpoParams"


  def convertChromosomeRepresentation(chromosomeRepresentation: java.util.List[HPOBaseParam]): List[HPOBaseParam] = {
    import org.bertspark.implicits._

    require(chromosomeRepresentation.nonEmpty, "Chromosome representation should not be empty")
    val params: List[HPOBaseParam] = chromosomeRepresentation
    params
  }
}