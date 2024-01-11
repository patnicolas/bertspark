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

import org.apache.commons.math3.genetics._
import org.apache.spark.sql.SparkSession
import org.bertspark.config.ExecutionMode
import org.bertspark.hpo.{ClassifierTrainHPO, ClassifierTrainHPOConfig}
import org.slf4j._

/**
 *
 * @param numSubModels
 * @param sparkSession Implicit reference to the current Spark context
 */
final class GASearch(implicit sparkSession: SparkSession) {
  import org.apache.commons.math3.genetics.{Chromosome, ElitisticListPopulation, GeneticAlgorithm, OnePointCrossover, Population, TournamentSelection}
  import GASearch._

  ExecutionMode.setHpo

  // GA algorithm
  private[this] val ga = new GeneticAlgorithm(
    new OnePointCrossover[Int](),
    defaultXoverRate,
    new RandomKeyMutation(),
    defaultMutationRate,
    new TournamentSelection(tournementArity)
  )

  private[this] val classifierHPOConfiguration = ClassifierTrainHPO.loadClassifierHPOConfiguration

  //  Initial randomly selected population
  private[this] val initialPopulation: Population = {
    val _hpoConfig: Option[ClassifierTrainHPOConfig] = classifierHPOConfiguration.map(_.getClassifierTrainHPOConfig)
    val hpoParameters = _hpoConfig.map(_.getHPOParamsList).getOrElse(List.empty[HPOBaseParam])

    if(hpoParameters.isEmpty)
      throw new IllegalArgumentException("Failed to retrieve HPO parameters")

    val populationSize = classifierHPOConfiguration.map(_.populationSize).getOrElse(20)
    val initial = List.fill(populationSize)(
      new HPOChromosome(_hpoConfig.map(_.getNumSubModels).getOrElse(300), new HPOParameters(hpoParameters))
    )
    import org.bertspark.implicits._
    val initialPop: java.util.List[Chromosome] = initial
    new ElitisticListPopulation(initialPop, (populationSize<<1), defaultElitistRate)
  }

  private[this] val stoppingCondition =
    new GAStoppingCondition(classifierHPOConfiguration.map(_.numCycles).getOrElse(60))

  def apply(): Boolean = {
    val finalPopulation = ga.evolve(initialPopulation, stoppingCondition)

    val bestChromosome: Chromosome = finalPopulation.getFittestChromosome
    logger.info(bestChromosome.toString)
    save(bestChromosome)
    true
  }

  private def save(chromosome: Chromosome): Unit = chromosome match {
    case bestChromosome: HPOChromosome => bestChromosome.savetoFs
    case _ => logger.error(s"Chromosome type, ${chromosome.getClass.getName} is incorrect")
  }
}


final object GASearch {
  final val logger: Logger = LoggerFactory.getLogger("HPOGASearch")

  private final val rand = new scala.util.Random(34L)


  trait HPOBaseParam {
    type T
    val values: Array[T]
    val label: String

    def getValue: T

    final def getLabel: String = label

    protected[this] var currentIndex: Int = rand.nextInt(values.length)


    def updateCurrentIndex(newIndex: Int): Unit = currentIndex = newIndex

    def convert: Float = currentIndex.toFloat / values.size

    def convert(ratio: Float): Unit = {
      val converted = (ratio * values.size + 0.5F).floor.toInt
      currentIndex = if (converted < 0) 0 else if (converted >= values.size) values.size - 1 else converted
    }

    def state: String = s"$label,${values(currentIndex)}"

    override def toString: String = s"$label: ${values.mkString(" ")}"
  }


  final class HPOParamString(override val label: String, override val values: Array[String]) extends HPOBaseParam {
    type T = String
    override def getValue: String = values(currentIndex)
  }

  final class HPOParamInt(override val label: String, override val values: Array[Int]) extends HPOBaseParam {
    type T = Int
    override def getValue: Int = values(currentIndex)
  }

  final class HPOParamFloat(override val label: String, override val values: Array[Float]) extends HPOBaseParam {
    type T = Float
    override def getValue: Float = values(currentIndex)
  }

  final class HPOParamArrayInt(
    override val label: String,
    override val values: Array[Array[Int]]) extends HPOBaseParam {
    type T = Array[Int]
    override def getValue: Array[Int] = values(currentIndex)
    override def state: String = s"$label:${values(currentIndex).mkString(" ")}"
  }
}
