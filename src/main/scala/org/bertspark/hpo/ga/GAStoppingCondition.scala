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

import java.util.concurrent.atomic.AtomicInteger
import org.apache.commons.math3.genetics.{Chromosome, Population, StoppingCondition}
import org.apache.commons.math3.util.Precision
import org.bertspark.config.MlopsConfiguration
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.hpo.ga.GASearch.logger
import org.bertspark.util.io.LocalFileUtil
import org.bertspark.RuntimeSystemMonitor
import scala.collection.mutable.ListBuffer


/**
 * Define the stopping condition for the Genetic algorithm. The best chromosome and fitness are displayed for
 * each reproduction cycle.
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class GAStoppingCondition(numGenerations: Int) extends StoppingCondition with RuntimeSystemMonitor {
  private[this] val generation = new AtomicInteger(0)
  private[this] val fittestChromosomes = ListBuffer[(String, Double)]()

  override def isSatisfied(population: Population): Boolean = {
    val fittestChromosome = population.getFittestChromosome()
    val generationCnt = generation.incrementAndGet()
    logger.info(s"Generation: $generationCnt with fittest chromosome: ${fittestChromosome.toString}")

    val fitness = fittestChromosome.fitness()
    // Save to local file system
    saveToFS(fittestChromosome, fitness)

    // Record and display memory and CPU/GPU usage
    logDebug(
      logger,
      {
        val metricsSummary = allMetrics("GA stopping")
        s"$generationCnt: ${if(metricsSummary.nonEmpty) metricsSummary else ""}"
      }
    )
    Precision.equals(fitness, 0.8, 0.1) && generation.get >= numGenerations
  }

  private def saveToFS(fittestChromosome: Chromosome, accuracy: Double): Unit = {
    val runId = MlopsConfiguration.mlopsConfiguration.runId
    val content = s"GA cycle: ${generation.get()}\nAccuracy: $accuracy\n${fittestChromosome.toString}"
    GASearch.logger.info(content)

    // Collect the fittest chromosome and sort them in decreasing order of accuracy or fitness
    // for further analysis.
    fittestChromosomes.append((content, accuracy))
    val sortedChromosomes = fittestChromosomes.sortWith(_._2 > _._2)
    LocalFileUtil.Save.local(s"output/hpoga-$runId", sortedChromosomes.map(_._1).mkString("\n\n"))
  }
}


