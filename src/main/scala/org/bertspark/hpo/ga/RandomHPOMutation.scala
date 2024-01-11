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

import org.apache.commons.math3.genetics.{Chromosome, GeneticAlgorithm, RandomKeyMutation}
import org.bertspark.hpo.ga.GASearch.HPOBaseParam


private[bertspark] final class RandomHPOMutation extends RandomKeyMutation {

    override def mutate(chromosome: Chromosome): Chromosome = {
      chromosome match {
        case hpoChromosome: HPOChromosome => {
          val representation: java.util.List[HPOBaseParam] = hpoChromosome.getHpoParameters.chromosomeRepresentation
          val randIndex = GeneticAlgorithm.getRandomGenerator().nextInt(representation.size)

          val randParam: HPOBaseParam = representation.get(randIndex)
          randParam.updateCurrentIndex(GeneticAlgorithm.getRandomGenerator().nextInt(randParam.values.size))
         // representation.set(randIndex, randParam)
          hpoChromosome.newFixedLengthChromosome((representation))
        }

        case _ =>
          throw new IllegalStateException((s"Mutating chromosome has incorrect type ${chromosome.getClass.getName}"))

      }
    }
}


