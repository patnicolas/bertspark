package org.bertspark.hpo.ga

import org.bertspark.hpo.ClassifierTrainHPO.loadHPOConfiguration
import org.scalatest.flatspec.AnyFlatSpec

private[ga] final class RandomHPOMutationTest extends AnyFlatSpec {

  it should "Succeed mutating a HPO chromosome" in {
    import org.bertspark.implicits._

    val RandomHPOMutation = new RandomHPOMutation

    loadHPOConfiguration.foreach(
      classifierTrainHPOConfig => {
        val chromosome = classifierTrainHPOConfig.getHPOChromosome
        val mutatedChromosome = RandomHPOMutation.mutate(chromosome)
        println(chromosome.toString)
        println(mutatedChromosome.toString)
      }
    )
  }
}
