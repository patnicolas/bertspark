package org.bertspark.dl.config

import ai.djl.training.optimizer._
import ai.djl.training.tracker._
import ai.djl.training.tracker.WarmUpTracker.Mode
import org.bertspark.config.OptimizerConfig

/**
 * Configuration of Optimizers
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final object Optimizers {
  def sgd(lr: Float, momentum: Float): Optimizer = {
    val lrTracker = Tracker.fixed(lr)
    Optimizer.sgd.setLearningRateTracker(lrTracker).optMomentum(momentum).build
  }

  def adamGrad(lr: Float, epsilon: Float): Optimizer = {
    val lrTracker = Tracker.fixed(lr)
    Optimizer.adagrad().optLearningRateTracker(lrTracker).optEpsilon(epsilon).build
  }

  def adam(lr: Float, beta1: Float, beta2: Float, epsilon: Float): Optimizer = {
    val lrTracker = Tracker.fixed(lr)
    Optimizer.adam
        .optLearningRateTracker(lrTracker)
        .optBeta1(beta1)
        .optBeta2(beta2)
        .optEpsilon(epsilon)
        .build
  }

  def adam(optimizerConfig: OptimizerConfig, decayStepsFactor: Int): Optimizer =
    adam(optimizerConfig.baseLr, optimizerConfig.epsilon, optimizerConfig.numSteps, decayStepsFactor)

  def adam(lr: Float, epsilon: Float, numSteps: Int, decayStepsFactor: Int): Optimizer = {
    val mainTracker = PolynomialDecayTracker.builder
        .setBaseValue(lr)
        .setEndLearningRate(lr / numSteps)
        .setDecaySteps(numSteps * decayStepsFactor)
        .optPower(1.0F)
        .build

    val learningRateTracker = WarmUpTracker.builder
        .optWarmUpBeginValue(0F)
        .optWarmUpSteps(numSteps)
        .optWarmUpMode(Mode.LINEAR)
        .setMainTracker(mainTracker)
        .build

    Adam.builder.optEpsilon(epsilon).optLearningRateTracker(learningRateTracker).build
  }

  def rmsProp(lr: Float, rho: Float, momentum: Float, epsilon: Float, centered: Boolean): Optimizer = {
    val lrTracker = Tracker.fixed(lr)
    Optimizer.rmsprop()
        .optLearningRateTracker(lrTracker)
        .optRho(rho)
        .optMomentum(momentum)
        .optEpsilon(epsilon)
        .optCentered(centered)
        .build
  }
}
