package org.bertspark.dl.config

import ai.djl.nn.norm.BatchNorm
import ai.djl.nn.Block

/**
 * Configuration of batch normalization neural block
 *
 * @param blockType Type of normalization
 * @param axis      Axis (Shape) the normalization applies
 * @param center    Is batch normalized around its center?
 * @param epsilon   Epsilon factor to be learned
 * @param momentum  Momentum factor to be learned
 * @param scale     Scale factor to be learned
 * @author Patrick Nicolas
 * @version 0.1
 */
case class BatchNormConfig(
  override val blockType: String,
  axis: Int,
  center: Boolean,
  epsilon: Float,
  momentum: Float,
  scale: Boolean
) extends BlockConfig {

  @throws(clazz = classOf[UnsupportedOperationException])
  override def apply(): Block = blockType match {
    case org.bertspark.dl.batchNormLbl =>
      BatchNorm.builder
          .optAxis(axis)
          .optCenter(center)
          .optEpsilon(epsilon)
          .optMomentum(momentum)
          .optScale(scale)
          .build
    case _ =>
      throw new UnsupportedOperationException(s"Batch norm type $blockType is not supported")
  }

  override def toString: String =
    s"conv_${getId}_(Axis: $axis, Center: $center, Epsilon $epsilon, Momentum: $momentum, Scale:$scale)"
}
