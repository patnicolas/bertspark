package org.bertspark.dl.config

import ai.djl.nn.Block

/**
 * Generic configuration of a neural block (DJL). The type (or class) of block is defined by blockType.
 * The key method, apply() generate a DJL block from the configuration of type 'blockType'
 *
 * @author Patrick Nicolas
 * @version 0.1
 */

private[bertspark] trait BlockConfig {
self =>
  protected[this] val blockType: String

  def apply(): Block

  final def getId: String = blockType
}
