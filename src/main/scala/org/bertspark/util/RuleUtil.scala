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
package org.bertspark.util

/**
 * Define the generic recursive tree mode as either a rule condition or a rule action
 * @tparam T Type used in the condition and action clauses
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] sealed trait RuleUtil[+T] {
  self =>
}

private[bertspark] final class RuleAction[T](value: T) extends RuleUtil[T] {
  final def getValue: T = value
}

private[bertspark] final class RuleCondition[T](
  ruleId: String,
  var value: T = null.asInstanceOf[T],
  var thenRule: RuleUtil[T] = null.asInstanceOf[RuleUtil[T]],
  var elseRule: RuleUtil[T] = null.asInstanceOf[RuleUtil[T]]) extends RuleUtil[T] {

  final def getRuleId: String = ruleId

  override def toString: String = s"$ruleId, $value, ${thenRule.toString}, ${elseRule.toString}"
}