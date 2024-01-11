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
package org.bertspark.nlp.token

import org.bertspark.nlp.token.TokenizerPreProcessor.TextPreprocessor


/**
 * Original Bert tokenizer extended to support abbreviations and other relevant pre-processing functions
 * @param textPreprocessor Configuration for the preprocessing bert tokenizer
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class ExtBERTTokenizer private (
) extends ai.djl.modality.nlp.bert.BertTokenizer
    with TokenizerPreProcessor {

  override def tokenize(input: String): java.util.List[String] = {
    val cleansed = TextPreprocessor()(input)
    super.tokenize(cleansed)
  }

  override def apply(input: String): java.util.List[String] = tokenize(input)
}



private[bertspark] final object ExtBERTTokenizer {
  def apply(): ExtBERTTokenizer = new ExtBERTTokenizer
}