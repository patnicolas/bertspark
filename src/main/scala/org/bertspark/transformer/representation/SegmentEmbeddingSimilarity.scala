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
package org.bertspark.transformer.representation

import org.bertspark.config.ExecutionMode
import org.bertspark.nlp.token.SentencesBuilder
import org.bertspark.transformer.representation.SegmentEmbeddingSimilarity.logger
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.nlp.tokenSeparator
import org.bertspark.nlp.trainingset.ContextualDocument
import org.bertspark.transformer.representation
import org.bertspark.transformer.representation.EmbeddingSimilarity.ModelSimilarity
import org.bertspark.util.io.LocalFileUtil
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable.ListBuffer


/**
 * Computes the similarity of segment embeddings
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final class SegmentEmbeddingSimilarity extends EmbeddingSimilarity  {
  private[this] val segEmbeddingsBuf = ListBuffer[(String, Seq[Array[Float]])]()
  private[this] val segTokensBuf = ListBuffer[(String, Seq[String])]()

  /**
   * Insert a context document as one element used in the computation of the similarity
   * @param contextDocument Contextual document as loaded from S3 bucket
   */
  def += (contextDocument: ContextualDocument): Unit =
    if(ExecutionMode.isSimilarity) {
      val sentencesBuilder = SentencesBuilder()
      val contextSentences = sentencesBuilder.apply(contextDocument)
      val segContents = contextSentences.map{
        case (ctx, text) => if(ctx.isEmpty) text else if(text.isEmpty) text else s"$ctx $text"
      }
      segTokensBuf.append((contextDocument.id, segContents))
    }



  def += (docId: String, embeddingValues: Array[Float]): Unit =
    if(ExecutionMode.isSimilarity) {
      val embSize = mlopsConfiguration.getEmbeddingsSize.toInt
      val segEmbeddings = (0 until embeddingValues.size by embSize).foldLeft(ListBuffer[Array[Float]]())(
        (xs, index) =>  {
          xs.append(embeddingValues.slice(index, index+embSize))
          xs
        }
      )
      segEmbeddingsBuf.append((docId, segEmbeddings))
    }

  override def similarity(numSamples: Int = 40): ModelSimilarity =
    if(ExecutionMode.isSimilarity) {
      val segTokensMap = segTokensBuf.toMap
      val segmentAndEmbeddings = segEmbeddingsBuf.flatMap{
        case (id, segEmbeddings) => {
          val segmentText = segTokensMap.getOrElse(id, {
            logger.warn(s"Cannot extract segment similarity for document $id")
            Seq.empty[String]
          })
          segmentText.zip(segEmbeddings)
        }
      }
      val segmentAndEmbeddingsStr = segmentAndEmbeddings.map{
        case (text, embedding) => s"$text\n${embedding.mkString(" ")}"
      }.mkString("\n\n")
      val tokensAndEmbeddings = segmentAndEmbeddings.map{
        case (text, embedding) => (text.split(tokenSeparator), embedding)
      }

      val tokenEmbeddingSim = (0 until numSamples).map(
        _ => {
          val r1 = scala.util.Random.nextInt(numSamples)
          val r2 =  scala.util.Random.nextInt(numSamples)
          val (tokens1, embedding1) = tokensAndEmbeddings(r1)
          val (tokens2, embedding2) = tokensAndEmbeddings(r2)
          val tokenSim = jaccard[String](tokens1, tokens2)
          val orderedTokenSim = orderedJaccard[String](tokens1, tokens2)
          val embeddingSim = representation.similarity(embedding1, embedding2)
          val cosineSim = representation.cosine(embedding1, embedding2)
          (tokenSim, orderedTokenSim, embeddingSim, cosineSim)
        }
      )
      val tokenEmbeddingSimStr = tokenEmbeddingSim.map{
        case (tokenSim, orderedTokenSim, embeddingSim, cosineSim) =>
          s"$tokenSim,$orderedTokenSim,$embeddingSim,$cosineSim"
      }.mkString("\n")

      // Segregate the embedding sharing the tokens
      val (identicalTokenEmbeddings, differenceTokenEmbedding) = tokenEmbeddingSim.partition(_._1 == 1.0)
      val aveIdenticalEmbeddings = identicalTokenEmbeddings.map(_._2).sum/identicalTokenEmbeddings.size
      val aveDifferentEmbeddings = differenceTokenEmbedding.map(_._2).sum/differenceTokenEmbedding.size
      val runId = mlopsConfiguration.runId
      LocalFileUtil.Save.local(
        s"output/similarities-$runId.csv",
        s"Token sim,Ordered token sim,Embedding sim,Cosine\n$tokenEmbeddingSimStr")
      LocalFileUtil.Save.local(s"output/segmentAndEmbeddings-$runId.txt", segmentAndEmbeddingsStr)
      ModelSimilarity(aveIdenticalEmbeddings, aveDifferentEmbeddings)
    }
    else
      ModelSimilarity()
}


private[bertspark] final object SegmentEmbeddingSimilarity  {
  final private val logger: Logger = LoggerFactory.getLogger("SegmentEmbeddingSimilarity")

  lazy val segmentEmbeddingsSimilarity = new SegmentEmbeddingSimilarity
}