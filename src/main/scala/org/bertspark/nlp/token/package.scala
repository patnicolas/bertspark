package org.bertspark.nlp


/**
 * Classes and methods for tokenization, segmentation of notes, TF-IDF ranking, document extraction. ...
 * @author Patrick Nicolas
 * @version 0.1
 */
package object token {

  trait PretrainingInput {
    self =>
    def getId: String
  }
}
