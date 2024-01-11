package org.bertspark.util

import org.scalatest.flatspec.AnyFlatSpec

class EncryptionUtilTest extends AnyFlatSpec {

  it should "Succeed encrypting/decrypting password" in {
    val password = "CFR110J+bRrmw9iXLUkw9gxIgxRl2c4c9qO190A5"
    val encrypted = EncryptionUtil(password).getOrElse("")
    println(encrypted)
    val decrypted = EncryptionUtil.unapply(encrypted).getOrElse("")
    assert(password == decrypted)
  }
}
