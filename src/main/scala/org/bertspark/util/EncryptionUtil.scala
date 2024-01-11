package org.bertspark.util

import javax.crypto.spec.{IvParameterSpec, SecretKeySpec}
import javax.crypto.{AEADBadTagException, BadPaddingException, Cipher, IllegalBlockSizeException}
import scala.util.hashing.MurmurHash3
import org.apache.commons.codec.binary.Base64.{decodeBase64, encodeBase64String}
import org.slf4j.{Logger, LoggerFactory}


object EncryptionUtil {
  final val log: Logger = LoggerFactory.getLogger("EncryptionUtil")

  private final val AesLabel = "AES"
  private final val EncodingScheme = "UTF-8"
  private final val key = "aesEncryptorABCD"
  private final val initVector = "aesInitVectorABC"
  private val iv = new IvParameterSpec(initVector.getBytes(EncodingScheme))
  private val keySpec = new SecretKeySpec(key.getBytes(), AesLabel)
  private val cipher = Cipher.getInstance("AES/CBC/PKCS5Padding")


  /**
   * Generate a hashed encryption for a word or group of words using
   * Murmur3 Hash. If a supplement character is provided then it replace the negative sign
   * @param content Content to be encrypted
   * @param supplement Optional supplement character to replace the - sign in the hash integer value
   * @return Encrypted content
   */
  final def hashEncrypt(content: String, supplement: String = ""): String =
    if(content.nonEmpty)
      if(supplement.nonEmpty) {
        val value = MurmurHash3.stringHash(content)
        if(value < 0) s"xx${supplement}${-value}" else s"xx${value}"
      }
      else
        MurmurHash3.stringHash(content).toString
    else
      ""

  /**
   * Encrypt a string or content using AES and Base64 bytes representation
   * @param content String to be encrypted
   * @return Optional encrypted string
   */
  def apply(content: String): Option[String] =  try {
    cipher.init(Cipher.ENCRYPT_MODE, keySpec, iv)

    val encrypted = cipher.doFinal(content.getBytes)
    Some(encodeBase64String(encrypted))
  } catch {
    case e: IllegalStateException =>
      log.error(e.toString)
      None
    case e: IllegalBlockSizeException =>
      log.error(e.toString)
      None
    case e: BadPaddingException =>
      log.error(e.toString)
      None
    case e: Exception =>
      log.error(e.toString)
      None
  }


  /**
   * Decryption of content
   * @param content Encrypted content
   * @return Optional decrypted string
   */
  def unapply(content: String): Option[String] = try {
    cipher.init(Cipher.DECRYPT_MODE, keySpec, iv)
    val originalContent = cipher.doFinal(decodeBase64(content))
    Some(new String(originalContent))
  } catch{
    case e: IllegalStateException =>
      log.error(e.toString)
      None
    case e: IllegalBlockSizeException =>
      log.error(e.toString)
      None
    case e: BadPaddingException =>
      log.error(e.toString)
      None
    case e: Exception =>
      log.error(e.toString)
      None
  }
}
