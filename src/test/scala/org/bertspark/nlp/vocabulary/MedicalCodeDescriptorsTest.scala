package org.bertspark.nlp.vocabulary

import org.bertspark.config.MlopsConfiguration.vocabulary
import org.bertspark.delay
import org.scalatest.flatspec.AnyFlatSpec



private[vocabulary] final class MedicalCodeDescriptorsTest extends AnyFlatSpec {
  import org.bertspark.nlp.vocabulary.MedicalCodeDescriptors._
  import MedicalCodeDescriptorsTest._


  it should "Succeed extracting code descriptors associated with a claim" in {
    val claim = "71275 26,GC I71.6-74174 26,GC I71.6-G9637 I71.6-G9637 I71.6"
    val descriptors = CodeDescriptorMap.getClaimDescriptors(claim)
    println(descriptors.mkString(" "))
  }

  ignore should "Succeed extracting left and right ICD codes" in {
    println(s"Left icds:\n${leftIcds.mkString(" ")}")
    println(s"Right icds:\n${rightIcds.mkString(" ")}")
  }

  ignore should "Succeed getting vocabulary" in {
    val token = "heart"
    val index = vocabulary.getIndex(token)
    assert(token == vocabulary.getToken(index))
  }

  ignore should "Succeed generating medical code descriptor terms" in {
    MedicalCodeDescriptors.build
    val medicalCodeDescriptorMap = CodeDescriptorMap.loadMedicalDescriptorsMap
    delay(2000L)
    val keyWords1 = getDescriptor("71046", medicalCodeDescriptorMap)
    println(keyWords1.mkString(" "))
    println("X-ray of chest, 2 views")

    val keyWords2 = getDescriptor("80076", medicalCodeDescriptorMap)
    println(keyWords2.mkString(" "))
    println("Hepatic function panel - Measurement of aspartate amino transferase")

    val keyWords3 = getDescriptor("72295", medicalCodeDescriptorMap)
    println(keyWords3.mkString(" "))
    println("Radiological supervision and interpretation for lumbar discography")
  }

  ignore should "Succeed extracting CPT and ICD terms" in {
    val cptIcdTerms = getCptIcdTerms
    println(cptIcdTerms.mkString("\n"))
  }

  ignore should "Succeed retrieving the CPT descriptors map"  in {
    val cptDescriptors = getCptDescriptors
    val headIndices = s"${cptDescriptors.head._1}: ${cptDescriptors.head._2.mkString(" ")}"
    println(headIndices)
    val headTerms = s"${cptDescriptors.head._1}: ${cptDescriptors.head._2.mkString(" ")}"
    println(headTerms)
  }

  ignore should "Succeed retrieving the ICD descriptors map"  in {
    val icdDescriptors = getIcdDescriptors
    val headIndices = s"${icdDescriptors.head._1}: ${icdDescriptors.head._2.mkString(" ")}"
    println(headIndices)
    val headTerms = s"${icdDescriptors.head._1}: ${icdDescriptors.head._2.mkString(" ")}"
    println(headTerms)
  }

  ignore should "Succeed generating, storing and loading medical code descriptors" in {
    delay(2000L)
    val codeDescriptorMap = CodeDescriptorMap.loadMedicalDescriptorsMap
    val head = s"${codeDescriptorMap.head._1}: ${codeDescriptorMap.head._2.mkString(" ")}"
    println(head)
  }
}


private[vocabulary] final object MedicalCodeDescriptorsTest {

  def getDescriptor(cpt: String, medicalCodeDescriptorMap: Map[String, Array[String]]): Array[String] =  {
    val keys = medicalCodeDescriptorMap.get(cpt).get
    println(cpt)
    keys
  }

}