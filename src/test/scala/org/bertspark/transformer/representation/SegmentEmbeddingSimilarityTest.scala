package org.bertspark.transformer.representation

import org.bertspark.nlp.trainingset.ContextualDocument
import org.scalatest.flatspec.AnyFlatSpec


private[representation] final class SegmentEmbeddingSimilarityTest extends AnyFlatSpec {

  it should "Succeed comparing Jaccard and unorderedJaccard" in {
    import SegmentEmbeddingSimilarityTest._
    val x = Array[String]("GGG", "AAA", "EEE", "BBB", "HHH", "CCC", "EEE", "FFF", "HHH", "DDD")
    val y = x.reverse
    val z = Array[String]("AAA", "BBB", "CCC", "DDD")
    val t = z.reverse
    val u = Array.fill(x.length)("AAA")
    val v = Array.fill(4)("AAA")
    val w = Array.fill(4)("ZZZ")

    val result =s"""${dump("Jaccard ", x, x, jaccard[String](x, x))}
         |${dump("OJaccard", x, x, orderedJaccard[String](x, x))}
         |${dump("Jaccard ", x, y, jaccard[String](x, y))}
         |${dump("OJaccard", x, y, orderedJaccard[String](x, y))}
         |${dump("Jaccard ", x, z, jaccard[String](x, z))}
         |${dump("OJaccard", x, z, orderedJaccard[String](x, z))}
         |${dump("Jaccard ", x, t, jaccard[String](x, t))}
         |${dump("OJaccard", x, t, orderedJaccard[String](x, t))}
         |${dump("Jaccard ", x, u, jaccard[String](x, u))}
         |${dump("OJaccard", x, u, orderedJaccard[String](x, u))}
         |${dump("Jaccard ", x, v, jaccard[String](x, v))}
         |${dump("OJaccard", x, v, orderedJaccard[String](x, v))}
         |${dump("Jaccard ", x, w, jaccard[String](x, w))}
         |${dump("OJaccard", x, w, orderedJaccard[String](x, w))}
         """.stripMargin
    println(result)
  }

  it should "Succeed computing similarity of segments/sentences" in {
    val id = "1"
    val contextVariables1 = Array[String](
      "2_age","m _gender","cornerstone_cust","sum_client","diagnostic_modality","er_pos","73110_cpt","26_mod","complete","x-ray","of","wrist","","minimum","views"
    )
    val contextVariables2 = contextVariables1.drop(4)
    val text1 = "sumner regional medical center patient name tracy la ##uderdale birth date xnum xnum xnum ss ##n xnum xnum xnum re ferring doctor abe ##ll ##j os hua reading doctor blom ##quist gus ##tav visit no mxnum order no xnum xnum exam date xnum xnum xnum exam xnum xnum department report report number initials date time use ##r unit number acc t number patient name age sex dict ##ating md ordering md rad xnum xnum xnum xnum xnum xnum interface mxnum mxnum la ##uderdale tracy dew ##ayne xnum blom ##quist gus ##tav md abe ##jos or ##d xnum xnum access ion xnum xnum cat rad pro ##c name wrist lt min xnum ##v xnum procedure forearm lt xnum ##v xnum sept ##ember xnum xnum xnum xnum wrist lt min xnum ##v xnum sept ##ember xnum xnum xnum xnum comparison none indication s pain without trauma injury sectionfindings no elbow joint effusion no fracture or dislocation no radiopaque foreign body no soft tissue swelling conc ##lusion no abnormality identified dict ated by blom ##quist gus ##tav md on sept ##ember xnum xnum at xnum xnum approved by blom ##quist gus ##tav md on sept ##ember xnum xnum at xnum xnum this report was veri fied electronic ally"
    val text2 = "sumner regional medical center patient name birth la ##uderdale birth date xnum xnum xnum ss ##n xnum re ferring doctor abe ##ll ##j os hua reading doctor blom ##quist gus ##tav visit no mxnum order no xnum xnum exam date xnum xnum xnum exam xnum xnum department report report number initials date time use ##r unit number acc t number patient name age sex dict ##ating md ordering md rad xnum xnum xnum xnum xnum xnum interface mxnum mxnum la ##uderdale tracy dew ##ayne xnum blom ##quist gus ##tav md abe ##jos or ##d xnum xnum access ion xnum xnum cat rad pro ##c name wrist lt min xnum ##v xnum procedure forearm lt xnum ##v xnum sept ##ember xnum xnum xnum xnum wrist lt min xnum ##v xnum sept ##ember xnum xnum xnum xnum comparison none indication s pain without trauma injury sectionfindings no elbow joint effusion no fracture or dislocation no radiopaque foreign body no soft tissue swelling conc ##lusion no abnormality identified dict ated by blom ##quist gus ##tav md on sept ##ember xnum xnum at xnum xnum approved by blom ##quist gus ##tav md on sept ##ember xnum xnum at xnum xnum this report was veri fied electronic ally"

    val contextualDocument1 = ContextualDocument("1", contextVariables1, text1)
    val contextualDocument2 = ContextualDocument("2", contextVariables2, text1)
    val contextualDocument3 = ContextualDocument("3", contextVariables1, text2)


  }
}

private[representation] final object SegmentEmbeddingSimilarityTest {
  def dump(str: String, x: Array[String], y: Array[String], j: Double): String =
    s"$str: ${x.mkString(",")}  ${y.mkString(",")} - $j"
}
