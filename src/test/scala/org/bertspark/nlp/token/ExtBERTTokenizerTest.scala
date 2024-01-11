package org.bertspark.nlp.token

import org.scalatest.flatspec.AnyFlatSpec

private[token] final class ExtBERTTokenizerTest extends AnyFlatSpec{

  it should "Succeed processing a given text" in {
    import org.bertspark.implicits._

    val input = "\\r\\n\\r\\nDate of Birth :04/04/1946\\r\\n\\r\\nMedical Record Number :RAM374862\\r\\n\\r\\nEXAM: LUMBAR SPINE 2 VIEWS?\\r\\n\\r\\nCLINICAL: LOW BACK PAIN.\\r\\n\\r\\nFINDINGS: NO PRIOR STUDIES. MINIMAL RIGHT CONVEX LUMBAR SCOLIOSIS. THE LUMBAR\\r\\nVERTEBRAL BODIES ARE OTHERWISE NORMAL IN HEIGHT AND ALIGNMENT. NO ACUTE\\r\\nFRACTURES OR SUBLUXATIONS. SEVERE DEGENERATIVE DISC DISEASE IS PRESENT\\r\\nTHROUGHOUT THE LUMBAR SPINE WITH OSTEOPHYTOSIS AND SEVERE DISC SPACE NARROWING\\r\\nWITH ENDPLATE SCLEROSIS. MULTILEVEL FACET JOINT DEGENERATIVE CHANGES ARE MOST\\r\\nPRONOUNCED AT L5-S1. BILATERAL RENAL CALCULI. RIGHT UPPER QUADRANT SURGICAL\\r\\nCLIPS.\\r\\n\\r\\nIMPRESSION:\\r\\n\\r\\nSEVERE MULTILEVEL DEGENERATIVE DISC DISEASE.\\r\\n\\r\\nBILATERAL RENAL CALCULI SUSPECTED.\\r\\n\\r\\nSIGNER NAME: DANIEL FOX\\r\\nSIGNED: 8/20/2019 11:51 AM EST\\r\\nWORKSTATION NAME: FINAODS-PC \\r\\nSIGNED BY DANIEL AT 8/20/2019 6:51:09 AM\\r\\n\\r\\n\\r\\nELECTRONICALLY SIGNED BY: DANIEL,  ON 20190820065109-0500\\r\\n"
    val preProcessedBertTokenizer = ExtBERTTokenizer()
    val tokens = listOf[String](preProcessedBertTokenizer.tokenize(input))
    println(tokens.mkString("\n"))
  }
}
