package org.bertspark.hpo

import org.scalatest.flatspec.AnyFlatSpec

class HPOSearchTest extends AnyFlatSpec{

  it should "Succeed unsqueeze global index" in {
    val randomSearch = new HPORandomSearch(
        Array[Array[Int]](
        Array[Int](0, 1),
        Array[Int](0, 1, 2, 3),
        Array[Int](0, 1, 2)
      )
    )

    var globalIndex = 2
    var result = randomSearch.decode(globalIndex)
    assert(result._1 == 1)
    assert(result._2 == 0)

    globalIndex = 0
    result = randomSearch.decode(globalIndex)
    assert(result._1 == 0)
    assert(result._2 == 0)

    globalIndex = 1
    result = randomSearch.decode(globalIndex)
    assert(result._1 == 0)
    assert(result._2 == 1)

    globalIndex = 3
    result = randomSearch.decode(globalIndex)
    assert(result._1 == 1)
    assert(result._2 == 1)

    globalIndex = 7
    result = randomSearch.decode(globalIndex)
    assert(result._1 == 2)
    assert(result._2 == 1)

    globalIndex = 8
    result = randomSearch.decode(globalIndex)
    assert(result._1 == 2)
    assert(result._2 == 2)
  }
}
