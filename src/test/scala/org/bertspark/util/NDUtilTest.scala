package org.bertspark.util

import ai.djl.engine.Engine
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.ndarray.{NDArray, NDList, NDManager}
import org.bertspark.util.plot.{MLinePlot, MPlotConfiguration}
import org.bertspark.util.NDUtil.{display, str}
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random



private[util] final class NDUtilTest extends AnyFlatSpec {

  it should "Succeed unbatchify NDArray" in {


  }

  it should "Succeed implementing ND array vector type conversion" in {
    val ndManager = NDManager.newBaseManager()
    var input = Array[Int](3, 5, 6)

    import org.bertspark.util.NDUtil.IntNDArray._
    val ndArray: NDArray = fromVec(ndManager, input)
    input = null
    val vec: Array[Int] = toVec(ndArray)
    println(s"NDArray vec conversion\n${vec.mkString(" ")}")
    assert(vec(1) == vec(1))
    ndManager.close()
  }


  it should "Succeed implementing ND array matrix type conversion" in {
    val ndManager = NDManager.newBaseManager()
    val input = Array[Array[Int]](
      Array[Int](3, 5, 6),
      Array[Int](1, -3, 8)
    )
    import org.bertspark.util.NDUtil.IntNDArray._
    val ndArray: NDArray = fromMatrix(ndManager, input)
    val matrix: Array[Array[Int]] = toMatrix(ndArray)
    println(s"NDArray matrix conversion\n${matrix.map(_.mkString(" ")).mkString("\n")}")
    assert(matrix(1)(2) == input(1)(2))
    ndManager.close()
  }


  it should "Succeed implementing ND array matrix Float conversion" in {
    val ndManager = NDManager.newBaseManager()
    val input = Array[Array[Float]](
      Array[Float](3.0F, 5.0F, 6.0F),
      Array[Float](1.0F, -3.0F, 8.0F)
    )
    import org.bertspark.util.NDUtil.FloatNDArray._
    val ndArray: NDArray = fromMatrix(ndManager, input)
    val matrix: Array[Array[Float]] = toMatrix(ndArray)
    println(s"NDArray matrix conversion\n${matrix.map(_.mkString(" ")).mkString("\n")}")
    assert(matrix(1)(2) == input(1)(2))
    ndManager.close()
  }

  it should "Succeed implementing ND list type conversion - vectors " in {
    val ndManager = NDManager.newBaseManager()
    val input = Array[Array[Int]](
      Array[Int](3, 5, 6),
      Array[Int](1, -3, 8)
    )

    import org.bertspark.util.NDUtil.IntNDList._
    val ndList: NDList = fromVec(ndManager, input)
    val values = toVec(ndList)
    println(s"NDList vector conversion\n${values.map(_.mkString(" ")).mkString("\n")}")
    assert(values(1)(2) == input(1)(2))
    ndManager.close()
  }


  it should "Succeed implementing ND list type int conversion - matrices" in {
    val ndManager = NDManager.newBaseManager()
    val input1 = Array[Array[Int]](
      Array[Int](3, 5, 6),
      Array[Int](1, -3, 8)
    )
    val input2 = input1.map(ar => ar.take(2).map(_ * 2))

    import org.bertspark.util.NDUtil.IntNDList._
    val ndList: NDList = fromMatrix(ndManager, Array[Array[Array[Int]]](input2, input2))
    val matrix: Array[Array[Array[Int]]] = toMatrix(ndList)
    println(s"NList matrices conversion:\n${matrix.map(_.map(_.mkString(" ")).mkString("\n")).mkString("\n\n")}")
    ndManager.close()
  }


  it should "Succeed implementing ND list type float conversion - matrices" in {
    val ndManager = NDManager.newBaseManager()
    val input1 = Array[Array[Float]](
      Array[Float](3.0F, 5.0F, 6.0F),
      Array[Float](1.0F, -3.0F, 8.0F)
    )
    val input2 = input1.map(_.map(_ * 2.0F))

    import org.bertspark.util.NDUtil.FloatNDList._
    val ndList: NDList = fromMatrix(ndManager, Array[Array[Array[Float]]](input2, input2))
    val matrix: Array[Array[Array[Float]]] = toMatrix(ndList)
    println(s"NList matrices conversion:\n${matrix.map(_.map(_.mkString(" ")).mkString("\n")).mkString("\n\n")}")
    ndManager.close()
  }


  ignore should "Succeed evaluating memory leaks" in {
    def compute(ndManager: NDManager): Float = {
      val subNdManager = ndManager.newSubManager()

      val x = Array.fill(50000)(Random.nextFloat())
      val ndArray = subNdManager.create(x)
      ndArray.attach(subNdManager)
      val ndMean = ndArray.mean()
      ndMean.attach(subNdManager)
      val mean = ndMean.toFloatArray.head
      
      subNdManager.close()
      Thread.sleep(20L)
      mean
    }
    val ndManager = NDManager.newBaseManager()
    (0 until 10000).foreach(_ => compute(ndManager))

    ndManager.close()
  }

  ignore should "Compute similarities for a vector" in {
    val ndManager = NDManager.newBaseManager()
    val values1 = Array[Float](0.4F, -0.5F, 1.0F, 0.6F, 0.8F, 0.6F)
    val values2 = Array[Float](0.39F, -0.6F, 0.0F, 0.6F, 0.81F, 0.66F)
    val values3 = values1.map(- _)
    val ndValues1 = ndManager.create(values1)
    val ndValues2 = ndManager.create(values2)
    val ndValues3 = ndManager.create(values3)

    val cosine1 = NDUtil.computeSimilarity(ndValues1, ndValues1, "cosine", values1.size)
    val euclidean1 = NDUtil.computeSimilarity(ndValues1, ndValues1, "euclidean", values1.size)
    val jaccard1 = NDUtil.computeSimilarity(ndValues1, ndValues1, "jaccard", values1.size)

    val cosine2 = NDUtil.computeSimilarity(ndValues1, ndValues2, "cosine", values1.size)
    val euclidean2 = NDUtil.computeSimilarity(ndValues1, ndValues2, "euclidean", values1.size)
    val jaccard2 = NDUtil.computeSimilarity(ndValues1, ndValues2, "jaccard", values1.size)

    val cosine3 = NDUtil.computeSimilarity(ndValues1, ndValues3, "cosine", values1.size)
    val euclidean3 = NDUtil.computeSimilarity(ndValues1, ndValues3, "euclidean",values1.size)
    val jaccard3 = NDUtil.computeSimilarity(ndValues1, ndValues3, "jaccard", values1.size)

    val str = s"""Identical vector: $cosine1, $euclidean1, $jaccard1
                 |Opposite vector:  $cosine3, $euclidean3, $jaccard3
                 |Partial           $cosine2, $euclidean2, $jaccard2""".stripMargin

    println(str)
    ndManager.close()
  }

  ignore should "Succeed displaying tensors and matrices" in {
    val ndManager = NDManager.newBaseManager()
    def createWordEmbedding(bias: Float): Array[Array[Float]] = {
      val embedding1 = Array.fill(8)(1.0F+bias)
      val embedding2 = Array.fill(8)(1.7F+bias)
      val embedding3 = Array.fill(8)(0.3F+bias)
      Array[Array[Float]](
        embedding1, embedding2, embedding3
      )
    }
    val embeddings1 = createWordEmbedding(0.6F)
    val embeddings2 = createWordEmbedding(0.1F)

    val ndEmbeddings1: NDArray = ndManager.create(embeddings1)
    println(s"Matrix:\n${display(ndEmbeddings1)}")

    val ndEmbeddings2 = ndManager.create(embeddings2)
    val expanded1 = ndEmbeddings1.expandDims(0)
    val expanded2 = ndEmbeddings2.expandDims(0)
    val results: NDArray = NDUtil.concat(Array[NDArray](expanded1, expanded2))
    println(s"\n3D tensor:\n${display(results)}")
    ndManager.close()
  }



  it should "Succeed stack batchying" in {
    val ndManager = NDManager.newBaseManager()
    val inputs = Array[NDList](
      new NDList(ndManager.create(Array[Float](0.4F, 0.8F, 0.1F))),
      new NDList(ndManager.create(Array[Float](0.5F, 1.0F, 0.0F)))
    )
    val ndOutput: NDList = NDUtil.batchify(inputs)
    val shapes: Array[Long] = ndOutput.get(0).getShape.getShape
    println(s"Stack batchy ${shapes.mkString(" ")}")
    ndManager.close
  }

  it should "Succeed concatenate NDLists axis 0" in {
    val ndManager = NDManager.newBaseManager()
    val inputs = Array[NDList](
      new NDList(ndManager.create(Array[Float](0.4F, 0.8F, 0.1F))),
      new NDList(ndManager.create(Array[Float](0.5F, 1.0F, 0.0F)))
    )
    val ndOutput = NDUtil.concat(inputs)
    val shapes: Array[Long] = ndOutput.get(0).getShape.getShape
    println(s"Shapes concatenate axis 0 ${shapes.mkString(" ")}")
    ndManager.close
  }


  ignore should "Succeed computing gradients" in {
    val ndManager = NDManager.newBaseManager

    val gc = Engine.getInstance.newGradientCollector
    val x = ndManager.arange(0.1F, 20.0F, 0.1F)
    x.setRequiresGradient(true)
    val y = x.sin.mul(x).mul(2)
    gc.backward(y)
    val grad = x.getGradient.toFloatArray.size
    val plotConfig = MPlotConfiguration("Gradient", "x-label", "y-label", "params")
    val linePlot = new MLinePlot(plotConfig)
    val yInput = ("y=x.sin(x)", y.toFloatArray)
    val gInput = ("dy/dx", x.getGradient.toFloatArray)

    linePlot(x.toFloatArray, yInput, gInput)

    ndManager.close
  }

  ignore should "Succeed manipulating shapes" in {

    val ndManager = NDManager.newBaseManager

    val x = ndManager.arange(0F, 24F, 1.0F, DataType.FLOAT32)
    println(x.getShape)
    println(str(x))
    val y = x.reshape(2, 12)
    println(y.getShape)
    println(str(y))
    val z = x.reshape(4, 6)
    println(z.getShape)
    val t = x.reshape(4, -1)
    println(t.getShape)
    val u = x.reshape(-1, 3)
    println(u.getShape)
    val v = x.reshape(3, 4, 2)
    println(str(v))
    ndManager.close
  }

  ignore should "Succeed generating random distribution" in {
    val ndManager = NDManager.newBaseManager

    val values = ndManager.randomNormal(0F, 10F, new Shape(4, 2), DataType.FLOAT32)
    println(str(values))

    val values2 = ndManager.randomUniform(0F, 10F, new Shape(4, 2), DataType.FLOAT32)
    println(str(values2))
  }

  ignore should "Succeed perform operation on NDArray" in {
    val ndManager = NDManager.newBaseManager

    val x = ndManager.create(Array.tabulate(6)(_.toFloat))
    val y = ndManager.create(Array.tabulate(6)(n => n.toFloat+8.0F))
    x.addi(y)
    println(str(x))
    ndManager.close
  }


  ignore should "Succeed perform in-place operation on NDArray" in {
    val ndManager = NDManager.newBaseManager

    val x = ndManager.create(Array.tabulate(6)(_.toFloat))
    val y = ndManager.create(Array.tabulate(6)(n => n.toFloat+8.0F))
    val z = x.add(y)
    val z0: Float = z.getFloat(0)
    assert(z0 == x.getFloat(0) + y.getFloat(0), "Failed execute in place summation")
    println(str(z))
    ndManager.close
  }

  ignore should "Succeed perform indexing on NDArray" in {
    val ndManager = NDManager.newBaseManager

    val x = ndManager.create(Array.tabulate(24)(_.toFloat))
    val y = x.reshape(6, 4)
    val z = y.get("0:2")
    println(str(y))
    println(str(z))
    val t = y.get("3:4")
    println(str(t))
    ndManager.close
  }

  ignore should "Succeed evaluating boolean logic for NDArray" in {
    val ndManager = NDManager.newBaseManager

    val x = ndManager.create(Array.tabulate(6)(_.toFloat))
    val y = ndManager.create(Array.tabulate(6)(_.toFloat))
    val z = x.eq(y)
    z.toBooleanArray.foreach(item => assert(item == true))
    ndManager.close
  }
}