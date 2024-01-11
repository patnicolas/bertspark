package org.bertspark.classifier.training

import org.scalatest.flatspec.AnyFlatSpec


private[training] final class ClassifierTrainingSetFilterTest extends AnyFlatSpec {

  ignore should "Succeed filtering classifier training set by num records" in   {
    import org.bertspark.implicits._

    val s3FeedbackFolder = "feedbacksProd/SMALL"
    val numRecords = 64
    val classifierTrainingSetFilter = ClassifierTrainingSetFilter(
      Set.empty[String],
      Set.empty[String],
      numRecords,
      -1,
      -1)
    val filteredByNumRecordDS = ClassifierTrainingSetFilter.filter(s3FeedbackFolder, classifierTrainingSetFilter)
    println(s"Filtering by num records ${filteredByNumRecordDS.count()} completed with\n${filteredByNumRecordDS.take(2).mkString("\n")}")
  }

  ignore should "Succeed filtering classifier training set by customers" in   {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val customers = Set[String]("AMB", "CMBS", "Cornerstone")
    val s3FeedbackFolder = "feedbacksProd/SMALL"
    val classifierTrainingSetFilter = ClassifierTrainingSetFilter(
      customers,
      Set.empty[String],
      32,
      -1,
      -1)
    val filteredByCustomerDS = ClassifierTrainingSetFilter.filter(s3FeedbackFolder, classifierTrainingSetFilter)
    assert(filteredByCustomerDS.count() <= 32)
    val filteredByCustomers = filteredByCustomerDS.map(_.context.customer).collect().mkString("\n")
    println(s"Filtering by customers ${filteredByCustomerDS.count()} completed with\n$filteredByCustomers")

  }

  ignore should "Succeed filtering classifier training set by sub models" in   {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val subModels = Set[String]("7025F-77067 26-77063 26", "7025F-77063 26-77067 26", "32557")
    val s3FeedbackFolder = "feedbacksProd/SMALL"
    val classifierTrainingSetFilter = ClassifierTrainingSetFilter(
      Set.empty[String],
      subModels,
      32,
      -1,
      -1)
    val filteredBySubModelDS = ClassifierTrainingSetFilter.filter(s3FeedbackFolder, classifierTrainingSetFilter)
    assert(filteredBySubModelDS.count() <= 32)
    val filteredBySubModels = filteredBySubModelDS.map(rec => s"${rec.context.customer}-${rec.id}").collect().mkString("\n")
    println(s"Filtering by sub models ${filteredBySubModelDS.count()} completed with\n$filteredBySubModels")

  }

  it should "Succeed filtering classifier training set by number of records per labels" in   {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3FeedbackFolder = "feedbacksProd/SMALL"
    val classifierTrainingSetFilter = ClassifierTrainingSetFilter(
      Set.empty[String],
      Set.empty[String],
      512,
      2,
      8)
    val filteredByFrequenciesDS = ClassifierTrainingSetFilter.filter(s3FeedbackFolder, classifierTrainingSetFilter)
    val filteredByFrequencies = filteredByFrequenciesDS.map(rec => s"${rec.context.customer}-${rec.id}").collect().mkString("\n")
    println(s"Filtering by frequencies ${filteredByFrequenciesDS.count()} completed with\n$filteredByFrequencies")
  }
}
