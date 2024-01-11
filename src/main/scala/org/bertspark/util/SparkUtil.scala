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
package org.bertspark.util

import org.apache.hadoop.fs._
import org.apache.spark.sql._
import org.apache.spark.rdd.RDD
import org.bertspark.util.io.S3Util
import org.slf4j.{Logger, LoggerFactory}
import scala.reflect.ClassTag

/**
  * Singleton for Wrapping of methods to manipulate HDFS files
  * {{{
  *   Test existing of HDFS entry
  *   Delete HDFS entry
  *   Compress then save HDFS content
  * }}}
  * @author Patrick Nicolas
  * @version 0.1
  */
private[bertspark] final object SparkUtil {
	final val log: Logger = LoggerFactory.getLogger("SparkUtil")


	/**
	  * Test if this hdfs file exists
	  * @param hdfsFileName Name of the hdfs file
	  * @param sparkSession implicit reference to the current SparkSession
	  * @return true if the file exists, false otherwise
	  */
	final def exists(hdfsFileName: String)(implicit sparkSession: SparkSession): Boolean = {
		val hdfsConfig = sparkSession.sparkContext.hadoopConfiguration
		val fs = FileSystem.get(hdfsConfig)
		val hdfsPath = new Path(hdfsFileName)

		fs.exists(hdfsPath)
	}

	/**
	  * Delete an existing HDFS file
	  * @param hdfsFileName Name of the hdfs file
	  * @param sparkSession implicit reference to the current SparkSession
	  * @return true if the file exists, false otherwise
	  */
	final def deleteHdfsFile(hdfsFileName: String)(implicit sparkSession: SparkSession): Boolean = {
		val hdfsConfig = sparkSession.sparkContext.hadoopConfiguration
		val fs = FileSystem.get(hdfsConfig)
		val hdfsPath = new Path(hdfsFileName)

		val isFileExists = fs.exists(hdfsPath)
		if (isFileExists)
			fs.delete(hdfsPath, true)
		isFileExists
	}

	/**
	  * Save and compress a given data set into HDFS
	  * @param dataset Input data set
	  * @param hdfsOutput Name of HDFS file
	  * @param sparkSession Implicit reference to the current Spark context
	  * @tparam T Type of elements of the data set
	  */
	final def saveAndCompress[T](
		dataset: Dataset[T],
		hdfsOutput: String
	)(implicit sparkSession: SparkSession): Unit =
		dataset.write
			.mode(SaveMode.Append)
			.option("codec", "org.apache.hadoop.io.compress.GzipCodec")
			.json(hdfsOutput)

	/**
	  * Save and compress a given data set extracted from an HDFS file
	  * @param hdfsInput Input HDFS file
	  * @param hdfsOutput Name of HDFS file
	  * @param sparkSession Implicit reference to the current Spark context
	  * @param encoder Encoder for type of elements contained in the input HDFS file
	  * @tparam T Type of elements of the data set
	  */
	final def saveAndCompress[T](
		hdfsInput: String,
		hdfsOutput: String
	)(implicit sparkSession: SparkSession, encoder: Encoder[T]): Unit = {

		val dataset: Dataset[T] = sparkSession.read.json(hdfsInput).as[T]
		saveAndCompress(dataset, hdfsOutput)
	}

	/**
	  * Optimized join of two data sets
	  * @param tDS First input data set
	  * @param tDSKey key for first data set
	  * @param uDS Second input data set
	  * @param uDSKey key for Second data set
	  * @tparam T Type of elements of the first data set
	  * @tparam U Type of elements of the second data set
	  * @return Data set of pair (T, U)
	  */
	final def sortingJoin[T, U](
		tDS: Dataset[T],
		tDSKey: String,
		uDS: Dataset[U],
		uDSKey: String
	)(implicit sparkSession: SparkSession): Dataset[(T, U)] = {
		val sortedTDS = tDS.repartition(tDS(tDSKey)).sortWithinPartitions(tDS(tDSKey)).cache()
		val sortedUDS = uDS.repartition(uDS(uDSKey)).sortWithinPartitions(uDS(uDSKey)).cache()
		sortedTDS.joinWith(sortedUDS, sortedTDS(tDSKey) === sortedUDS(uDSKey), joinType = "inner")
	}

	/**
	  * Optimized join of two data sets followed by aggregatioin
	  * @param tDS First input data set
	  * @param tDSKey key for first data set
	  * @param uDS Second input data set
	  * @param uDSKey key for Second data set
	  * @param aggr Aggregation operation
	  * @tparam T Type of elements of the first data set
	  * @tparam U Type of elements of the second data set
	  * @tparam V Type of elements of the aggregated data set
	  * @return Data set of pair (T, U)
	  */
	final def sortingJoin[T, U, V](
		tDS: Dataset[T],
		tDSKey: String,
		uDS: Dataset[U],
		uDSKey: String,
		aggr: (T, U) => V
	)(implicit sparkSession: SparkSession, encoder: Encoder[V]): Dataset[V] = {
		val sortedTDS = tDS.repartition(tDS(tDSKey)).sortWithinPartitions(tDS(tDSKey)).cache()
		val sortedUDS = uDS.repartition(uDS(uDSKey)).sortWithinPartitions(uDS(uDSKey)).cache()
		sortedTDS.joinWith(sortedUDS, sortedTDS(tDSKey) === sortedUDS(uDSKey), joinType = "inner")
			.map { case (u, v) => aggr(u, v) }
	}

	/**
	  * Efficient grouping of data using reduce by key
	  * @param key Extract the key from data type T
	  * @param reducer Reducer function within a partition (T, T) => T
	  * @param ds Input data set
	  * @param sparkSession Implicit reference to the current Spark context
	  * @param encoder Implicit encoder for the type T
	  * @tparam T Type of the elements of data set
	  * @tparam K Type of key
	  * @return Grouped data set as RDD[T]
	  */
	def groupBy[T: ClassTag, K: ClassTag](
		key: T => K,
		reducer: (T, T) => T,
		ds: Dataset[T]
	)(implicit sparkSession: SparkSession, encoder: Encoder[T], encoder2: Encoder[(K, T)]): RDD[T] =
		groupByKey[T, K](key, reducer, ds).map(_._2)



	/**
	* Efficient grouping of data using reduce by key
	* @param key Extract the key from data type T
	* @param reducer Reducer function within a partition (T, T) => T
	* @param ds Input data set
	* @param sparkSession Implicit reference to the current Spark context
	* @param encoder Implicit encoder for the type T
	* @tparam T Type of the elements of data set
	* @tparam K Type of key
	* @return Grouped data set as RDD[(K, T])
	*/
	def groupByKey[T: ClassTag, K: ClassTag](
		key: T => K,
		reducer: (T, T) => T,
		ds: Dataset[T]
	)(implicit sparkSession: SparkSession, encoder: Encoder[T], encoder2: Encoder[(K, T)]): RDD[(K, T)] = {
		val keyedDS = ds.map(t => (key(t), t))
		keyedDS.rdd.reduceByKey(reducer(_,_))
	}

	/**
	  * Singleton that wraps the conversion between S3 JSON file and Parquet format
	  * @note Design pattern Context bound
	  */
	final object JsonToParquet {
		/**
		  * Load a data set stored in JSON format then stored into Parquet format
		  * @param s3InputJson Name of folder in default bucket for JSON file
		  * @param s3OutputParquet Name of output folder in default bucket for Parquet file
		  * @tparam T Type of elements in the data set
		  * @return true if conversion succeed, false, otherwise
		  */
		final def apply[T](
			s3InputJson: String,
			s3OutputParquet: String
		)(implicit sparkSession: SparkSession, encoder: Encoder[T]): Boolean = try {
			val header = false
			val ds = S3Util.s3ToDataset[T](s3InputJson, header, fileFormat = "json")
			S3Util.datasetToS3[T](
				ds,
				s3OutputParquet,
				header,
				fileFormat = "parquet",
				toAppend = false,
				numPartitions = -1
			)
			true
		}
		catch {
			case e: IllegalStateException =>
				log.error(e.getMessage)
				false
		}

		/**
		  * Load a data set stored in Parquet format then stored into JSON format
		  * @param s3InputParquet Name of folder in default bucket for parquet file
		  * @param s3OutputJson Name of output folder in default bucket for JSON file
		  * @tparam T Type of elements in the data set
		  * @return true if conversion succeed, false, otherwise
		  */
		final def unapply[T](
			s3InputParquet: String,
			s3OutputJson: String
		)(implicit sparkSession: SparkSession, encoder: Encoder[T]): Boolean = try {
			val ds = S3Util.s3ToDataset[T](s3InputParquet, header = false, fileFormat = "parquet")
			S3Util.datasetToS3[T](ds, s3OutputJson, header = false, fileFormat = "json", toAppend = false, numPartitions = -1)
			true
		}
		catch {
			case e: IllegalStateException =>
				log.error(e.getMessage)
				false
		}
	}


	/**
	 * Splitting algorithm that use sequence of notes loaded from S3 prediction requests
	 * @param inputDS  Input data set to be split
	 * @param batchSize Size of batch to be processed
	 * @tparam T Type of elements of the data set
	 * @return  Sequence of chunk (split) data sets
	 */
	final def sequentialSplit[T](inputDS: Dataset[T], batchSize: Int): Seq[Dataset[T]] = {
		import scala.collection.mutable.ListBuffer

		var cursorDS = inputDS
		var cursor = inputDS.count()
		val buf = new ListBuffer[Dataset[T]]()

		while(cursor > 0) {
			val newDS = cursorDS.limit(batchSize).persist().cache()
			buf += newDS

			val remainingDS = cursorDS.except(newDS)
			cursorDS = remainingDS
			cursor -= batchSize
		}
		buf.toSeq
	}


	/**
	 *  Split an input data set into small data set of size 'batchSize' selected randoml
	 * @param inputDS  Input data set to be split
	 * @param batchSize Size of batch to be processed
	 * @tparam T Type of elements of the data set
	 * @return  Sequence of chunk (split) data sets
	 */
	final def randomSplit[T](inputDS: Dataset[T], batchSize: Int): Array[Dataset[T]] = {
		val numRecords = inputDS.count()
		val numSplits = (numRecords.toDouble / batchSize).ceil.toInt

		if ((numRecords < 0 || numRecords > 3) && numSplits >= 3) {
			val splitter = Array.fill(numSplits)(1.0 / numSplits)
			inputDS.randomSplit(splitter, 42L)
		}
		else {
			log.error(s"Number of records $numRecords or number of splits: $numSplits is out of range")
			Array.empty[Dataset[T]]
		}
	}

}
