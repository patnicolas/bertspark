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
package org.bertspark.util.io

import ai.djl.aws.s3.S3RepositoryFactory
import ai.djl.repository.Repository
import com.amazonaws._
import com.amazonaws.auth._
import com.amazonaws.services.s3._
import com.amazonaws.services.s3.model._
import java.io._
import org.apache.spark.SparkException
import org.apache.spark.sql._
import org.bertspark.config.MlopsConfiguration.mlopsConfiguration
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.delay
import org.bertspark.util.EncryptionUtil
import org.bertspark.util.io.LocalFileUtil.CSV_SEPARATOR
import org.slf4j._
import scala.collection.mutable.{HashMap, ListBuffer}
import scala.reflect.ClassTag
import software.amazon.awssdk.auth.credentials.{AwsBasicCredentials, StaticCredentialsProvider}
import software.amazon.awssdk.services.s3.S3Client
import sun.java2d.marlin.MarlinUtils.logInfo

/**
 * Singleton generic utility for Logging
 * {{{
 *   The credentials (Access and secret keys are defined in the configuration file as
 *   encrypted using AES algorithm
 * }}}
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] object S3Util {
  final private val logger: Logger = LoggerFactory.getLogger("S3Util")
  final val defaultNumPartitions = 4

  final private def defaultMaxKeys = 5000

  val accessKey = EncryptionUtil.unapply(mlopsConfiguration.storageConfig.encryptedS3AccessKey).getOrElse(
		throw new IllegalStateException(
			s"Failed to decrypt access key for S3 ${mlopsConfiguration.storageConfig.encryptedS3AccessKey}"
		)
	)
  val secretKey = EncryptionUtil.unapply(mlopsConfiguration.storageConfig.encryptedS3SecretKey).getOrElse(
		throw new IllegalStateException(
			s"Failed to decrypt secret key for S3 ${mlopsConfiguration.storageConfig.encryptedS3SecretKey}"
		)
	)
  val currentRegion = "us-east-2"

  private lazy val credentials = new BasicAWSCredentials(accessKey, secretKey)
  private lazy val s3Client = AmazonS3ClientBuilder
      .standard()
      .withCredentials(new AWSStaticCredentialsProvider(credentials))
      .withRegion(currentRegion)
      .build

  final def getS3Client: AmazonS3 = s3Client


	lazy val s3DjlClient: S3Client = {
		import software.amazon.awssdk.auth.credentials.AwsBasicCredentials
		import software.amazon.awssdk.regions.Region
		val awsBasicCredentials = AwsBasicCredentials.create(accessKey, secretKey)

		// AwsCredentialsProvider
		val credProvider: StaticCredentialsProvider = StaticCredentialsProvider.create(awsBasicCredentials)
		val client = S3Client.builder
				.credentialsProvider(credProvider)
				.region(Region.of(currentRegion))
				.build()
		Repository.registerRepositoryFactory(new S3RepositoryFactory(client))
		client
	}




	/**
   * Store the content of a data set into S3 file for a given S3bucket
   * {{{
   *    The output dataset is repartitioned only if the argument numPartitions is > 0, before be saved into S3
   *    The original partition is preserved if the argument numPartitions <= 0
   *   }}}
	 *
	  * @param s3Bucket S3Bucket used for output file
	  * @param inputDataset Data set with parameterized type
	  * @param s3OutputPath S3 absolute path
	  * @param header Flag to specify if a header is to be added
	  * @param fileFormat "json", "csv" , "text"
	  * @param toAppend Content will be appended if true, overwritten otherwise
	  * @param numPartitions Number of partitions to store data in S3
	  * @param sparkSession Implicit reference to the current Spark context
	  * @param encoder      Encoder for the Parameterized type T
	  * @tparam T T Type of element of the dataset
	  */
	@throws(clazz = classOf[IllegalStateException])
	def datasetToS3[T](
		s3Bucket: String,
		inputDataset: Dataset[T],
		s3OutputPath: String,
		header: Boolean,
		fileFormat: String,
		toAppend: Boolean,
		numPartitions: Int
	)(implicit sparkSession: SparkSession, encoder: Encoder[T]): Unit = {
		import sparkSession.implicits._

		val accessConfig = inputDataset.sparkSession.sparkContext.hadoopConfiguration

		accessConfig.set("fs.s3a.access.key", accessKey)
		accessConfig.set("fs.s3a.secret.key", secretKey)
		val headerStr = if (header) "true" else "false"

		try {
			// We repartition only if is is > 0
			val partitionedDS = if (numPartitions > 0) inputDataset.repartition(numPartitions) else inputDataset

			fileFormat match {
				case "text" => partitionedDS
						.map(_.toString)
						.write
						.format("json")
						.option("header", headerStr)
						.mode(if (toAppend) SaveMode.Append else SaveMode.Overwrite)
						.save(path = s"s3a://$s3Bucket/$s3OutputPath")
				case "csv" => partitionedDS
						.map(_.toString)
						.write
						.format("csv")
						.mode(if (toAppend) SaveMode.Append else SaveMode.Overwrite)
						.save(path = s"s3a://$s3Bucket/$s3OutputPath")
				case "json" => partitionedDS
						.write
						.format(fileFormat)
						.option("header", header)
						.mode(if (toAppend) SaveMode.Append else SaveMode.Overwrite)
						.save(path = s"s3a://$s3Bucket/$s3OutputPath")
				case _ => throw new IllegalStateException(s"Incorrect file format $fileFormat")
			}
		}
		catch {
			case e: AmazonS3Exception => throw new IllegalStateException(e.getMessage)
			case e: FileNotFoundException => new IllegalStateException(e.getMessage)
			case e: SparkException => new IllegalStateException(e.getMessage)
			case e: Exception => new IllegalStateException(e.getMessage)
		}
	}


	/**
	 * Store the content of a data set into S3 file for a the default S3 folder specified in the configuration target
	 * {{{
	 *    The output dataset is repartitioned only if the argument numPartitions is > 0, before be saved into S3
	 *    The original partition is preserved if the argument numPartitions <= 0
	 * }}}
	 *
	 * @param inputDataset Data set with parameterized type
	 * @param s3OutputPath S3 absolute path
	 * @param header Flag to specify if a header is to be added
	 * @param fileFormat "json", "csv" , "text"
	 * @param toAppend Content will be appended if true, overwritten otherwise
	 * @param numPartitions Number of partitions to store data in S3
	 * @param sparkSession Implicit reference to the current Spark context
	 * @param encoder Encoder for the Parameterized type T
	 * @tparam T T Type of element of the dataset
	 */
	@throws(clazz = classOf[IllegalStateException])
	def datasetToS3[T](
		inputDataset: Dataset[T],
		s3OutputPath: String,
		header: Boolean,
		fileFormat: String,
		toAppend: Boolean,
		numPartitions: Int
	)(implicit sparkSession: SparkSession, encoder: Encoder[T]): Unit =
		datasetToS3[T](
			mlopsConfiguration.storageConfig.s3Bucket,
			inputDataset,
			s3OutputPath,
			header,
			fileFormat,
			toAppend,
			numPartitions
		)


	/**
	  * Load content of a S3 folder into a dataset for further processing on Spark cluster
 *
	  * @param s3BucketName Name of the bucket
	  * @param s3InputFile  absolute name of S3 file
	  * @param sparkSession Implicit reference to the current Spark context
	  * @param encoder      Encoder for the Parameterized type T
	  * @tparam T Type of element of the dataset
	  * @return Data set
	  */
	@throws(clazz = classOf[IllegalStateException])
	def s3ToDataset[T](
		s3BucketName: String,
		s3InputFile: String,
		header: Boolean,
		fileFormat: String
	)(implicit sparkSession: SparkSession, encoder: Encoder[T]): Dataset[T] = {
		import sparkSession.implicits._

		val loadDS = Seq[T]().toDS()
		val accessConfig = loadDS.sparkSession.sparkContext.hadoopConfiguration
		accessConfig.set("fs.s3a.access.key", accessKey)
		accessConfig.set("fs.s3a.secret.key", secretKey)
		val headerStr = if (header) "true" else "false"

		try {
			if (fileFormat == "json") {
				val inputSchema = loadDS.schema
				sparkSession.read
					.format(fileFormat)
					.option("header", headerStr)
					.schema(inputSchema)
					.load(path = s"s3a://$s3BucketName/$s3InputFile")
					.as[T]
			}
			else
				sparkSession.read
					.format(fileFormat)
					.option("header", headerStr)
					.load(path = s"s3a://$s3BucketName/$s3InputFile}")
					.as[T]
		}
		catch {
			case e: FileNotFoundException =>
				throw new IllegalStateException(s"S3 file not found exception: ${e.getMessage}")
			case e: AmazonS3Exception => throw new IllegalStateException(s"S3 Amazon exception: ${e.getMessage}")
			case e: SparkException => throw new IllegalStateException(s"S3 Spark exception: ${e.getMessage}")
			case e: Exception => throw new IllegalStateException(s"S3 Undefined exception: ${e.getMessage}")
		}
	}



	/**
		* Load content of a S3 folder into a dataset for further processing on Spark cluster,
		* using the S3 bucket defined by the target configuration parameter
		*
		* @param s3InputFile  absolute name of S3 file
		* @param sparkSession Implicit reference to the current Spark context
		* @param encoder      Encoder for the Parameterized type T
		* @tparam T Type of element of the dataset
		* @return Data set
		*/
	def s3ToDataset[T](s3InputFile: String)(implicit sparkSession: SparkSession, encoder: Encoder[T]): Dataset[T] =
		s3ToDataset[T](mlopsConfiguration.storageConfig.s3Bucket, s3InputFile, header = false, fileFormat = "json")


		/**
	 	* Load content of a S3 folder into a dataset for further processing on Spark cluster,
	 	* using the S3 bucket defined by the target configuration parameter
	 	*
	 	* @param s3InputFile  absolute name of S3 file
		* @param header Flag to specify if a header is required
		* @param fileFormat File format "json"
	 	* @param sparkSession Implicit reference to the current Spark context
	 	* @param encoder      Encoder for the Parameterized type T
	 	* @tparam T Type of element of the dataset
	 	* @return Data set
	 	*/
	def s3ToDataset[T](
		s3InputFile: String,
		header: Boolean,
		fileFormat: String
	)(implicit sparkSession: SparkSession, encoder: Encoder[T]): Dataset[T] =
		s3ToDataset[T](mlopsConfiguration.storageConfig.s3Bucket, s3InputFile, header, fileFormat)


	/**
	 * Extract a data frame from a CSV file in S3 with predefined S3 bucket list
 *
	 * @param s3InputFile Input CSV file
	 * @param header Flag to specify if the file has header
	 * @param sparkSession Implicit reference to the current spark context
	 * @throws IllegalStateException If loading the CSV file fails
	 * @return Data frame
	 */
	@throws(clazz = classOf[IllegalStateException])
	def s3CSVToDataFrame(
		s3InputFile: String,
		header: Boolean
	)(implicit sparkSession: SparkSession): DataFrame = {
		s3CSVToDataFrame(s3InputFile, header, mlopsConfiguration.storageConfig.s3Bucket, true)
	}

	/**
	  * Extract a data frame from a CSV file in S3
 *
	  * @param s3InputFile CSV input file
	  * @param header Flag to specify if the file has header
	  * @param s3Bucket Name of the S3 bucket
	  * @param isMultiLine Specify if some of the fields are multi-line
	  * @param sparkSession Implicit reference to the current spark context
	  * @throws IllegalStateException If loading the CSV file fails
	  * @return Data frame
	  */
	@throws(clazz = classOf[IllegalStateException])
	def s3CSVToDataFrame(
		s3InputFile: String,
		header: Boolean,
		s3Bucket: String,
		isMultiLine: Boolean
	)(implicit sparkSession: SparkSession): DataFrame = {
		import sparkSession.implicits._

		val loadDS = Seq[String]().toDS()
		val accessConfig = loadDS.sparkSession.sparkContext.hadoopConfiguration

		try {
			accessConfig.set("fs.s3a.access.key", accessKey)
			accessConfig.set("fs.s3a.secret.key", secretKey)
			val headerStr = if (header) "true" else "false"

			sparkSession
				.read
				.format("csv")
				.option("header", headerStr)
				.option("delimiter", CSV_SEPARATOR)
				.option("multiLine", isMultiLine)
				.load(path = s"s3a://$s3Bucket/$s3InputFile")
		}
		catch {
			case e: FileNotFoundException => throw new IllegalStateException(e.getMessage)
			case e: SparkException => throw new IllegalStateException(e.getMessage)
			case e: Exception => throw new IllegalStateException(e.getMessage)
		}
	}


	/**
		* Convert data set to local file
		* @param s3SourceFolder Source S3 folder
		* @param fsTargetPath local target files
		* @param sparkSession Implicit reference to the current spark context
		* @param encoder Encoder for the type of data set
		* @tparam T Type of  element of the data set
		* @return Number of records
		*/
	def s3DatasetToFs[T](
		s3SourceFolder: String,
		fsTargetPath: String,
		numRecordsPerFile: Int = 50000)(implicit sparkSession: SparkSession, encoder: Encoder[T]): Unit = try {
		val ds: Dataset[T] = s3ToDataset[T](s3SourceFolder)

		val splits = if(ds.count > 10000L) ds.randomSplit(Array.fill(5)(0.2)) else Array[Dataset[T]](ds)
		splits.foreach(
			splitDS => {
				val iter = splitDS.toLocalIterator()
				val acc = ListBuffer[T]()
				var cnt: Int = 0
				var index = 0

				while(iter.hasNext) {
					val record = iter.next()
					acc.append(record)
					cnt += 1
					if(cnt % 500 == 0)
						logDebug(logger, msg = s"Loaded $cnt records from $s3SourceFolder, to $fsTargetPath")

					if(cnt % numRecordsPerFile == 0) {
						index += 1
						val fsTargetFile = s"$fsTargetPath/part-$index"
						logDebug(logger,msg = s"Save $fsTargetFile into $fsTargetPath")
						LocalFileUtil.Save.local(fsTargetFile, acc.mkString("\n"))
						acc.clear()
						delay(timeInMillis = 1000L)
					}
				}
			}
		)
	}
	catch {
		case e: IllegalArgumentException => logger.error(e.getMessage)
	}






	/**
	  * Load content of a S3 folder into a structured data stream for further processing on Spark cluster
	  * @param s3InputFile  absolute name of S3 file
	  * @param sparkSession Implicit reference to the current Spark context
	  * @param encoder      Encoder for the Parameterized type T
	  * @tparam T Type of element of the dataset
	  * @return Dataset DStream
	  */
	@throws(clazz = classOf[IllegalStateException])
	def s3ToStream[T](
		s3InputFile: String,
		header: Boolean,
		fileFormat: String
	)(implicit sparkSession: SparkSession, encoder: Encoder[T]): Dataset[T] = {
		import sparkSession.implicits._

		val loadDS = Seq[T]().toDS()
		val accessConfig = loadDS.sparkSession.sparkContext.hadoopConfiguration

		accessConfig.set("fs.s3a.access.key", accessKey)
		accessConfig.set("fs.s3a.secret.key", secretKey)
		val headerStr = if (header) "true" else "false"

		try {
			val inputSchema = loadDS.schema
			sparkSession.readStream
				.format(fileFormat)
				.option("header", headerStr)
				.schema(inputSchema)
				.load(path = s"s3a://${mlopsConfiguration.storageConfig.encryptedS3SecretKey}/${s3InputFile}")
				.as[T]
		}
		catch {
			case e: FileNotFoundException => throw new IllegalStateException(e.getMessage)
			case e: SparkException => throw new IllegalStateException(e.getMessage)
			case e: Throwable => throw new IllegalStateException(e.getMessage)
		}
	}

	final def exists(bucketName: String, prefix: String): Boolean = try {
		import org.bertspark.implicits._

		val objects = s3Client.listObjects(new ListObjectsRequest().withBucketName(bucketName).withPrefix(prefix))
		val summaries: java.util.List[S3ObjectSummary] = objects.getObjectSummaries
		val ls: Iterable[S3ObjectSummary] = listOf[S3ObjectSummary](summaries)
		ls.nonEmpty
	}
	catch {
		case e: Exception =>
			logger.error(s"Failed to test existence for $bucketName/$prefix  ${e.getMessage}")
			false
	}

	/**
	  * Delete data contains in a s3://$bucketName/prefix
 *
	  * @param client AWS authenticated client
	  * @param bucketName Name of the bucket
	  * @param prefix Prefix (s3Folder) to be deleted
	  * @return true is successful, false otherwise
	  */
	def delete(client: AmazonS3, bucketName: String, prefix: String): Boolean = try {
		import org.bertspark.implicits._

		val objectList = client.listObjects(bucketName, prefix)
		val objectSummaryList = listOf[S3ObjectSummary](objectList.getObjectSummaries)
		objectSummaryList.foreach(entry => client.deleteObject(bucketName, entry.getKey))
		true
	}
	catch {
		case e: AmazonServiceException =>
			logger.error(s"AWS service failed in deleting $bucketName/$prefix ${e.getMessage}")
			false
		case e: SdkClientException =>
			logger.error(s"SDK client failed in deleting $bucketName/$prefix ${e.getMessage}")
			false
		case e: Exception =>
			logger.error(s"Failed to delete $bucketName/$prefix ${e.getMessage}")
			false
	}

	/**
	  * Delete data contains in a s3://$bucketName/prefix
 *
	  * @param bucketName Name of the bucket
	  * @param prefix Prefix (s3Folder) to be deleted
	  * @return true is successful, false otherwise
	  */
	def delete(bucketName: String, prefix: String): Boolean = delete(s3Client, bucketName, prefix)

	/**
	  * Delete data contains in a s3://$bucketName/prefix for which the bucketName is defined
	  * in configuration file.
	  * @param prefix Prefix (s3Folder) to be deleted
	  * @return true is successful, false otherwise
	  */
	def delete(prefix: String): Boolean = delete(
		s3Client,
		mlopsConfiguration.storageConfig.encryptedS3SecretKey,
		prefix
	)

	/**
	  * Copy content of a folder into another folder within a bucket
 *
	  * @param s3BucketName Name of the bucket
	  * @param fromS3Folder Source folder
	  * @param toS3Fold   r Destination folder
	  * @throws IllegalStateException If operation failed or connectivity is lost
	  */
	@throws(clazz = classOf[IllegalStateException])
	def copy(s3BucketName: String, fromS3Folder: String, toS3Folder: String): Unit = try {
		s3Client.copyObject(s3BucketName, fromS3Folder, s3BucketName, toS3Folder)
	}
	catch {
		case e: AmazonServiceException =>
			throw new IllegalStateException(s"S3, Failed to copy $fromS3Folder in ${s3BucketName} ${e.getMessage}")
		case e: SdkClientException =>
			throw new IllegalStateException(s"S3, broken connection copy $fromS3Folder in${s3BucketName} ${e.getMessage}")
		case e: IOException =>
			throw new IllegalStateException(s"S3, IO failed to copy $fromS3Folder in ${s3BucketName} ${e.getMessage}")
	}


	/**
	 * Move content from one folder to another folder within the same bucket.
	 * {{{
	 *   The steps are
	 *   - Copy the content of fromS3Folder to toS3Folder
	 *   - Delete content of fromS3Folder
	 * }}}
	 *
	 * @param s3BucketName Bucket name
	 * @param fromS3Folder S3 source folder
	 * @param toS3Fold   r S3 destination folder
	 * @return Number of S3 files/records moved
	 */
	def moveFolder(s3BucketName: String, fromS3Folder: String, toS3Folder: String): Int = try {
		val keys = getS3Keys(s3BucketName, fromS3Folder)

		// If we find some keys if(keys.nonEmpty)
		keys.map(
			key =>
				try {
					moveFile(s3BucketName, key, toS3Folder)
					1
				}
				catch {
					case e: IllegalStateException =>
						logger.error(s"Failed to move S3 $key")
						0
				}
		).sum
	}
	catch {
		case e: AmazonServiceException =>
			throw new IllegalStateException(s"S3, Failed to move $fromS3Folder in $s3BucketName ${e.getMessage}")
		case e: SdkClientException =>
			throw new IllegalStateException(s"S3, broken connection moving $fromS3Folder in${s3BucketName} ${e.getMessage}")
		case e: IOException =>
			throw new IllegalStateException(s"S3, IO failed to moving $fromS3Folder in ${s3BucketName} ${e.getMessage}")
	}


	/**
	 * Move content of this file to another folder within the same bucket.
	 * {{{
	 *   The steps are
	 *   - Copy the content of file to a new folder
	 *   - Delete source file
	 * }}}
	 *
	 * @param s3BucketName Bucket name
	 * @param s3SourceFile S3 file to be moved to new folder
	 * @param toS3Folder S3 destination folder
	 * @return Number of S3 files/records moved
	 */
	@throws(clazz = classOf[IllegalStateException])
	def moveFile(s3BucketName: String, s3SourceFile: String, toS3Folder: String): Unit = try {
		val lastSeparatorIndex = s3SourceFile.lastIndexOf("/")

		if(lastSeparatorIndex != -1) {
			val destFile = s3SourceFile.substring(lastSeparatorIndex +1)
			if(destFile.nonEmpty) {
				val destPath = s"$toS3Folder/$destFile"
				s3Client.copyObject(s3BucketName, s3SourceFile, s3BucketName, destPath)
				s3Client.deleteObject(s3BucketName, s3SourceFile)
			}
		}
		else
			logger.error(s"Could not extract key from $s3SourceFile")
	}
	catch {
		case e: AmazonServiceException =>
			throw new IllegalStateException(s"S3, Failed to move $s3SourceFile in ${s3BucketName} ${e.getMessage}")
		case e: SdkClientException =>
			throw new IllegalStateException(s"S3, broken connection moving $s3SourceFile in${s3BucketName} ${e.getMessage}")
		case e: IOException =>
			throw new IllegalStateException(s"S3, IO failed to moving $s3SourceFile in ${s3BucketName} ${e.getMessage}")
	}

	@throws(clazz = classOf[IllegalStateException])
	final def getS3Keys(client: AmazonS3, bucketName: String): Iterable[String] = try {
		import org.bertspark.implicits._

		val objects = client.listObjects(new ListObjectsRequest().withBucketName(bucketName))
		val summaries = objects.getObjectSummaries
		val ls: Iterable[S3ObjectSummary] = listOf[S3ObjectSummary](summaries)
		ls.map(_.getKey)
	}
	catch {
		case e: AmazonServiceException =>
			throw new IllegalStateException(s"S3, Failed to get S3 keys in ${bucketName} ${e.getMessage}")
		case e: SdkClientException =>
			throw new IllegalStateException(s"S3, broken connection to get S3 keys in ${bucketName} ${e.getMessage}")
		case e: IOException =>
			throw new IllegalStateException(s"S3, Could not write to get S3 keys in ${bucketName} ${e.getMessage}")
	}

	/**
	  * Get the list of keys from a given bucket name, a prefix (root directory), date interval [fromDate, toDate]
	  * and maximum number of keys per object listing
 *
	  * @param s3BucketName Name of AWS/S3 buckets
	  * @param s3Prefix Prefix or parent folder
	  * @param fromToDates Optional pair (fromDate, toDate) that defined the interval to extract the records.
	  *                    The format should follow convention ml.s3DateFormat
	  * @param maxKeys Maximum number of keys per object listing
	  * @return Sequence of keys/S3 paths
	  */
	@throws(clazz = classOf[IllegalStateException])
	final def getS3Keys(
		s3BucketName: String,
		s3Prefix: String,
		fromToDates: Option[(String, String)],
		maxKeys: Int): Iterable[String] = {
			val ls = getS3SummariesWithDates(s3BucketName, s3Prefix, fromToDates, maxKeys)
			ls.map(_.getKey)
		}

	/**
	 * Get the list of keys from a given bucket name and prefix (root directory)
 *
	 * @param s3BucketName Name of AWS/S3 buckets
	 * @param s3Prefix Prefix or parent folder
	 * @return Sequence of keys/S3 paths
	 */
	@throws(clazz = classOf[IllegalStateException])
	final def getS3Keys(
		s3BucketName: String,
		s3Prefix: String): Iterable[String] = {
		val ls = _getS3Summaries[String](s3BucketName, s3Prefix, None, defaultMaxKeys)
		ls.map(_.getKey)
	}

	/**
		* Retrieve the file with the maximum key for a given pattern for S3 folder
		* @param s3BucketName Name of S3 bucker
		* @param s3Prefix S3 prefix for the keys to search
		* @param pattern Pattern used to search the key
		* @param ranking Function to extract rank from a S3 key
		* @return Key with the highest rank or empty key
		*/
	final def getMaxKey(s3BucketName: String, s3Prefix: String, pattern: String, ranking: String => Int): String = {
		val keys = getS3Keys(s3BucketName, s3Prefix)
		if(keys.nonEmpty)
			keys.maxBy(
				key => {
					val keyHead = key.substring(key.length-pattern.length)
					ranking(keyHead)
				}
			)
		else {
			logger.warn(s"Failed to find a max key for $s3Prefix with $pattern pattern")
			""
		}
	}


	/**
	  * Retrieve the object summaries associated to a prefix and bucket. This implementation relies
	  * on paging mechanism to get the next batch of objects
 *
	  * @param s3BucketName Name of AWS/S3 buckets
	  * @param s3Prefix Prefix or parent folder
	  * @param maxKeys Maximum number of keys returned
	  * @return Sequence of object summary (-1) retrieve all the objects
	  */
	final def getS3Summaries(s3BucketName: String, s3Prefix: String, maxKeys: Int): List[S3ObjectSummary] =
		_getS3Summaries[String](s3BucketName, s3Prefix, None,  maxKeys)


	final def getS3Summaries(s3BucketName: String, s3Prefix: String): List[S3ObjectSummary] =
		_getS3Summaries[String](s3BucketName, s3Prefix, None,  maxKeys = defaultMaxKeys)

	/**
	 * {{{
	 * Retrieve the object summaries associated to a prefix and bucket. This implementation relies
	 * on paging mechanism to get the next batch of objects.
	 * The maximum number of keys is applied to each object listing extracted from the given bucket using
	 * a S3 path (or prefix).
	 * The condition on the time frame specified by argument fromToDates time > fromTime && time <= toTime
	 * }}}
	 *
	 * @param s3BucketName Name of AWS/S3 buckets
	 * @param s3Prefix Prefix or parent folder
	 * @param fromToDates Pair (fromDate, toDate) that defined the interval to extract the records.
	 *                    The format should follow convention ml.s3DateFormat
	 * @param maxKeys Maximum number of keys returned
	 * @return Sequence of object summary (-1) retrieve all the objects
	 * @note  We do not assume that the order of from date and to date is correct.
	 */
	@throws(clazz = classOf[IllegalStateException])
	final def getS3SummariesWithDates(
		s3BucketName: String,
		s3Prefix: String,
		fromToDates: Option[(String, String)],
		maxKeys: Int): List[S3ObjectSummary] = _getS3Summaries[String](s3BucketName ,s3Prefix, fromToDates, maxKeys)


	@throws(clazz = classOf[IllegalStateException])
	final def getS3SummariesWithTimeMillis(
		s3BucketName: String,
		s3Prefix: String,
		fromToTimeMillis: Option[(Long, Long)],
		maxKeys: Int): List[S3ObjectSummary] = _getS3Summaries[Long](s3BucketName ,s3Prefix, fromToTimeMillis, maxKeys)


	final def _getS3Summaries[T](
		s3BucketName: String,
		s3Prefix: String,
		fromToDates: Option[(T, T)],
		maxKeys: Int): List[S3ObjectSummary] = try {
		import org.bertspark.implicits._
		import scala.collection.mutable.ListBuffer

		val listObjectRequest = new ListObjectsRequest().withBucketName(s3BucketName).withPrefix(s3Prefix)
		var objectsListing: ObjectListing = s3Client.listObjects(listObjectRequest)
		objectsListing.setMaxKeys(maxKeys)

		val keyList = new ListBuffer[S3ObjectSummary]()
		val ls: Iterable[S3ObjectSummary] = listOf[S3ObjectSummary](objectsListing.getObjectSummaries)
		keyList.appendAll(ls)

		// Apply the iterator on objects list
		while (objectsListing.isTruncated) {
			objectsListing = s3Client.listNextBatchOfObjects(objectsListing)
			// Set maximum keys..
			objectsListing.setMaxKeys(maxKeys)
			// Java 2 Scala list conversion
			val _ls = listOf[S3ObjectSummary](objectsListing.getObjectSummaries)
			keyList.appendAll(_ls)
		}

		// If the filter per date is defined.....
		fromToDates.map {
			case (from, to) =>
				import org.bertspark.util.DateUtil.simpleS3TimeStamp

				val fromTimeMillis = from match {
					case _from: Long => _from
					case _from: String => simpleS3TimeStamp(_from)
					case _ => throw new IllegalStateException(s"get Summaries $from type unsupported")
				}
				val toTimeMillis = to match {
					case _to: Long => _to
					case _to: String => simpleS3TimeStamp(_to)
					case _ => throw new IllegalStateException(s"get Summaries $to type unsupported")
				}

				keyList.toList.filter(
					objSummary => {
						val convertedTime = simpleS3TimeStamp(objSummary.getLastModified.toString)
						// We do not assume that the time interval was properly specified.
						if(fromTimeMillis < toTimeMillis)
							convertedTime > fromTimeMillis && convertedTime <= toTimeMillis
						else
							convertedTime <= fromTimeMillis && convertedTime > toTimeMillis
					}
				)
		}.getOrElse(keyList.toList)
	}
	catch {
		case e: AmazonServiceException =>
			throw new IllegalStateException(s"S3, Failed to get S3 keys in ${s3BucketName} ${e.getMessage}")
		case e: SdkClientException =>
			throw new IllegalStateException(s"S3, broken connection to get S3 keys in ${s3BucketName} ${e.getMessage}")
		case e: IOException =>
			throw new IllegalStateException(s"S3, Could not write to get S3 keys in ${s3BucketName} ${e.getMessage}")
	}


	/**
	  * Get the list of keys from a given bucket name and prefix (root directory)
 *
	  * @param s3BucketName Name of AWS buckets
	  * @param s3Prefix Prefix or root directory
	  * @param s3Suffix last section of the key
	  * @return Sequence of keys
	  */
	final def getS3Keys(s3BucketName: String, s3Prefix: String, s3Suffix: String): Iterable[String] = {
		val regexStr = s3Prefix +"/([a-zA-Z0-9/]+)/" + s3Suffix
    val regex = regexStr.r
    try {
      getS3Keys(s3BucketName, s3Prefix)
          .filter(_.nonEmpty)
          .flatMap(regex.findFirstMatchIn(_).map(_.group(1)))
          .toSeq
          .distinct
    }
    catch {
      case e: AmazonServiceException =>
        throw new IllegalStateException(s"S3, Failed to get keys from ${s3BucketName}/$s3Prefix ${e.getMessage}")
      case e: SdkClientException =>
        throw new IllegalStateException(s"S3, broken connection to  ${s3BucketName}/$s3Prefix ${e.getMessage}")
      case e: IOException =>
        throw new IllegalStateException(s"S3, Could not write  ${s3BucketName}/$s3Prefix ${e.getMessage}")
    }
  }

  /**
   * Load a dataset from S3, repartitions it then save into a new S3 folder
   *
   * @param s3SourceFolder      Folder of source of HDFS data
   * @param s3DestinationFolder Destination of HDFS data
   * @param numPartitions       Number of new partitions
   * @param fileFormat          Format of the file (i.e. CSV, JSON....)
   * @param sparkSession        Implicit reference to the current Spark context
   * @param encoder             Implicit encoder
   * @tparam T Type of data set element
   * @throws IllegalArgumentException if Number of partitions is out of bounds
   * @throws IllegalStateException    If read or write operation on S3 fails
   */
  @throws(clazz = classOf[IllegalArgumentException])
  @throws(clazz = classOf[IllegalStateException])
  def s3Repartition[T](
    s3SourceFolder: String,
    s3DestinationFolder: String,
    numPartitions: Int,
    fileFormat: String = "json"
  )(implicit sparkSession: SparkSession, encoder: Encoder[T]): Unit =
    s3Transform(s3SourceFolder, s3DestinationFolder, numPartitions, fileFormat, toAppend = false, None)

  /**
   * Load a dataset from S3, re-partitions it then save into a new S3 folder
   *
   * @param s3SourceFolder      Folder of source of HDFS data
   * @param s3DestinationFolder Destination of HDFS data
   * @param numPartitions       Number of new partitions
   * @param fileFormat          Format of the file (i.e. CSV, JSON....)
   * @param sparkSession        Implicit reference to the current Spark context
   * @param encoder             Implicit encoder
   * @tparam T Type of data set element
   * @throws IllegalArgumentException if Number of partitions is out of bounds
   * @throws IllegalStateException    If read or write operation on S3 fails
   */
  @throws(clazz = classOf[IllegalStateException])
  def s3Transform[T](
    s3SourceFolder: String,
    s3DestinationFolder: String,
    numPartitions: Int,
    fileFormat: String = "json",
    toAppend: Boolean = false,
    transform: Option[T => T]
  )(implicit sparkSession: SparkSession, encoder: Encoder[T]): Unit = try {
    require(numPartitions > 0 && numPartitions < 1024, s"Number of partitions for repartitioning $numPartitions should be [1, 1023]")

    val initialDS = s3ToDataset[T](s3InputFile = s3SourceFolder, header = false, fileFormat = "json")
    val ds = transform.map(t => initialDS.map(t(_))).getOrElse(initialDS)

    datasetToS3[T](
      ds,
      s3OutputPath = s3DestinationFolder,
      header = false,
      fileFormat,
      toAppend,
      numPartitions
    )
  }
  catch {case e: IllegalArgumentException => logger.error(e.getMessage)}

  def upload(bucketName: String, s3Folder: String, content: String): Option[String] =
    load(bucketName, s3Folder, content)

	def upload(s3Folder: String, content: String): Option[String] =
		load(mlopsConfiguration.storageConfig.s3Bucket, s3Folder, content)



  private def load(bucketName: String, key: String, content: String): Option[String] = try {
    val metadata = new ObjectMetadata
    metadata.setContentLength(content.length.asInstanceOf[Long])

    val inputStream = new ByteArrayInputStream(content.getBytes)
    val putObjectRequest = new PutObjectRequest(bucketName, key, inputStream, metadata)
    s3Client.putObject(putObjectRequest)
    Some(key)
  }
  catch {
    case e: AmazonServiceException =>
      logger.error(s"S3, Failed to upload $key into bucket $bucketName ${e.getMessage}")
      None
    case e: SdkClientException =>
      logger.error(s"S3, Broken connection to $key into bucket $bucketName ${e.getMessage}")
      None
    case e: IOException =>
      logger.error(s"S3, Could not write $key into bucket $bucketName ${e.getMessage}")
      None
		case e: Exception =>
			logger.error(s"S3, Undefined exception $key into bucket $bucketName ${e.getMessage}")
			None
  }

	def cpS3ToFs(bucketName: String, key: String, fsFilename: String): Unit = try {
		val client = getS3Client
		val objectRequest = new GetObjectRequest(bucketName, key)
		val file = new File(fsFilename)
		val objectMetadata = client.getObject(objectRequest, file)
		logDebug(logger, s"Copy ${objectMetadata.getContentLength} bytes from $key to $fsFilename")
	} catch {
		case e: AmazonServiceException =>
			logger.error(s"S3, Failed to upload $key into bucket $bucketName ${e.getMessage}")
		case e: SdkClientException =>
			logger.error(s"S3, broken connection to $key into bucket $bucketName ${e.getMessage}")
		case e: IOException =>
			logger.error(s"S3, Could not write $key into bucket $bucketName ${e.getMessage}")
	}

  /**
   * Download the content of a key or file
   *
   * @param bucketName Name of S3 bucket
   * @param key        Key or name of the file
   * @return Optional content
   */
  def download(bucketName: String, key: String): Option[String] = try {
    val client = getS3Client
    val objectRequest = new GetObjectRequest(bucketName, key)
    val obj = client.getObject(objectRequest)
    val s3Content = obj.getObjectContent
    val stream = new InputStreamReader(s3Content)
    val bufferedReader = new BufferedReader(stream)
    val iter = bufferedReader.lines.iterator

    val collector = new StringBuilder
    while (iter.hasNext) {
      collector.append(iter.next).append("\n")
    }
    Some(collector.toString)
  }
  catch {
    case e: AmazonServiceException =>
      logger.error(s"S3, download AWS $key for $bucketName ${e.getMessage}")
			e.printStackTrace()
      None
    case e: SdkClientException =>
      logger.error(s"S3, download SDK $key for $bucketName ${e.getMessage}")
			e.printStackTrace()
      None
    case e: IOException =>
      logger.error(s"S3, download IO $key for $bucketName ${e.getMessage}")
			e.printStackTrace()
      None
		case e: Exception =>
			logger.error(s"S3, download $key for $bucketName ${e.getMessage}")
			e.printStackTrace()
			None
  }

	def downloadCollection(bucketName: String, key: String): Option[Seq[String]] = try {
		import scala.collection.JavaConverters._

		val client = getS3Client
		val objectRequest = new GetObjectRequest(bucketName, key)
		val obj = client.getObject(objectRequest)
		val s3Content = obj.getObjectContent
		val stream = new InputStreamReader(s3Content)
		val bufferedReader = new BufferedReader(stream)

		val iter: java.util.Iterator[String] = bufferedReader.lines.iterator
		val output = iter.asScala.toArray
		Some(output)
	}
	catch {
		case e: AmazonServiceException =>
			logger.error(s"S3 downloadCollection AWS $key for $bucketName ${e.getMessage}")
			e.printStackTrace()
			None
		case e: SdkClientException =>
			logger.error(s"S3 downloadCollection SDK $key for $bucketName ${e.getMessage}")
			e.printStackTrace()
			None
		case e: IOException =>
			logger.error(s"S3 downloadCollection IO $key for $bucketName ${e.getMessage}")
			e.printStackTrace()
			None
		case e: Exception =>
			logger.error(s"S3 downloadCollection $key for $bucketName ${e.getMessage}")
			e.printStackTrace()
			None
	}

	/**
		* Download the CSV fields from a S3 folder
		* @param bucketName Name of S3 bucket
		* @param key S3 key or absolute path for folder
		* @return Optional sequence of fields
		*/
	def downloadCSVFields(bucketName: String, key: String): Option[Seq[Array[String]]] =
		downloadCollection(bucketName, key).map(_.map(_.split(",")))

	/**
	 * Transfer from a local file to a S3 bucket and path
	 * @param sourceFilename Name of the original, source
	 * @param s3Bucket Name of S3 bucket
	 * @param s3Path Name of S3 path
	 */
	@throws(clazz = classOf[IllegalStateException])
	def fsToS3(sourceFilename: String, s3Bucket: String, s3Path: String): Unit = try {
		val start = System.currentTimeMillis()
		val fsFile = new File(sourceFilename)
		val len = fsFile.length()*0.001

		logger.info(s"Start uploading $sourceFilename with $len KB to $s3Path")
		val client = getS3Client
		val putRequest = new PutObjectRequest(s3Bucket, s3Path, fsFile)
		client.putObject(putRequest)
		logger.info(s"$sourceFilename copied to $s3Path completed after ${(System.currentTimeMillis()-start)*0.001} secs.")
	}
	catch {
		case e: AmazonServiceException =>
			throw new IllegalStateException(s"AWS service From $sourceFilename to $s3Bucket/$s3Path:  ${e.getMessage}")
		case e: SdkClientException =>
			throw new IllegalStateException(s"AWS SDK From $sourceFilename to $s3Bucket/$s3Path:   ${e.getMessage}")
		case e: IOException =>
			throw new IllegalStateException(s"IO failure From $sourceFilename to $s3Bucket/$s3Path:  ${e.getMessage}")
		case e: Exception =>
			throw new IllegalStateException(s"Undefined failure From $sourceFilename to $s3Bucket/$s3Path:  ${e.getMessage}")
	}

	@throws(clazz = classOf[IllegalStateException])
	def fsToS3(fsSrcFilename: String, s3DestPath: String): Unit =
		fsToS3(fsSrcFilename, mlopsConfiguration.storageConfig.s3Bucket, s3DestPath)

	/**
	 * Transfer a file from S3 to a local
	 * @param destFilename local file name
	 * @param s3Bucket Bucket containing the path to store the transferred file.
	 * @param s3Path Path of S3 folder
	 * @throws IllegalStateException In case the S3 folder could not be loaded..
	 */
	@throws(clazz = classOf[IllegalStateException])
	def s3ToFs(destFilename: String, s3Bucket: String, s3Path: String, isText: Boolean): Unit =
		if(isText) {
			val completed = download(s3Bucket, s3Path)
					.map(LocalFileUtil.Save.local(destFilename, _))
					.getOrElse(throw new IllegalStateException(s"S3toFS failed for $s3Path"))
			logger.debug(s"Copy from $s3Path to local $destFilename completion: $completed")
		}
		else
			cpS3ToFs(s3Bucket, s3Path, destFilename)



	/**
	 * Transfer a file from S3 to a local
	 * @param destFilename local file name
	 * @param s3Path Path of S3 folder
	 * @throws IllegalStateException In case the S3 folder could not be loaded..
	 */
	@throws(clazz = classOf[IllegalStateException])
	def s3ToFs(destFilename: String, s3Path: String, isText: Boolean): Unit =
		s3ToFs(destFilename, mlopsConfiguration.storageConfig.s3Bucket, s3Path, isText)


	/**
	 * Extract the highest digit included in the S3 keys
	 * @param keys S3 keys
	 * @param numDigits Number of digit in the counter
	 * @param suffix Suffix after the counter in the key, if any
	 * @return Highest digit contained in the key
	 */
	def getHighestCount(keys: Seq[String], numDigits: Int, suffix: String = ""): Int =
		if(keys.nonEmpty)
			keys.map(
				key => {
					if(suffix.nonEmpty) {
						val suffixIndex = key.indexOf(suffix)
						val digitStr = key.substring(suffixIndex - numDigits, suffixIndex)
						digitStr.toInt
					}
					else {
						val digitStr = key.substring(key.length - numDigits)
						digitStr.toInt
					}
				}
			).max
		else 0

	/**
	 * Generic upload and download to/from a S3 folder using CSV format
	 * @param s3Folder Targeted S3 folder
	 * @param keyFct Function to convert String to Key of type T
	 * @param valueFct Function to convert String to Value of type U
	 * @tparam K Type of key
	 * @tparam V Type of value
	 *
	 * @author Patrick Nicolas
	 * @version 0.1
	 */
	final class S3UploadDownloadMap[K, V](s3Folder: String, keyFct: String => K, valueFct: String => V) {

		def upload(subModelOutputSizeMap: HashMap[K, V]): Unit = try {
			S3Util.upload(
				mlopsConfiguration.storageConfig.s3Bucket,
				s3Folder,
				subModelOutputSizeMap.map{
					case (key, value) => s"${key.toString},${value.toString}"
				}.mkString("\n")
			)
		}
		catch {
			case e: AmazonServiceException =>
				throw new IllegalStateException(s"Upload AWS service ${mlopsConfiguration.storageConfig.s3Bucket}/$s3Folder  ${e.getMessage}")
			case e: SdkClientException =>
				throw new IllegalStateException(s"Upload AWS SDK ${mlopsConfiguration.storageConfig.s3Bucket}/$s3Folder  ${e.getMessage}")
			case e: IOException =>
				throw new IllegalStateException(s"Upload IO failure ${mlopsConfiguration.storageConfig.s3Bucket}/$s3Folder ${e.getMessage}")
			case e: Exception =>
				throw new IllegalStateException(s"Upload Undefined failure ${mlopsConfiguration.storageConfig.s3Bucket}/$s3Folder ${e.getMessage}")
		}

		def download: HashMap[K, V] = try {
			S3Util.download(
				mlopsConfiguration.storageConfig.s3Bucket,
				s3Folder
			)   .map(_.split("\n").map(_.split(CSV_SEPARATOR)))
					.map(_.foldLeft(HashMap[K, V]()) ((hMap, ar) => hMap += ( (keyFct(ar.head), valueFct(ar(1)))))
					).getOrElse({
						logger.error(s"Failed to load the sub model classes map from $s3Folder")
						HashMap.empty[K, V]
				}
			)
		}
		catch {
			case e: AmazonServiceException =>
				throw new IllegalStateException(s"Download AWS service ${mlopsConfiguration.storageConfig.s3Bucket}/$s3Folder  ${e.getMessage}")
			case e: SdkClientException =>
				throw new IllegalStateException(s"Download AWS SDK ${mlopsConfiguration.storageConfig.s3Bucket}/$s3Folder  ${e.getMessage}")
			case e: IOException =>
				throw new IllegalStateException(s"Download IO failure ${mlopsConfiguration.storageConfig.s3Bucket}/$s3Folder ${e.getMessage}")
			case e: Exception =>
				throw new IllegalStateException(s"Download Undefined failure ${mlopsConfiguration.storageConfig.s3Bucket}/$s3Folder ${e.getMessage}")
		}
	}


	/**
	 *
	 * @param s3Folder
	 * @param keyFunc
	 * @param stringize Convert the type K to a string
	 * @tparam K
	 */
	final class S3UploadDownloadSeq[K: ClassTag](
		s3Folder: String,
		keyFunc: String => K,
		stringize: K => String = (k: K) => k.toString) {

		def upload(sequence: Array[K]): Unit = try {
			S3Util.upload(
				mlopsConfiguration.storageConfig.s3Bucket,
				s3Folder,
				sequence.map(stringize(_)).mkString("\n")
			)
		} catch {
			case e: IllegalArgumentException => logger.error(s"S3UploadDownloadSeq.upload ${e.getMessage}")
		}

		def download: Array[K] = try {
			S3Util.download(
				mlopsConfiguration.storageConfig.s3Bucket,
				s3Folder
			).map(_.split("\n").map(keyFunc(_))).getOrElse(Array.empty[K])
		}
		catch {
			case e: IllegalArgumentException =>
				logger.error(s"S3UploadDownloadSeq.download ${e.getMessage}")
				Array.empty[K]
		}
	}
}
