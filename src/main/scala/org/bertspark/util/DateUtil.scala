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

import java.util.Date
import org.slf4j.{Logger, LoggerFactory}


/**
 * Simple date utility functions
 * @author Patrick Nicolas
 * @version 0.1
 */
private[bertspark] final object DateUtil {
  final val log: Logger = LoggerFactory.getLogger("DateUtil")

  final val oneMinuteIntervalSec = 60
  final val oneHourIntervalSec = 3600
  final val halfDayIntervalSec = 12*3600
  final val oneDayIntervalSec = 24*3600

  final val oneMinuteIntervalMilliSec = oneMinuteIntervalSec*1000L
  final val oneHourIntervalMilliSec = oneHourIntervalSec*1000L
  final val halfDayIntervalMilliSec = halfDayIntervalSec*1000L
  final val oneDayIntervalMilliSec = oneDayIntervalSec*1000L

  final val defaultTimeFormat =  "MM-dd-yyyy-HH.MM.SS"
  final val internalDateFormat = "MM-dd-yyyy"
  final val requestDateFormat = "yyyy-MM-dd"
  final val s3DateFormat = "EEE MMM dd HH:mm:ss zzz yyyy"


  /**
   * Extract the date associated with this time stamp (Long) and date format
   * @param timeMilliSeconds Time stamp
   * @param formatter Date-String formatter
   * @return Date associated with the time stamp
   */
  final def longToDate(timeMilliSeconds: Long, formatter: String): String = {
    import java.text.SimpleDateFormat

    val simpleDateFormat = new SimpleDateFormat(formatter)
    simpleDateFormat.format(if(timeMilliSeconds > 0) new Date(timeMilliSeconds) else new Date())
  }

  /**
   * retrieve the date associated with the current time stamp using the default time format
   * @return
   */
  final def longToDate: String = longToDate(-1L, defaultTimeFormat)

  final def simpleLongToDate: String = longToDate(-1L, internalDateFormat)


  /**
   * Generic conversion of date into absolute time in milliseconds
   * @param inputDate Input date using the pattern, 'pattern'
   * @param dateFormat Pattern or format for the extraction of the date
   * @return Time in milliseconds associated with the date, -1L if parser failed
   */
  final def simpleTimeStampWithPattern(inputDate: String, dateFormat: String): Long = {
    import java.text.SimpleDateFormat
    import java.text.ParseException

    val simpleDateFormat = new SimpleDateFormat(dateFormat)
    try {
      val date = simpleDateFormat.parse(inputDate);
      date.getTime
    } catch {
      case e: ParseException =>
        log.error(e.toString)
        -1L
    }
  }

  /**
   * Generic conversion of date into absolute time in milliseconds
   * @param inputDate Input date using the pattern, 'pattern'
   * @return Time in milliseconds associated with the date, -1L if parser failed
   */
  final def timeStamp(inputDate: String): Long = simpleTimeStampWithPattern(inputDate, defaultTimeFormat)


  /**
   * Generic conversion of dates in requests .. using format 'yyyy-MM-dd' into absolute time in milliseconds
   * @param inputDate Input date of format 'yyyy-MM-dd'
   * @return Time in milliseconds
   */
  final def requestTimeStamp(inputDate: String): Long = simpleTimeStampWithPattern(inputDate, requestDateFormat)

  /**
   * Conversion of a date string using internal date format into a time in millis
   * @param inputDate Input date using internal date format
   * @return Time in millis if succeeds, -1L in case of ParseException error
   */
  final def s3TimeStamp(inputDate: String): Long = simpleTimeStampWithPattern(inputDate, internalDateFormat)


  /**
   * Conversion of a date string using S3 date format into a time in millis
   * @param inputDate Input date using S3 date format
   * @return Time in millis if succeeds, -1L in case of ParseException error
   */
  final def simpleS3TimeStamp(inputDate: String): Long = simpleTimeStampWithPattern(inputDate, s3DateFormat)


  /**
   * Retrieve the date of the next day
   * @param inputDate Current day
   * @param dateFormat Date format used in the conversion
   * @return Date of next day
   */
   def getNextDay(inputDate: String, dateFormat: String): String = {
    val timeMilliSeconds = simpleTimeStampWithPattern(inputDate, dateFormat)
    val nextTimeMilliSeconds = timeMilliSeconds + oneDayIntervalMilliSec
    longToDate(nextTimeMilliSeconds, dateFormat)
  }

  /**
   * Retrieve the date of the previous day
   * @param inputDate Current day
   * @param dateFormat Date format used in the conversion
   * @return Date of next day
   */
  def getPreviousDay(inputDate: String, dateFormat: String): String = {
    val timeMilliSeconds = simpleTimeStampWithPattern(inputDate, dateFormat)
    val nextTimeMilliSeconds = timeMilliSeconds - oneDayIntervalMilliSec
    longToDate(nextTimeMilliSeconds, dateFormat)
  }
}
