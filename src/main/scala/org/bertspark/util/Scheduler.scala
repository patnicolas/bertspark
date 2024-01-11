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
/*
import akka.actor.{Actor, Props}
import org.mlops.util.Scheduler.ScheduledTask
import org.slf4j.{Logger, LoggerFactory}
import scala.concurrent.duration.DurationLong
import scala.language.postfixOps

/**
 * Parameterized scheduler that uses Akka actor and dispatchers
 * @param task Task to be executed at given interval
 * @param intervalSec Interval between execution in seconds
 * @param initialDelay  Time the scheduler executes the first time (default value: 20 seconds)
 * @tparam T Type of task which should implement the SchedulerTask
 *
 * @author Patrick Nicolas
 * @version 0.1
 */
private[mlops] final class Scheduler[T <: ScheduledTask](task: T, intervalSec: Long, initialDelay: Int = 20) {
  import Scheduler._
  import org.mlops.implicits._
  import scala.concurrent.duration.DurationInt
  import system.dispatcher

  private[this] val scheduleActor = system.actorOf(Props(new ScheduleActor(task)), name = s"${task.name}_actor")
  private[this] val cancellableScheduler = system.scheduler.schedule(
    initialDelay seconds,
    intervalSec seconds,
    scheduleActor,
    schedulingMessage)

  def cancel: Unit = cancellableScheduler.cancel
}


/**
 * Singleton to define Scheduling actors and default variables
 */
private[mlops] final object Scheduler {
  final private val logger: Logger = LoggerFactory.getLogger("Scheduler")
  final private val dateTimeFormat = "yyyy-MM-dd:hh.mm.ss"

  final private val schedulingMessage = "schedulingMessage"

  trait ScheduledTask {
    val name: String
    def execute: Unit
  }

  /**
   * Actors invoked for a give scheduled task
   * @param scheduledTask Task to be executed at given interval...
   */
  final class ScheduleActor(scheduledTask: ScheduledTask) extends Actor {
    def receive: Receive = {
      case `schedulingMessage` =>
        logInfo(logger,  s"Scheduler starts task ${scheduledTask.name}")
        scheduledTask.execute
        logInfo(logger,  s"Scheduler completed task ${scheduledTask.name}")
    }
  }


  /**
   * Compute the delay between the current time in milliseconds and the scheduled time in the day for the first
   * and subsequent executions.
   * @param timeInDay Time in the day
   * @throws IllegalArgumentException if time in day is out of bounds
   * @return A valid time in secs if successful, -1 otherwise
   */
  @throws(clazz = classOf[Exception])
  @throws(clazz = classOf[IllegalArgumentException])
  def timeIntervalToScheduledTimeInSec(timeInDay: Int): Int =  {
    require(timeInDay >= 0 && timeInDay < 25, s"Scheduled time in the day $timeInDay should be [0, 24]")

    val schedulerTimeMillisInDay = timeInDay*3600*1000L
    val currentTimeMillis = System.currentTimeMillis
    val currentDateTime = DateUtil.longToDate(currentTimeMillis, dateTimeFormat)

    val dateIndex = currentDateTime.indexOf(":")
    val startDayDateTime = currentDateTime.substring(0, dateIndex)
    val startDayMillis = DateUtil.simpleTimeStampWithPattern(startDayDateTime, "yyyy-MM-dd")
    val scheduledTime = startDayMillis + schedulerTimeMillisInDay

    val scheduledDateTime = DateUtil.longToDate(scheduledTime, dateTimeFormat)
    logInfo(logger,  s"Current time: $currentDateTime, Scheduled time = $scheduledDateTime")
    ((scheduledTime-currentTimeMillis)*0.001).toInt
  }
}

 */
