/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Aug 21, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file ThreadedKFVio.cpp
 * @brief Source file for the ThreadedKFVio class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <map>
#include <fstream>
#include <iomanip>

#include <glog/logging.h>

#include <okvis/ThreadedKFVio.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ImuError.hpp>

#ifndef OKVIS_N_THREADS
#define OKVIS_N_THREADS 4 // default works well for quadcore i7
#endif

/// \brief okvis Main namespace of this package.
namespace okvis {

static const int max_camera_input_queue_size = 10;
static const okvis::Duration temporal_imu_data_overlap(0.02);  // overlap of imu data before and after two consecutive frames [seconds]

#ifdef USE_MOCK
// Constructor for gmock.
ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters, okvis::MockVioBackendInterface& estimator,
    okvis::MockVioFrontendInterface& frontend)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      estimator_(estimator),
      frontend_(frontend),
      parameters_(parameters),
      maxImuInputQueueSize_(60) {
  init();
}
#else
// Constructor.
ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      estimator_(),
      frontend_(parameters.nCameraSystem.numCameras()),
      parameters_(parameters),
      maxImuInputQueueSize_(
          2 * max_camera_input_queue_size * parameters.imu.rate
              / parameters.sensors_information.cameraRate) {
  setBlocking(false);
  init();
}
#endif

// Initialises settings and calls startThreads().
void ThreadedKFVio::init() {
  assert(parameters_.nCameraSystem.numCameras() > 0);
  numCameras_ = parameters_.nCameraSystem.numCameras();
  numCameraPairs_ = 1;

  frontend_.setBriskDetectionOctaves(parameters_.optimization.detectionOctaves);
  frontend_.setBriskDetectionThreshold(parameters_.optimization.detectionThreshold);
  frontend_.setBriskDetectionMaximumKeypoints(parameters_.optimization.maxNoKeypoints);

  lastOptimizedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)
  lastAddedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)

  estimator_.addImu(parameters_.imu);
  size_t i = 0;
  for (; i < numCameras_; ++i) {
    // parameters_.camera_extrinsics is never set (default 0's)...
    // do they ever change?
    estimator_.addCamera(parameters_.camera_extrinsics);
    cameraMeasurementsReceived_.emplace_back(
          std::shared_ptr<threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> > >
          (new threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> >()));
  }
  for (; i < numCameras_ + parameters_.trackerNCameraSystem.numCameras(); ++i) {
    // parameters_.camera_extrinsics is never set (default 0's)...
    // do they ever change?
    cameraMeasurementsReceived_.emplace_back(
          std::shared_ptr<threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> > >
          (new threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> >()));
  }
  estimator_.addGps(parameters_.gps);
  
  // set up windows so things don't crash on Mac OS
  if(parameters_.visualization.displayImages){
    for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
      std::stringstream windowname;
      windowname << "OKVIS camera " << im;
      cv::namedWindow(windowname.str());
    }
  }
  
  // // MBZIRC: target keypoint locations
  // const double scale = parameters_.targetSizeMetres/1.5;
  // const double s_45 = std::sin(45 * M_PI / 180.0);
  // const double l = scale * 0.33;
  // const double r = scale * 0.1 / std::sqrt(2);

  // Eigen::Matrix2Xd xPoints(2,12);
  // xPoints << r, r + l * s_45, l * s_45, 0, -l * s_45, -r - l * s_45, -r, -r
  //     - l * s_45, -l * s_45, 0, l * s_45, r + l * s_45, 0, l * s_45, r
  //     + l * s_45, r, r + l * s_45, l * s_45, 0, -l * s_45, -r - l * s_45, -r, -r
  //     - l * s_45, -l * s_45;
  // frontend_.setTargetKeypoints(xPoints);
  // frontend_.setTargetSize(parameters_.targetSizeMetres);
  // estimator_.setTargetKeypoints(xPoints);

  startThreads();
}

std::vector<std::shared_ptr<ThreadedKFVio::MarkerMeasurement>> markerMeasurements;

// Start all threads.
void ThreadedKFVio::startThreads() {

  markerMeasurements.resize(parameters_.trackerNCameraSystem.numCameras());

  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.emplace_back(&ThreadedKFVio::frameConsumerLoop, this, i);
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.emplace_back(&ThreadedKFVio::matchingLoop, this);
  }
  imuConsumerThread_ = std::thread(&ThreadedKFVio::imuConsumerLoop, this);
  positionConsumerThread_ = std::thread(&ThreadedKFVio::positionConsumerLoop,
                                        this);
  gpsConsumerThread_ = std::thread(&ThreadedKFVio::gpsConsumerLoop, this);
  magnetometerConsumerThread_ = std::thread(
      &ThreadedKFVio::magnetometerConsumerLoop, this);
  differentialConsumerThread_ = std::thread(
      &ThreadedKFVio::differentialConsumerLoop, this);

  // algorithm threads
  visualizationThread_ = std::thread(&ThreadedKFVio::visualizationLoop, this);
  optimizationThread_ = std::thread(&ThreadedKFVio::optimizationLoop, this);
  publisherThread_ = std::thread(&ThreadedKFVio::publisherLoop, this);
}

// Destructor. This calls Shutdown() for all threadsafe queues and joins all threads.
ThreadedKFVio::~ThreadedKFVio() {
  for (size_t i = 0; i < numCameras_+ parameters_.trackerNCameraSystem.numCameras(); ++i) {
    cameraMeasurementsReceived_.at(i)->Shutdown();
  }
  keypointMeasurements_.Shutdown();
  matchedFrames_.Shutdown();
  imuMeasurementsReceived_.Shutdown();
  optimizationResults_.Shutdown();
  visualizationData_.Shutdown();
  imuFrameSynchronizer_.shutdown();
  positionMeasurementsReceived_.Shutdown();
  gpsPositionMeasurementsReceived_.Shutdown();

  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.at(i).join();
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.at(i).join();
  }
  imuConsumerThread_.join();
  positionConsumerThread_.join();
  gpsConsumerThread_.join();
  magnetometerConsumerThread_.join();
  differentialConsumerThread_.join();
  visualizationThread_.join();
  optimizationThread_.join();
  publisherThread_.join();

/*for (size_t i = 0; i < numCameras_; ++i) {
  okvis::kinematics::Transformation T_SC;
  estimator_.getCameraSensorStates(estimator_.currentFrameId(), i, T_SC);
  std::stringstream s;
  s << T_SC.T();
  LOG(INFO) << "T_SC[" << i <<"]:\n" << s.str();
}*/
#ifndef DEACTIVATE_TIMERS
  LOG(INFO) << okvis::timing::Timing::print();
#endif
}

// Add a new image.
bool ThreadedKFVio::addImage(const okvis::Time & stamp, size_t cameraIndex,
                             const cv::Mat & image,
                             const cv::Mat & depthImage,
                             const std::vector<cv::KeyPoint> * keypoints,
                             bool* /*asKeyframe*/) {
  assert(cameraIndex<numCameras_);

  if (lastAddedImageTimestamp_ > stamp
      && fabs((lastAddedImageTimestamp_ - stamp).toSec())
          > parameters_.sensors_information.frameTimestampTolerance) {
    LOG(ERROR)
        << "Received image from the past. Dropping the image.";
    return false;
  }
  lastAddedImageTimestamp_ = stamp;

  std::shared_ptr<okvis::CameraMeasurement> frame = std::make_shared<
      okvis::CameraMeasurement>();
  frame->measurement.image = image;
  frame->measurement.depthImage = depthImage;
  frame->timeStamp = stamp;
  frame->sensorId = cameraIndex;

  if (keypoints != nullptr) {
    frame->measurement.deliversKeypoints = true;
    frame->measurement.keypoints = *keypoints;
  } else {
    frame->measurement.deliversKeypoints = false;
  }

  if (blocking_) {
    cameraMeasurementsReceived_[cameraIndex]->PushBlockingIfFull(frame, 1);
    return true;
  } else {
    cameraMeasurementsReceived_[cameraIndex]->PushNonBlockingDroppingIfFull(
        frame, max_camera_input_queue_size);
    return cameraMeasurementsReceived_[cameraIndex]->Size() == 1;
  }
}

// Add a new image for target tracking.
bool ThreadedKFVio::addTrackingImage(const okvis::Time & stamp, size_t cameraIndex,
                              const cv::Mat & image) {
  std::shared_ptr<okvis::CameraMeasurement> frame = std::make_shared<
      okvis::CameraMeasurement>();
  frame->measurement.image = image;
  frame->timeStamp = stamp;
  frame->sensorId = cameraIndex;

  if (blocking_) {
    cameraMeasurementsReceived_[cameraIndex]->PushBlockingIfFull(frame, 1);
    return true;
  } else {
    cameraMeasurementsReceived_[cameraIndex]->PushNonBlockingDroppingIfFull(
        frame, 1);
    return cameraMeasurementsReceived_[cameraIndex]->Size() == 1;
  }
}

// Add an abstracted image observation.
bool ThreadedKFVio::addKeypoints(
    const okvis::Time & /*stamp*/, size_t /*cameraIndex*/,
    const std::vector<cv::KeyPoint> & /*keypoints*/,
    const std::vector<uint64_t> & /*landmarkIds*/,
    const cv::Mat & /*descriptors*/,
    bool* /*asKeyframe*/) {
  OKVIS_THROW(
      Exception,
      "ThreadedKFVio::addKeypoints() not implemented anymore since changes to _keypointMeasurements queue.");
  return false;
}

// Add an IMU measurement.
bool ThreadedKFVio::addImuMeasurement(const okvis::Time & stamp,
                                      const Eigen::Vector3d & alpha,
                                      const Eigen::Vector3d & omega) {

  okvis::ImuMeasurement imu_measurement;
  imu_measurement.measurement.accelerometers = alpha;
  imu_measurement.measurement.gyroscopes = omega;
  imu_measurement.timeStamp = stamp;

//std::cout << stamp.toSec() << " " << omega.transpose() << " " << alpha.transpose() << std::endl;

  if (blocking_) {
    imuMeasurementsReceived_.PushBlockingIfFull(imu_measurement, 1);
    return true;
  } else {
    imuMeasurementsReceived_.PushNonBlockingDroppingIfFull(
        imu_measurement, maxImuInputQueueSize_);
    return imuMeasurementsReceived_.Size() == 1;
  }
}

// Add a position measurement.
void ThreadedKFVio::addPositionMeasurement(const okvis::Time & stamp,
                                           const Eigen::Vector3d & position,
                                           const Eigen::Matrix3d & positionCovariance) {
  okvis::PositionMeasurement position_measurement;
  position_measurement.measurement.position = position;
  position_measurement.measurement.positionCovariance = positionCovariance;
  position_measurement.timeStamp = stamp;

  if (blocking_) {
    positionMeasurementsReceived_.PushBlockingIfFull(position_measurement, 1);
    return;
  } else {
    positionMeasurementsReceived_.PushNonBlockingDroppingIfFull(
        position_measurement, maxPositionInputQueueSize_);
    return;
  }
}

// // Add a GPS measurement.
// void ThreadedKFVio::addGpsMeasurement(const okvis::Time & stamp,
//                                       double lat_wgs84_deg, double lon_wgs84_deg,
//                                       double alt_wgs84,
//                                       const Eigen::Matrix3d & positionCovarianceENU) {
//   okvis::GpsPositionMeasurement position_measurement;
//   position_measurement.measurement.lat_wgs84 = lat_wgs84_deg;
//   position_measurement.measurement.lon_wgs84 = lon_wgs84_deg;
//   position_measurement.measurement.alt_wgs84 = alt_wgs84;
//   position_measurement.measurement.positionCovarianceENU = positionCovarianceENU;
//   position_measurement.timeStamp = stamp;

//   if (blocking_) {
//     gpsPositionMeasurementsReceived_.PushBlockingIfFull(position_measurement, 1);
//     return;
//   } else {
//     gpsPositionMeasurementsReceived_.PushNonBlockingDroppingIfFull(
//         position_measurement, maxGpsPositionInputQueueSize_);
//    return;
//   }
// }

// Add a magnetometer measurement.
void ThreadedKFVio::addMagnetometerMeasurement(const okvis::Time &,
                                               const Eigen::Vector3d &, double) {
  OKVIS_THROW(Exception, "Magnetometer measurements not supported")
}

// Add a static pressure measurement.
void ThreadedKFVio::addBarometerMeasurement(const okvis::Time &, double, double) {

  OKVIS_THROW(Exception, "Barometer measurements not supported")
}

// Add a differential pressure measurement.
void ThreadedKFVio::addDifferentialPressureMeasurement(const okvis::Time &,
                                                       double, double) {

  OKVIS_THROW(Exception, "Differential pressure measurements not supported")
}

// Set the blocking variable that indicates whether the addMeasurement() functions
// should return immediately (blocking=false), or only when the processing is complete.
void ThreadedKFVio::setBlocking(bool blocking) {
  blocking_ = blocking;
  // disable time limit for optimization
  if(blocking_) {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    estimator_.setOptimizationTimeLimit(-1.0,parameters_.optimization.max_iterations);
  }
}

// Loop to process frames from camera with index cameraIndex
void ThreadedKFVio::frameConsumerLoop(size_t cameraIndex) {
  std::shared_ptr<okvis::CameraMeasurement> frame;
  std::shared_ptr<okvis::MultiFrame> multiFrame;
  TimerSwitchable beforeDetectTimer("1.1 frameLoopBeforeDetect"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer("1.1.1 waitForFrameSynchronizerMutex"+std::to_string(cameraIndex),true);
  TimerSwitchable addNewFrameToSynchronizerTimer("1.1.2 addNewFrameToSynchronizer"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForStateVariablesMutexTimer("1.1.3 waitForStateVariablesMutex"+std::to_string(cameraIndex),true);
  TimerSwitchable propagationTimer("1.1.4 propagationTimer"+std::to_string(cameraIndex),true);
  TimerSwitchable detectTimer("1.2 detectAndDescribe"+std::to_string(cameraIndex),true);
  TimerSwitchable afterDetectTimer("1.3 afterDetect"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer2("1.3.1 waitForFrameSynchronizerMutex2"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForMatchingThreadTimer("1.4 waitForMatchingThread"+std::to_string(cameraIndex),true);


  for (;;) {
    // get data and check for termination request
    if (cameraMeasurementsReceived_[cameraIndex]->PopBlocking(&frame) == false) {
      return;
    }
    beforeDetectTimer.start();
    {  // lock the frame synchronizer
      waitForFrameSynchronizerMutexTimer.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer.stop();
      // add new frame to frame synchronizer and get the MultiFrame containing it
      addNewFrameToSynchronizerTimer.start();
      multiFrame = frameSynchronizer_.addNewFrame(frame);
      addNewFrameToSynchronizerTimer.stop();
    }  // unlock frameSynchronizer only now as we can be sure that not two states are added for the same timestamp
    okvis::kinematics::Transformation T_WS;
    okvis::Time lastTimestamp;
    okvis::SpeedAndBias speedAndBiases;
    // copy last state variables
    {
      waitForStateVariablesMutexTimer.start();
      std::lock_guard<std::mutex> lock(lastState_mutex_);
      waitForStateVariablesMutexTimer.stop();
      T_WS = lastOptimized_T_WS_;
      speedAndBiases = lastOptimizedSpeedAndBiases_;
      lastTimestamp = lastOptimizedStateTimestamp_;
    }

    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime = multiFrame->timestamp()
        + temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime = lastTimestamp - temporal_imu_data_overlap;

    OKVIS_ASSERT_TRUE_DBG(Exception,imuDataBeginTime < imuDataEndTime,"imu data end time is smaller than begin time.");

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(
      okvis::Time(imuDataEndTime)) == false)  {
      return;
    }
    OKVIS_ASSERT_TRUE_DBG(Exception,
                          imuDataEndTime < imuMeasurements_.back().timeStamp,
                          "Waiting for up to date imu data seems to have failed!");

    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,
                                                           imuDataEndTime);

    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0) {
      beforeDetectTimer.stop();
      continue;
    }

    if (imuData.front().timeStamp > frame->timeStamp) {
      LOG(WARNING) << "Frame is newer than oldest IMU measurement. Dropping it.";
      beforeDetectTimer.stop();
      continue;
    }

    // get T_WC(camIndx) for detectAndDescribe()
    if (estimator_.numFrames() == 0) {
      // first frame ever
      bool success = okvis::Estimator::initPoseFromImu(imuData, T_WS);
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        lastOptimized_T_WS_ = T_WS;
        lastOptimizedSpeedAndBiases_.setZero();
        lastOptimizedSpeedAndBiases_.segment<3>(6) = imu_params_.a0;
        lastOptimizedStateTimestamp_ = multiFrame->timestamp();
      }
      OKVIS_ASSERT_TRUE_DBG(Exception, success,
          "pose could not be initialized from imu measurements.");
      if (!success) {
        beforeDetectTimer.stop();
        continue;
      }
    } else {
      // get old T_WS
      propagationTimer.start();
      okvis::ceres::ImuError::propagation(imuData, parameters_.imu, T_WS,
                                          speedAndBiases, lastTimestamp,
                                          multiFrame->timestamp());
      propagationTimer.stop();
    }

    okvis::kinematics::Transformation T_WC = T_WS
        * (*parameters_.nCameraSystem.T_SC(frame->sensorId));
    beforeDetectTimer.stop();
    detectTimer.start();

//     if(cameraIndex==0) {
//       for(size_t t=0; t<parameters_.trackerNCameraSystem.numCameras(); ++t) {
//         okvis::kinematics::Transformation T_WC_trackerCam = T_WS
//             * (*parameters_.trackerNCameraSystem.T_SC(t));
//         std::shared_ptr<okvis::CameraMeasurement> fr;
//         markerMeasurements.at(t).reset(new MarkerMeasurement());
//         if(cameraMeasurementsReceived_[numCameras_+t]->PopNonBlocking(&fr)){
// // 	  std::cout << "received tracker image "  << numCameras_+t << std::endl;
//           kinematics::Transformation T_WT;
//           {
//             std::lock_guard<std::mutex> lock(trackerPrediction_mutex_);
//             estimator_.getPredictedTargetState(multiFrame->timestamp(), T_WT);
//             //std::cout << T_WT.T() << std::endl;
//           }
//           //std::cout << fr->measurement.image.cols << std::endl;
//           const okvis::kinematics::Transformation T_CT_init = T_WC_trackerCam.inverse()*T_WT;
// //          std::cout << std::endl << "frontend_.trackDetectTarget start!" << std::endl;
//           frontend_.trackDetectTarget(
//               fr->sensorId, fr->measurement.image, multiFrame->timestamp(), parameters_,
//               T_CT_init, markerMeasurements.at(t)->keypointMatches,
//               markerMeasurements.at(t)->T_CT,
//               markerMeasurements.at(t)->successfulRedetection);
// //          std::cout << "frontend_.trackDetectTarget done!" << std::endl << std::endl;
//           //std::cout <<"no. matches = " << markerMeasurements.at(t)->keypointMatches.size() << std::endl;
//         }
//       }
//     }
    //OKVIS_ASSERT_TRUE(Exception, frame->sensorId == cameraIndex,"very bad");
    if(!parameters_.nCameraSystem.isVirtual(frame->sensorId)) {
      frontend_.detectAndDescribe(frame->sensorId, multiFrame, T_WC, nullptr, nullptr);
    }
    detectTimer.stop();
    afterDetectTimer.start();

    bool push = false;
    {  // we now tell frame synchronizer that detectAndDescribe is done for MF with our timestamp
      waitForFrameSynchronizerMutexTimer2.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer2.stop();
      frameSynchronizer_.detectionEndedForMultiFrame(multiFrame->id());

      if (frameSynchronizer_.detectionCompletedForAllCameras(
          multiFrame->id())) {
//        LOG(INFO) << "detection completed for multiframe with id "<< multi_frame->id();
        push = true;
      }
    }  // unlocking frame synchronizer
    afterDetectTimer.stop();
    if (push) {
      // use queue size 1 to propagate a congestion to the _cameraMeasurementsReceived queue
      // and check for termination request
      waitForMatchingThreadTimer.start();
      if (keypointMeasurements_.PushBlockingIfFull(multiFrame, 1) == false) {
        return;
      }
      markerMeasurements_.PushNonBlockingDroppingIfFull(markerMeasurements, 1);
      waitForMatchingThreadTimer.stop();
    }
  }
}

// Loop that matches frames with existing frames.
void ThreadedKFVio::matchingLoop()
{
  TimerSwitchable prepareToAddStateTimer("2.1 prepareToAddState", true);
  TimerSwitchable waitForOptimizationTimer("2.2 waitForOptimization", true);
  TimerSwitchable addStateTimer("2.3 addState", true);
  TimerSwitchable matchingTimer("2.4 matching", true);

  for (;;) {
    // get new frame
    std::shared_ptr<okvis::MultiFrame> frame;
    std::vector<std::shared_ptr<MarkerMeasurement> > markers;

    // get data and check for termination request
    if (keypointMeasurements_.PopBlocking(&frame) == false)
      return;
    // get data and check for termination request
    markerMeasurements_.PopNonBlocking(&markers);

    prepareToAddStateTimer.start();
    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime = frame->timestamp() + temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime = lastAddedStateTimestamp_
        - temporal_imu_data_overlap;

    OKVIS_ASSERT_TRUE_DBG(Exception,imuDataBeginTime < imuDataEndTime,
        "imu data end time is smaller than begin time." <<
        "current frametimestamp " << frame->timestamp() << " (id: " << frame->id() <<
        "last timestamp         " << lastAddedStateTimestamp_ << " (id: " << estimator_.currentFrameId());

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(
        okvis::Time(imuDataEndTime)) == false)
      return;OKVIS_ASSERT_TRUE_DBG(Exception,
        imuDataEndTime < imuMeasurements_.back().timeStamp,
        "Waiting for up to date imu data seems to have failed!");

    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,
                                                           imuDataEndTime);

    prepareToAddStateTimer.stop();
    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0)
      continue;

    // make sure that optimization of last frame is over.
    // TODO If we didn't actually 'pop' the _matchedFrames queue until after optimization this would not be necessary
    {
      waitForOptimizationTimer.start();
      std::unique_lock<std::mutex> l(estimator_mutex_);
      while (!optimizationDone_)
        optimizationNotification_.wait(l);
      waitForOptimizationTimer.stop();
      addStateTimer.start();
      okvis::Time t0Matching = okvis::Time::now();
      bool asKeyframe = false;
      if (estimator_.addStates(frame, imuData, asKeyframe)) {
        lastAddedStateTimestamp_ = frame->timestamp();
        addStateTimer.stop();
      } else {
        LOG(ERROR)<< "Failed to add state! will drop multiframe.";
        addStateTimer.stop();
        continue;
      }

      // // -- add position measurements
      // {
      //   std::lock_guard<std::mutex> positionLock(positionMeasurements_mutex_);
      //   for (auto it = positionMeasurements_.begin();
      //       it != positionMeasurements_.end();) {
      //     bool success = false;
      //     if (it->timeStamp < frame->timestamp()) {
      //       success = estimator_.addPositionMeasurement(
      //           it->timeStamp, it->measurement.position,
      //           it->measurement.positionCovariance);
      //       if (!success) {
      //         LOG(WARNING)<< "Failed to add position measurement!";
      //       }
      //       it = positionMeasurements_.erase(it);
      //     } else {
      //       ++it;
      //     }
      //   }
      // }

      // // -- add GPS position measurements
      // {
      //   std::lock_guard<std::mutex> positionLock(
      //       gpsPositionMeasurements_mutex_);
      //   for (auto it = gpsPositionMeasurements_.begin();
      //       it != gpsPositionMeasurements_.end();) {
      //     bool success = false;
      //     if (it->timeStamp < frame->timestamp()) {
      //       //std::cout << "adding stamp " << frame->timestamp() << std::endl;
      //       success = estimator_.addGpsMeasurement(
      //           it->timeStamp, it->measurement.lat_wgs84,
      //           it->measurement.lon_wgs84, it->measurement.alt_wgs84,
      //           it->measurement.positionCovarianceENU);
      //       if (!success) {
      //         LOG(WARNING)<< "Failed to add gps position measurement!";
      //       }
      //       it = gpsPositionMeasurements_.erase(it);
      //     } else {
      //       ++it;
      //     }
      //   }
      // }

      // -- matching keypoints, initialising landmarks etc.
      okvis::kinematics::Transformation T_WS;
      estimator_.get_T_WS(frame->id(), T_WS);
      matchingTimer.start();
      frontend_.dataAssociationAndInitialization(estimator_, T_WS, parameters_,
                                                 map_, frame, &asKeyframe);
      // write tracks
      for (size_t i = 0; i < frame->numFrames(); ++i) {
        if (csvTracksFiles_[i]) {
          for (size_t k = 0; k < frame->numKeypoints(i); ++k) {
            cv::KeyPoint keypoint;
            frame->getCvKeypoint(i, k, keypoint);
            std::stringstream time;
            time << frame->timestamp().sec << std::setw(9)
                << std::setfill('0') << frame->timestamp().nsec;
            *csvTracksFiles_[i] << time.str() << ", " << std::setprecision(19)
                << frame->landmarkId(i, k) << ", " << std::scientific
                << std::setprecision(18) << keypoint.pt.x << ", "
                                         << keypoint.pt.y << ", "
                                         << keypoint.size / 16.0 << std::endl;
          }
        }
      }

      matchingTimer.stop();
      if (asKeyframe)
        estimator_.setKeyframe(frame->id(), asKeyframe);
      if (!blocking_) {
        double timeLimit = parameters_.optimization
            .timeLimitForMatchingAndOptimization
            - (okvis::Time::now() - t0Matching).toSec();
        estimator_.setOptimizationTimeLimit(
            std::max<double>(0.0, timeLimit),
            parameters_.optimization.min_iterations);
      }

      // for(size_t i=0; i<markers.size(); ++i) {
      //   //std::cout<<"."<<std::endl;
      //   //feed to estimator
      //   std::lock_guard<std::mutex> lock(trackerPrediction_mutex_);
      //   auto camera = std::static_pointer_cast<const cameras::PinholeCameraBase>(
      //       parameters_.trackerNCameraSystem.cameraGeometry(i))
      //           ->undistortedPinholeCamera();
      //   if(markers.at(i)->successfulRedetection) {
      //     estimator_.addTargetMeasurement(
      //         0, frame->id(), i + numCameras_,
      //         *parameters_.trackerNCameraSystem.T_SC(i), camera,
      //         markers.at(i)->keypointMatches, &markers.at(i)->T_CT);
      //   } else {
      //     estimator_.addTargetMeasurement(
      //         0, frame->id(), i + numCameras_,
      //         *parameters_.trackerNCameraSystem.T_SC(i), camera,
      //         markers.at(i)->keypointMatches);
      //   }
      // }


      optimizationDone_ = false;
    }  // unlock estimator_mutex_

    // use queue size 1 to propagate a congestion to the _matchedFrames queue
    if (matchedFrames_.PushBlockingIfFull(frame, 1) == false)
      return;
  }
}

// Loop to process IMU measurements.
void ThreadedKFVio::imuConsumerLoop() {
  okvis::ImuMeasurement data;
  TimerSwitchable processImuTimer("0 processImuMeasurements",true);
  for (;;) {
    // get data and check for termination request
    if (imuMeasurementsReceived_.PopBlocking(&data) == false)
      return;
    processImuTimer.start();
    okvis::Time start;
    const okvis::Time* end;  // do not need to copy end timestamp
    {
      std::lock_guard<std::mutex> imuLock(imuMeasurements_mutex_);
      OKVIS_ASSERT_TRUE(Exception,
                        imuMeasurements_.empty()
                        || imuMeasurements_.back().timeStamp < data.timeStamp,
                        "IMU measurement from the past received");

      if (parameters_.publishing.publishImuPropagatedState) {
        if (!repropagationNeeded_ && imuMeasurements_.size() > 0) {
          start = imuMeasurements_.back().timeStamp;
        } else if (repropagationNeeded_) {
          std::lock_guard<std::mutex> lastStateLock(lastState_mutex_);
          start = lastOptimizedStateTimestamp_;
          T_WS_propagated_ = lastOptimized_T_WS_;
          speedAndBiases_propagated_ = lastOptimizedSpeedAndBiases_;
          repropagationNeeded_ = false;
        } else
          start = okvis::Time(0, 0);
        end = &data.timeStamp;
      }
      imuMeasurements_.push_back(data);
    }  // unlock _imuMeasurements_mutex

    // notify other threads that imu data with timeStamp is here.
    imuFrameSynchronizer_.gotImuData(data.timeStamp);

    if (parameters_.publishing.publishImuPropagatedState) {
      Eigen::Matrix<double, 15, 15> covariance;
      Eigen::Matrix<double, 15, 15> jacobian;

      if(frontend_.isInitialized()) {
        // only propagate if initialised (will drift crazily otherwise)
        frontend_.propagation(imuMeasurements_, imu_params_, T_WS_propagated_,
                              speedAndBiases_propagated_, start, *end, &covariance,
                              &jacobian);
      }

      OptimizationResults result;
      result.stamp = *end;
      result.T_WS = T_WS_propagated_;
      {
        std::lock_guard<std::mutex> lastStateLock(lastState_mutex_);
        result.T_GW = T_GW_;
      }
      result.speedAndBiases = speedAndBiases_propagated_;
      result.omega_S = imuMeasurements_.back().measurement.gyroscopes
          - speedAndBiases_propagated_.segment<3>(3);
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        result.vector_of_T_SCi.push_back(
            okvis::kinematics::Transformation(
                *parameters_.nCameraSystem.T_SC(i)));
      }
      result.onlyPublishLandmarks = false;
      optimizationResults_.PushNonBlockingDroppingIfFull(result,1);
    }
    processImuTimer.stop();
  }
}

// Loop to process position measurements.
void ThreadedKFVio::positionConsumerLoop() {
  okvis::PositionMeasurement data;
  for (;;) {
    // get data and check for termination request
    if (positionMeasurementsReceived_.PopBlocking(&data) == false)
      return;
    // collect
    {
      std::lock_guard<std::mutex> positionLock(positionMeasurements_mutex_);
      positionMeasurements_.push_back(data);
    }
  }
}

// Loop to process GPS measurements.
void ThreadedKFVio::gpsConsumerLoop() {
  okvis::GpsPositionMeasurement data;
  for (;;) {
    // get data and check for termination request
    if (gpsPositionMeasurementsReceived_.PopBlocking(&data) == false)
      return;
    // collect
    {
      std::lock_guard<std::mutex> positionLock(gpsPositionMeasurements_mutex_);
      gpsPositionMeasurements_.push_back(data);
    }
  }
}

// Loop to process magnetometer measurements.
void ThreadedKFVio::magnetometerConsumerLoop() {
}

// Loop to process differential pressure measurements.
void ThreadedKFVio::differentialConsumerLoop() {
}

// Loop that visualizes completed frames.
void ThreadedKFVio::visualizationLoop() {
  okvis::VioVisualizer visualizer_(parameters_);
  for (;;) {
    VioVisualizer::VisualizationData::Ptr new_data;
    if (visualizationData_.PopBlocking(&new_data) == false)
      return;
    //visualizer_.showDebugImages(new_data);
    std::vector<cv::Mat> out_images(parameters_.nCameraSystem.numCameras());
    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
      out_images[i] = visualizer_.drawMatches(new_data, i);
    }
  displayImages_.PushNonBlockingDroppingIfFull(out_images,1);
  }
}

// trigger display (needed because OSX won't allow threaded display)
void ThreadedKFVio::display() {
  std::vector<cv::Mat> out_images;
  if (displayImages_.Size() == 0)
  return;
  if (displayImages_.PopBlocking(&out_images) == false)
    return;
  // draw
  for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
    std::stringstream windowname;
    windowname << "OKVIS camera " << im;
    cv::imshow(windowname.str(), out_images[im]);
  }

//// MBZIRC -hack
  //cv::imshow("detection", frontend_.debugImage);

  cv::waitKey(1);
}

// Get a subset of the recorded IMU measurements.
okvis::ImuMeasurementDeque ThreadedKFVio::getImuMeasurments(
    okvis::Time& imuDataBeginTime, okvis::Time& imuDataEndTime) {
  // sanity checks:
  // if end time is smaller than begin time, return empty queue.
  // if begin time is larger than newest imu time, return empty queue.
  if (imuDataEndTime < imuDataBeginTime
      || imuDataBeginTime > imuMeasurements_.back().timeStamp)
    return okvis::ImuMeasurementDeque();

  std::lock_guard<std::mutex> lock(imuMeasurements_mutex_);
  // get iterator to imu data before previous frame
  okvis::ImuMeasurementDeque::iterator first_imu_package = imuMeasurements_
      .begin();
  okvis::ImuMeasurementDeque::iterator last_imu_package =
      imuMeasurements_.end();
  // TODO go backwards through queue. Is probably faster.
  for (auto iter = imuMeasurements_.begin(); iter != imuMeasurements_.end();
      ++iter) {
    // move first_imu_package iterator back until iter->timeStamp is higher than requested begintime
    if (iter->timeStamp <= imuDataBeginTime)
      first_imu_package = iter;

    // set last_imu_package iterator as soon as we hit first timeStamp higher than requested endtime & break
    if (iter->timeStamp >= imuDataEndTime) {
      last_imu_package = iter;
      // since we want to include this last imu measurement in returned Deque we
      // increase last_imu_package iterator once.
      ++last_imu_package;
      break;
    }
  }

  // create copy of imu buffer
  return okvis::ImuMeasurementDeque(first_imu_package, last_imu_package);
}

// Remove IMU measurements from the internal buffer.
int ThreadedKFVio::deleteImuMeasurements(const okvis::Time& eraseUntil) {
  std::lock_guard<std::mutex> lock(imuMeasurements_mutex_);
  if (imuMeasurements_.front().timeStamp > eraseUntil)
    return 0;

  okvis::ImuMeasurementDeque::iterator eraseEnd;
  int removed = 0;
  for (auto it = imuMeasurements_.begin(); it != imuMeasurements_.end(); ++it) {
    eraseEnd = it;
    if (it->timeStamp >= eraseUntil)
      break;
    ++removed;
  }

  imuMeasurements_.erase(imuMeasurements_.begin(), eraseEnd);

  return removed;
}

// Loop that performs the optimization and marginalisation.
void ThreadedKFVio::optimizationLoop() {
  TimerSwitchable optimizationTimer("3.1 optimization",true);
  TimerSwitchable marginalizationTimer("3.2 marginalization",true);
  TimerSwitchable afterOptimizationTimer("3.3 afterOptimization",true);

  for (;;) {
    std::shared_ptr<okvis::MultiFrame> frame_pairs;
    VioVisualizer::VisualizationData::Ptr visualizationDataPtr;
    okvis::Time deleteImuMeasurementsUntil(0, 0);
    if (matchedFrames_.PopBlocking(&frame_pairs) == false)
      return;
    OptimizationResults result;
    {
      std::lock_guard<std::mutex> l(estimator_mutex_);
      optimizationTimer.start();
      //if(frontend_.isInitialized()){
        estimator_.optimize(parameters_.optimization.max_iterations, std::max(1,OKVIS_N_THREADS/2), false);
      //}
      /*if (estimator_.numFrames() > 0 && !frontend_.isInitialized()){
        // undo translation
        for(size_t n=0; n<estimator_.numFrames(); ++n){
          okvis::kinematics::Transformation T_WS_0;
          estimator_.get_T_WS(estimator_.frameIdByAge(n),T_WS_0);
          Eigen::Matrix4d T_WS_0_mat = T_WS_0.T();
          T_WS_0_mat.topRightCorner<3,1>().setZero();
          estimator_.set_T_WS(estimator_.frameIdByAge(n),okvis::kinematics::Transformation(T_WS_0_mat));
          okvis::SpeedAndBias sb_0 = okvis::SpeedAndBias::Zero();
          if(estimator_.getSpeedAndBias(estimator_.frameIdByAge(n), 0, sb_0)){
            sb_0.head<3>().setZero();
            estimator_.setSpeedAndBias(estimator_.frameIdByAge(n), 0, sb_0);
          }
        }
      }*/

      optimizationTimer.stop();

      // get timestamp of last frame in IMU window. Need to do this before marginalization as it will be removed there (if not keyframe)
      if (estimator_.numFrames()
          > size_t(parameters_.optimization.numImuFrames)) {
        deleteImuMeasurementsUntil = estimator_.multiFrame(
            estimator_.frameIdByAge(parameters_.optimization.numImuFrames))
            ->timestamp() - temporal_imu_data_overlap;
      }

      marginalizationTimer.start();
      estimator_.applyMarginalizationStrategy(
          parameters_.optimization.numKeyframes,
          parameters_.optimization.numImuFrames, result.transferredLandmarks);
      marginalizationTimer.stop();
      afterOptimizationTimer.start();

      // now actually remove measurements
      deleteImuMeasurements(deleteImuMeasurementsUntil);

      // saving optimized state and saving it in OptimizationResults struct
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        estimator_.get_T_WS(frame_pairs->id(), lastOptimized_T_WS_);
        estimator_.getSpeedAndBias(frame_pairs->id(), 0,
                                   lastOptimizedSpeedAndBiases_);
        estimator_.get_T_GW(frame_pairs->id(), T_GW_);
        result.T_GW = T_GW_;
        lastOptimizedStateTimestamp_ = frame_pairs->timestamp();

        // if we publish the state after each IMU propagation we do not need to publish it here.
        if (!parameters_.publishing.publishImuPropagatedState) {
          result.T_WS = lastOptimized_T_WS_;
          result.speedAndBiases = lastOptimizedSpeedAndBiases_;
          result.stamp = lastOptimizedStateTimestamp_;
          result.onlyPublishLandmarks = false;
        }
        else
          result.onlyPublishLandmarks = true;
        double velocityUncertainty = 0.0;
        result.T_WT_success = estimator_.get_T_WT(frame_pairs->id(), 0,result.T_WT,result.v_WT_W, result.omega_WT_W, velocityUncertainty);
        if(velocityUncertainty > 1.0) {
          result.T_WT_success = false; // make sure to have a proper speed estimate before outputing this
        }
        estimator_.getLandmarks(result.landmarksVector);

        // copy a whole lot of stuff, if a dense stereo callback is registered.
        if(denseStereoCallback_) {
          for (size_t k=0; k< estimator_.numFrames(); ++k) {
            FrameInfo info;
            uint64_t frameId = estimator_.frameIdByAge(estimator_.numFrames()-k-1);
            info.timestamp = estimator_.timestamp(frameId);
            estimator_.get_T_WS(frameId, info.T_WS);
            const size_t numCameras = parameters_.nCameraSystem.numCameras();
            info.images.resize(numCameras);
            info.depthImages.resize(numCameras);
            info.T_SC_i.resize(numCameras);
            for(size_t i=0; i<numCameras; ++i) {
              estimator_.getCameraSensorStates(frameId,i,info.T_SC_i.at(i));
              info.images.at(i) = estimator_.multiFrame(frameId)->image(i);
              if(parameters_.nCameraSystem.isDepthCamera(i)) {
                info.depthImages.at(i) = estimator_.multiFrame(frameId)->depthImage(i);
              }
            }
            info.isKeyframe = estimator_.isKeyframe(frameId);
            result.frameInfos.push_back(info);
          }
        }

        repropagationNeeded_ = true;
      }

      if (parameters_.visualization.displayImages) {
        // fill in information that requires access to estimator.
        visualizationDataPtr = VioVisualizer::VisualizationData::Ptr(
            new VioVisualizer::VisualizationData());
        visualizationDataPtr->observations.resize(frame_pairs->numKeypoints());
        okvis::MapPoint landmark;
        okvis::ObservationVector::iterator it = visualizationDataPtr
            ->observations.begin();
        for (size_t camIndex = 0; camIndex < frame_pairs->numFrames();
            ++camIndex) {
          for (size_t k = 0; k < frame_pairs->numKeypoints(camIndex); ++k) {
            OKVIS_ASSERT_TRUE_DBG(Exception,it != visualizationDataPtr->observations.end(),"Observation-vector not big enough");
            it->keypointIdx = k;
            frame_pairs->getKeypoint(camIndex, k, it->keypointMeasurement);
            frame_pairs->getKeypointSize(camIndex, k, it->keypointSize);
            it->cameraIdx = camIndex;
            it->frameId = frame_pairs->id();
            it->landmarkId = frame_pairs->landmarkId(camIndex, k);
            if (estimator_.isLandmarkAdded(it->landmarkId)) {
              estimator_.getLandmark(it->landmarkId, landmark);
              it->landmark_W = landmark.point;
              if (estimator_.isLandmarkInitialized(it->landmarkId))
                it->isInitialized = true;
              else
                it->isInitialized = false;
            } else {
              it->landmark_W = Eigen::Vector4d(0, 0, 0, 0);  // set to infinity to tell visualizer that landmark is not added
            }
            ++it;
          }
        }
        visualizationDataPtr->keyFrames = estimator_.multiFrame(
            estimator_.currentKeyframeId());
        estimator_.get_T_WS(estimator_.currentKeyframeId(),
                            visualizationDataPtr->T_WS_keyFrame);
      }

      optimizationDone_ = true;
    }  // unlock mutex
    optimizationNotification_.notify_all();

    if (!parameters_.publishing.publishImuPropagatedState) {
      // adding further elements to result that do not access estimator.
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        result.vector_of_T_SCi.push_back(
            okvis::kinematics::Transformation(
                *parameters_.nCameraSystem.T_SC(i)));
      }
    }
    optimizationResults_.Push(result);

    // adding further elements to visualization data that do not access estimator
    if (parameters_.visualization.displayImages) {
      visualizationDataPtr->currentFrames = frame_pairs;
      visualizationData_.PushNonBlockingDroppingIfFull(visualizationDataPtr, 1);
    }
    afterOptimizationTimer.stop();
  }
}

// Loop that publishes the newest state and landmarks.
void ThreadedKFVio::publisherLoop() {
  for (;;) {
    // get the result data
    OptimizationResults result;
    if (optimizationResults_.PopBlocking(&result) == false)
      return;

    // call all user callbacks
    if(parameters_.publishing.referenceFrame == FrameName::G) {
      // we need to change stuff into G frame
std::cout << "wrong" << std::endl;
      result.T_WS = result.T_GW * result.T_WS;
      result.speedAndBiases.head<3>() = result.T_GW.C() * result.speedAndBiases.head<3>();
    }
    if (stateCallback_ && !result.onlyPublishLandmarks)
      stateCallback_(result.stamp, result.T_WS);
    if (fullStateCallback_ && !result.onlyPublishLandmarks)
      fullStateCallback_(result.stamp, result.T_WS, result.speedAndBiases,
                         result.omega_S);
    if (fullStateCallbackWithExtrinsics_ && !result.onlyPublishLandmarks)
      fullStateCallbackWithExtrinsics_(result.stamp, result.T_WS,
                                       result.speedAndBiases, result.omega_S,
                                       result.vector_of_T_SCi);
    if(denseStereoCallback_ && !result.frameInfos.empty()){
      denseStereoCallback_(result.frameInfos);
    }

    if (landmarksCallback_ && !result.landmarksVector.empty()) {
        if(parameters_.publishing.referenceFrame == FrameName::G){
            for(size_t i=0; i<result.landmarksVector.size(); ++i){
                result.landmarksVector[i].point = result.T_GW*result.landmarksVector[i].point;
            }
            for(size_t i=0; i<result.transferredLandmarks.size(); ++i){
                result.transferredLandmarks[i].point = result.T_GW*result.transferredLandmarks[i].point;//*(1.0/result.transferredLandmarks[i].point[3]));
            }
        }
        landmarksCallback_(result.stamp, result.landmarksVector,
                           result.transferredLandmarks);  //TODO(gohlp): why two maps?
    }
    if(markerCallback_ && result.onlyPublishLandmarks && result.T_WT_success) {
      if(parameters_.publishing.referenceFrame == FrameName::G) {
        markerCallback_(result.stamp, result.T_GW * result.T_WT, result.T_GW.C() * result.v_WT_W, result.T_GW.C() * result.omega_WT_W);
      } else {
        markerCallback_(result.stamp, result.T_WT, result.v_WT_W, result.omega_WT_W);
      }
    }
  }
}

}  // namespace okvis
