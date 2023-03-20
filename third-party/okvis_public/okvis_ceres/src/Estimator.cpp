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
 *  Created on: Dec 30, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file Estimator.cpp
 * @brief Source file for the Estimator class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <glog/logging.h>
#include <okvis/Estimator.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/PositionError.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/ceres/ode/ode.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor if a ceres map is already available.
Estimator::Estimator(std::shared_ptr<okvis::ceres::Map> mapPtr)
    : mapPtr_(mapPtr),
      referencePoseId_(0),
      cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      huberLossFunctionPtr_(new ::ceres::HuberLoss(1)),
      marginalizationResidualId_(0) {
  keypointLocations_.col(0) = Eigen::Vector2d(-0.0707 / 6.66, 0.0);
  keypointLocations_.col(1) = Eigen::Vector2d(0.0707 / 6.66, 0.0);
  keypointLocations_.col(2) = Eigen::Vector2d(0.0, -0.0707 / 6.66);
  keypointLocations_.col(3) = Eigen::Vector2d(0.0, 0.0707 / 6.66);
}

// The default constructor.
Estimator::Estimator()
    : mapPtr_(new okvis::ceres::Map()),
      referencePoseId_(0),
      cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      huberLossFunctionPtr_(new ::ceres::HuberLoss(1)),
      marginalizationResidualId_(0) {
  keypointLocations_.col(0) = Eigen::Vector2d(-0.0707 / 6.66, 0.0);
  keypointLocations_.col(1) = Eigen::Vector2d(0.0707 / 6.66, 0.0);
  keypointLocations_.col(2) = Eigen::Vector2d(0.0, -0.0707 / 6.66);
  keypointLocations_.col(3) = Eigen::Vector2d(0.0, 0.0707 / 6.66);
}

Estimator::~Estimator() {}

// Add a camera to the configuration. Sensors can only be added and never removed.
int Estimator::addCamera(const ExtrinsicsEstimationParameters& extrinsicsEstimationParameters) {
  extrinsicsEstimationParametersVec_.push_back(extrinsicsEstimationParameters);
  return extrinsicsEstimationParametersVec_.size() - 1;
}

// Add an IMU to the configuration.
int Estimator::addImu(const ImuParameters& imuParameters) {
  if (imuParametersVec_.size() > 1) {
    LOG(ERROR) << "only one IMU currently supported";
    return -1;
  }
  imuParametersVec_.push_back(imuParameters);
  return imuParametersVec_.size() - 1;
}

// Add a position sensor to the configuration.

int Estimator::addGpsPositionSensor(const okvis::PositionSensorParameters& positionSensorParameters) {
  if (positionSensorParametersVec_.size() > 1) {
    LOG(ERROR) << "only one position sensor currently supported";
    return -1;
  }
  positionSensorParametersVec_.push_back(positionSensorParameters);
  return positionSensorParametersVec_.size() - 1;
}

// Add a GPS to the configuration.
int Estimator::addGps(const okvis::GpsParameters& gpsParameters) {
  if (gpsParametersVec_.size() > 1) {
    LOG(ERROR) << "only one GPS currently supported";
    return -1;
  }
  gpsParametersVec_.push_back(gpsParameters);
  localCartesian_.Reset(gpsParameters.lat0, gpsParameters.lon0, gpsParameters.alt0);
  return gpsParametersVec_.size() - 1;
}

// Remove all cameras from the configuration
void Estimator::clearCameras() { extrinsicsEstimationParametersVec_.clear(); }

// Remove all IMUs from the configuration.
void Estimator::clearImus() { imuParametersVec_.clear(); }

// Remove all GPSs from the configuration.
void Estimator::clearGpss() { gpsParametersVec_.clear(); }

// Add a pose to the state.
bool Estimator::addStates(okvis::MultiFramePtr multiFrame, const okvis::ImuMeasurementDeque& imuMeasurements, bool asKeyframe) {
  // note: this is before matching...
  // TODO !!
  okvis::kinematics::Transformation T_WS;
  okvis::SpeedAndBias speedAndBias;
  if (statesMap_.empty()) {
    // in case this is the first frame ever, let's initialize the pose:
    bool success0 = initPoseFromImu(imuMeasurements, T_WS);
    OKVIS_ASSERT_TRUE_DBG(Exception, success0, "pose could not be initialized from imu measurements.");
    if (!success0) return false;
    speedAndBias.setZero();
    speedAndBias.segment<3>(6) = imuParametersVec_.at(0).a0;
  } else {
    // get the previous states
    uint64_t T_WS_id = statesMap_.rbegin()->second.id;
    uint64_t speedAndBias_id = statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).at(0).at(ImuSensorStates::SpeedAndBias).id;
    OKVIS_ASSERT_TRUE_DBG(Exception, mapPtr_->parameterBlockExists(T_WS_id), "this is an okvis bug. previous pose does not exist.");
    T_WS = std::static_pointer_cast<ceres::PoseParameterBlock>(mapPtr_->parameterBlockPtr(T_WS_id))->estimate();
    // OKVIS_ASSERT_TRUE_DBG(
    //    Exception, speedAndBias_id,
    //    "this is an okvis bug. previous speedAndBias does not exist.");
    speedAndBias = std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(mapPtr_->parameterBlockPtr(speedAndBias_id))->estimate();

    // propagate pose and speedAndBias
    int numUsedImuMeasurements = ceres::ImuError::propagation(imuMeasurements, imuParametersVec_.at(0), T_WS, speedAndBias,
                                                              statesMap_.rbegin()->second.timestamp, multiFrame->timestamp());
    OKVIS_ASSERT_TRUE_DBG(Exception, numUsedImuMeasurements > 1, "propagation failed");
    if (numUsedImuMeasurements < 1) {
      LOG(INFO) << "numUsedImuMeasurements=" << numUsedImuMeasurements;
      return false;
    }
  }

  // create a states object:
  States states(asKeyframe, multiFrame->id(), multiFrame->timestamp());

  // check if id was used before
  OKVIS_ASSERT_TRUE_DBG(Exception, statesMap_.find(states.id) == statesMap_.end(), "pose ID" << states.id << " was used before!");

  // create global states
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock(new okvis::ceres::PoseParameterBlock(T_WS, states.id, multiFrame->timestamp()));
  states.global.at(GlobalStates::T_WS).exists = true;
  states.global.at(GlobalStates::T_WS).id = states.id;

  if (statesMap_.empty()) {
    referencePoseId_ = states.id;  // set this as reference pose
    if (!mapPtr_->addParameterBlock(poseParameterBlock, ceres::Map::Pose6d)) {
      return false;
    }
  } else {
    if (!mapPtr_->addParameterBlock(poseParameterBlock, ceres::Map::Pose6d)) {
      return false;
    }
  }

  // add to buffer
  statesMap_.insert(std::pair<uint64_t, States>(states.id, states));
  multiFramePtrMap_.insert(std::pair<uint64_t, okvis::MultiFramePtr>(states.id, multiFrame));

  // the following will point to the last states:
  std::map<uint64_t, States>::reverse_iterator lastElementIterator = statesMap_.rbegin();
  lastElementIterator++;

  // initialize new sensor states
  // cameras:
  for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
    SpecificSensorStatesContainer cameraInfos(2);
    cameraInfos.at(CameraSensorStates::T_SCi).exists = true;
    cameraInfos.at(CameraSensorStates::Intrinsics).exists = false;
    if (((extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_translation < 1e-12) ||
         (extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_orientation < 1e-12)) &&
        (statesMap_.size() > 1)) {
      // use the same block...
      cameraInfos.at(CameraSensorStates::T_SCi).id = lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id;
    } else {
      const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicsParameterBlockPtr(new okvis::ceres::PoseParameterBlock(T_SC, id, multiFrame->timestamp()));
      if (!mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr, ceres::Map::Pose6d)) {
        return false;
      }
      cameraInfos.at(CameraSensorStates::T_SCi).id = id;
    }
    // update the states info
    statesMap_.rbegin()->second.sensors.at(SensorStates::Camera).push_back(cameraInfos);
    states.sensors.at(SensorStates::Camera).push_back(cameraInfos);
  }

  // IMU states are automatically propagated.
  for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
    SpecificSensorStatesContainer imuInfo(2);
    imuInfo.at(ImuSensorStates::SpeedAndBias).exists = true;
    uint64_t id = IdProvider::instance().newId();
    std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock(
        new okvis::ceres::SpeedAndBiasParameterBlock(speedAndBias, id, multiFrame->timestamp()));

    if (!mapPtr_->addParameterBlock(speedAndBiasParameterBlock)) {
      return false;
    }
    imuInfo.at(ImuSensorStates::SpeedAndBias).id = id;
    statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).push_back(imuInfo);
    states.sensors.at(SensorStates::Imu).push_back(imuInfo);
  }

  // depending on whether or not this is the very beginning, we will add priors or relative terms to the last state:
  if (statesMap_.size() == 1) {
    // let's add a prior
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
    information(5, 5) = 1.0e6;
    information(0, 0) = 1.0e4;
    information(1, 1) = 1.0e4;
    information(2, 2) = 1.0e4;
    std::shared_ptr<ceres::PoseError> poseError(new ceres::PoseError(T_WS, information));
    /*auto id2= */ mapPtr_->addResidualBlock(poseError, NULL, poseParameterBlock);
    // std::cout << "================== fixing pose ======================" << std::endl;
    // mapPtr_->isJacobianCorrect(id2,1.0e-6);

    // sensor states
    for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
      double translationStdev = extrinsicsEstimationParametersVec_.at(i).sigma_absolute_translation;
      double translationVariance = translationStdev * translationStdev;
      double rotationStdev = extrinsicsEstimationParametersVec_.at(i).sigma_absolute_orientation;
      double rotationVariance = rotationStdev * rotationStdev;
      if (translationVariance > 1.0e-16 && rotationVariance > 1.0e-16) {
        const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
        std::shared_ptr<ceres::PoseError> cameraPoseError(new ceres::PoseError(T_SC, translationVariance, rotationVariance));
        // add to map
        mapPtr_->addResidualBlock(cameraPoseError, NULL,
                                  mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id));
        // mapPtr_->isJacobianCorrect(id,1.0e-6);
      } else {
        mapPtr_->setParameterBlockConstant(states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id);
      }
    }
    for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
      Eigen::Matrix<double, 6, 1> variances;
      // get these from parameter file
      const double sigma_bg = imuParametersVec_.at(0).sigma_bg;
      const double sigma_ba = imuParametersVec_.at(0).sigma_ba;
      std::shared_ptr<ceres::SpeedAndBiasError> speedAndBiasError(new ceres::SpeedAndBiasError(speedAndBias, 0.01, sigma_bg * sigma_bg, sigma_ba * sigma_ba));
      // add to map
      mapPtr_->addResidualBlock(speedAndBiasError, NULL,
                                mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
      // std::cout << "================== fixing speed and bias ======================" << speedAndBias.transpose() << std::endl;
      // mapPtr_->isJacobianCorrect(id,1.0e-6);
    }
  } else {
    // add IMU error terms
    for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
      std::shared_ptr<ceres::ImuError> imuError(
          new ceres::ImuError(imuMeasurements, imuParametersVec_.at(i), lastElementIterator->second.timestamp, states.timestamp));
      /*::ceres::ResidualBlockId id = */ mapPtr_->addResidualBlock(
          imuError, NULL, mapPtr_->parameterBlockPtr(lastElementIterator->second.id),
          mapPtr_->parameterBlockPtr(lastElementIterator->second.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id),
          mapPtr_->parameterBlockPtr(states.id), mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
      // imuError->setRecomputeInformation(false);
      // mapPtr_->isJacobianCorrect(id,1.0e-9);
      // imuError->setRecomputeInformation(true);
    }

    // add relative sensor state errors
    for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
      if (lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id !=
          states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id) {
        // i.e. they are different estimated variables, so link them with a temporal error term
        double dt = (states.timestamp - lastElementIterator->second.timestamp).toSec();
        double translationSigmaC = extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_translation;
        double translationVariance = translationSigmaC * translationSigmaC * dt;
        double rotationSigmaC = extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_orientation;
        double rotationVariance = rotationSigmaC * rotationSigmaC * dt;
        std::shared_ptr<ceres::RelativePoseError> relativeExtrinsicsError(new ceres::RelativePoseError(translationVariance, rotationVariance));
        mapPtr_->addResidualBlock(
            relativeExtrinsicsError, NULL,
            mapPtr_->parameterBlockPtr(lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id),
            mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id));
        // mapPtr_->isJacobianCorrect(id,1.0e-6);
      }
    }
    // only camera. this is slightly inconsistent, since the IMU error term contains both
    // a term for global states as well as for the sensor-internal ones (i.e. biases).
    // TODO: magnetometer, pressure, ...
  }

  return true;
}

// Add a landmark.
bool Estimator::addLandmark(uint64_t landmarkId, const Eigen::Vector4d& landmark) {
  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> pointParameterBlock(new okvis::ceres::HomogeneousPointParameterBlock(landmark, landmarkId));
  if (!mapPtr_->addParameterBlock(pointParameterBlock, okvis::ceres::Map::HomogeneousPoint)) {
    return false;
  }

  // remember
  double dist = std::numeric_limits<double>::max();
  if (fabs(landmark[3]) > 1.0e-8) {
    dist = (landmark / landmark[3]).head<3>().norm();  // euclidean distance
  }
  landmarksMap_.insert(std::pair<uint64_t, MapPoint>(landmarkId, MapPoint(landmarkId, landmark, 0.0, dist)));
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "bug: inconsistend landmarkdMap_ with mapPtr_.");
  return true;
}

// Remove an observation from a landmark.
bool Estimator::removeObservation(::ceres::ResidualBlockId residualBlockId) {
  const ceres::Map::ParameterBlockCollection parameters = mapPtr_->parameters(residualBlockId);
  const uint64_t landmarkId = parameters.at(1).first;
  // remove in landmarksMap
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);
  for (std::map<okvis::KeypointIdentifier, uint64_t>::iterator it = mapPoint.observations.begin(); it != mapPoint.observations.end();) {
    if (it->second == uint64_t(residualBlockId)) {
      it = mapPoint.observations.erase(it);
    } else {
      it++;
    }
  }
  // remove residual block
  mapPtr_->removeResidualBlock(residualBlockId);
  return true;
}

// Remove an observation from a landmark, if available.
bool Estimator::removeObservation(uint64_t landmarkId, uint64_t poseId, size_t camIdx, size_t keypointIdx) {
  if (landmarksMap_.find(landmarkId) == landmarksMap_.end()) {
    for (PointMap::iterator it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      LOG(INFO) << it->first << ", no. obs = " << it->second.observations.size();
    }
    LOG(INFO) << landmarksMap_.at(landmarkId).id;
  }
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "landmark not added");

  okvis::KeypointIdentifier kid(poseId, camIdx, keypointIdx);
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);
  std::map<okvis::KeypointIdentifier, uint64_t>::iterator it = mapPoint.observations.find(kid);
  if (it == landmarksMap_.at(landmarkId).observations.end()) {
    return false;  // observation not present
  }

  // remove residual block
  mapPtr_->removeResidualBlock(reinterpret_cast<::ceres::ResidualBlockId>(it->second));

  // remove also in local map
  mapPoint.observations.erase(it);

  return true;
}

// // Add a position measurement.
// bool Estimator::addPositionMeasurement(const okvis::Time& stamp, const Eigen::Vector3d& position, const Eigen::Matrix3d& positionCovariance) {
//   if (statesMap_.size() < 2) {
//     return false;
//   }
//   // find the right point in time
//   auto it = statesMap_.rbegin();
//   auto it2 = statesMap_.rbegin();
//   it2++;
//   bool found = false;
//   while (it2 != statesMap_.rend()) {
//     if (it2->second.timestamp <= stamp && it->second.timestamp >= stamp) {
//       found = true;
//       break;
//     }
//     ++it;
//     ++it2;
//   }

//   // make sure we found the right point
//   if (!found) {
//     return false;  // we probably want a boolean return value, so return false here.
//   }

//   // get the related IMU measurements
//   ImuMeasurementDeque imuMeasurements;
//   auto res = mapPtr_->residuals(it2->first);
//   for (auto resit = res.begin(); resit != res.end(); ++resit) {
//     std::shared_ptr<okvis::ceres::ImuError> imuError = std::dynamic_pointer_cast<okvis::ceres::ImuError>(resit->errorInterfacePtr);
//     if (imuError) {
//       if (imuError->imuMeasurements().front().timeStamp < stamp && imuError->imuMeasurements().back().timeStamp > stamp) {
//         imuMeasurements = imuError->imuMeasurements();
//       }
//     }
//   }

//   // make sure we fount IMU measurements
//   if (!(imuMeasurements.size() > 0)) {
//     return false;  // we probably want a boolean return value, so return false here.
//   }

//   // create the error term
//   std::shared_ptr<::ceres::CostFunction> positionError(
//       new okvis::ceres::PositionError(position, positionCovariance, okvis::PositionSensorParameters{gpsParametersVec_.at(0).antennaOffset, true},
//                                       imuMeasurements, imuParametersVec_.at(0), it2->second.timestamp, stamp));

//   // create global pose alignemnt if not existing
//   if (!positionSensorAlignmentParameterBlock_) {
//     positionSensorAlignmentParameterBlock_.reset(new ceres::PoseParameterBlock(okvis::kinematics::Transformation(position, Eigen::Quaterniond::Identity()),
//                                                                                IdProvider::instance::newId(), okvis::Time(0)));
//     mapPtr_->addParameterBlock(positionSensorAlignmentParameterBlock_, okvis::ceres::Map::Pose4d);
//     // mapPtr_->setParameterBlockConstant(positionSensorAlignmentParameterBlock_);
//   }

//   // obtain all parameter blocks
//   std::shared_ptr<ceres::ParameterBlock> speedAndBiasParameterBlockPtr;
//   getSensorStateParameterBlockPtr(it2->first, 0, SensorStates::Imu, ImuSensorStates::SpeedAndBias, speedAndBiasParameterBlockPtr);

//   mapPtr_->addResidualBlock(positionError, huberLossFunctionPtr_.get(), mapPtr_->parameterBlockPtr(it2->second.id), speedAndBiasParameterBlockPtr,
//                             positionSensorAlignmentParameterBlock_);

//   // std::cout << "added" << positionCovariance.diagonal().transpose() << std::endl;
//   std::cout << "added " << position.transpose() << " at t=" << stamp << std::endl;

//   return true;
// }

// // Add a GPS measurement.
// bool Estimator::addGpsMeasurement(const okvis::Time& stamp, double lat_wgs84_deg, double lon_wgs84_deg, double alt_wgs84,
//                                   const Eigen::Matrix3d& positionCovarianceENU) {
//   // std::cout << "received position measurement for timestamp "<<stamp<<std::endl;
//   // std::cout << "wgs84 :      " << lat_wgs84_deg << "N " << lon_wgs84_deg << "E " << alt_wgs84 << " m" << std::endl;
//   double x, y, z;
//   localCartesian_.Forward(lat_wgs84_deg, lon_wgs84_deg, alt_wgs84, x, y, z);
//   // std::cout << "Cartesian  : " << x << ", " << y << ", " << z << std::endl;
//   // std::cout << "Variances  : " << positionCovarianceENU(0,0) << ", " << positionCovarianceENU(1,1) << ", " << positionCovarianceENU(2,2) << std::endl;
//   const Eigen::Vector3d position(x, y, z);

//   if (statesMap_.size() < 2) {
//     return false;
//   }
//   // find the right point in time
//   auto it = statesMap_.rbegin();
//   auto it2 = statesMap_.rbegin();
//   it2++;
//   bool found = false;
//   while (it2 != statesMap_.rend()) {
//     if (it2->second.timestamp <= stamp && it->second.timestamp >= stamp) {
//       found = true;
//       break;
//     }
//     ++it;
//     ++it2;
//   }

//   // make sure we found the right point
//   if (!found) {
//     return false;  // we probably want a boolean return value, so return false here.
//   }

//   // get the related IMU measurements
//   ImuMeasurementDeque imuMeasurements;
//   auto res = mapPtr_->residuals(it2->first);
//   for (auto resit = res.begin(); resit != res.end(); ++resit) {
//     std::shared_ptr<okvis::ceres::ImuError> imuError = std::dynamic_pointer_cast<okvis::ceres::ImuError>(resit->errorInterfacePtr);
//     if (imuError) {
//       if (imuError->imuMeasurements().front().timeStamp < stamp && imuError->imuMeasurements().back().timeStamp > stamp) {
//         imuMeasurements = imuError->imuMeasurements();
//       }
//     }
//   }

//   // make sure we fount IMU measurements
//   if (!(imuMeasurements.size() > 0)) {
//     return false;  // we probably want a boolean return value, so return false here.
//   }

//   // create the error term
//   std::shared_ptr<::ceres::CostFunction> positionError(
//       new okvis::ceres::PositionError(position, positionCovarianceENU, okvis::PositionSensorParameters{gpsParametersVec_.at(0).antennaOffset, true},
//                                       imuMeasurements, imuParametersVec_.at(0), it2->second.timestamp, stamp));

//   // create global pose alignemnt if not existing
//   bool isUnremovable = false;
//   if (!positionSensorAlignmentParameterBlock_) {
//     positionSensorAlignmentParameterBlock_.reset(new ceres::PoseParameterBlock(okvis::kinematics::Transformation(position, Eigen::Quaterniond::Identity()),
//                                                                                IdProvider::instance::newId(), okvis::Time(0)));
//     mapPtr_->addParameterBlock(positionSensorAlignmentParameterBlock_, okvis::ceres::Map::Pose4d);
//     isUnremovable = true;
//     // mapPtr_->setParameterBlockConstant(positionSensorAlignmentParameterBlock_);
//   }

//   // obtain all parameter blocks
//   std::shared_ptr<ceres::ParameterBlock> speedAndBiasParameterBlockPtr;
//   getSensorStateParameterBlockPtr(it2->first, 0, SensorStates::Imu, ImuSensorStates::SpeedAndBias, speedAndBiasParameterBlockPtr);

//   mapPtr_->addResidualBlock(positionError, huberLossFunctionPtr_.get(), mapPtr_->parameterBlockPtr(it2->second.id), speedAndBiasParameterBlockPtr,
//                             positionSensorAlignmentParameterBlock_);

//   if (isUnremovable) {
//     std::cout << "##############################unremovable pose id" << it2->first << " - " << it2->second.id << std::endl;
//     it2->second.isUnremovable = true;
//     it2->second.isKeyframe = true;
//   }

//   if (!gpsInitialised_) {
//     // make all frames with GPS keyframes, in order not to possibly lose all GPS measurements
//     it2->second.isKeyframe = true;
//   }

//   // std::cout << "added" << positionCovarianceENU.diagonal().transpose() << std::endl;
//   std::cout << "added " << position.transpose() << std::endl;

//   return true;
// }

// bool Estimator::getPredictedTargetState(const okvis::Time& stamp, kinematics::Transformation& T_WT) const {
//   const double dt = (stamp - t_last_target_prediction_).toSec();
//   // std::cout << stamp << "vs" << t_last_target_measurement_ << std::endl;
//   // prediction -- state
//   Eigen::Vector3d r_pred = T_WT_.r() + v_WT_W_ * dt;
//   Eigen::Quaterniond q_pred = T_WT_.q();
//   // omega and v stay (constant velocity model)
//   // assign new pose
//   T_WT = kinematics::Transformation(r_pred, q_pred);
//   return (stamp-t_last_target_measurement_).toSec() < 2.0;
// }

// bool Estimator::addTargetMeasurement(int targetId, uint64_t poseId, size_t camIdx, const okvis::kinematics::Transformation& T_SCi,
//                                      const cameras::PinholeCamera<cameras::NoDistortion>& camera, const std::vector<std::pair<size_t, cv::Point2f>>& matches,
//                                      const okvis::kinematics::Transformation* T_CT) {
//   // std::cout <<"no. matches = " << matches.size() << std::endl;
//   //

//   // prepare
//   okvis::kinematics::Transformation T_WS;
//   get_T_WS(poseId, T_WS);
//   // getCameraSensorStates(poseId, camIdx, T_SCi);
//   const okvis::Time t = timestamp(poseId);
//   const double dt = (t - t_last_target_prediction_).toSec();

//   if (dt < 0.0) {
//     std::cout << "target measurement in the past received -- ignoring" << std::endl;
//     return false;
//   }

//   // for disambiguation: T_TT due to rotation ambiguity, thanks MBZIRC organizers
//   okvis::kinematics::Transformation T_TT;
//   bool predicted = false;
//   if (T_CT) {
//     // pose measurement -- get the most plausible one (due to ambiguity, thanks MBZIRC organizers):
//     std::vector<okvis::kinematics::Transformation, Eigen::aligned_allocator<okvis::kinematics::Transformation>> T_WT_hypotheses(4);
//     okvis::kinematics::Transformation T_WT = T_WS * T_SCi * (*T_CT);
//     okvis::kinematics::Transformation T_delta = T_WT * T_WT_.inverse();
//     double score = fabs(asin(T_delta.q().vec().norm()) * 2.0);
//     // std::cout <<  180.0/M_PI*score << " ";

//     for (size_t h = 1; h < 4; ++h) {
//       okvis::kinematics::Transformation T_TT_tmp(
//           Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond(cos(h * M_PI / 4.0), 0, 0, sin(h * M_PI / 4.0)));  // 90 degree rotations around z-axis in target frame
//       const okvis::kinematics::Transformation T_WT_tmp = T_WS * T_SCi * (T_TT_tmp * T_CT->inverse()).inverse();
//       T_delta = T_WT_tmp * T_WT_.inverse();
//       double score_tmp = fabs(asin(T_delta.q().vec().norm()) * 2.0);
//       // std::cout << 180.0/M_PI*score_tmp << " ";
//       if (score_tmp < score) {
//         score = score_tmp;
//         T_WT = T_WT_tmp;
//         T_TT = T_TT_tmp;
//       }
//     }
//     // std::cout << "score [deg] = " << 180.0/M_PI*score << std::endl;
//     // std::cout << "measured " << T_WT.r().transpose() << " : " << T_WT.q().coeffs().transpose() << std::endl;

//     // std::cout << "=== dt = "<< dt << std::endl;

//     // do we need to re-initialise?
//     if (dt > 2.0 || !targetInitialised_) {
//       std::cout << "RE-INIT TARGET STATE" << std::endl;
//       T_WT_ = T_WT;
//       v_WT_W_ = Eigen::Vector3d::Zero();  // Tracked target linear velocity.
//       P_T_.setIdentity();
//       P_T_.block<3, 3>(6, 6) *= 4;
//       P_T_.block<3, 3>(3, 3) *= 0.01;
//       t_last_target_measurement_ = t;  // last observation timestamp
//       targetInitialised_ = true;
//     } else {
//       // call prediction
//       predictTarget(dt);
//       t_last_target_prediction_ = t;
//       predicted = true;

//       // update
//       okvis::kinematics::Transformation dp = T_WT * T_WT_.inverse();
//       Eigen::Matrix<double, 6, 1> r;
//       const Eigen::Vector3d dtheta = 2 * dp.q().coeffs().head<3>();
//       r.head<3>() = T_WT.r() - T_WT_.r();
//       r.tail<3>() = dtheta;
//       Eigen::Matrix<double, 6, 9> H = Eigen::Matrix<double, 6, 9>::Zero();
//       H.topLeftCorner<3, 3>().setIdentity();
//       H.block<3, 3>(3, 3) = okvis::kinematics::plus(dp.q()).topLeftCorner<3, 3>();
//       Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Identity();
//       R.topLeftCorner<3, 3>() *= sigma_r_ * sigma_r_;
//       R.bottomRightCorner<3, 3>() *= sigma_alpha_ * sigma_alpha_;
//       Eigen::Matrix<double, 6, 6> S = H * P_T_ * H.transpose() + R;
//       const Eigen::Matrix<double, 9, 6> K = P_T_ * H.transpose() * S.inverse();

//       // update -- state
//       const Eigen::Matrix<double, 9, 1> delta_x = K * r;
//       const Eigen::Matrix<double, 6, 1> delta_p = delta_x.head<6>();
//       /////////////////// DEBGUG
//       T_WT_.oplus(delta_p);  // update pose.
//       v_WT_W_ += delta_x.segment<3>(6);

//       /*std::cout << "T_WT: "<< std::endl
//                 << "r: " << T_WT_.r().transpose() << ";" << std::endl
//                 << "q: " << T_WT_.q().coeffs().transpose() << ";" << std::endl
//                 << "v_WT_W: " << v_WT_W_.transpose() << ";" << std::endl;*/

//       // update -- covariance (Joseph form for stability)
//       Eigen::Matrix<double, 9, 9> L = (Eigen::Matrix<double, 9, 9>::Identity() - K * H);
//       P_T_ -= K * H * P_T_;

//       // remember time
//       t_last_target_measurement_ = t;  // last observation timestamp
//     }
//   }

//   // now apply the keypoint measurements
//   if (!targetInitialised_) {
//     return false;
//   }

//   if (!predicted) {
//     predictTarget(dt);
//     t_last_target_prediction_ = t;
//   }

//   for (size_t p = 0; p < matches.size(); ++p) {
//     //    std::cout << "processing keypoint p=" << p << std::endl;

//     // transform the point to camera coordinates
//     okvis::kinematics::Transformation T_CW = T_SCi.inverse() * T_WS.inverse();
//     Eigen::Vector4d p_T(0, 0, 0, 1);
//     p_T.head<2>() = keypointLocations_.col(matches[p].first);
//     Eigen::Vector4d p_C = T_CW * T_WT_ * T_TT * p_T;

//     // project
//     Eigen::Vector2d point2d;
//     Eigen::Matrix<double, 2, 4> U;

//     Eigen::Matrix<double, 2, 9> H_numDiff = Eigen::Matrix<double, 2, 9>::Zero();
//     if (false) {
//       // Jacobian verification
//       for (size_t i = 0; i < 6; ++i) {
//         Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
//         delta[i] = 1.0e-7;
//         auto T_WT_p = T_WT_;
//         auto T_WT_m = T_WT_;
//         T_WT_p.oplus(delta);
//         T_WT_m.oplus(-delta);
//         Eigen::Vector4d p_C_p = T_CW * T_WT_p * T_TT * p_T;
//         Eigen::Vector4d p_C_m = T_CW * T_WT_m * T_TT * p_T;
//         Eigen::Vector2d point2d_p;
//         Eigen::Vector2d point2d_m;
//         camera.projectHomogeneous(p_C_p, &point2d_p, &U);
//         camera.projectHomogeneous(p_C_m, &point2d_m, &U);
//         H_numDiff.col(i) = (point2d_p - point2d_m) / 2.0e-7;
//       }
//     }

//     if (camera.projectHomogeneous(p_C, &point2d, &U) != cameras::CameraBase::ProjectionStatus::Successful) {
//       // invalid, skip
//       continue;
//     }

//     // residual
//     Eigen::Vector2d y = Eigen::Vector2d(matches[p].second.x, matches[p].second.y) - point2d;

//     // std::cout << y.transpose() << " : " << Eigen::Vector2d(matches[p].second.x, matches[p].second.y).transpose() << " : " << point2d.transpose() <<
//     // std::endl;

//     Eigen::Matrix<double, 2, 9> H = Eigen::Matrix<double, 2, 9>::Zero();
//     Eigen::Matrix<double, 4, 6> J = Eigen::Matrix<double, 4, 6>::Zero();
//     J.topLeftCorner<3, 3>().setIdentity();
//     J.topRightCorner<3, 3>() = -kinematics::crossMx(T_WT_.C() * (T_TT * p_T).head<3>());
//     H.topLeftCorner<2, 6>() = U * T_CW.T() * J;

//     // Residual covar
//     Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * sigma_pt_ * sigma_pt_;
//     Eigen::Matrix2d S = H * P_T_ * H.transpose() + R;
//     //    std::cout << "chi2=" <<y.transpose()*S.inverse()*y << std::endl;
//     //std::cout << "err=" << y.transpose() << std::endl;

//     if (y.transpose() * S.inverse() * y > 9) {
//       // invalid, skip
//       std::cout << "chi2=" << y.transpose() * S.inverse() * y << ",  y=" << y.transpose() << std::endl;
//       continue;
//     }

//     // std::cout << "H=\n"<<H << std::endl;
//     // std::cout << "H_numDiff=\n"<< H_numDiff << std::endl;

//     const Eigen::Matrix<double, 9, 2> K = P_T_ * H.transpose() * S.inverse();

//     // update -- state
//     const Eigen::Matrix<double, 9, 1> delta_x = K * y;
//     const Eigen::Matrix<double, 6, 1> delta_p = delta_x.head<6>();
//     T_WT_.oplus(delta_p);  // update pose.
//     v_WT_W_ += delta_x.segment<3>(6);

//     // std::cout << "fused pt " << T_WT_.r().transpose() << " : " << T_WT_.q().coeffs().transpose() << " : " << v_WT_W_.transpose() << std::endl;

//     // std::cout << delta_x.transpose() << std::endl;

//     /*std::cout << "T_WT: "<< std::endl
//               << "r: " << T_WT_.r().transpose() << ";" << std::endl
//               << "q: " << T_WT_.q().coeffs().transpose() << ";" << std::endl
//               << "v_WT_W: " << v_WT_W_.transpose() << ";" << std::endl
//               << "omega_WT_W: " << omega_WT_W_.transpose() << std::endl;*/

//     // update -- covariance
//     P_T_ -= K * H * P_T_;

//     // remember time
//     t_last_target_measurement_ = t;  // last observation timestamp
//   }

//   return true;
// }

// void Estimator::predictTarget(double dt) {

// //std::cout << "predictTarget " << dt << std::endl;
//   // prediction -- state
//   Eigen::Vector3d r_pred = T_WT_.r() + v_WT_W_ * dt;
//   Eigen::Quaterniond q_pred = T_WT_.q();
//   // omega and v stay (constant velocity model)

//   // prediction -- covariance
//   Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
//   F.block<3, 3>(0, 6) = dt * Eigen::Matrix3d::Identity();
//   P_T_ = F * P_T_ * F.transpose();  // propagate

//   P_T_.block<2, 2>(3, 3) += sigma_alpha_c_ * sigma_alpha_c_ * (dt)*Eigen::Matrix2d::Identity() * 0.01;  // process noise
//   P_T_(5, 5) *= sigma_alpha_c_ * sigma_alpha_c_ * (dt);
//   P_T_.block<3, 3>(6, 6) += sigma_v_c_ * sigma_v_c_ * (dt)*Eigen::Matrix3d::Identity();  // process noise

//   // assign new pose
//   T_WT_ = kinematics::Transformation(r_pred, q_pred);
// }

/**
 * @brief Does a vector contain a certain element.
 * @tparam Class of a vector element.
 * @param vector Vector to search element in.
 * @param query Element to search for.
 * @return True if query is an element of vector.
 */
template <class T>
bool vectorContains(const std::vector<T>& vector, const T& query) {
  for (size_t i = 0; i < vector.size(); ++i) {
    if (vector[i] == query) {
      return true;
    }
  }
  return false;
}

// Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
// The new number of frames in the window will be numKeyframes+numImuFrames.
bool Estimator::applyMarginalizationStrategy(size_t numKeyframes, size_t numImuFrames, okvis::MapPointVector& removedLandmarks) {
  // keep the newest numImuFrames
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  for (size_t k = 0; k < numImuFrames; k++) {
    rit++;
    if (rit == statesMap_.rend()) {
      // nothing to do.
      return true;
    }
  }

  bool reDoFixation = false;

  // so this must be the newest keyframe in the KF window
  uint64_t currentKfId = rit->first;

  // remove linear marginalizationError, if existing
  if (marginalizationErrorPtr_ && marginalizationResidualId_) {
    bool success = mapPtr_->removeResidualBlock(marginalizationResidualId_);
    OKVIS_ASSERT_TRUE_DBG(Exception, success, "could not remove marginalization error");
    marginalizationResidualId_ = 0;
    if (!success) return false;
  }

  // these will keep track of what we want to marginalize out.
  std::vector<uint64_t> paremeterBlocksToBeMarginalized;
  std::vector<bool> keepParameterBlocks;

  if (!marginalizationErrorPtr_) {
    marginalizationErrorPtr_.reset(new ceres::MarginalizationError(*mapPtr_.get()));
  }

  // distinguish if we marginalize everything or drop observations and lose states
  std::vector<uint64_t> loseStates;
  std::vector<uint64_t> allLinearisedPoses;
  std::vector<uint64_t> marginaliseStates;
  size_t countedKeyframes = 0;
  while (rit != statesMap_.rend()) {
    allLinearisedPoses.push_back(rit->second.id);
    if (countedKeyframes >= numKeyframes && rit->second.isKeyframe) {
      marginaliseStates.push_back(rit->second.id);
    }
    if (!rit->second.isKeyframe) {
      loseStates.push_back(rit->second.id);
    } else {
      countedKeyframes++;
    }
    ++rit;  // check the next frame
  }

  // handle states to lose
  for (size_t k = 0; k < loseStates.size(); ++k) {
    std::shared_ptr<ceres::ImuError> imuError_0;
    ceres::Map::ParameterBlockCollection pars_0;
    std::shared_ptr<ceres::ImuError> imuError_1;
    ceres::Map::ParameterBlockCollection pars_1;
    auto res = mapPtr_->residuals(loseStates[k]);
    for (auto& r : res) {
      // drop all observations
      if (std::dynamic_pointer_cast<ceres::ReprojectionError2dBase>(r.errorInterfacePtr)) {
        auto lmId = mapPtr_->parameters(r.residualBlockId)[1].first;
        removeObservation(r.residualBlockId);
        if (landmarksMap_[lmId].observations.size() == 0) {
          landmarksMap_.erase(lmId);
          mapPtr_->removeParameterBlock(lmId);
        }
      }

      // drop position errors
      if (std::dynamic_pointer_cast<ceres::PositionError>(r.errorInterfacePtr)) {
        mapPtr_->removeResidualBlock(r.residualBlockId);
      }

      // find both IMU errors
      auto imuError = std::dynamic_pointer_cast<ceres::ImuError>(r.errorInterfacePtr);
      if (imuError) {
        auto pars = mapPtr_->parameters(r.residualBlockId);
        mapPtr_->removeResidualBlock(r.residualBlockId);  // remove from map in any case
        if ((imuError->t1() - timestamp(loseStates[k])).toSec() == 0.0) {
          imuError_0 = imuError;  // the one earlier in time
          pars_0 = pars;
        } else {
          imuError_1 = imuError;  // must be the one later in time
          pars_1 = pars;
        }
      }
    }
    // link IMU measurements and lose states
    OKVIS_ASSERT_TRUE(Exception, imuError_0, "no IMU error found");
    OKVIS_ASSERT_TRUE(Exception, imuError_1, "no IMU error found");
    okvis::SpeedAndBias speedAndBias;
    getSpeedAndBias(loseStates[k], 0, speedAndBias);
    imuError_0->append(okvis::kinematics::Transformation() /* not needed */, speedAndBias, imuError_1->imuMeasurements(), imuError_1->t1());
    OKVIS_ASSERT_TRUE(Exception, pars_0[2].first == pars_1[0].first, "bug");
    OKVIS_ASSERT_TRUE(Exception, pars_0[3].first == pars_1[1].first, "bug");
    mapPtr_->removeParameterBlock(pars_0[2].second);  // lose pose
    mapPtr_->removeParameterBlock(pars_0[3].second);  // lose speed and bias
    mapPtr_->addResidualBlock(imuError_0, NULL, pars_0[0].second, pars_0[1].second, pars_1[2].second, pars_1[3].second);

    // add remaining error terms of the sensor states.
    size_t i = SensorStates::Camera;
    std::map<uint64_t, States>::iterator it = statesMap_.find(loseStates[k]);
    for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
      size_t k = CameraSensorStates::T_SCi;
      if (!it->second.sensors[i][j][k].exists) {
        continue;
      }
      if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if (checkit->second.sensors[i][j][k].exists && checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id) {
        continue;
      }
      it->second.sensors[i][j][k].exists = false;  // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.sensors[i][j][k].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
        if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }

    // book-keeping
    statesMap_.erase(loseStates[k]);
  }

  // marginalize everything but pose:
  for (size_t k = 0; k < marginaliseStates.size(); ++k) {
    std::map<uint64_t, States>::iterator it = statesMap_.find(marginaliseStates[k]);
    for (size_t i = 0; i < it->second.global.size(); ++i) {
      if (i == GlobalStates::T_WS) {
        continue;  // we do not remove the pose here.
      }
      if (!it->second.global[i].exists) {
        continue;  // if it doesn't exist, we don't do anything.
      }
      if (mapPtr_->parameterBlockPtr(it->second.global[i].id)->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if (checkit->second.global[i].exists && checkit->second.global[i].id == it->second.global[i].id) {
        continue;
      }
      it->second.global[i].exists = false;  // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[i].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.global[i].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
        if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }
    // add all error terms of the sensor states.
    for (size_t i = 0; i < it->second.sensors.size(); ++i) {
      for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
        for (size_t k = 0; k < it->second.sensors[i][j].size(); ++k) {
          if (i == SensorStates::Camera && k == CameraSensorStates::T_SCi) {
            continue;  // we do not remove the extrinsics pose here.
          }
          if (!it->second.sensors[i][j][k].exists) {
            continue;
          }
          if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)->fixed()) {
            continue;  // we never eliminate fixed blocks.
          }
          std::map<uint64_t, States>::iterator checkit = it;
          checkit++;
          // only get rid of it, if it's different
          if (checkit->second.sensors[i][j][k].exists && checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id) {
            continue;
          }
          it->second.sensors[i][j][k].exists = false;  // remember we removed
          paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
          keepParameterBlocks.push_back(false);
          ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.sensors[i][j][k].id);
          for (size_t r = 0; r < residuals.size(); ++r) {
            std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
                std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
            auto positionError = std::dynamic_pointer_cast<ceres::PositionError>(residuals[r].errorInterfacePtr);
            if (positionError) {
              if (!gpsInitialised_) {
                // we ignore all position errors outside the window
                mapPtr_->removeResidualBlock(residuals[r].residualBlockId);
              }
            } else if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
              marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
            }
          }
        }
      }
    }
  }
  // marginalize ONLY pose now:
  for (size_t k = 0; k < marginaliseStates.size(); ++k) {
    std::map<uint64_t, States>::iterator it = statesMap_.find(marginaliseStates[k]);

    // schedule removal - but always keep the very first frame.
    if (!it->second.isUnremovable) {
      it->second.global[GlobalStates::T_WS].exists = false;  // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[GlobalStates::T_WS].id);
      keepParameterBlocks.push_back(false);
      // std::cout << "removing pose " << it->second.global[GlobalStates::T_WS].id << std::endl;
    } else {
      // std::cout << "############################################ skipping removal of " << marginaliseStates[k] << std::endl;
      continue;
    }

    // add remaing error terms
    ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.global[GlobalStates::T_WS].id);

    for (size_t r = 0; r < residuals.size(); ++r) {
      /*if(std::dynamic_pointer_cast<ceres::PositionError>(residuals[r].errorInterfacePtr)){
        std::cout << "HAA------------------"<< std::endl;
      }*/
      if (std::dynamic_pointer_cast<ceres::PoseError>(residuals[r].errorInterfacePtr)) {  // avoids linearising initial pose error
        mapPtr_->removeResidualBlock(residuals[r].residualBlockId);
        if (!positionSensorAlignmentParameterBlock_) reDoFixation = true;
        continue;
      }
      std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError = std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
      if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
        marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
      }
    }

    // add remaining error terms of the sensor states.
    size_t i = SensorStates::Camera;
    for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
      size_t k = CameraSensorStates::T_SCi;
      if (!it->second.sensors[i][j][k].exists) {
        continue;
      }
      if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if (checkit->second.sensors[i][j][k].exists && checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id) {
        continue;
      }
      it->second.sensors[i][j][k].exists = false;  // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.sensors[i][j][k].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
        if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }

    // now finally we treat all the observations.
    {
      for (PointMap::iterator pit = landmarksMap_.begin(); pit != landmarksMap_.end();) {
        ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(pit->first);

        // first check if we can skip
        bool skipLandmark = true;
        bool hasNewObservations = false;
        bool justDelete = false;
        bool marginalize = true;
        bool errorTermAdded = false;
        std::map<uint64_t, bool> visibleInFrame;
        size_t obsCount = 0;
        for (size_t r = 0; r < residuals.size(); ++r) {
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
          if (reprojectionError) {
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
            // since we have implemented the linearisation to account for robustification,
            // we don't kick out bad measurements here any more like
            // if(vectorContains(allLinearizedFrames,poseId)){ ...
            //   if (error.transpose() * error > 6.0) { ... removeObservation ... }
            // }
            if (vectorContains(marginaliseStates, poseId)) {
              skipLandmark = false;
            }
            if (poseId >= currentKfId) {
              marginalize = false;
              hasNewObservations = true;
            }
            if (vectorContains(allLinearisedPoses, poseId)) {
              visibleInFrame.insert(std::pair<uint64_t, bool>(poseId, true));
              obsCount++;
            }
          }
        }

        if (residuals.size() == 0) {
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        if (skipLandmark) {
          pit++;
          continue;
        }

        // so, we need to consider it.
        for (size_t r = 0; r < residuals.size(); ++r) {
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
          if (reprojectionError) {
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
            if ((vectorContains(marginaliseStates, poseId) && hasNewObservations) || (!vectorContains(allLinearisedPoses, poseId) && marginalize)) {
              // ok, let's ignore the observation.
              removeObservation(residuals[r].residualBlockId);
              residuals.erase(residuals.begin() + r);
              r--;
            } else if (marginalize && vectorContains(allLinearisedPoses, poseId)) {
              // TODO: consider only the sensible ones for marginalization
              if (obsCount < 2) {  // visibleInFrame.size()
                removeObservation(residuals[r].residualBlockId);
                residuals.erase(residuals.begin() + r);
                r--;
              } else {
                // add information to be considered in marginalization later.
                errorTermAdded = true;
                marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId, false);
              }
            }
            // check anything left
            if (residuals.size() == 0) {
              justDelete = true;
              marginalize = false;
            }
          }
        }

        if (justDelete) {
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }
        if (marginalize && errorTermAdded) {
          paremeterBlocksToBeMarginalized.push_back(pit->first);
          keepParameterBlocks.push_back(false);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        pit++;
      }
    }

    // update book-keeping and go to the next frame
    // if(it != statesMap_.begin()){ // let's remember that we kept the very first pose
    if (true) {  ///// DEBUG
      multiFramePtrMap_.erase(it->second.id);
      statesMap_.erase(it->second.id);
    }
  }

  // now apply the actual marginalization
  if (paremeterBlocksToBeMarginalized.size() > 0) {
    std::vector<::ceres::ResidualBlockId> addedPriors;
    marginalizationErrorPtr_->marginalizeOut(paremeterBlocksToBeMarginalized, keepParameterBlocks);
  }

  // update error computation
  if (paremeterBlocksToBeMarginalized.size() > 0) {
    marginalizationErrorPtr_->updateErrorComputation();
  }

  // add the marginalization term again
  if (marginalizationErrorPtr_->num_residuals() == 0) {
    marginalizationErrorPtr_.reset();
  }
  if (marginalizationErrorPtr_) {
    std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>> parameterBlockPtrs;
    marginalizationErrorPtr_->getParameterBlockPtrs(parameterBlockPtrs);
    marginalizationResidualId_ = mapPtr_->addResidualBlock(marginalizationErrorPtr_, NULL, parameterBlockPtrs);
    OKVIS_ASSERT_TRUE_DBG(Exception, marginalizationResidualId_, "could not add marginalization error");
    if (!marginalizationResidualId_) return false;
  }

  if (positionSensorAlignmentParameterBlock_ && !gpsInitialised_) {
    if (statesMap_.begin()->second.isUnremovable) {
      okvis::kinematics::Transformation T_WS_0;
      get_T_WS(statesMap_.begin()->first, T_WS_0);
      okvis::kinematics::Transformation T_WS;
      get_T_WS(statesMap_.rbegin()->first, T_WS);
      if ((T_WS.r() - T_WS_0.r()).head<2>().norm() > 2.0) {
        statesMap_.begin()->second.isUnremovable = false;
        mapPtr_->setParameterBlockConstant(positionSensorAlignmentParameterBlock_);
        gpsInitialised_ = true;
        // std::cout << "================== GPS INITIALISED ======================" << std::endl;
      }
    }
  }

  if (reDoFixation) {
    /*if(positionSensorAlignmentParameterBlock_){
      mapPtr_->setParameterBlockConstant(positionSensorAlignmentParameterBlock_);
    }*/
    // finally fix the first pose properly
    // mapPtr_->resetParameterization(statesMap_.begin()->first, ceres::Map::Pose3d);
    // std::cout << "================== refixing ======================" << std::endl;
    okvis::kinematics::Transformation T_WS_0;
    get_T_WS(statesMap_.begin()->first, T_WS_0);
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
    information(5, 5) = 1.0e6;
    information(0, 0) = 1.0e4;
    information(1, 1) = 1.0e4;
    information(2, 2) = 1.0e4;
    std::shared_ptr<ceres::PoseError> poseError(new ceres::PoseError(T_WS_0, information));
    mapPtr_->addResidualBlock(poseError, NULL, mapPtr_->parameterBlockPtr(statesMap_.begin()->first));
  }

  /*for(auto it = statesMap_.begin(); it!=statesMap_.end(); ++it){
    printStates(it->first, std::cout);
  }*/

  return true;
}

// Prints state information to buffer.
void Estimator::printStates(uint64_t poseId, std::ostream& buffer) const {
  buffer << "GLOBAL: ";
  for (size_t i = 0; i < statesMap_.at(poseId).global.size(); ++i) {
    if (statesMap_.at(poseId).global.at(i).exists) {
      uint64_t id = statesMap_.at(poseId).global.at(i).id;
      if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << "(";
      buffer << "id=" << id << ":";
      buffer << mapPtr_->parameterBlockPtr(id)->typeInfo();
      if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << ")";
      buffer << ", ";
    }
  }
  buffer << "SENSOR: ";
  for (size_t i = 0; i < statesMap_.at(poseId).sensors.size(); ++i) {
    for (size_t j = 0; j < statesMap_.at(poseId).sensors.at(i).size(); ++j) {
      for (size_t k = 0; k < statesMap_.at(poseId).sensors.at(i).at(j).size(); ++k) {
        if (statesMap_.at(poseId).sensors.at(i).at(j).at(k).exists) {
          uint64_t id = statesMap_.at(poseId).sensors.at(i).at(j).at(k).id;
          if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << "(";
          buffer << "id=" << id << ":";
          buffer << mapPtr_->parameterBlockPtr(id)->typeInfo();
          if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << ")";
          buffer << ", ";
        }
      }
    }
  }
  buffer << std::endl;
}

// Initialise pose from IMU measurements. For convenience as static.
bool Estimator::initPoseFromImu(const okvis::ImuMeasurementDeque& imuMeasurements, okvis::kinematics::Transformation& T_WS) {
  // set translation to zero, unit rotation
  T_WS.setIdentity();

  if (imuMeasurements.size() == 0) return false;

  // acceleration vector
  Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin(); it < imuMeasurements.end(); ++it) {
    acc_B += it->measurement.accelerometers;
  }
  acc_B /= double(imuMeasurements.size());
  Eigen::Vector3d e_acc = acc_B.normalized();

  // align with ez_W:
  Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
  Eigen::Matrix<double, 6, 1> poseIncrement;
  poseIncrement.head<3>() = Eigen::Vector3d::Zero();
  poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();
  double angle = std::acos(ez_W.transpose() * e_acc);
  poseIncrement.tail<3>() *= angle;
  T_WS.oplus(-poseIncrement);

  return true;
}

// Start ceres optimization.
#ifdef USE_OPENMP
void Estimator::optimize(size_t numIter, size_t numThreads, bool verbose)
#else
void Estimator::optimize(size_t numIter, size_t /*numThreads*/,
                         bool verbose)  // avoid warning since numThreads unused
#warning openmp not detected, your system may be slower than expected
#endif

{
  // assemble options
  mapPtr_->options.linear_solver_type = ::ceres::SPARSE_SCHUR;
  // mapPtr_->options.initial_trust_region_radius = 1.0e4;
  // mapPtr_->options.initial_trust_region_radius = 2.0e6;
  // mapPtr_->options.preconditioner_type = ::ceres::IDENTITY;
  mapPtr_->options.trust_region_strategy_type = ::ceres::DOGLEG;
// mapPtr_->options.trust_region_strategy_type = ::ceres::LEVENBERG_MARQUARDT;
// mapPtr_->options.use_nonmonotonic_steps = true;
// mapPtr_->options.max_consecutive_nonmonotonic_steps = 10;
// mapPtr_->options.function_tolerance = 1e-12;
// mapPtr_->options.gradient_tolerance = 1e-12;
// mapPtr_->options.jacobi_scaling = false;
#ifdef USE_OPENMP
  mapPtr_->options.num_threads = numThreads;
#endif
  mapPtr_->options.max_num_iterations = numIter;

  if (verbose) {
    mapPtr_->options.minimizer_progress_to_stdout = true;
  } else {
    mapPtr_->options.minimizer_progress_to_stdout = false;
  }

  // call solver
  mapPtr_->solve();

  // okvis::kinematics::Transformation T_GW;
  // get_T_GW(0,T_GW);
  // std::cout<<T_GW.q().coeffs().transpose()<<std::endl;

  // update landmarks
  {
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      Eigen::MatrixXd H(3, 3);
      mapPtr_->getLhs(it->first, H);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(H);
      Eigen::Vector3d eigenvalues = saes.eigenvalues();
      const double smallest = (eigenvalues[0]);
      const double largest = (eigenvalues[2]);
      if (smallest < 1.0e-12) {
        // this means, it has a non-observable depth
        it->second.quality = 0.0;
      } else {
        // OK, well constrained
        it->second.quality = sqrt(smallest) / sqrt(largest);
      }

      // update coordinates
      it->second.point = std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(mapPtr_->parameterBlockPtr(it->first))->estimate();
    }
  }

  // summary output
  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
}

// Set a time limit for the optimization process.
bool Estimator::setOptimizationTimeLimit(double timeLimit, int minIterations) {
  if (ceresCallback_ != nullptr) {
    if (timeLimit < 0.0) {
      // no time limit => set minimum iterations to maximum iterations
      ceresCallback_->setMinimumIterations(mapPtr_->options.max_num_iterations);
      return true;
    }
    ceresCallback_->setTimeLimit(timeLimit);
    ceresCallback_->setMinimumIterations(minIterations);
    return true;
  } else if (timeLimit >= 0.0) {
    ceresCallback_ = std::unique_ptr<okvis::ceres::CeresIterationCallback>(new okvis::ceres::CeresIterationCallback(timeLimit, minIterations));
    mapPtr_->options.callbacks.push_back(ceresCallback_.get());
    return true;
  }
  // no callback yet registered with ceres.
  // but given time limit is lower than 0, so no callback needed
  return true;
}

// getters
// Get a specific landmark.
bool Estimator::getLandmark(uint64_t landmarkId, MapPoint& mapPoint) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  if (landmarksMap_.find(landmarkId) == landmarksMap_.end()) {
    OKVIS_THROW_DBG(Exception, "landmark with id = " << landmarkId << " does not exist.")
    return false;
  }
  mapPoint = landmarksMap_.at(landmarkId);
  return true;
}

// Checks whether the landmark is initialized.
bool Estimator::isLandmarkInitialized(uint64_t landmarkId) const {
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "landmark not added");
  return std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(mapPtr_->parameterBlockPtr(landmarkId))->initialized();
}

// Get a copy of all the landmarks as a PointMap.
size_t Estimator::getLandmarks(PointMap& landmarks) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks = landmarksMap_;
  return landmarksMap_.size();
}

// Get a copy of all the landmark in a MapPointVector. This is for legacy support.
// Use getLandmarks(okvis::PointMap&) if possible.
size_t Estimator::getLandmarks(MapPointVector& landmarks) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks.clear();
  landmarks.reserve(landmarksMap_.size());
  for (PointMap::const_iterator it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    landmarks.push_back(it->second);
  }
  return landmarksMap_.size();
}

// Get pose for a given pose ID.
bool Estimator::get_T_WS(uint64_t poseId, okvis::kinematics::Transformation& T_WS) const {
  if (!getGlobalStateEstimateAs<ceres::PoseParameterBlock>(poseId, GlobalStates::T_WS, T_WS)) {
    return false;
  }

  return true;
}

bool Estimator::get_T_WT(uint64_t poseId, uint64_t targetId, okvis::kinematics::Transformation& T_WT, Eigen::Vector3d& v_W, Eigen::Vector3d& omega_W,
                         double& velocityUncertainty) const {
  v_W = v_WT_W_;
  omega_W.setZero();
  const double dt = (timestamp(poseId) - t_last_target_prediction_).toSec();
  velocityUncertainty = sqrt(P_T_.bottomRightCorner<3, 3>().norm());
  const Eigen::Vector3d r_pred = T_WT_.r() + v_WT_W_ * dt;
  const Eigen::Quaterniond q_pred = T_WT_.q();

  // assign new pose
  T_WT = kinematics::Transformation(r_pred, q_pred);

//std::cout << dt << std::endl;

  return (timestamp(poseId) - t_last_target_measurement_).toSec() < 1.8;
}

// Feel free to implement caching for them...
// Get speeds and IMU biases for a given pose ID.
bool Estimator::getSpeedAndBias(uint64_t poseId, uint64_t imuIdx, okvis::SpeedAndBias& speedAndBias) const {
  if (!getSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias, speedAndBias)) {
    return false;
  }
  return true;
}

bool Estimator::get_T_GW(uint64_t /*poseId*/, okvis::kinematics::Transformation& T_GW) const {
  if (!positionSensorAlignmentParameterBlock_) {
    T_GW.setIdentity();
    return false;
  }

  T_GW = positionSensorAlignmentParameterBlock_->estimate();

  return true;
}

// Get camera states for a given pose ID.
bool Estimator::getCameraSensorStates(uint64_t poseId, size_t cameraIdx, okvis::kinematics::Transformation& T_SCi) const {
  return getSensorStateEstimateAs<ceres::PoseParameterBlock>(poseId, cameraIdx, SensorStates::Camera, CameraSensorStates::T_SCi, T_SCi);
}

// Get the ID of the current keyframe.
uint64_t Estimator::currentKeyframeId() const {
  for (std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin(); rit != statesMap_.rend(); ++rit) {
    if (rit->second.isKeyframe) {
      return rit->first;
    }
  }
  OKVIS_THROW_DBG(Exception, "no keyframes existing...");
  return 0;
}

// Get the ID of an older frame.
uint64_t Estimator::frameIdByAge(size_t age) const {
  std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin();
  for (size_t i = 0; i < age; ++i) {
    ++rit;
    OKVIS_ASSERT_TRUE_DBG(Exception, rit != statesMap_.rend(), "requested age " << age << " out of range.");
  }
  return rit->first;
}

// Get the ID of the newest frame added to the state.
uint64_t Estimator::currentFrameId() const {
  OKVIS_ASSERT_TRUE_DBG(Exception, statesMap_.size() > 0, "no frames added yet.")
  return statesMap_.rbegin()->first;
}

// Checks if a particular frame is still in the IMU window
bool Estimator::isInImuWindow(uint64_t frameId) const {
  if (statesMap_.at(frameId).sensors.at(SensorStates::Imu).size() == 0) {
    return false;  // no IMU added
  }
  return statesMap_.at(frameId).sensors.at(SensorStates::Imu).at(0).at(ImuSensorStates::SpeedAndBias).exists;
}

// Set pose for a given pose ID.
bool Estimator::set_T_WS(uint64_t poseId, const okvis::kinematics::Transformation& T_WS) {
  if (!setGlobalStateEstimateAs<ceres::PoseParameterBlock>(poseId, GlobalStates::T_WS, T_WS)) {
    return false;
  }

  return true;
}

// Set the speeds and IMU biases for a given pose ID.
bool Estimator::setSpeedAndBias(uint64_t poseId, size_t imuIdx, const okvis::SpeedAndBias& speedAndBias) {
  return setSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias, speedAndBias);
}

// Set the transformation from sensor to camera frame for a given pose ID.
bool Estimator::setCameraSensorStates(uint64_t poseId, size_t cameraIdx, const okvis::kinematics::Transformation& T_SCi) {
  return setSensorStateEstimateAs<ceres::PoseParameterBlock>(poseId, cameraIdx, SensorStates::Camera, CameraSensorStates::T_SCi, T_SCi);
}

// Set the homogeneous coordinates for a landmark.
bool Estimator::setLandmark(uint64_t landmarkId, const Eigen::Vector4d& landmark) {
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_->parameterBlockPtr(landmarkId);
#ifndef NDEBUG
  std::shared_ptr<ceres::HomogeneousPointParameterBlock> derivedParameterBlockPtr =
      std::dynamic_pointer_cast<ceres::HomogeneousPointParameterBlock>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception, "wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(landmark);
  ;
#else
  std::static_pointer_cast<ceres::HomogeneousPointParameterBlock>(parameterBlockPtr)->setEstimate(landmark);
#endif

  // also update in map
  landmarksMap_.at(landmarkId).point = landmark;
  return true;
}

// Set the landmark initialization state.
void Estimator::setLandmarkInitialized(uint64_t landmarkId, bool initialized) {
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "landmark not added");
  std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(mapPtr_->parameterBlockPtr(landmarkId))->setInitialized(initialized);
}

// private stuff
// getters
bool Estimator::getGlobalStateParameterBlockPtr(uint64_t poseId, int stateType, std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).global.at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW(Exception, "pose with id = " << id << " does not exist.")
    return false;
  }

  stateParameterBlockPtr = mapPtr_->parameterBlockPtr(id);
  return true;
}
template <class PARAMETER_BLOCK_T>
bool Estimator::getGlobalStateParameterBlockAs(uint64_t poseId, int stateType, PARAMETER_BLOCK_T& stateParameterBlock) const {
  // convert base class pointer with various levels of checking
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
  if (!getGlobalStateParameterBlockPtr(poseId, stateType, parameterBlockPtr)) {
    return false;
  }
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr = std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    LOG(INFO) << "--" << parameterBlockPtr->typeInfo();
    std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
    OKVIS_THROW_DBG(Exception, "wrong pointer type requested: requested " << info->typeInfo() << " but is of type" << parameterBlockPtr->typeInfo())
    return false;
  }
  stateParameterBlock = *derivedParameterBlockPtr;
#else
  stateParameterBlock = *std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
#endif
  return true;
}
template <class PARAMETER_BLOCK_T>
bool Estimator::getGlobalStateEstimateAs(uint64_t poseId, int stateType, typename PARAMETER_BLOCK_T::estimate_t& state) const {
  PARAMETER_BLOCK_T stateParameterBlock;
  if (!getGlobalStateParameterBlockAs(poseId, stateType, stateParameterBlock)) {
    return false;
  }
  state = stateParameterBlock.estimate();
  return true;
}

bool Estimator::getSensorStateParameterBlockPtr(uint64_t poseId, int sensorIdx, int sensorType, int stateType,
                                                std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).sensors.at(sensorType).at(sensorIdx).at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }
  stateParameterBlockPtr = mapPtr_->parameterBlockPtr(id);
  return true;
}
template <class PARAMETER_BLOCK_T>
bool Estimator::getSensorStateParameterBlockAs(uint64_t poseId, int sensorIdx, int sensorType, int stateType, PARAMETER_BLOCK_T& stateParameterBlock) const {
  // convert base class pointer with various levels of checking
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
  if (!getSensorStateParameterBlockPtr(poseId, sensorIdx, sensorType, stateType, parameterBlockPtr)) {
    return false;
  }
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr = std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
    OKVIS_THROW_DBG(Exception, "wrong pointer type requested: requested " << info->typeInfo() << " but is of type" << parameterBlockPtr->typeInfo())
    return false;
  }
  stateParameterBlock = *derivedParameterBlockPtr;
#else
  stateParameterBlock = *std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
#endif
  return true;
}
template <class PARAMETER_BLOCK_T>
bool Estimator::getSensorStateEstimateAs(uint64_t poseId, int sensorIdx, int sensorType, int stateType, typename PARAMETER_BLOCK_T::estimate_t& state) const {
  PARAMETER_BLOCK_T stateParameterBlock;
  if (!getSensorStateParameterBlockAs(poseId, sensorIdx, sensorType, stateType, stateParameterBlock)) {
    return false;
  }
  state = stateParameterBlock.estimate();
  return true;
}

template <class PARAMETER_BLOCK_T>
bool Estimator::setGlobalStateEstimateAs(uint64_t poseId, int stateType, const typename PARAMETER_BLOCK_T::estimate_t& state) {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).global.at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_->parameterBlockPtr(id);
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr = std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception, "wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(state);
#else
  std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr)->setEstimate(state);
#endif
  return true;
}

template <class PARAMETER_BLOCK_T>
bool Estimator::setSensorStateEstimateAs(uint64_t poseId, int sensorIdx, int sensorType, int stateType, const typename PARAMETER_BLOCK_T::estimate_t& state) {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).sensors.at(sensorType).at(sensorIdx).at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_->parameterBlockPtr(id);
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr = std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception, "wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(state);
#else
  std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr)->setEstimate(state);
#endif
  return true;
}

}  // namespace okvis
