/*
 * PositionError.hpp
 *
 *  Created on: 24 Jul 2015
 *      Author: sleutene
 */

#ifndef INCLUDE_OKVIS_CERES_POSITIONERROR_HPP_
#define INCLUDE_OKVIS_CERES_POSITIONERROR_HPP_

#include <vector>
#include <mutex>
#include "ceres/ceres.h"
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Variables.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// \brief Absolute error of a position. Uses IMU preintegrals for time alignment.
class PositionError :
    public ::ceres::SizedCostFunction<3 /* number of residuals */,
        7 /* size of first parameter, i.e. pose */,
        9 /* size of second parameter, i.e. speed and bias */,
        7 /* size of third parameter, i.e. alignment pose*/>,
    public ErrorInterface
{
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The base in ceres we derive from
  typedef ::ceres::SizedCostFunction<3, 7, 9, 7> base_t;

  /// \brief Trivial default constructor
  PositionError()
  {
  }

  /// \brief Trivial destructor.
  virtual ~PositionError()
  {
  }

  /// \brief Construct with measurements and parameters.
  /// \@param[in] positionMeasurement The 3d position measured in frame G.
  /// \@param[in] positionCovariance The position measurement covariance.
  /// \@param[in] positionMeasurementParameters Parameters (containing the antenna offset).
  /// \@param[in] imuMeasurements All the IMU measurements.
  /// \@param[in] imuParameters The parameters to be used.
  /// \@param[in] t_0 Start time: the time of the estimated states.
  /// \@param[in] t_1 End time: the time of the measurement.
  PositionError(
      const Eigen::Vector3d & positionMeasurement,
      const Eigen::Matrix3d & positionCovariance,
      const okvis::PositionSensorParameters & positionSensorParameters,
      const okvis::ImuMeasurementDeque & imuMeasurements,
      const okvis::ImuParameters & imuParameters, const okvis::Time& t_0,
      const okvis::Time& t_1);

  /// \brief Set the position measurement including parameters.
  /// \@param[in] positionMeasurement The 3d position measured in frame G.
  /// \@param[in] positionCovariance The position measurement covariance.
  /// \@param[in] positionMeasurementParameters Parameters (containing the antenna offset).
  void setPositionMeasurement(
      const Eigen::Vector3d & positionMeasurement,
      const Eigen::Matrix3d & positionCovariance,
      const okvis::PositionSensorParameters & positionSensorParameters);

  /// \brief Set IMU measurements as used for propagation to the actual timestamp.
  /// \@param[in] imuMeasurements All the IMU measurements.
  /// \@param[in] imuParameters The parameters to be used.
  /// \@param[in] t_0 Start time: the time of the estimated states.
  /// \@param[in] t_1 End time: the time of the measurement.
  void setImuMeasurements(const okvis::ImuMeasurementDeque & imuMeasurements,
                          const okvis::ImuParameters & imuParameters,
                          const okvis::Time& t_0, const okvis::Time& t_1);

  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @warning This is not actually const, since the re-propagation must somehow be stored...
   * @param[in] T_WS Start pose.
   * @param[in] speedAndBiases Start speed and biases.
   * @return Number of integration steps.
   */
  int redoPreintegration(const okvis::kinematics::Transformation& T_WS,
                         const okvis::SpeedAndBias & speedAndBiases) const;

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const;

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const
  {
    return 3;
  }

  /// \brief Number of parameter blocks.
  virtual size_t parameterBlocks() const
  {
    return parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  size_t parameterBlockDim(size_t parameterBlockId) const
  {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const
  {
    return "PositionError";
  }

 protected:
  // parameters
  okvis::PositionSensorParameters positionSensorParameters_;  ///< The position parameters.
  okvis::ImuParameters imuParameters_;  ///< The IMU parameters.

  // measurements
  Eigen::Vector3d positionMeasurement_;
  okvis::ImuMeasurementDeque imuMeasurements_;  ///< The IMU measurements used. Must be spanning t0_ - t1_.

  // times
  okvis::Time t0_;  ///< The start time (i.e. time of the estimated set of states).
  okvis::Time t1_;  ///< The end time (i.e. time of the measurement set of states).

  // preintegration stuff. the mutable is a TERRIBLE HACK, but what can I do.
  mutable std::mutex preintegrationMutex_;  //< Protect access of intermediate results.
  // increments (initialise with identity)
  mutable Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);  ///< Intermediate result
  mutable Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero();  ///< Intermediate result
  mutable Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero();  ///< Intermediate result
  mutable Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero();  ///< Intermediate result
  mutable Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero();  ///< Intermediate result

  // cross matrix accumulatrion
  mutable Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero();  ///< Intermediate result

  // sub-Jacobians
  mutable Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero();  ///< Intermediate result
  mutable Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero();  ///< Intermediate result
  mutable Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero();  ///< Intermediate result

  /// \brief Reference biases that are updated when called redoPreintegration.
  mutable SpeedAndBiases speedAndBiases_ref_ = SpeedAndBiases::Zero();

  mutable bool redo_ = true;  ///< Keeps track of whether or not this redoPreintegration() needs to be called.
  mutable int redoCounter_ = 0;  ///< Counts the number of preintegrations for statistics.

  // information matrix and its square root
  Eigen::Matrix3d information_;  ///< The information matrix for this error term.
  Eigen::Matrix3d squareRootInformation_;  ///< The square root information matrix for this error term.
};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_POSITIONERROR_HPP_ */
