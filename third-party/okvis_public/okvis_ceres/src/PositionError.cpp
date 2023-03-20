/*
 * PositionError.cpp
 *
 *  Created on: 25 Jul 2015
 *      Author: lestefan
 */

#include <okvis/ceres/PositionError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>

namespace okvis {
namespace ceres {

// Construct with measurements and parameters.
PositionError::PositionError(
    const Eigen::Vector3d & positionMeasurement,
    const Eigen::Matrix3d & positionCovariance,
    const okvis::PositionSensorParameters & positionSensorParameters,
    const okvis::ImuMeasurementDeque & imuMeasurements,
    const okvis::ImuParameters & imuParameters, const okvis::Time& t_0,
    const okvis::Time& t_1)
    : positionSensorParameters_(positionSensorParameters),
      imuParameters_(imuParameters),
      positionMeasurement_(positionMeasurement),
      imuMeasurements_(imuMeasurements),
      t0_(t_0),
      t1_(t_1) {
  // set the appropriate information
  information_ = positionCovariance.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<Eigen::Matrix3d> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

// Set the position measurement including parameters..
void PositionError::setPositionMeasurement(
    const Eigen::Vector3d & positionMeasurement,
    const Eigen::Matrix3d & positionCovariance,
    const okvis::PositionSensorParameters & positionSensorParameters) {
  positionMeasurement_ = positionMeasurement;
  positionSensorParameters_ = positionSensorParameters;
  // set the appropriate information
  information_ = positionCovariance.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<Eigen::Matrix3d> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

// Set IMU measurements as used for propagation to the actual timestamp.
void PositionError::setImuMeasurements(
    const okvis::ImuMeasurementDeque & imuMeasurements,
    const okvis::ImuParameters & imuParameters, const okvis::Time& t_0,
    const okvis::Time& t_1) {
  imuMeasurements_ = imuMeasurements;
  imuParameters_ = imuParameters;
  t0_ = t_0;
  t1_ = t_1;
}

// Propagates pose, speeds and biases with given IMU measurements.
int PositionError::redoPreintegration(
    const okvis::kinematics::Transformation& /*T_WS*/,
    const okvis::SpeedAndBias & speedAndBiases) const {
  // ensure unique access
  std::lock_guard<std::mutex> lock(preintegrationMutex_);

  // now the propagation
  okvis::Time time = t0_;
  okvis::Time end = t1_;

  // sanity check:
  assert(imuMeasurements_.front().timeStamp<=time);
  if (!(imuMeasurements_.back().timeStamp >= end))
    return -1;  // nothing to do...

  // increments (initialise with identity)
  Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);
  C_integral_ = Eigen::Matrix3d::Zero();
  C_doubleintegral_ = Eigen::Matrix3d::Zero();
  acc_integral_ = Eigen::Vector3d::Zero();
  acc_doubleintegral_ = Eigen::Vector3d::Zero();

  // cross matrix accumulatrion
  cross_ = Eigen::Matrix3d::Zero();

  // sub-Jacobians
  dalpha_db_g_ = Eigen::Matrix3d::Zero();
  dv_db_g_ = Eigen::Matrix3d::Zero();
  dp_db_g_ = Eigen::Matrix3d::Zero();

  double Delta_t = 0;
  bool hasStarted = false;
  int i = 0;
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements_.begin();
      it != imuMeasurements_.end(); ++it) {

    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements_.end()) {
      nexttime = t1_;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t1_;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }
    Delta_t += dt;

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double sigma_g_c = imuParameters_.sigma_g_c;
    double sigma_a_c = imuParameters_.sigma_a_c;

    if (fabs(omega_S_0[0]) > imuParameters_.g_max
        || fabs(omega_S_0[1]) > imuParameters_.g_max
        || fabs(omega_S_0[2]) > imuParameters_.g_max
        || fabs(omega_S_1[0]) > imuParameters_.g_max
        || fabs(omega_S_1[1]) > imuParameters_.g_max
        || fabs(omega_S_1[2]) > imuParameters_.g_max) {
      sigma_g_c *= 100;
      LOG(WARNING)<< "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParameters_.a_max
        || fabs(acc_S_0[1]) > imuParameters_.a_max
        || fabs(acc_S_0[2]) > imuParameters_.a_max
        || fabs(acc_S_1[0]) > imuParameters_.a_max
        || fabs(acc_S_1[1]) > imuParameters_.a_max
        || fabs(acc_S_1[2]) > imuParameters_.a_max) {
      sigma_a_c *= 100;
      LOG(WARNING)<< "acc saturation";
    }

    // actual propagation
    // orientation:
    Eigen::Quaterniond dq;
    const Eigen::Vector3d omega_S_true = (0.5 * (omega_S_0 + omega_S_1)
        - speedAndBiases.segment<3>(3));
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = okvis::kinematics::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;
    // rotation matrix integral:
    const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
    const Eigen::Vector3d acc_S_true = (0.5 * (acc_S_0 + acc_S_1)
        - speedAndBiases.segment<3>(6));
    const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 = acc_integral_
        + 0.5 * (C + C_1) * acc_S_true * dt;
    // rotation matrix double integral:
    C_doubleintegral_ += C_integral_ * dt + 0.25 * (C + C_1) * dt * dt;
    acc_doubleintegral_ += acc_integral_ * dt
        + 0.25 * (C + C_1) * acc_S_true * dt * dt;

    // Jacobian parts
    dalpha_db_g_ += C_1 * okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d cross_1 = dq.inverse().toRotationMatrix() * cross_
        + okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
    const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
    Eigen::Matrix3d dv_db_g_1 = dv_db_g_
        + 0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
    dp_db_g_ += dt * dv_db_g_
        + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

    // memory shift
    Delta_q_ = Delta_q_1;
    C_integral_ = C_integral_1;
    acc_integral_ = acc_integral_1;
    cross_ = cross_1;
    dv_db_g_ = dv_db_g_1;
    time = nexttime;

    ++i;

    if (nexttime == t1_)
      break;

  }

  // store the reference (linearisation) point
  speedAndBiases_ref_ = speedAndBiases;

  return i;
}

// error term and Jacobian implementation
// This evaluates the error term and additionally computes the Jacobians.
bool PositionError::Evaluate(double const* const * parameters,
                             double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
bool PositionError::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {

  // get pose
  const okvis::kinematics::Transformation T_WS_0(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4],
                         parameters[0][5]));
  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();

  // get speed and bias
  okvis::SpeedAndBias speedAndBiases_0;
  for (size_t i = 0; i < 9; ++i) {
    speedAndBiases_0[i] = parameters[1][i];
  }

  // get the global alignment pose
  const okvis::kinematics::Transformation T_GW(
      Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]),
      Eigen::Quaterniond(parameters[2][6], parameters[2][3], parameters[2][4],
                         parameters[2][5]));

  // call the propagation (if needed)
  const double Delta_t = (t1_ - t0_).toSec();
  Eigen::Matrix<double, 6, 1> Delta_b;
  // ensure unique access
  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);
    Delta_b = speedAndBiases_0.tail<6>() - speedAndBiases_ref_.tail<6>();
  }
  redo_ = redo_ || (Delta_b.head<3>().norm() * Delta_t > 0.0001);
  if (redo_) {
    redoPreintegration(T_WS_0, speedAndBiases_0);
    redoCounter_++;
    Delta_b.setZero();
    redo_ = false;
    /*if (redoCounter_ > 1) {
      std::cout << "pre-integration no. " << redoCounter_ << std::endl;
    }*/
  }

  // ensure unique access
  std::lock_guard<std::mutex> lock(preintegrationMutex_);

  // We compute the Jacobian dp/d[delta_p,delta_alpha,delta_sb]
  // in any case, since we might need a part of it for the error computation
  // holds for all states, including d/dalpha, d/db_g, d/db_a
  Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Identity();
  F.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
      C_WS_0 * acc_doubleintegral_);
  F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * Delta_t;
  F.block<3, 3>(0, 9) = C_WS_0 * dp_db_g_;
  F.block<3, 3>(0, 12) = -C_WS_0 * C_doubleintegral_;
  F.block<3, 3>(3, 9) = -C_WS_0 * dalpha_db_g_;
  F.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(C_WS_0 * acc_integral_);
  F.block<3, 3>(6, 9) = C_WS_0 * dv_db_g_;
  F.block<3, 3>(6, 12) = -C_WS_0 * C_integral_;

  // actual propagation output:
  const Eigen::Vector3d g_W = imuParameters_.g
      * Eigen::Vector3d(0, 0, 6371009).normalized();
  okvis::kinematics::Transformation T_WS(
      T_WS_0.r() + speedAndBiases_0.head<3>() * Delta_t
          + C_WS_0 * (acc_doubleintegral_) - 0.5 * g_W * Delta_t * Delta_t,
      T_WS_0.q() * Delta_q_);
  okvis::SpeedAndBias speedAndBiases = speedAndBiases_0;
  speedAndBiases.head<3>() += C_WS_0 * (acc_integral_) - g_W * Delta_t;

  // account for linearisation:
  T_WS.oplus(F.block<6,6>(0,9)*Delta_b);
  speedAndBiases.head<3>() += F.block<3,6>(6,9)*Delta_b;

  // now we are ready to compute the error:
  const Eigen::Vector3d p_SP_S = positionSensorParameters_.positionSensorOffset;
  const Eigen::Vector3d error = positionMeasurement_
      - T_GW.r() - T_GW.C()*(T_WS.r() + T_WS.C() * p_SP_S);

  //std::cout << error.transpose() << std::endl;

  // weigh it
  Eigen::Map<Eigen::Vector3d > weighted_error(residuals);
  weighted_error = squareRootInformation_*error;

  // assign Jacobian, if requested TODO
  if (jacobians != NULL) {
    // Jacobains using chain rule...
    if(jacobians != NULL){
        if(jacobians[0] != NULL) {
          Eigen::Map<Eigen::Matrix<double,3,7,Eigen::RowMajor> > J0(jacobians[0]);
          Eigen::Matrix<double,3,6,Eigen::RowMajor> J0_minimal;
          J0_minimal=-T_GW.C()*F.topLeftCorner<3,6>();
          J0_minimal+=T_GW.C()*okvis::kinematics::crossMx(T_WS.C()*p_SP_S)*F.block<3,6>(3,0);

          // pseudo inverse of the local parametrization Jacobian:
          Eigen::Matrix<double,6,7,Eigen::RowMajor> J_lift;
          PoseLocalParameterization::liftJacobian(parameters[0],J_lift.data());

          // hallucinate Jacobian w.r.t. state
          J0= squareRootInformation_*J0_minimal*J_lift;

          if(jacobiansMinimal!=NULL){
            if(jacobiansMinimal[0]!=NULL){
              Eigen::Map<Eigen::Matrix<double,3,6,Eigen::RowMajor> > J0_minimal_mapped(jacobiansMinimal[0]);
              J0_minimal_mapped= squareRootInformation_*J0_minimal;
            }
          }
        }
        if(jacobians[1] != NULL) {
          Eigen::Map<Eigen::Matrix<double,3,9,Eigen::RowMajor> > J1(jacobians[1]);
          Eigen::Matrix<double,3,9,Eigen::RowMajor> J1_minimal;
          J1_minimal=-T_GW.C()*F.topRightCorner<3,9>();
          J1_minimal+=T_GW.C()*okvis::kinematics::crossMx(T_WS.C()*p_SP_S)*F.block<3,9>(3,6);
          J1= squareRootInformation_*J1_minimal;

          if(jacobiansMinimal!=NULL){
            if(jacobiansMinimal[1]!=NULL){
              Eigen::Map<Eigen::Matrix<double,3,9,Eigen::RowMajor> > J1_minimal_mapped(jacobiansMinimal[1]);
              J1_minimal_mapped= squareRootInformation_*J1_minimal;
            }
          }
        }
        if(jacobians[2] != NULL) {
          Eigen::Map<Eigen::Matrix<double,3,7,Eigen::RowMajor> > J2(jacobians[2]);
          Eigen::Matrix<double,3,6,Eigen::RowMajor> J2_minimal;

          J2_minimal.topLeftCorner<3,3>()=-Eigen::Matrix3d::Identity();
          J2_minimal.topRightCorner<3,3>()=okvis::kinematics::crossMx(T_GW.C()*(T_WS.C()*p_SP_S+T_WS.r()));

          // pseudo inverse of the local parametrization Jacobian:
          Eigen::Matrix<double,6,7,Eigen::RowMajor> J_lift;
          PoseLocalParameterization::liftJacobian(parameters[2],J_lift.data());

          // hallucinate Jacobian w.r.t. state
          J2= squareRootInformation_*J2_minimal*J_lift;

          if(jacobiansMinimal!=NULL){
            if(jacobiansMinimal[2]!=NULL){
              Eigen::Map<Eigen::Matrix<double,3,6,Eigen::RowMajor> > J2_minimal_mapped(jacobiansMinimal[2]);
              J2_minimal_mapped= squareRootInformation_*J2_minimal;
            }
          }
        }
      }
  }

  return true;
}

}  // namespace ceres
}  // namespace okvis
