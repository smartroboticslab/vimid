/*
 * SPDX-FileCopyrightText: 2022 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2022 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/

#include <Eigen/Core>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Estimator.hpp>
#include <okvis/VioParametersReader.hpp>
#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/ceres/ImuError.hpp>
#include "dataLoader.h"
#include <boost/filesystem.hpp>
#include <math_utils.h>
#include <commons.h>


int main(){
    Eigen::Vector3d r(1, 2, 3);
    Eigen::Quaterniond q(1, 1, 1, 1);
    okvis::kinematics::Transformation T(r, q);
    Eigen::Matrix4d T_eigen = T.T();
    std::cout << "Origin" << std::endl << T_eigen << std::endl;
    Eigen::Matrix<double, 6, 1> dt = Eigen::Matrix<double, 6, 1>::Ones();
    dt = dt * 1;

    okvis::kinematics::Transformation T_update_1(T_eigen);
    T_update_1.oplus(dt);
    std::cout << "First" << std::endl << T_update_1.T() << std::endl;

    okvis::kinematics::Transformation T_update_2;
    T_update_2.oplus(dt);
    std::cout << T_update_2.T() << std::endl;
    std::cout << "Second" << std::endl << T_update_2.T() * T_eigen << std::endl;


    // Load all IMU measurements
    std::string path_ = "/home/ryf/slam/dataset/indoor";
    dataLoader loader(path_);
    okvis::VioParametersReader vio_parameters_reader("/home/ryf/slam/vimid/config/config_indoor.yaml");
    okvis::VioParameters parameters_;
    vio_parameters_reader.getParameters(parameters_);

    // Open the IMU file
    std::ifstream imu_file_(path_ + "/imu0/data.csv");
    if (!imu_file_.good()){
        std::cout << "no imu file found at " << path_+"/imu0/data.csv" << std::endl;
    }
    okvis::Time start_(0.0);
    okvis::Time t_imu = start_;
    int number_of_lines = 0;
    std::string line_;
    std::getline(imu_file_, line_); // set reading position to the second line
    std::shared_ptr<okvis::threadsafe::ThreadSafeQueue<okvis::ImuMeasurement>> imuMeasurementsPtr_;
    imuMeasurementsPtr_.reset(new okvis::threadsafe::ThreadSafeQueue<okvis::ImuMeasurement>);
    while (std::getline(imu_file_, line_)){
        ++number_of_lines;

        std::stringstream stream(line_);
        std::string s;
        std::getline(stream, s, ',');
        std::string nanoseconds = s.substr(s.size() - 9, 9);
        std::string seconds = s.substr(0, s.size() - 9);

        Eigen::Vector3d gyr;
        for (int j = 0; j < 3; ++j) {
            std::getline(stream, s, ',');
            gyr[j] = std::stof(s);
        }

        Eigen::Vector3d acc;
        for (int j = 0; j < 3; ++j) {
            std::getline(stream, s, ',');
            acc[j] = std::stof(s);
        }
        t_imu = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));

        // Create an imu measurement
        okvis::ImuMeasurement imu_measurement;
        imu_measurement.measurement.accelerometers = acc;
        imu_measurement.measurement.gyroscopes = gyr;
        imu_measurement.timeStamp = t_imu;

        // Add to the queue
        imuMeasurementsPtr_->PushBlockingIfFull(imu_measurement, 120000);
    }
    imu_file_.close();

    std::cout << "No. IMU measurements: " << number_of_lines << std::endl;
    if (number_of_lines <= 0) {
        std::cout << "no imu messages present in " << path_+"/imu0/data.csv" << std::endl;
    }

    // Add keyframes vector
    std::vector<okvis::Time> times;
    okvis::Time latest(0);
    int num_camera_images = 0;
    std::string folder(path_ + "/cam0/data");

    std::vector<std::string> image_names_;
    for (auto it = boost::filesystem::directory_iterator(folder);
         it != boost::filesystem::directory_iterator(); it++) {
        if (!boost::filesystem::is_directory(it->path())) {  //we eliminate directories
            num_camera_images++;
            image_names_.push_back(it->path().filename().string());
        } else {
            continue;
        }
    }
    if (num_camera_images == 0) {
        std::cout << "no images at " << folder << std::endl;
    }
    std::cout << "No. cam0 images: " << num_camera_images << std::endl;
    // the filenames are not going to be sorted. So do this here
    std::sort(image_names_.begin(), image_names_.end());

    auto cam_iterators_ = image_names_.begin();

    // Set state variable
    Eigen::Matrix4d T_WC_0 = Eigen::Matrix4d::Identity();   // T_WC 0
    Eigen::Matrix4d T_WC_1 = Eigen::Matrix4d::Identity();   // T_WC 1
    Eigen::Matrix4d T_WS_0 = Eigen::Matrix4d::Identity();   // T_WS 0
    Eigen::Matrix4d T_WS_1 = Eigen::Matrix4d::Identity();   // T_WS 1
    Eigen::Vector3d W_v_WS = Eigen::Vector3d::Zero();       // velocity
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();           // gyroscope biases
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();           // accelerometer biases

    // Set useful variable holders
    bool isFirst = true;
    Eigen::Matrix<double, 15, 1> linearizationPoint = Eigen::Matrix<double, 15, 1>::Zero();
    Eigen::Matrix<double, 30, 30> JtJ = Eigen::Matrix<double, 30, 30>::Zero();
    Eigen::Matrix<double, 30, 1> JtR = Eigen::Matrix<double, 30, 1>::Zero();

    // Set transformation T_SC
    Eigen::Matrix4d T_SC = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_CS = Eigen::Matrix4d::Identity();
    T_SC.block<3, 1>(0, 3) = Eigen::Vector3d(-0.0302200001, 0.0074000000, 0.0160200000);
    T_CS = T_SC.inverse();

    // Initialize useful values
    Eigen::Matrix<double, 15, 1> b_star0 = Eigen::Matrix<double, 15, 1>::Zero();
    Eigen::Matrix<double ,15 ,15> H_star = Eigen::Matrix<double ,15 ,15>::Identity();
    H_star.block<6,6>(0,0) = Eigen::Matrix<double, 6, 6>::Identity(); // pose
    H_star.block<3,3>(6,6) = Eigen::Matrix3d::Identity();
    H_star.block<3,3>(9,9) = Eigen::Matrix3d::Identity() * 1.0 / (0.03 * 0.03);
    H_star.block<3,3>(12,12) = Eigen::Matrix3d::Identity() * 1.0 / (0.01 * 0.01);

    // Set okvis variables
    okvis::Time prevTimeStamp_;
    okvis::Time currTimeStamp_;
    okvis::Time viTimeStamp_;
    okvis::ImuMeasurementDeque imuMeasurements_;
    okvis::ImuMeasurementDeque imuData_;

    int tick = 0;
    while (true) {
        tick ++;
        std::cout << "Tick " << tick << std::endl;
        // Load next gt pose
        Matrix4 gtPose = loader.loadNextGTPose();
        // Load IMU measurements before current viTimeStamp
        // Check if at the end
        if (cam_iterators_ == image_names_.end()) {
            std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
            break;
        }

        // Obtain current viTimeStamp
        std::string nanoseconds = cam_iterators_->substr(cam_iterators_->size() - 13, 9);
        std::string seconds = cam_iterators_->substr(0, cam_iterators_->size() - 13);
        viTimeStamp_ = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));
        std::cout << "Current Time: " << viTimeStamp_ << std::endl;
        if (start_ == okvis::Time(0.0)) {
            start_ = viTimeStamp_;
        }

        // Record the new viTimeStamp
        prevTimeStamp_ = currTimeStamp_;
        currTimeStamp_ = viTimeStamp_;

        okvis::Duration imu_overlap_dur(0.02);
        okvis::Time imu_overlap_time(0, 20000000);
        okvis::Time imuBeginTime(0, 0);
        if (prevTimeStamp_ >= imu_overlap_time) {
            imuBeginTime = prevTimeStamp_ - imu_overlap_dur;
        }
        okvis::Time imuEndTime;
        if(tick == 1) {
            // on the first run through, IMU measurements are only used to estimate gravity vector
            // (therefore, only want measurements up to the current time)
            imuEndTime = currTimeStamp_;
        }
        else {
            imuEndTime = currTimeStamp_ + imu_overlap_dur;
        }

        // Add all new IMU measurements to the deque
        okvis::ImuMeasurement data;
        bool waitForImu = true;
        while(waitForImu) {
            if(imuMeasurementsPtr_->PopNonBlocking(&data)) {
                imuMeasurements_.push_back(data);
                if(data.timeStamp > imuEndTime) {
                    waitForImu = false;
                }
            }
        }

        // Create a transformation deque (imuData) of times between previous viTimeStamp and current viTimeStamp
        auto first_msg = imuMeasurements_.begin();
        auto last_msg = imuMeasurements_.end();
        --last_msg; // last_msg actually refers to the past-the-end element of the deque... we want the actual end

        // Move the first_msg iterator to the first message within the time window
        if(tick != 1) { // want to include the very first measurement
            while(first_msg->timeStamp <= imuBeginTime && first_msg != imuMeasurements_.end()) {
                ++first_msg;
            }
        }

        // Move the last_msg iterator to the last message within the time window
        while(last_msg->timeStamp >= imuEndTime && last_msg != imuMeasurements_.begin() && last_msg > first_msg) {
            --last_msg;
        }

        if(last_msg != imuMeasurements_.end()) {
            ++last_msg;
        }

        imuData_ = okvis::ImuMeasurementDeque(first_msg, last_msg);
        cam_iterators_++;

        if (tick == 1) {
            // Initialize the IMU to camera transformation
            Eigen::Vector3d acc = Eigen::Vector3d::Zero();
            for (auto it = imuData_.begin(); it < imuData_.end(); it++){
                acc += it->measurement.accelerometers;
            }
            acc /= double(imuData_.size());
            acc /= double(imuData_.size());
            Eigen::Vector3d acc_norm = acc.normalized();
            Eigen::Vector3d grav(0.0, 0.0, 1.0);
            Eigen::Vector3d omega = grav.cross(acc_norm).normalized();
            double angle = std::acos(grav.transpose() * acc_norm);
            omega *= angle;

            Eigen::Matrix3d init_pose = rodrigues(-omega);
            linearizationPoint.segment<3>(0) = Eigen::Vector3d(0.0, 0.0, 0.0);
            Eigen::AngleAxisd aa;
            aa = init_pose;
            linearizationPoint.segment<3>(3) = aa.angle() * aa.axis();

            // Set initialized pose
            T_WS_1.block<3,3>(0,0) = init_pose;
            T_WS_0 = T_WS_1;
            T_WC_0 = T_WS_0 = T_WS_1 * T_SC;
            T_WC_1 = T_WS_1;
        } else {
            // Jacobians
            double *jacobians[4];
            Eigen::Matrix<double,15,7,Eigen::RowMajor> J0 = Eigen::Matrix<double,15,7,Eigen::RowMajor>::Zero();
            Eigen::Matrix<double,15,9,Eigen::RowMajor> J1 = Eigen::Matrix<double,15,9,Eigen::RowMajor>::Zero();
            Eigen::Matrix<double,15,7,Eigen::RowMajor> J2 = Eigen::Matrix<double,15,7,Eigen::RowMajor>::Zero();
            Eigen::Matrix<double,15,9,Eigen::RowMajor> J3 = Eigen::Matrix<double,15,9,Eigen::RowMajor>::Zero();
            jacobians[0] = J0.data();
            jacobians[1] = J1.data();
            jacobians[2] = J2.data();
            jacobians[3] = J3.data();
            double* jacobiansMinimal[4];
            Eigen::Matrix<double,15,6,Eigen::RowMajor> J0min = Eigen::Matrix<double,15,6,Eigen::RowMajor>::Zero();
            Eigen::Matrix<double,15,9,Eigen::RowMajor> J1min = Eigen::Matrix<double,15,9,Eigen::RowMajor>::Zero();
            Eigen::Matrix<double,15,6,Eigen::RowMajor> J2min = Eigen::Matrix<double,15,6,Eigen::RowMajor>::Zero();
            Eigen::Matrix<double,15,9,Eigen::RowMajor> J3min = Eigen::Matrix<double,15,9,Eigen::RowMajor>::Zero();
            jacobiansMinimal[0] = J0min.data();
            jacobiansMinimal[1] = J1min.data();
            jacobiansMinimal[2] = J2min.data();
            jacobiansMinimal[3] = J3min.data();
            Eigen::Matrix<double, 15, 1> residuals = Eigen::Matrix<double, 15, 1>::Zero();

            // Previous state
            okvis::kinematics::Transformation prevState;
            okvis::SpeedAndBias prevSpeedAndBias;
            prevState.set(T_WS_1);
            prevSpeedAndBias.segment<3>(0) = W_v_WS;
            prevSpeedAndBias.segment<3>(3) = bg;
            prevSpeedAndBias.segment<3>(6) = ba;

            // Current state
            okvis::kinematics::Transformation currState;
            okvis::SpeedAndBias currSpeedAndBias;
            currState = prevState;
            currSpeedAndBias = prevSpeedAndBias;

            // Instantiate the IMU cost function
            okvis::ceres::ImuError *imu_cost_function;
            imu_cost_function = new okvis::ceres::ImuError(imuData_,
                                                           parameters_.imu,
                                                           prevTimeStamp_,
                                                           currTimeStamp_);

            std::cout << "Start to iterate for solving the pose" << std::endl;
            for (int i = 0; i < 10; i++){
                // Calculate the IMU jacobians for the current pose estimate
                // std::cout << "Iter: " << i << std::endl;

                // Set up new current state and speedAndBias
                // T_WS_0 = T_WC_0 * T_CS;
                prevState.set(T_WS_0);

                // T_WS_1 = T_WC_1 * T_CS;
                currState.set(T_WS_1);

                // Set up the paramter blocks
                okvis::ceres::PoseParameterBlock poseParameterBlock_0(prevState, 0, prevTimeStamp_);
                okvis::ceres::PoseParameterBlock poseParameterBlock_1(currState, 2, currTimeStamp_);
                okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_0(prevSpeedAndBias, 1, prevTimeStamp_);
                okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_1(currSpeedAndBias, 3, currTimeStamp_);
                double *parameters[4];
                parameters[0] = poseParameterBlock_0.parameters();
                parameters[1] = speedAndBiasParameterBlock_0.parameters();
                parameters[2] = poseParameterBlock_1.parameters();
                parameters[3] = speedAndBiasParameterBlock_1.parameters();

                // Call evaluateWithMinimalJacobians on imu_cost function
                bool check = imu_cost_function->EvaluateWithMinimalJacobians(parameters, residuals.data(), jacobians, jacobiansMinimal);
                if(!check){
                    std::cout << "Did not obtain Jacobians correctly!" << std::endl;
                }

                // Set the Jacobians
                Eigen::Matrix<double, 15, 15> j_imu0 = Eigen::Matrix<double, 15, 15>::Zero();
                Eigen::Matrix<double, 15, 15> j_imu1 = Eigen::Matrix<double, 15, 15>::Zero();
                j_imu0.block<15,6>(0,0) = J0min;
                j_imu0.block<15,9>(0,6) = J1min;
                j_imu1.block<15,6>(0,0) = J2min;
                j_imu1.block<15,9>(0,6) = J3min;
                // Eigen::Matrix<double, 15, 15> correction = Eigen::Matrix<double, 15, 15>::Identity();
                // Eigen::Matrix<double, 3, 1> W_r_C_S = T_WC_0.block<3,3>(0,0) * T_CS.block<3,1>(0,3);
                // correction.block<3,3>(0,3) = -skew(W_r_C_S);
                // j_imu0 = j_imu0 * correction;

                // correction = Eigen::Matrix<double, 15, 15>::Identity();
                // W_r_C_S = T_WC_1.block<3,3>(0,0) * T_CS.block<3,1>(0,3);
                // correction.block<3,3>(0,3) = -skew(W_r_C_S);
                // j_imu1 = j_imu1 * correction;

                if (false) {
                    Eigen::Matrix<double, 15, 1> res_ref = residuals;

                    Eigen::Matrix<double, 30, 1> dir = Eigen::Matrix<double, 30, 1>::Ones();
                    Eigen::Matrix<double, 30, 1> delta30 = 1e-8 * dir;

                    okvis::kinematics::Transformation T_update_old(T_WC_0);
                    T_update_old.oplus(delta30.segment<6>(0));
                    okvis::kinematics::Transformation T_update_new(T_WC_1);
                    T_update_new.oplus(delta30.segment<6>(15));

                    Eigen::Matrix4d T_WC_0_test = T_update_old.T();
                    Eigen::Matrix4d T_WC_1_test = T_update_new.T();

                    okvis::SpeedAndBias prevSpeedAndBias_test = prevSpeedAndBias + delta30.segment<9>(6);
                    okvis::SpeedAndBias currSpeedAndBias_test = currSpeedAndBias + delta30.segment<9>(21);

                    Eigen::Matrix4d T_WS_0_test = T_WC_0_test * T_CS;
                    Eigen::Matrix4d T_WS_1_test = T_WC_1_test * T_CS;

                    okvis::kinematics::Transformation prevState_test;
                    okvis::kinematics::Transformation currState_test;
                    prevState_test.set(T_WS_0_test);
                    currState_test.set(T_WS_1_test);

                    okvis::ceres::PoseParameterBlock poseParameterBlock_0_test(prevState_test, 0, prevTimeStamp_);
                    okvis::ceres::PoseParameterBlock poseParameterBlock_1_test(currState_test, 2, currTimeStamp_);
                    okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_0_test(prevSpeedAndBias_test, 1, prevTimeStamp_);
                    okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_1_test(currSpeedAndBias_test, 3, currTimeStamp_);
                    double* parameters_test[4];
                    parameters_test[0]=poseParameterBlock_0_test.parameters();
                    parameters_test[1]=speedAndBiasParameterBlock_0_test.parameters();
                    parameters_test[2]=poseParameterBlock_1_test.parameters();
                    parameters_test[3]=speedAndBiasParameterBlock_1_test.parameters();

                    double* jacobians_test[4];
                    Eigen::Matrix<double,15,7,Eigen::RowMajor> J0_test;
                    Eigen::Matrix<double,15,9,Eigen::RowMajor> J1_test;
                    Eigen::Matrix<double,15,7,Eigen::RowMajor> J2_test;
                    Eigen::Matrix<double,15,9,Eigen::RowMajor> J3_test;
                    jacobians_test[0]=J0_test.data();
                    jacobians_test[1]=J1_test.data();
                    jacobians_test[2]=J2_test.data();
                    jacobians_test[3]=J3_test.data();
                    double* jacobiansMinimal_test[4];
                    Eigen::Matrix<double,15,6,Eigen::RowMajor> J0min_test;
                    Eigen::Matrix<double,15,9,Eigen::RowMajor> J1min_test;
                    Eigen::Matrix<double,15,6,Eigen::RowMajor> J2min_test;
                    Eigen::Matrix<double,15,9,Eigen::RowMajor> J3min_test;
                    jacobiansMinimal_test[0]=J0min_test.data();
                    jacobiansMinimal_test[1]=J1min_test.data();
                    jacobiansMinimal_test[2]=J2min_test.data();
                    jacobiansMinimal_test[3]=J3min_test.data();
                    Eigen::Matrix<double, 15, 1> residuals_test = Eigen::Matrix<double, 15, 1>::Zero();

                    // Call evaluateWithMinimalJacobians on imu_cost function
                    check = imu_cost_function->EvaluateWithMinimalJacobians(parameters_test, residuals_test.data(), jacobians_test, jacobiansMinimal_test);
                    if(!check) {
                        std::cout << "Did not obtain Jacobians correctly!" << std::endl;
                    }

                    Eigen::Matrix<double, 15, 1> res_new = residuals_test;
                    Eigen::Matrix<double, 15, 1> res_dir = (res_new - res_ref) * 1e8;

                    std::cout << "Residual direction numerical: " << res_dir.transpose() << std::endl;
                    std::cout << "Residual direction analytical: " << (j_imu0 * dir.segment<15>(0) + j_imu1 * dir.segment<15>(15)).transpose() << std::endl;
                    // std::cout << "Analytical direction 2: " << (j_imu0_test * dir.segment<15>(0) + j_imu1_test * dir.segment<15>(15)).transpose() << std::endl;
                    std::cout << "Norm Numerical = " << (res_dir.segment<3>(0)).norm() << std::endl;
                    std::cout << "Norm Analytical = " << ((j_imu0 * dir.segment<15>(0) + j_imu1 * dir.segment<15>(15)).segment<3>(0)).norm() << std::endl;
                }


                // Solve for the update
                // Calculate DeltaChi (diff between prev pose and linearization point)
                Eigen::Matrix<double, 15, 1> DeltaChi = Eigen::Matrix<double, 15, 1>::Zero();

                DeltaChi.segment<3>(0) = T_WC_0.block<3,1>(0,3) - linearizationPoint.segment<3>(0);
                Eigen::Quaterniond q_lp;
                if(linearizationPoint.segment<3>(3).isZero(1e-8)) {
                    std::cout << "linearizationPoint close to zero..." << std::endl;
                    Eigen::Matrix<double, 3, 3> eye = Eigen::Matrix<double, 3, 3>::Identity();
                    Eigen::Quaterniond q_lp_temp(eye);
                    q_lp = q_lp_temp;
                    // std::cout << "inside isZero" << std::endl;
                }
                else {
                    Eigen::AngleAxisd aa_lp(linearizationPoint.segment<3>(3).norm(), linearizationPoint.segment<3>(3).normalized());
                    Eigen::Quaterniond q_lp_temp(aa_lp);
                    q_lp = q_lp_temp;
                }
                Eigen::Quaterniond q(T_WC_0.block<3,3>(0,0));
                Eigen::AngleAxisd aa_diff(q * q_lp.inverse());
                DeltaChi.segment<3>(3) = aa_diff.angle() * aa_diff.axis();
                DeltaChi.segment<9>(6) = prevSpeedAndBias - linearizationPoint.segment<9>(6);

                // Calcualte b_star
                Eigen::Matrix<double, 15, 1> b_star = b_star0 + H_star * DeltaChi;

                // Construct the complete Jacobian
                JtJ.block<15,15>(0,0) = H_star + j_imu0.transpose().eval() * j_imu0;
                JtJ.block<15,15>(0,15) = j_imu0.transpose().eval() * j_imu1;
                JtJ.block<15,15>(15,0) = j_imu1.transpose().eval() * j_imu0;
                JtJ.block<15,15>(15,15) = j_imu1.transpose().eval() * j_imu1;
                JtR.segment<15>(0) = b_star + j_imu0.transpose().eval() * residuals;
                JtR.segment<15>(15) = j_imu1.transpose().eval() * residuals;

                // Enforce symmetry, condition the Hessian...
                JtJ = (0.5 * (JtJ + JtJ.transpose().eval()).eval());

                // Solve the system
                Eigen::Matrix<double, 30, 1> delta30;
                Eigen::LDLT<Eigen::Matrix<double, 30, 30>> ldlt(JtJ);
                if (ldlt.info() != Eigen::Success) {
                    std::cout << "bad30" << std::endl;
                }
                else {
                    // nothing yet
                }
                delta30 = -ldlt.solve(JtR);

                // Compute the update (T_WC)
                // okvis::kinematics::Transformation T_update_old(T_WC_0);
                // T_update_old.oplus(delta30.segment<6>(0));
                // T_WC_0 = T_update_old.T();
                // okvis::kinematics::Transformation T_update_new(T_WC_1);
                // T_update_new.oplus(delta30.segment<6>(15));
                // T_WC_1 = T_update_new.T();

                // Compute the update (T_WS)
                okvis::kinematics::Transformation T_update_old(T_WS_0);
                T_update_old.oplus(delta30.segment<6>(0));
                T_WS_0 = T_update_old.T();
                okvis::kinematics::Transformation T_update_new(T_WS_1);
                T_update_new.oplus(delta30.segment<6>(15));
                T_WS_1 = T_update_new.T();
                T_WC_0 = T_WS_0 * T_SC;
                T_WC_1 = T_WS_1 * T_SC;

                // okvis::kinematics::Transformation T_update_old;
                // T_update_old.oplus(delta30.segment<6>(0));
                // T_WC_0 = T_update_old.T() * T_WC_0;
                // okvis::kinematics::Transformation T_update_new;
                // T_update_new.oplus(delta30.segment<6>(15));
                // T_WC_1 = T_update_new.T() * T_WC_1;


                // Update the velocity and biases based on the optimization results
                prevSpeedAndBias.segment<3>(0) += delta30.segment<3>(6);
                prevSpeedAndBias.segment<3>(3) += delta30.segment<3>(9);
                prevSpeedAndBias.segment<3>(6) += delta30.segment<3>(12);

                currSpeedAndBias.segment<3>(0) += delta30.segment<3>(21);
                currSpeedAndBias.segment<3>(3) += delta30.segment<3>(24);
                currSpeedAndBias.segment<3>(6) += delta30.segment<3>(27);

            }
            free(imu_cost_function);
            // Update the state
            std::cout << "Updating the state..." << std::endl;

            // Find the covariance of the new state
            Eigen::Matrix<double, 15, 15> V = JtJ.block<15,15>(0,0);
            Eigen::Matrix<double, 15, 15> V1 = 0.5 * (V + V.transpose()); // enforce symmetry

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> saes(V1);

            double epsilon = std::numeric_limits<double>::epsilon();
            double tol = epsilon * V1.cols() * saes.eigenvalues().array().maxCoeff();

            Eigen::Matrix<double, 15, 15> Vinv =  (saes.eigenvectors()) * Eigen::Matrix<double, 15, 1>((
                    saes.eigenvalues().array() > tol).select(
                    saes.eigenvalues().array().inverse(), 0)).asDiagonal() * (saes.eigenvectors().transpose());

            H_star = JtJ.block<15,15>(15,15) - JtJ.block<15,15>(15,0) * Vinv * JtJ.block<15,15>(0,15);
            H_star = (0.5 * (H_star + H_star.transpose().eval())).eval(); // enforce symmetry

            b_star0 = JtR.segment<15>(15) - JtJ.block<15,15>(15,0) * Vinv * JtR.segment<15>(0);

            // Store the linearization point
            linearizationPoint.segment<3>(0) = T_WC_1.block<3,1>(0,3);
            Eigen::AngleAxisd aa;
            aa = T_WC_1.block<3,3>(0,0);
            linearizationPoint.segment<3>(3) = aa.angle() * aa.axis();
            linearizationPoint.segment<9>(6) = currSpeedAndBias;

            // Update the speedAndBias
            W_v_WS = currSpeedAndBias.segment<3>(0); // okvis expects velocity in the world frame
            bg = currSpeedAndBias.segment<3>(3);
            ba = currSpeedAndBias.segment<3>(6);
        }

        okvis::kinematics::Transformation T_WC_1_es(T_WC_1);
        Matrix4 estimate = fromOkvisToMidFusion(T_WC_1_es);
        printMatrix4("Estimation: ", estimate);
        printMatrix4("Gt: ", gtPose);
        std::cout << "==============================" << std::endl;
        // imu_mea.record(imu_mea.estimation_file_, currTimeStamp_, T_WC_1_es);
        // imu_mea.record(imu_mea.gt_file_, currTimeStamp_, fromMidFusionToOkvis(gtPose));

        // Delete from main deque all imu measurements before the previous viTimeStamp with 0.02s buffer
        static const okvis::Time overlap(0, 20000000);
        static const okvis::Time zero(0, 0);
        okvis::Duration eraseUntil;
        if (prevTimeStamp_ > overlap) {
            eraseUntil = prevTimeStamp_ - overlap;
        }
        auto eraseEnd = imuMeasurements_.begin();
        while (eraseEnd->timeStamp - zero < eraseUntil) {
            ++eraseEnd;
        }
        if (eraseEnd != imuMeasurements_.begin()) {
            --eraseEnd; // go back to the previous position so < prevTimeStamp - overlap
        }
        imuMeasurements_.erase(imuMeasurements_.begin(), eraseEnd);
    }


    return 0;
}