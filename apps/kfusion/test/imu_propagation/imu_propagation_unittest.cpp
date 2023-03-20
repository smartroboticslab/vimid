/*
 * SPDX-FileCopyrightText: 2022 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2022 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/

#include <Eigen/Core>
#include <okvis/Measurements.hpp>
#include <okvis/VioParametersReader.hpp>
#include <okvis/kinematics/Transformation.hpp>

#include "imuMeasurement.h"


int main(){
    // Frame 0 pose
    Eigen::Vector3d r_0(1.00248e-10, 9.34187e-11, -8.56055e-11);
    Eigen::Quaterniond q_0(-0.304418, 0.949457, 0.0765416, 0.00137806);
    okvis::kinematics::Transformation T_WC_0(r_0, q_0);

    // Frame 1 pose
    Eigen::Vector3d r_1(0.0174067, 2.30992e-05, 0.00585296);
    Eigen::Quaterniond q_1(-0.299111, 0.95193, 0.0657514, -0.00622389);
    okvis::kinematics::Transformation T_WC_1(r_1, q_1);

    // Load IMU measurements
    std::string path = "/home/ryf/slam/vimid/apps/kfusion/vimid_test/imu_propagation";
    okvis::VioParametersReader vio_parameters_reader("/home/ryf/slam/vimid/config/config_indoor.yaml");
    okvis::VioParameters parameters;
    vio_parameters_reader.getParameters(parameters);
    imuMeasurement imu_measurement(parameters);

    // Open the IMU file
    std::ifstream imu_file(path + "/imu0/data.csv");
    if (!imu_file.good()){
        std::cout << "no imu file found at " << path+"/imu0/data.csv" << std::endl;
        return false;
    }
    int number_of_lines = 0;
    std::string line;

    // set reading position to second line
    imu_file.clear();
    imu_file.seekg(0, std::ios::beg);
    getline(imu_file, line);
    okvis::Time start(10.071590000);
    okvis::Time end(10.543694000);
    okvis::Time t_imu = start;
    while(getline(imu_file, line)){
        std::stringstream stream(line);
        std::string s;
        getline(stream, s, ',');

        long int time = std::stol(s);
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
        
        if (t_imu < start){
            continue;
        }
        else if (t_imu >= end){
            break;
        }
        std::cout << "IMU loader: Loading time " << t_imu << " ..." << std::endl;

        if (t_imu - start + okvis::Duration(1.0) > okvis::Duration(0.0)) {
            imu_measurement.addImuMeasurement(t_imu, acc, gyr, start, end);
        }
    } 

    Matrix4 camera_pose = imu_measurement.fromOkvisTransformation(T_WC_0);
    imu_measurement.propagation(camera_pose);
    Matrix4 estimate = imu_measurement.fromOkvisTransformation(imu_measurement.T_WC_);
    Matrix4 gt = imu_measurement.fromOkvisTransformation(T_WC_1);

    Eigen::Vector3d r = (imu_measurement.T_WC_).r();
    Eigen::Matrix3d C = (imu_measurement.T_WC_).C();
    Eigen::Quaterniond q(C);
    std::cout << r[0] << "," << r[1] << "," << r[2] << "," << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << std::endl;
    printMatrix4("Estimation: ", estimate);
    printMatrix4("Gt: ", gt);
    imu_measurement.record(imu_measurement.estimation_file_, imu_measurement.start_, T_WC_0);
    imu_measurement.record(imu_measurement.estimation_file_, imu_measurement.end_, imu_measurement.T_WC_);
    imu_measurement.record(imu_measurement.gt_file_, imu_measurement.start_, T_WC_0);
    imu_measurement.record(imu_measurement.gt_file_, imu_measurement.end_, T_WC_1);
    return 0;
}