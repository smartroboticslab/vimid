/*
 * SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2023 Dimos Tzoumanikas
 * SPDX-FileCopyrightText: 2023 Binbin Xu
 * SPDX-FileCopyrightText: 2023 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef DATARECORDER_H
#define DATARECORDER_H

#include <sys/stat.h>
#include <string>
#include <fstream>
#include <memory>
#include <atomic>
#include <iostream>
#include <thread>
#include <chrono>
#include <Eigen/Core>
#include <okvis/VioInterface.hpp>
#include <okvis/Time.hpp>


class dataRecorder
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    dataRecorder();
    dataRecorder(const std::string folder, const bool use_imu);
    ~dataRecorder();

    void initRecData(bool use_imu);
    void stopRecData();
    void recCamPose(const okvis::Time &timestamp, const okvis::kinematics::Transformation &T_WC);
    void recCamPose(const long int &timestamp, const okvis::kinematics::Transformation &T_WC);
    

private:
    std::string output_path;
    std::ofstream *camCsv;
    std::ofstream *insCsv;

    bool isRecData = true;

    // file helpers
    bool file_exists(const std::string &path);
    bool directory_exists(const std::string &path);
    void make_directory(const std::string &path);
};



#endif // DATARECORDER_H