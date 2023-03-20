/*
 * SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2023 Binbin Xu
 * SPDX-FileCopyrightText: 2023 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/
#ifndef DATALOADER_H
#define DATALOADER_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <config.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math_utils.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/VioParametersReader.hpp>
#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/Duration.hpp>
#include <okvis/Time.hpp>


class dataLoader
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int index_;    // counter for mask-rcnn
    long int time_;   // original name of the frame
    cv::Mat *preInputRGB_ = nullptr;
    cv::Mat *preInputDepth_ = nullptr;
    okvis::Time viTimeStamp_;
    okvis::Time prevTimeStamp_;
    okvis::Time currTimeStamp_;
    okvis::Time imuBeginTime_;
    bool processFirst_ = true;
    okvis::ImuMeasurementDeque imuMeasurements_;
    okvis::ImuMeasurementDeque imuData_;

    std::shared_ptr<okvis::threadsafe::ThreadSafeQueue<okvis::ImuMeasurement>> imuMeasurementsPtr_;
    
    // imu files
    okvis::VioParameters parameters_;
    okvis::VioParametersReader vio_parameters_reader_;
    okvis::Duration deltaT_;
    unsigned int numCameras_ = 1;
    std::string line_;
    std::ifstream imu_file_;
    std::vector<std::string> image_names_;
    std::vector<std::string>::iterator cam_iterators_;
    int counter_;
    okvis::Time start_;


    dataLoader();
    dataLoader(const std::string path);
    dataLoader(Configuration *c);
    ~dataLoader();

    bool initRGBDLoader(const std::string rgbPath, const std::string depthPath);
    bool initImuLoader(const std::string imuConfigFile);
    bool initGTLoader(const std::string gtFile);
    bool initMaskLoader(const std::string maskPath);

    bool checkNextRGBDFrame();

    cv::Mat loadNextRGBFrame();
    cv::Mat loadNextDepthFrame();
    bool loadImuMeasurements();
    void deleteImuMeasurements();
    Matrix4 loadNextGTPose();


private:
    std::string path_;
    std::ifstream *rgbdCsv_;    // read RGB-D image
    std::ifstream *gtCsv_;      // read ground truth pose
    std::ifstream *rgbdAssociateCsv_;

    int offset_;
    bool useAssociate_;         // judge if use the associate file

    // data files
    std::string rgbFramePath_;
    std::string depthFramePath_;
    std::string gtPosePath_;

    bool isFirst_ = true;

    // file helpers
    bool file_exists(const std::string &path);
    bool directory_exists(const std::string &path);
};

#endif // DATALOADER_H