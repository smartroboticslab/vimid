/*
 * SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2023 Binbin Xu
 * SPDX-FileCopyrightText: 2023 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef VIMID_KEYFRAME_H
#define VIMID_KEYFRAME_H


#include <set>
#include "opencv2/core/core.hpp"
#include "math_utils.h"
#include "tracking.h"



class KeyFrame
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  KeyFrame();

  KeyFrame(long unsigned int mnId, cv::Mat &img);

  KeyFrame(long unsigned int mnId, cv::Mat &img, TrackingPointer rgbdTracker);

  // Set functions
  void SetPose(const Matrix4 &T_WC);
  void SetInformation(const Eigen::Matrix<double,6,6> &JtJ);
  void SetVertex(float3 *model_vertex);
  void SetNormal(float3 *model_normal);
  void SetRGBDTracker(TrackingPointer rgbdTracker);

  // Get functions
  Matrix4 GetPose();


  // Extract features and descriptors
  bool ExtractAndDescribe(bool isDraw);


public:

    // static long unsigned int nNextId;
    long unsigned int mnId;
    long unsigned int mnRelId;
    // const long unsigned int mnFrameId;

    // Image
    cv::Mat RGB_;
    uint2 imgSize_;
    // Vertex and normal
    float **l_I_;
    float **l_gradx_;
    float **l_grady_;
    float3 *model_vertex_;
    float3 *model_normal_;
    float3 **l_vertex_;
    float3 **l_normal_;
    float3 **icp_cov_pyramid_;
    // Background outlier mask
    std::vector<cv::Mat> bg_outlier_mask_;
    TrackingPointer rgbdTracker_ = nullptr;

    // Features
    cv::Mat descriptors_;

    // SE3 pose
    Matrix4 T_WC_;
    Matrix4 T_CW_;

    // Information matrix
    Eigen::Matrix<double,6,6> infoMat_;
};


#endif //VIMID_KEYFRAME_H
