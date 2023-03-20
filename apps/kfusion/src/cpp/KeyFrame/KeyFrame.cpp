/*
 * SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2023 Binbin Xu
 * SPDX-FileCopyrightText: 2023 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/


#include "KeyFrame.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>


KeyFrame::KeyFrame() {}

KeyFrame::KeyFrame(long unsigned int mnId, cv::Mat &img) : mnId(mnId), RGB_(img.clone()) {

  // Information matrix
  infoMat_ = Eigen::Matrix<double,6,6>::Identity();
}


KeyFrame::KeyFrame(long unsigned int mnId, cv::Mat &img, TrackingPointer rgbdTracker) :
                   mnId(mnId), RGB_(img.clone()) {

  rgbdTracker_ = rgbdTracker;

  imgSize_ = rgbdTracker_->getImgSize();
  bg_outlier_mask_ = rgbdTracker_->getBgOutMask();

  l_I_ = (float **) calloc(sizeof(float *) * rgbdTracker_->getGNOptIter().size(), 1);
  l_gradx_ = (float **) calloc(sizeof(float *) * rgbdTracker_->getGNOptIter().size(), 1);
  l_grady_ = (float **) calloc(sizeof(float *) * rgbdTracker_->getGNOptIter().size(), 1);
  model_vertex_ = (float3 *) calloc(sizeof(float3) * (imgSize_.x * imgSize_.y), 1);
  model_normal_ = (float3 *) calloc(sizeof(float3) * (imgSize_.x * imgSize_.y), 1);
  l_vertex_ = (float3 **) calloc(sizeof(float3 *) * rgbdTracker_->getGNOptIter().size(), 1);
  l_normal_ = (float3 **) calloc(sizeof(float3 *) * rgbdTracker_->getGNOptIter().size(), 1);
  icp_cov_pyramid_ = (float3 **) calloc(sizeof(float3 *) * rgbdTracker_->getGNOptIter().size(), 1);
  std::vector<uint2> localImgSize;

  for (unsigned int level = 0; level < rgbdTracker_->getGNOptIter().size(); ++level) {
    uint2 localimagesize = make_uint2(imgSize_.x / (int) pow(2, level), imgSize_.y / (int) pow(2, level));
    localImgSize.push_back(localimagesize); //from fine to coarse

    l_I_[level] = (float *) calloc(sizeof(float) * (imgSize_.x * imgSize_.y) / (int) pow(2, level), 1);
    l_gradx_[level] = (float *) calloc(sizeof(float) * (imgSize_.x * imgSize_.y) / (int) pow(2, level), 1);
    l_grady_[level] = (float *) calloc(sizeof(float) * (imgSize_.x * imgSize_.y) / (int) pow(2, level), 1);
    l_vertex_[level] = (float3 *) calloc(sizeof(float3) * (imgSize_.x * imgSize_.y) / (int) pow(2, level), 1);
    l_normal_[level] = (float3 *) calloc(sizeof(float3) * (imgSize_.x * imgSize_.y) / (int) pow(2, level), 1);
    icp_cov_pyramid_[level] = (float3 *) calloc(sizeof(float3) * (imgSize_.x * imgSize_.y) / (int) pow(2, level), 1);
  }

  for (int level = rgbdTracker_->getGNOptIter().size() - 1; level >= 0; --level) {
    memcpy(l_I_[level], rgbdTracker_->get_l_I()[level], sizeof(float) * localImgSize[level].x * localImgSize[level].y);
    memcpy(l_gradx_[level], rgbdTracker_->get_gradx()[level], sizeof(float) * localImgSize[level].x * localImgSize[level].y);
    memcpy(l_grady_[level], rgbdTracker_->get_grady()[level], sizeof(float) * localImgSize[level].x * localImgSize[level].y);
    memcpy(l_vertex_[level], rgbdTracker_->get_l_vertex()[level], sizeof(float3) * localImgSize[level].x * localImgSize[level].y);
    memcpy(l_normal_[level], rgbdTracker_->get_l_normal()[level], sizeof(float3) * localImgSize[level].x * localImgSize[level].y);
    memcpy(icp_cov_pyramid_[level], rgbdTracker_->get_icp_cov_pyramid()[level], sizeof(float3) * localImgSize[level].x * localImgSize[level].y);
  }

  // Information matrix
  infoMat_ = Eigen::Matrix<double,6,6>::Identity();
}


void KeyFrame::SetPose(const Matrix4 &T_WC) {
  T_WC_ = T_WC;
}

void KeyFrame::SetInformation(const Eigen::Matrix<double,6,6> &JtJ) {
  infoMat_ = JtJ.eval();
}

void KeyFrame::SetVertex(float3 *model_vertex) {
  memcpy(model_vertex_, model_vertex, sizeof(float) * imgSize_.x * imgSize_.y);
}

void KeyFrame::SetNormal(float3 *model_normal) {
  memcpy(model_normal_, model_normal, sizeof(float) * imgSize_.x * imgSize_.y);
}

void KeyFrame::SetRGBDTracker(TrackingPointer rgbdTracker) {
  rgbdTracker_ = rgbdTracker;
}

Matrix4 KeyFrame::GetPose() {
  return T_WC_;
}


bool KeyFrame::ExtractAndDescribe(bool isDraw) {
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();

  // Detect Oriented FAST corners
  detector->detect(RGB_, keypoints);

  // Compute BRIEF descriptor
  extractor->compute(RGB_, keypoints, descriptors);

  if (isDraw) {
    cv::Mat out;
    cv::drawKeypoints(RGB_, keypoints, out, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB features", out);
    // cv::waitKey(0);
  }

  descriptors_ = descriptors.clone();
  return true;
}
