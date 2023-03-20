/*
 * SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2023 Binbin Xu
 * SPDX-FileCopyrightText: 2023 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef VIMID_POSE_ESTIMATION_H
#define VIMID_POSE_ESTIMATION_H

#include "math_utils.h"
#include "OdometryProvider.h"
#include "commons.h"
#include "timings.h"
#include <memory>
#include <Eigen/Core>

// #include <sophus/se3.hpp>



/// \brief Project 2d pixels to 3d points in camera coordinate
static cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K){
  return cv::Point2d(
          (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
          (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}


/// \brief Find feature matches using ORB features
static void find_feature_matches(cv::Mat img_1, cv::Mat img_2,
                                 std::vector<cv::KeyPoint> &keypoints_1,
                                 std::vector<cv::KeyPoint> &keypoints_2,
                                 std::vector<cv::DMatch> &matches,
                                 bool refine,
                                 bool render){
  cv::Mat descriptors_1, descriptors_2;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();    // using ORB features
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  detector->compute(img_1, keypoints_1, descriptors_1);
  detector->compute(img_2, keypoints_2, descriptors_2);

  matcher->match(descriptors_1, descriptors_2, matches);

  auto min_max = std::minmax_element(matches.begin(), matches.end(),
                                     [](const cv::DMatch &m1, const cv::DMatch &m2)
                                     {return m1.distance < m2.distance;});
  double min_dist = min_max.first->distance;

  if (render) {
    cv::Mat img_match;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    cv::imshow("all matches", img_match);
  }

  if (refine) {
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
      // In case the min_dist is very low, we set 30.0 for the minimum boundary
      if (matches[i].distance <= std::max(2 * min_dist, 20.0)) {
        good_matches.push_back(matches[i]);
      }
    }
    matches = good_matches;
    if (render) {
      cv::Mat img_goodmatch;
      cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
      cv::imshow("good matches", img_goodmatch);
    }
  }
}

/// \brief Pose estimation using 2d-2d epipolar geometry
// static void pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
//                           std::vector<cv::KeyPoint> keypoints_2,
//                           std::vector<cv::DMatch> matches,
//                           const cv::Mat K,
//                           cv::Mat &R, cv::Mat &t){
//
//   // 1. Calculate the fundamental matrix and essential matrix
//   std::vector<cv::Point2f> points1;
//   std::vector<cv::Point2f> points2;
//   for (int i = 0; i < (int)matches.size(); i++) {
//     points1.push_back(keypoints_1[matches[i].queryIdx].pt);
//     points2.push_back(keypoints_2[matches[i].trainIdx].pt);
//   }
//
//   cv::Mat fundamental_matrix;
//   fundamental_matrix = cv::findFundamentalMat(points1, points2, CV_FM_RANSAC);
//
//   cv::Point2d principle_point(K.at<double>(0, 2), K.at<double>(1, 2));
//   double focal_length = K.at<double>(0, 0);
//   cv::Mat essential_matrix;
//   essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principle_point);
//
//   // 2. Recover the transformation from essential matrix
//   cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principle_point);
// }


/// \brief Pose estimation using 3d-3d ICP
static void pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1,
                                 const std::vector<cv::Point3f> &pts2,
                                 Eigen::Matrix3d &R,
                                 Eigen::Vector3d &t){

  // 1. Eliminate the centroid coordinates
  cv::Point3f p1, p2;
  int N = pts1.size();
  for (int i = 0; i < N; i++){
    p1 += pts1[i];
    p2 += pts2[i];
  }
  p1 = cv::Point3f(cv::Vec3f(p1) / N);
  p2 = cv::Point3f(cv::Vec3f(p2) / N);

  std::vector<cv::Point3f> q1(N), q2(N);
  for (int i = 0; i < N; i++){
    q1[i] = pts1[i] - p1;
    q2[i] = pts2[i] - p2;
  }

  // 2. Compute q1*q2^T
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; i++){
    W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
  }

  // 3. SVD on W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  R = U * (V.transpose());
  t = Eigen::Vector3d(p1.x, p1.y, p1.z) - R * Eigen::Vector3d(p2.x, p2.y, p2.z);
}


/// \brief Pose estimation using 3d-2d BA using GN
// static void pose_estimation_3d2d(const std::vector<cv::Point3d> &pts1,
//                                  const std::vector<cv::Point2d> &pts2,
//                                  Eigen::Matrix3d &R,
//                                  Eigen::Vector3d &t){
//
// }



#endif //VIMID_POSE_ESTIMATION_H
