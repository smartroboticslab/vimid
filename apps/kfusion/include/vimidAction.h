/*
 * SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2023 Binbin Xu
 * SPDX-FileCopyrightText: 2023 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef VIMIDACTION_H
#define VIMIDACTION_H

#include <kernels.h>
#include <interface.h>
#include <stdint.h>
#include <cstddef>
#include <tick.h>
#include <vector>
#include <sstream>
#include <string>
#include <cstring>
#include <math_utils.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>
#include <getopt.h>
#include <perfstats.h>
#include <PowerMonitor.h>
#include <atomic>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <okvis/VioInterface.hpp>
#include <okvis/VioParametersReader.hpp>
#include <okvis/ThreadedKFVio.hpp>
#include <okvis/ZR300Reader.h>
#include "dataLoader.h"
#include "dataRecorder.h"

//read mask and class from mask-rcnn
#include "segmentation.h"

#ifndef __QT__

#include <draw.h>
#endif

DepthReader *createReader(Configuration *c, std::string filename = "");


/**
 * @brief This class saves and draws the pose.
 */
class PoseViewer
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  constexpr static const double imageSize = 500.0;

  /// \brief The default constructor.
  PoseViewer();

  // this we can register as a callback
  void publishFullStateAsCallback(
      const okvis::Time & /*t*/, const okvis::kinematics::Transformation & T_WS,
      const Eigen::Matrix<double, 9, 1> & speedAndBiases,
      const Eigen::Matrix<double, 3, 1> & /*omega_S*/);
  
  void display();

  void display(const okvis::kinematics::Transformation & T_WS);

 private:
  cv::Point2d convertToImageCoordinates(const cv::Point2d & pointInMeters) const;
 
  void drawPath();

  cv::Mat _image;
  std::vector<cv::Point2d> _path;
  std::vector<double> _heights;
  double _scale = 1.0;
  double _min_x = -0.5;
  double _min_y = -0.5;
  double _min_z = -0.5;
  double _max_x = 0.5;
  double _max_y = 0.5;
  double _max_z = 0.5;
  const double _frameScale = 0.2;  // [m]
  std::atomic_bool drawing_;
  std::atomic_bool showing_;
};


class vimidAction
{
  enum RunType {naive = 0, okvisLive, okvisDataset};
  enum OkvisDatasetType {realsense = 0, viSim};

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  virtual ~vimidAction();
  RunType runType_;
  vimidAction(int w, int h);
  void init(Configuration *c, Kfusion *k);
  void init(Configuration *c);
  void storeStats(int frame, double *timings, float3 pos, bool tracked, bool integrated);
  int procNaiveAFrame(/*bool processFrame, bool renderImages, bool reset*/);
  int procGtAFrame();
  int procVimidAFrame();
  int procMIDAFrame();
  void procOkvisAFrameCallback(const std::vector<okvis::VioInterface::FrameInfo,
                               Eigen::aligned_allocator<okvis::VioInterface::FrameInfo>> &frameInfos);
  void stateCb(const std::vector<okvis::VioInterface::FrameInfo,
               Eigen::aligned_allocator<okvis::VioInterface::FrameInfo>> &frameInfos);
  bool runNaive();
  bool runOkvisLive();
  bool runOkvisDataset();
  bool runGtVimid();
  bool runMIDFusion();
  bool runVimid();
  bool run();

  // void procOkvisAFrameCallback(
  //   const std::vector<okvis::VioInterface::FrameInfo,
  //                     Eigen::aligned_allocator<okvis::VioInterface::FrameInfo> >& frameInfos  
  // );

  Matrix4 fromOkvisTransformation(okvis::kinematics::Transformation& T_OKVIS);
  Matrix4 fromOkvisTransformation(okvis::kinematics::Transformation& T_WS,
                                  okvis::kinematics::Transformation& T_SC);
  bool depthTransformation(cv::Mat& mat, double scale = 1.0);
  bool disToUshortDepth(const cv::Mat& mat, uint16_t * depth, double scale = 1.0);
  bool rgbToUchar3Image(const cv::Mat& mat, uchar3 * image);
  bool checkOkvisDataset(std::string& path, OkvisDatasetType& type);

  // debug functions
  void printPose(Matrix4& p);

  //outlier check
  bool use_okvis_tracking_ = true;

 private:
  int width_, height_;
  uint2 inputSize_;
  uint2 computationSize_;
  PerfStats Stats_;
  PowerMonitor *powerMonitor_ = NULL;
  uint16_t *inputDepth_ = NULL;
  uchar3 *inputRGB_ = NULL;
  uchar4 *depthRender_ = NULL;
  uchar4 *raycastRender_ = NULL;
  uchar4 *trackRender_ = NULL;
  uchar4 *segmentRender_ = NULL;
  uchar4 *volumeRender_ = NULL;
  uchar4 *extraRender_ = NULL;
  uchar4 *mainViewRender_ = NULL;
  uchar4 *topViewRender_ = NULL;
  uchar4 *maskRender_ = NULL;
  DepthReader *reader_ = NULL;
  Kfusion *kfusion_ = NULL;
  Configuration *config_ = NULL;

  float3 init_pose_;
  std::ostream *logstream_ = &std::cout;
  std::ofstream logfilestream_;
  bool processFrame_;
  bool renderImages_;
  bool reset_;

  // for okvis tracking use
  okvis::ZR300Reader * rsReader_ = NULL;
  float rsFx_, rsFy_;
  float rsCx_, rsCy_;
  long int rsFrameIndex_;
  float rsDistortCoeff_[4];
  cv::Mat rsIntrinsics_;
  cv::Mat rsDistCoeffs_;

  int frameOffset_;
  bool firstFrame_;

  // for vimid IMU use
  okvis::Duration deltaT_;
  std::string path_;
  OkvisDatasetType dataType_;
  okvis::VioParameters parameters_;
  okvis::ThreadedKFVio* okvis_estimator_ = NULL;
  unsigned int numCameras_ = 1;
  std::string line_;
  std::ifstream imu_file_;
  std::vector < std::string > image_names_;
  std::vector < std::string > ::iterator cam_iterators_;
  int counter_;
  okvis::Time start_;

  // for okvis visulization
  // poseViewer * poseVis = NULL;
  okvis::kinematics::Transformation T_WS_;
  okvis::kinematics::Transformation T_WC_;
  okvis::kinematics::Transformation T_SC_;
  // file helpers
  bool file_exists(const std::string &path);
  bool directory_exists(const std::string &path);
  void make_directory(const std::string &path);

  //remove the outliers from raw depth information
  const int kInitFrameNumber_=10;
  const int kOutlierDepth_=0;
  cv::Mat filter_depth_;
  bool isInputDenoised_;
  //std::ofstream depth_diff_out_;

  void filterInputDepth(const cv::Mat &input_depth);
  void doSegmentation(const cv::Mat& depth);
  Segmentation segmentation_;
  // DepthDifferenceFlag* segmentationDepthResult;

  // const float icp_rgb_residual_ = 20.1f;
  // const float icp_rgb_residual_ = 4.01f;
  const float icp_rgb_residual_ = 5.0f;
  const float rgb_residual_ = 5.0f;
  // const float rgb_residual_ = 4.1f;

  // for data loading recording
  dataRecorder *recorder_ = NULL;
  dataLoader *loader_ = NULL;
};


#endif // VIMIDACTION_H