/*
 * SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2023 Binbin Xu
 * SPDX-FileCopyrightText: 2023 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/

#include <vimidAction.h>
#include <functional>
#include <sys/stat.h>
#include <okvis/VioParametersReader.hpp>
#include <okvis/ThreadedKFVio.hpp>
#include <boost/filesystem.hpp>
#include <okvis/ZR300Reader.h>

PoseViewer::PoseViewer()
{
  cv::namedWindow("OKVIS Top View");
  _image.create(imageSize, imageSize, CV_8UC3);
  drawing_ = false;
  showing_ = false;
}

void PoseViewer::publishFullStateAsCallback(
    const okvis::Time & /*t*/, const okvis::kinematics::Transformation &T_WS,
    const Eigen::Matrix<double, 9, 1> &speedAndBiases,
    const Eigen::Matrix<double, 3, 1> & /*omega_S*/)
{
  // just append the path
  Eigen::Vector3d r = T_WS.r();
  Eigen::Matrix3d C = T_WS.C();
  _path.push_back(cv::Point2d(r[0], r[1]));
  _heights.push_back(r[2]);
  // maintain scaling
  if (r[0] - _frameScale < _min_x)
    _min_x = r[0] - _frameScale;
  if (r[1] - _frameScale < _min_y)
    _min_y = r[1] - _frameScale;
  if (r[2] < _min_z)
    _min_z = r[2];
  if (r[0] + _frameScale > _max_x)
    _max_x = r[0] + _frameScale;
  if (r[1] + _frameScale > _max_y)
    _max_y = r[1] + _frameScale;
  if (r[2] > _max_z)
    _max_z = r[2];
  _scale = std::min(imageSize / (_max_x - _min_x), imageSize / (_max_y - _min_y));

  // draw it
  while (showing_)
  {
  }
  drawing_ = true;
  // erase
  _image.setTo(cv::Scalar(10, 10, 10));
  drawPath();
  // draw axes
  Eigen::Vector3d e_x = C.col(0);
  Eigen::Vector3d e_y = C.col(1);
  Eigen::Vector3d e_z = C.col(2);
  cv::line(
      _image,
      convertToImageCoordinates(_path.back()),
      convertToImageCoordinates(
          _path.back() + cv::Point2d(e_x[0], e_x[1]) * _frameScale),
      cv::Scalar(0, 0, 255), 1, CV_AA);
  cv::line(
      _image,
      convertToImageCoordinates(_path.back()),
      convertToImageCoordinates(
          _path.back() + cv::Point2d(e_y[0], e_y[1]) * _frameScale),
      cv::Scalar(0, 255, 0), 1, CV_AA);
  cv::line(
      _image,
      convertToImageCoordinates(_path.back()),
      convertToImageCoordinates(
          _path.back() + cv::Point2d(e_z[0], e_z[1]) * _frameScale),
      cv::Scalar(255, 0, 0), 1, CV_AA);

  // some text:
  std::stringstream postext;
  postext << "position = [" << r[0] << ", " << r[1] << ", " << r[2] << "]";
  cv::putText(_image, postext.str(), cv::Point(15, 15),
              cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  std::stringstream veltext;
  veltext << "velocity = [" << speedAndBiases[0] << ", " << speedAndBiases[1] << ", " << speedAndBiases[2] << "]";
  cv::putText(_image, veltext.str(), cv::Point(15, 35),
              cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

  drawing_ = false; // notify
}

void PoseViewer::display()
{
  while (drawing_)
  {
  }
  showing_ = true;
  cv::imshow("OKVIS Top View", _image);
  showing_ = false;
  // cv::waitKey(1);
}

void PoseViewer::display(const okvis::kinematics::Transformation &T_WS)
{
  // just append the path
  Eigen::Vector3d r = T_WS.r();
  Eigen::Matrix3d C = T_WS.C();
  _path.push_back(cv::Point2d(r[0], r[1]));
  _heights.push_back(r[2]);
  // maintain scaling
  if (r[0] - _frameScale < _min_x)
    _min_x = r[0] - _frameScale;
  if (r[1] - _frameScale < _min_y)
    _min_y = r[1] - _frameScale;
  if (r[2] < _min_z)
    _min_z = r[2];
  if (r[0] + _frameScale > _max_x)
    _max_x = r[0] + _frameScale;
  if (r[1] + _frameScale > _max_y)
    _max_y = r[1] + _frameScale;
  if (r[2] > _max_z)
    _max_z = r[2];
  _scale = std::min(imageSize / (_max_x - _min_x), imageSize / (_max_y - _min_y));

  // erase
  _image.setTo(cv::Scalar(10, 10, 10));
  drawPath();
  // draw axes
  Eigen::Vector3d e_x = C.col(0);
  Eigen::Vector3d e_y = C.col(1);
  Eigen::Vector3d e_z = C.col(2);
  cv::line(
      _image,
      convertToImageCoordinates(_path.back()),
      convertToImageCoordinates(
          _path.back() + cv::Point2d(e_x[0], e_x[1]) * _frameScale),
      cv::Scalar(0, 0, 255), 1, CV_AA);
  cv::line(
      _image,
      convertToImageCoordinates(_path.back()),
      convertToImageCoordinates(
          _path.back() + cv::Point2d(e_y[0], e_y[1]) * _frameScale),
      cv::Scalar(0, 255, 0), 1, CV_AA);
  cv::line(
      _image,
      convertToImageCoordinates(_path.back()),
      convertToImageCoordinates(
          _path.back() + cv::Point2d(e_z[0], e_z[1]) * _frameScale),
      cv::Scalar(255, 0, 0), 1, CV_AA);

  // some text:
  std::stringstream postext;
  postext << "position = [" << r[0] << ", " << r[1] << ", " << r[2] << "]";
  cv::putText(_image, postext.str(), cv::Point(15, 15),
              cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

  cv::imshow("OKVIS Top View", _image);
  // cv::imwrite("/home/ryf/slam/vimid/apps/kfusion/indoor/top.png", _image);
  cv::waitKey(1);
}

cv::Point2d PoseViewer::convertToImageCoordinates(const cv::Point2d &pointInMeters) const
{
  cv::Point2d pt = (pointInMeters - cv::Point2d(_min_x, _min_y)) * _scale;
  return cv::Point2d(pt.x, imageSize - pt.y); // reverse y for more intuitive top-down plot
}

void PoseViewer::drawPath()
{
  for (size_t i = 0; i + 1 < _path.size();)
  {
    cv::Point2d p0 = convertToImageCoordinates(_path[i]);
    cv::Point2d p1 = convertToImageCoordinates(_path[i + 1]);
    cv::Point2d diff = p1 - p0;
    if (diff.dot(diff) < 2.0)
    {
      _path.erase(_path.begin() + i + 1); // clean short segment
      _heights.erase(_heights.begin() + i + 1);
      continue;
    }
    double rel_height = (_heights[i] - _min_z + _heights[i + 1] - _min_z) * 0.5 / (_max_z - _min_z);
    cv::line(
        _image,
        p0,
        p1,
        rel_height * cv::Scalar(255, 0, 0) + (1.0 - rel_height) * cv::Scalar(0, 0, 255),
        1, CV_AA);
    i++;
  }
}

/// \brief Constructor of vimidAction
vimidAction::vimidAction(int w, int h) : width_(w), height_(h),
                                         processFrame_(true), renderImages_(true), reset_(false),
                                         frameOffset_(2), firstFrame_(true),
                                         rsFrameIndex_(-1),
                                         runType_(RunType::naive)
{
  inputSize_ = make_uint2(width_, height_);
  // for viSim
  // rsFx_ = 600.0;
  // rsFy_ = 600.0;
  // rsCx_ = 320.0;
  // rsCy_ = 240.0;
  // rsDistortCoeff_[0] = 0.0;
  // rsDistortCoeff_[1] = 0.0;
  // rsDistortCoeff_[2] = 0.0;
  // rsDistortCoeff_[3] = 0.0;

  // for indoor
  // rsFx_ = 390.598938;
  // rsFy_ = 390.598938;
  // rsCx_ = 320.581665;
  // rsCy_ = 237.712845;
  // rsDistortCoeff_[0] = 0.0;
  // rsDistortCoeff_[1] = 0.0;
  // rsDistortCoeff_[2] = 0.0;
  // rsDistortCoeff_[3] = 0.0;

  // for outside
  rsFx_ = 383.432;
  rsFy_ = 383.089;
  rsCx_ = 314.521;
  rsCy_ = 254.927;
  rsDistortCoeff_[0] = 0.0;
  rsDistortCoeff_[1] = 0.0;
  rsDistortCoeff_[2] = 0.0;
  rsDistortCoeff_[3] = 0.0;
}

/// \brief Initializer
void vimidAction::init(Configuration *c)
{
  computationSize_ = make_uint2(
      inputSize_.x / c->compute_size_ratio,
      inputSize_.y / c->compute_size_ratio);
  init_pose_ = c->initial_pos_factor * c->volume_size;
  Kfusion *kfusion = new Kfusion(computationSize_, c->volume_resolution,
                                 c->volume_size, init_pose_, c->pyramid, *c);
  config_ = c;
  kfusion_ = kfusion;
  powerMonitor_ = new PowerMonitor();
  if (c->okvis_config_file.empty())
  {
    reader_ = createReader(c);
  }
  // construction scene reader and input buffer
  inputDepth_ = (uint16_t *)malloc(sizeof(uint16_t) * width_ * height_);
  inputRGB_ = (uchar3 *)malloc(sizeof(uchar3) * width_ * height_);
  depthRender_ = (uchar4 *)malloc(sizeof(uchar4) * computationSize_.x * computationSize_.y);
  raycastRender_ = (uchar4 *)malloc(sizeof(uchar4) * computationSize_.x * computationSize_.y);
  trackRender_ = (uchar4 *)malloc(sizeof(uchar4) * computationSize_.x * computationSize_.y);
  segmentRender_ = (uchar4 *)malloc(sizeof(uchar4) * computationSize_.x * computationSize_.y);
  volumeRender_ = (uchar4 *)malloc(sizeof(uchar4) * computationSize_.x * computationSize_.y);
  extraRender_ = (uchar4 *)malloc(sizeof(uchar4) * computationSize_.x * computationSize_.y);
  mainViewRender_ = (uchar4 *)malloc(sizeof(uchar4) * computationSize_.x * computationSize_.y);
  topViewRender_ = (uchar4 *)malloc(sizeof(uchar4) * computationSize_.x * computationSize_.y);
  maskRender_ = (uchar4 *)malloc(sizeof(uchar4) * computationSize_.x * computationSize_.y);

  kfusion_->setPoseScale(c->initial_pos_factor);

  if (c->log_file != "")
  {
    logfilestream_.open(c->log_file.c_str());
    logstream_ = &logfilestream_;
  }

  logstream_->setf(std::ios::fixed, std::ios::floatfield);

  // record data
  std::string path = config_->okvis_dataset_path + "/results/";
  recorder_ = new dataRecorder(path, config_->use_imu);
}

/// \brief Destructor
vimidAction::~vimidAction()
{
  delete powerMonitor_;
  free(inputDepth_);
  free(inputRGB_);
  free(depthRender_);
  free(raycastRender_);
  free(trackRender_);
  free(segmentRender_);
  free(volumeRender_);
  free(extraRender_);
  free(mainViewRender_);
  free(topViewRender_);
  free(maskRender_);
  recorder_->~dataRecorder();
  loader_->~dataLoader();
}

/// \brief Store the performance statistics
void vimidAction::storeStats(int frame, double *timings, float3 pos,
                             bool tracked, bool integrated)
{
  Stats.sample("frame", frame, PerfStats::FRAME);
  Stats.sample("acquisition", timings[1] - timings[0], PerfStats::TIME);
  Stats.sample("preprocessing", timings[2] - timings[1], PerfStats::TIME);
  Stats.sample("tracking", timings[4] - timings[3], PerfStats::TIME);
  Stats.sample("segmentation", timings[3] - timings[2] + timings[5] - timings[4], PerfStats::TIME);
  Stats.sample("integration", timings[6] - timings[5], PerfStats::TIME);
  Stats.sample("raycasting", timings[7] - timings[6], PerfStats::TIME);
  Stats.sample("rendering", timings[8] - timings[7], PerfStats::TIME);
  Stats.sample("computation", timings[7] - timings[1], PerfStats::TIME);
  Stats.sample("total", timings[8] - timings[0], PerfStats::TIME);
  Stats.sample("X", pos.x, PerfStats::DISTANCE);
  Stats.sample("Y", pos.y, PerfStats::DISTANCE);
  Stats.sample("Z", pos.z, PerfStats::DISTANCE);
  Stats.sample("tracked", tracked, PerfStats::INT);
  Stats.sample("integrated", integrated, PerfStats::INT);
}

/// \brief Process naive one frame
int vimidAction::procNaiveAFrame()
{
  bool tracked, integrated, raycasted, segmented, hasmaskrcnn;
  double start, end, startCompute, endCompute;
  uint2 render_vol_size;
  double timings[9];
  float3 pos;
  int frame;
  const uint2 inputSize = (reader_ != NULL) ? reader_->getinputSize() : make_uint2(640, 480);
  float4 camera = (reader_ != NULL) ? (reader_->getK() / config_->compute_size_ratio) : make_float4(0.0);
  if (config_->camera_overrided)
  {
    camera = config_->camera / config_->compute_size_ratio;
  }
  if (reset_)
  {
    frameOffset_ = reader_->getFrameNumber();
  }
  bool finished = false;
  // start to save performance
  if (processFrame_)
  {
    Stats_.start();
  }
  Matrix4 pose;
  timings[0] = tock();
  if (processFrame_ && reader_->readNextDepthFrame(inputRGB_, inputDepth_) &&
      (kfusion_->MaskRCNN_next_frame(reader_->getFrameNumber() - frameOffset_,
                                     config_->maskrcnn_folder)))
  {
    frame = reader_->getFrameNumber() - frameOffset_;
    if (frame < kfusion_->segment_startFrame_)
    {
      std::cout << frame << " finished" << std::endl;
      return finished;
    }
    if (powerMonitor_ != NULL && !firstFrame_)
    {
      powerMonitor_->start();
    }
    timings[1] = tock();

    if (kfusion_->render_color_)
    {
      kfusion_->preprocessing(inputDepth_, inputRGB_, inputSize_, config_->bilateralFilter);
    }
    else
    {
      kfusion_->preprocessing(inputDepth_, inputSize_, config_->bilateralFilter);
    }
    timings[2] = tock();

    hasmaskrcnn = kfusion_->readMaskRCNN(camera, frame, config_->maskrcnn_folder);
    timings[3] = tock();
    tracked = kfusion_->tracking(camera, config_->tracking_rate, frame);
    pos = kfusion_->getPosition();
    pose = kfusion_->getPose();
    timings[4] = tock();

    segmented = kfusion_->segment(camera, frame, config_->maskrcnn_folder, hasmaskrcnn);
    timings[5] = tock();

    integrated = kfusion_->integration(camera, config_->integration_rate, config_->mu, frame);
    timings[6] = tock();

    raycasted = kfusion_->raycasting(camera, config_->mu, frame);
    timings[7] = tock();
  }
  else
  {
    if (processFrame_)
    {
      finished = true;
      timings[0] = tock();
    }
  }
  if (renderImages_)
  {
    int render_frame = (processFrame_ ? reader_->getFrameNumber() - frameOffset_ : 0);
    /////////////visualize the segmentation/////////
    kfusion_->renderTrack(trackRender_, kfusion_->getComputationResolution(), 2, frame);
    kfusion_->renderMaskMotionWithImage(segmentRender_, kfusion_->getComputationResolution(), *(kfusion_->frame_masks_), render_frame);
    kfusion_->renderVolume(volumeRender_, kfusion_->getComputationResolution(),
                           (processFrame_ ? reader_->getFrameNumber() - frameOffset_ : 0),
                           config_->rendering_rate, camera, 0.75 * config_->mu,
                           false);
    kfusion_->renderVolume(extraRender_, kfusion_->getComputationResolution(),
                           render_frame,
                           config_->rendering_rate, camera, 0.75 * config_->mu,
                           true);
    kfusion_->renderInstance(raycastRender_, kfusion_->getComputationResolution(),
                             (kfusion_->rendered_mask_), render_frame, "rendered_mask_");

    kfusion_->renderMaskWithImage(depthRender_, kfusion_->getComputationResolution(),
                                  (kfusion_->geo2mask_result), render_frame, "geo2mask_result");
    timings[8] = tock();
    std::cout << frame << " finished" << std::endl;
  }
  if (!finished)
  {
    if (powerMonitor_ != NULL && !firstFrame_)
    {
      powerMonitor_->sample();
    }

    float xt = pose.data[0].w - init_pose_.x;
    float yt = pose.data[1].w - init_pose_.y;
    float zt = pose.data[2].w - init_pose_.z;
    storeStats(frame, timings, pos, tracked, integrated);
    if (config_->no_gui || (config_->log_file != ""))
    {
      *logstream_ << reader_->getFrameNumber() << "\t" << xt << "\t" << yt << "\t" << zt << "\t" << std::endl;
    }

    if (config_->log_file != "")
    {
      kfusion_->save_poses(config_->log_file, reader_->getFrameNumber());
      kfusion_->save_times(config_->log_file, reader_->getFrameNumber(), timings);
    }
    firstFrame_ = false;
  }
  return (finished);
}

/// \brief Callback function for OKVIS
void vimidAction::procOkvisAFrameCallback(
    const std::vector<okvis::VioInterface::FrameInfo,
                      Eigen::aligned_allocator<okvis::VioInterface::FrameInfo>> &frameInfos)
{
  std::cout << "procOkvisAFrameCallback is running" << std::endl;
  if (frameInfos.empty())
  {
    return;
  }
  okvis::VioInterface::FrameInfo frameStack = frameInfos.back();
  if (frameStack.depthImages.empty() || frameStack.images.empty())
  {
    return;
  }

  // Record camera pose
  okvis::Time timestamp = frameStack.timestamp;
  std::cout << "Timestamp: " << timestamp << std::endl;
  T_WS_ = frameStack.T_WS;
  T_WC_ = T_WS_ * frameStack.T_SC_i.front();
  recorder_->recCamPose(timestamp, T_WC_);

  // Visualization
  static float duration = tick();
  bool tracked, integrated, raycasted, segmented, hasmaskrcnn, outlier_checked;
  double timings[9];
  float3 pos;
  int frame;

  float4 camera = config_->camera / config_->compute_size_ratio;
  if (config_->camera_overrided)
  {
    camera = config_->camera / config_->compute_size_ratio;
  }

  if (config_->okvis_mapKeyFrameOnly && !frameStack.isKeyframe)
  {
    return;
  }

  bool isRGBValid = rgbToUchar3Image(frameStack.images.front(), inputRGB_);
  bool isDepthValid = disToUshortDepth(frameStack.depthImages.front(), inputDepth_, 1.0 / 5.0);

  // cv::imshow("RGB", frameStack.images.front());
  // cv::waitKey(0);

  if (reset_)
  {
    frameOffset_ = rsFrameIndex_;
  }
  bool finished = false;

  if (processFrame_)
  {
    Stats_.start();
  }
  Matrix4 pose;
  timings[0] = tock();
  if (!isRGBValid || !isDepthValid)
  {
    return;
  }
  if (processFrame_ && kfusion_->MaskRCNN_next_frame(rsFrameIndex_ - frameOffset_,
                                                     config_->maskrcnn_folder))
  {
    frame = rsFrameIndex_ - frameOffset_;
    std::cout << "rsframeIndex: " << rsFrameIndex_ << std::endl;
    std::cout << "Current frame ID: " << frame << std::endl;
    if (frame < kfusion_->segment_startFrame_)
    {
      std::cout << frame << " finished" << std::endl;
      return;
    }
    if (powerMonitor_ != NULL && !firstFrame_)
      powerMonitor_->start();

    timings[1] = tock();

    if (kfusion_->render_color_)
    {
      kfusion_->preprocessing(inputDepth_, inputRGB_, inputSize_, config_->bilateralFilter);
    }
    else
    {
      kfusion_->preprocessing(inputDepth_, inputSize_, config_->bilateralFilter);
    }
    timings[2] = tock();

    hasmaskrcnn = kfusion_->readMaskRCNN(camera, frame, config_->maskrcnn_folder);
    timings[3] = tock();

    if (use_okvis_tracking_)
    {
      Matrix4 p = fromOkvisTransformation(T_WC_);
      float3 default_pos = make_float3(0.0f, 0.0f, 0.0f);
      // tracked = kfusion_->adaptPose(p, default_pos, config_->tracking_rate, frame);
      tracked = kfusion_->okvisTrackingTwo(camera, config_->tracking_rate, frame, p);
    }
    else
    {
      tracked = kfusion_->tracking(camera, config_->tracking_rate, frame);
    }
    pos = kfusion_->getPosition();
    pose = kfusion_->getPose();
    timings[4] = tock();

    segmented = kfusion_->segment(camera, frame, config_->maskrcnn_folder, hasmaskrcnn);
    timings[5] = tock();

    integrated = kfusion_->integration(camera, config_->integration_rate, config_->mu, frame);
    timings[6] = tock();

    raycasted = kfusion_->raycasting(camera, config_->mu, frame);
    timings[7] = tock();

    // rsFrameIndex_++;
  }
  else
  {
    if (processFrame_)
    {
      finished = true;
      timings[0] = tock();
    }
  }

  if (renderImages_)
  {
    int render_frame = (processFrame_ ? frame : 0);

    /////////////visualize the segmentation/////////
    kfusion_->renderTrack(trackRender_, kfusion_->getComputationResolution(), 2, frame);
    kfusion_->renderMaskMotionWithImage(segmentRender_, kfusion_->getComputationResolution(), *(kfusion_->frame_masks_), render_frame);
    kfusion_->renderVolume(volumeRender_, kfusion_->getComputationResolution(),
                           (processFrame_ ? frame : 0),
                           config_->rendering_rate, camera, 0.75 * config_->mu,
                           false);
    kfusion_->renderVolume(extraRender_, kfusion_->getComputationResolution(),
                           render_frame,
                           config_->rendering_rate, camera, 0.75 * config_->mu,
                           true);
    kfusion_->renderInstance(raycastRender_, kfusion_->getComputationResolution(),
                             (kfusion_->rendered_mask_), render_frame, "rendered_mask_");

    kfusion_->renderMaskWithImage(depthRender_, kfusion_->getComputationResolution(),
                                  (kfusion_->geo2mask_result), render_frame, "geo2mask_result");
    timings[8] = tock();
    std::cout << frame << " finished" << std::endl;
    std::cout << "==========================================" << std::endl;
  }

  if (!finished)
  {
    if (powerMonitor_ != NULL && !firstFrame_)
      powerMonitor_->sample();

    float xt = pose.data[0].w - init_pose_.x;
    float yt = pose.data[1].w - init_pose_.y;
    float zt = pose.data[2].w - init_pose_.z;
    storeStats(frame, timings, pos, tracked, integrated);
    if ((config_->no_gui) || (config_->log_file != ""))
    {
      *logstream_ << frame << "\t" << xt << "\t" << yt << "\t" << zt << "\t" << std::endl;
    }

    if (config_->log_file != "")
    {
      kfusion_->save_poses(config_->log_file, frame);
      kfusion_->save_times(config_->log_file, frame, timings);
    }

    // if (config->no_gui && (config->log_file == ""))
    //	Stats.print();
    firstFrame_ = false;
  }
  return;
}

/// \brief Process one frame using ground truth
int vimidAction::procGtAFrame()
{
  bool tracked, integrated, raycasted, segmented, hasmaskrcnn;
  double start, end, startCompute, endCompute;
  uint2 render_vol_size;
  double timings[9];
  float3 pos;
  int frame;
  float4 camera = config_->camera / config_->compute_size_ratio;
  if (config_->camera_overrided)
  {
    camera = config_->camera / config_->compute_size_ratio;
  }
  if (reset_)
  {
    frameOffset_ = 0;
  }
  bool finished = false;
  // start to save performance
  if (processFrame_)
  {
    Stats_.start();
  }
  Matrix4 pose;
  timings[0] = tock();

  if (processFrame_ && loader_->checkNextRGBDFrame() &&
      (kfusion_->MaskRCNN_next_frame(loader_->index_,
                                     config_->maskrcnn_folder)))
  {
    // load RGB and depth frame then check the validity
    cv::Mat rgbFrame = loader_->loadNextRGBFrame();
    cv::Mat depthFrame = loader_->loadNextDepthFrame();
    // convert RGB and depth frame to mid-fusion version
    bool isRGBValid = rgbToUchar3Image(rgbFrame, inputRGB_);
    depthTransformation(depthFrame, 1.0);
    bool isDepthValid = disToUshortDepth(depthFrame, inputDepth_, 1.0);
    // load next ground truth pose
    Matrix4 p = loader_->loadNextGTPose();

    frame = loader_->index_;
    if (frame < kfusion_->segment_startFrame_)
    {
      std::cout << frame << " finished" << std::endl;
      return finished;
    }

    if (powerMonitor_ != NULL && !firstFrame_)
    {
      powerMonitor_->start();
    }
    timings[1] = tock();

    if (kfusion_->render_color_)
    {
      kfusion_->preprocessing(inputDepth_, inputRGB_, inputSize_, config_->bilateralFilter);
    }
    else
    {
      kfusion_->preprocessing(inputDepth_, inputSize_, config_->bilateralFilter);
    }
    timings[2] = tock();

    hasmaskrcnn = kfusion_->readMaskRCNN(camera, frame, config_->maskrcnn_folder);
    timings[3] = tock();

    tracked = kfusion_->okvisTracking(camera, config_->tracking_rate, frame, p);
    pos = kfusion_->getPosition();
    pose = kfusion_->getPose();
    timings[4] = tock();

    segmented = kfusion_->segment(camera, frame, config_->maskrcnn_folder, hasmaskrcnn);
    timings[5] = tock();

    integrated = kfusion_->integration(camera, config_->integration_rate, config_->mu, frame);
    timings[6] = tock();

    raycasted = kfusion_->raycasting(camera, config_->mu, frame);
    timings[7] = tock();
  }
  else
  {
    if (processFrame_)
    {
      finished = true;
      timings[0] = tock();
    }
  }
  if (renderImages_)
  {
    int render_frame = (processFrame_ ? frame : 0);
    /////////////visualize the segmentation/////////
    kfusion_->renderTrack(trackRender_, kfusion_->getComputationResolution(), 2, frame);
    kfusion_->renderMaskMotionWithImage(segmentRender_, kfusion_->getComputationResolution(), *(kfusion_->frame_masks_), render_frame);
    kfusion_->renderVolume(volumeRender_, kfusion_->getComputationResolution(),
                           (processFrame_ ? frame : 0),
                           config_->rendering_rate, camera, 0.75 * config_->mu,
                           false);
    kfusion_->renderVolume(extraRender_, kfusion_->getComputationResolution(),
                           render_frame,
                           config_->rendering_rate, camera, 0.75 * config_->mu,
                           true);
    kfusion_->renderInstance(raycastRender_, kfusion_->getComputationResolution(),
                             (kfusion_->rendered_mask_), render_frame, "rendered_mask_");

    kfusion_->renderMaskWithImage(depthRender_, kfusion_->getComputationResolution(),
                                  (kfusion_->geo2mask_result), render_frame, "geo2mask_result");
    timings[8] = tock();
    std::cout << frame << " finished" << std::endl;
  }
  if (!finished)
  {
    if (powerMonitor_ != NULL && !firstFrame_)
    {
      powerMonitor_->sample();
    }

    float xt = pose.data[0].w - init_pose_.x;
    float yt = pose.data[1].w - init_pose_.y;
    float zt = pose.data[2].w - init_pose_.z;
    storeStats(frame, timings, pos, tracked, integrated);
    if (config_->no_gui || (config_->log_file != ""))
    {
      *logstream_ << frame << "\t" << xt << "\t" << yt << "\t" << zt << "\t" << std::endl;
    }

    if (config_->log_file != "")
    {
      kfusion_->save_poses(config_->log_file, frame);
      kfusion_->save_times(config_->log_file, frame, timings);
    }
    firstFrame_ = false;
  }
  return (finished);
}

/// \brief Process one frame using VI-MID-FUSION
int vimidAction::procVimidAFrame()
{
  float duration = tick();
  bool tracked, integrated, raycasted, segmented, hasmaskrcnn;
  double start, end, startCompute, endCompute;
  uint2 render_vol_size;
  double timings[9];
  float3 pos;
  int frame;
  float4 camera = config_->camera / config_->compute_size_ratio;
  if (config_->camera_overrided)
  {
    camera = config_->camera / config_->compute_size_ratio;
  }
  if (reset_)
  {
    frameOffset_ = 0;
  }
  bool finished = false;
  // start to save performance
  if (processFrame_)
  {
    // Stats_.start();
  }
  Matrix4 pose;
  timings[0] = tock();

  if (processFrame_ && loader_->checkNextRGBDFrame() &&
      (kfusion_->MaskRCNN_next_frame(loader_->index_,
                                     config_->maskrcnn_folder)))
  {

    // load RGB and depth frame then check the validity
    cv::Mat rgbFrame = loader_->loadNextRGBFrame();
    cv::Mat depthFrame = loader_->loadNextDepthFrame();

    // convert RGB and depth frame to mid-fusion version
    bool isRGBValid = rgbToUchar3Image(rgbFrame, inputRGB_);
    depthTransformation(depthFrame, 1.0);
    bool isDepthValid = disToUshortDepth(depthFrame, inputDepth_, 1.0 / 5.0);

    // load imu measurements before next frame
    loader_->loadImuMeasurements();

    // load next ground truth pose
    Matrix4 p = loader_->loadNextGTPose();

    frame = loader_->index_;
    if (frame < kfusion_->segment_startFrame_)
    {
      loader_->deleteImuMeasurements();
      std::cout << frame << " finished" << std::endl;
      return finished;
    }

    kfusion_->save_input(inputRGB_, inputSize_, frame - kfusion_->segment_startFrame_);

    if (powerMonitor_ != NULL && !firstFrame_)
    {
      // powerMonitor_->start();
    }
    timings[1] = tock();

    if (kfusion_->render_color_)
    {
      kfusion_->preprocessing(inputDepth_, inputRGB_, inputSize_, config_->bilateralFilter);
    }
    else
    {
      kfusion_->preprocessing(inputDepth_, inputSize_, config_->bilateralFilter);
    }
    timings[2] = tock();

    hasmaskrcnn = kfusion_->readMaskRCNN(camera, frame, config_->maskrcnn_folder);
    timings[3] = tock();

    tracked = kfusion_->vimidTracking(camera, config_->tracking_rate, frame, p, loader_->parameters_,
                                      loader_->imuData_, loader_->prevTimeStamp_,
                                      loader_->currTimeStamp_, config_->use_imu,
                                      timings);
    // tracked = kfusion_->tracking(camera, config_->tracking_rate, frame);
    timings[4] = tock();

    pos = kfusion_->getPosition();
    pose = kfusion_->getPose();
    T_WC_ = fromMidFusionToOkvis(pose);
    recorder_->recCamPose(loader_->currTimeStamp_, T_WC_);

    segmented = kfusion_->segment(camera, frame, config_->maskrcnn_folder, hasmaskrcnn);
    timings[5] = tock();

    integrated = kfusion_->integration(camera, config_->integration_rate, config_->mu, frame);
    timings[6] = tock();

    raycasted = kfusion_->raycasting(camera, config_->mu, frame);
    timings[7] = tock();

    // Delete from main deque all imu measurements before the previous viTimeStamp
    loader_->deleteImuMeasurements();
    loader_->preInputRGB_ = &rgbFrame;
    loader_->preInputDepth_ = &depthFrame;
    kfusion_->prevResizeRGB_ = kfusion_->currResizeRGB_;
  }
  else
  {
    if (processFrame_)
    {
      finished = true;
      timings[0] = tock();
    }
  }
  if (renderImages_)
  {
    int render_frame = (processFrame_ ? frame : 0);
    /////////////visualize the segmentation/////////
    kfusion_->renderTrack(trackRender_, kfusion_->getComputationResolution(), 2, frame);
    kfusion_->renderMaskMotionWithImage(segmentRender_, kfusion_->getComputationResolution(), *(kfusion_->frame_masks_), render_frame);
    kfusion_->renderVolume(volumeRender_, kfusion_->getComputationResolution(),
                           (processFrame_ ? frame : 0),
                           config_->rendering_rate, camera, 0.75 * config_->mu,
                           false);
    kfusion_->renderVolume(extraRender_, kfusion_->getComputationResolution(),
                           render_frame,
                           config_->rendering_rate, camera, 0.75 * config_->mu,
                           true);
    kfusion_->renderInstance(raycastRender_, kfusion_->getComputationResolution(),
                             (kfusion_->rendered_mask_), render_frame, "rendered_mask_");

    kfusion_->renderMaskWithImage(maskRender_, kfusion_->getComputationResolution(),
                                  (kfusion_->geo2mask_result), render_frame, "geo2mask_result");
    kfusion_->renderDepth(depthRender_, kfusion_->getComputationResolution(), frame);
    // Matrix4 viewPose_1 = kfusion_->getInitPose();
    // Matrix4 rot_3 = rotateY(-3.14/24, 0.0, 0.0, 0.0);
    // Matrix4 rot_4 = rotateX(3.14/6, 0, 0, 0);
    // Matrix4 rot_5 = rotateZ(3.14/24, 0, 0.4, 0);
    // viewPose_1 = kfusion_->getInitPose() * rot_3 *rot_4 * rot_5;
    // viewPose_1.data[2].w -= 0.5f;
    // kfusion_->renderMainView(mainViewRender_, kfusion_->getComputationResolution(),
    //                         (processFrame_ ? frame : 0),
    //                         config_->rendering_rate, camera, 0.75 * config_->mu,
    //                         true, viewPose_1);
    // Matrix4 rot_1 = rotateY(3.14/12, 0.0, -7.0, 1.5);
    // Matrix4 rot_2 = rotateX(-3.14/3, 0.0, 0.0, 0.0);
    // // Matrix4 rot_1 = rotateY(3.14/2.5, 0.0, 0.0, 0.0);
    // // Matrix4 rot_2 = rotateX(0.0, 0.0, 0.0, 0.0);
    // Matrix4 viewPose_2 = kfusion_->getInitPose() * rot_1 * rot_2;
    // kfusion_->renderMainView(topViewRender_, kfusion_->getComputationResolution(),
    //                          (processFrame_ ? frame : 0),
    //                          config_->rendering_rate, camera, 0.75 * config_->mu,
    //                          true, viewPose_2);
    timings[8] = tock();
    std::cout << frame << " finished" << std::endl;
  }
  if (!finished)
  {
    // if (powerMonitor_ != NULL && !firstFrame_) {
    //   powerMonitor_->sample();
    // }

    // float xt = pose.data[0].w - init_pose_.x;
    // float yt = pose.data[1].w - init_pose_.y;
    // float zt = pose.data[2].w - init_pose_.z;
    // storeStats(frame, timings, pos, tracked, integrated);
    // if (config_->no_gui || (config_->log_file != "")) {
    //   *logstream_ << frame << "\t" << xt << "\t" << yt << "\t" << zt << "\t" << std::endl;
    // }

    // if (config_->log_file != "") {
    //   kfusion_->save_poses(config_->log_file, loader_->currTimeStamp_);
    //   // kfusion_->save_poses(config_->log_file, frame);
    //   kfusion_->save_times(config_->log_file, frame, timings);
    // }
    firstFrame_ = false;
  }
  kfusion_->delete_wrong_obj();
  return (finished);
}

int vimidAction::procMIDAFrame()
{
  float duration = tick();
  bool tracked, integrated, raycasted, segmented, hasmaskrcnn;
  double start, end, startCompute, endCompute;
  uint2 render_vol_size;
  double timings[9];
  float3 pos;
  int frame;
  float4 camera = config_->camera / config_->compute_size_ratio;
  if (config_->camera_overrided)
  {
    camera = config_->camera / config_->compute_size_ratio;
  }
  if (reset_)
  {
    frameOffset_ = 0;
  }
  bool finished = false;
  // start to save performance
  if (processFrame_)
  {
    Stats_.start();
  }
  Matrix4 pose;
  timings[0] = tock();

  if (processFrame_ && loader_->checkNextRGBDFrame() &&
      (kfusion_->MaskRCNN_next_frame(loader_->index_,
                                     config_->maskrcnn_folder)))
  {
    // load RGB and depth frame then check the validity
    cv::Mat rgbFrame = loader_->loadNextRGBFrame();
    cv::Mat depthFrame = loader_->loadNextDepthFrame();
    // cv::imshow("RGB", rgbFrame);
    // cv::waitKey(0);
    // convert RGB and depth frame to mid-fusion version
    bool isRGBValid = rgbToUchar3Image(rgbFrame, inputRGB_);
    depthTransformation(depthFrame, 1.0);
    bool isDepthValid = disToUshortDepth(depthFrame, inputDepth_, 1.0 / 5.0);
    // load imu measurements before next frame
    loader_->loadImuMeasurements();
    // std::cout << "Current vi time: " << loader_->viTimeStamp_ << std::endl;

    // load next ground truth pose
    Matrix4 p = loader_->loadNextGTPose();

    frame = loader_->index_;
    if (frame < kfusion_->segment_startFrame_)
    {
      loader_->deleteImuMeasurements();
      std::cout << frame << " finished" << std::endl;
      return finished;
    }

    kfusion_->save_input(inputRGB_, inputSize_, frame);

    if (powerMonitor_ != NULL && !firstFrame_)
    {
      powerMonitor_->start();
    }
    timings[1] = tock();
    // if (frame > 80) {
    //   cv::waitKey(0);
    // }

    if (kfusion_->render_color_)
    {
      kfusion_->preprocessing(inputDepth_, inputRGB_, inputSize_, config_->bilateralFilter);
    }
    else
    {
      kfusion_->preprocessing(inputDepth_, inputSize_, config_->bilateralFilter);
    }
    timings[2] = tock();

    hasmaskrcnn = kfusion_->readMaskRCNN(camera, frame, config_->maskrcnn_folder);
    timings[3] = tock();
    tracked = kfusion_->tracking(camera, config_->tracking_rate, frame);
    pos = kfusion_->getPosition();
    pose = kfusion_->getPose();
    T_WC_ = fromMidFusionToOkvis(pose);
    recorder_->recCamPose(loader_->currTimeStamp_, T_WC_);
    timings[4] = tock();

    segmented = kfusion_->segment(camera, frame, config_->maskrcnn_folder, hasmaskrcnn);
    timings[5] = tock();

    integrated = kfusion_->integration(camera, config_->integration_rate, config_->mu, frame);
    timings[6] = tock();

    raycasted = kfusion_->raycasting(camera, config_->mu, frame);
    timings[7] = tock();
  }
  else
  {
    if (processFrame_)
    {
      finished = true;
      timings[0] = tock();
    }
  }
  if (renderImages_)
  {
    int render_frame = (processFrame_ ? frame : 0);
    /////////////visualize the segmentation/////////
    kfusion_->renderTrack(trackRender_, kfusion_->getComputationResolution(), 2, frame);
    kfusion_->renderMaskMotionWithImage(segmentRender_, kfusion_->getComputationResolution(), *(kfusion_->frame_masks_), render_frame);
    kfusion_->renderVolume(volumeRender_, kfusion_->getComputationResolution(),
                           (processFrame_ ? frame : 0),
                           config_->rendering_rate, camera, 0.75 * config_->mu,
                           false);
    kfusion_->renderVolume(extraRender_, kfusion_->getComputationResolution(),
                           render_frame,
                           config_->rendering_rate, camera, 0.75 * config_->mu,
                           true);
    kfusion_->renderInstance(raycastRender_, kfusion_->getComputationResolution(),
                             (kfusion_->rendered_mask_), render_frame, "rendered_mask_");

    kfusion_->renderMaskWithImage(maskRender_, kfusion_->getComputationResolution(),
                                  (kfusion_->geo2mask_result), render_frame, "geo2mask_result");
    kfusion_->renderDepth(depthRender_, kfusion_->getComputationResolution(), frame);
    timings[8] = tock();
    std::cout << frame << " finished" << std::endl;
  }
  if (!finished)
  {
    if (powerMonitor_ != NULL && !firstFrame_)
    {
      powerMonitor_->sample();
    }

    if (config_->log_file != "")
    {
      kfusion_->save_poses(config_->log_file, loader_->currTimeStamp_);
      // kfusion_->save_poses(config_->log_file, frame);
      kfusion_->save_times(config_->log_file, frame, timings);
    }
    firstFrame_ = false;
  }
  kfusion_->delete_wrong_obj();
  return (finished);
}

/// \brief Run naive mid-fusion
bool vimidAction::runNaive()
{
  if (!config_->no_gui)
  {
#ifdef __QT__
    qtLinkKinectQt(argc, argv, &kfusion_, &reader_, &config_, depthRender_, trackRender_, volumeRender_, inputRGB_);
#else
    if ((reader_ == NULL) || !(reader_->cameraActive))
    {
      std::cerr << "No valid input file specified\n";
      return false;
    }
    int idx = 0;
    while (procNaiveAFrame() == 0)
    {
      drawthem(extraRender_, depthRender_, trackRender_, volumeRender_,
               segmentRender_, raycastRender_, computationSize_, computationSize_,
               computationSize_, computationSize_, computationSize_, computationSize_);

      if (config_->in_debug & config_->dump_volume_file != "")
      {
        double start = tock();
        kfusion_->dump_mesh(config_->dump_volume_file + std::to_string(idx));
        double end = tock();
        Stats_.sample("meshing", end - start, PerfStats::TIME);
      }

      idx++;

      if (config_->pause)
        getchar();
    }
#endif
  }
  else
  {
    if ((reader_ == NULL) || !(reader_->cameraActive))
    {
      std::cerr << "No valid input file specified\n";
      return false;
    }
    int idx = 0;
    while (procNaiveAFrame() == 0)
    {
      if (config_->in_debug & config_->dump_volume_file != "")
      {
        double start = tock();
        kfusion_->dump_mesh(config_->dump_volume_file + std::to_string(idx));
        double end = tock();
        Stats_.sample("meshing", end - start, PerfStats::TIME);
      }
      idx++;
    }

    std::cout << __LINE__ << std::endl;
  }
  // ==========     DUMP VOLUME      =========

  if (config_->dump_volume_file != "")
  {
    double start = tock();
    kfusion_->dump_mesh(config_->dump_volume_file);
    double end = tock();
    Stats_.sample("meshing", end - start, PerfStats::TIME);
  }

  if (powerMonitor_ && powerMonitor_->isActive())
  {
    std::ofstream powerStream("power.rpt");
    powerMonitor_->powerStats.print_all_data(powerStream);
    powerStream.close();
  }
  std::cout << "{";
  Stats_.print_all_data(std::cout, false);
  std::cout << "}" << std::endl;

  return true;
}

/// \brief Run okvis + mid-fusion
bool vimidAction::runOkvisDataset()
{
  // check whether the config file is valid
  if (config_->okvis_config_file.empty() || !file_exists(config_->okvis_config_file))
  {
    std::cerr << "OKVIS Error: improper config file input!" << std::endl;
    return false;
  }

  // check whether the dataset path is valid
  if (config_->okvis_dataset_path.empty() || !directory_exists(config_->okvis_dataset_path))
  {
    std::cerr << "OKVIS Error: improper dataset path input!" << std::endl;
    return false;
  }

  // cache the path to dataset input
  std::string path = config_->okvis_dataset_path;
  OkvisDatasetType dataType = OkvisDatasetType::viSim;

  if (!checkOkvisDataset(path, dataType))
  {
    std::cerr << "OKVIS Error: improper dataset format received!" << std::endl;
    return false;
  }

  okvis::Duration deltaT(0.0);

  // read configuration file
  okvis::VioParametersReader vio_parameters_reader(config_->okvis_config_file);
  okvis::VioParameters parameters;
  vio_parameters_reader.getParameters(parameters);

  std::shared_ptr<okvis::ThreadedKFVio> okvis_estimator = std::make_shared<okvis::ThreadedKFVio>(parameters);

  PoseViewer poseViewer;
  okvis_estimator->setDenseStereoCallback(
      std::bind(&vimidAction::procOkvisAFrameCallback, this, std::placeholders::_1));
  okvis_estimator->setBlocking(false);

  const unsigned int numCameras = parameters.nCameraSystem.numCameras();

  // Open the IMU file
  std::string line;
  std::ifstream imu_file(path + "/imu0/data.csv");
  if (!imu_file.good())
  {
    LOG(ERROR) << "no imu file found at " << path + "/imu0/data.csv";
    return -1;
  }
  int number_of_lines = 0;
  while (std::getline(imu_file, line))
    ++number_of_lines;
  LOG(INFO) << "No. IMU measurements: " << number_of_lines - 1;
  if (number_of_lines - 1 <= 0)
  {
    LOG(ERROR) << "no imu messages present in " << path + "/imu0/data.csv";
    return -1;
  }
  // set reading position to second line
  imu_file.clear();
  imu_file.seekg(0, std::ios::beg);
  std::getline(imu_file, line);

  std::vector<okvis::Time> times;
  okvis::Time latest(0);
  int num_camera_images = 0;
  std::vector<std::vector<std::string>> image_names(numCameras);
  for (size_t i = 0; i < numCameras; ++i)
  {
    if (parameters.nCameraSystem.isVirtual(i))
    {
      continue; // do nothing if the current camera is virtual camera
    }
    num_camera_images = 0;
    std::string folder(path + "/cam" + std::to_string(i) + "/data");

    for (auto it = boost::filesystem::directory_iterator(folder);
         it != boost::filesystem::directory_iterator(); it++)
    {
      if (!boost::filesystem::is_directory(it->path()))
      { // we eliminate directories
        num_camera_images++;
        image_names.at(i).push_back(it->path().filename().string());
      }
      else
      {
        continue;
      }
    }

    if (num_camera_images == 0)
    {
      LOG(ERROR) << "no images at " << folder;
      return false;
    }

    LOG(INFO) << "No. cam " << i << " images: " << num_camera_images;
    // the filenames are not going to be sorted. So do this here
    std::sort(image_names.at(i).begin(), image_names.at(i).end());
  }

  std::vector<std::vector<std::string>::iterator> cam_iterators(numCameras);
  for (size_t i = 0; i < numCameras; ++i)
  {
    cam_iterators.at(i) = image_names.at(i).begin();
  }

  int counter = 0;
  okvis::Time start(0.0);

  while (true)
  {
    rsFrameIndex_ = counter;
    if (!config_->no_gui)
    {
      okvis_estimator->display();
      poseViewer.display(T_WC_);

      drawthem(extraRender_, depthRender_, trackRender_, volumeRender_,
               segmentRender_, raycastRender_, computationSize_, computationSize_,
               computationSize_, computationSize_, computationSize_, computationSize_);
    }

    // check if at the end
    for (size_t i = 0; i < numCameras; ++i)
    {
      if (parameters.nCameraSystem.isVirtual(i))
      {
        continue; // do nothing if the current camera is virtual camera
      }
      if (cam_iterators[i] == image_names[i].end())
      {
        std::cout << std::endl
                  << "Finished. Press any key to exit." << std::endl
                  << std::flush;
        cv::waitKey();
        break;
      }
    }

    /// add images
    okvis::Time t;

    for (size_t i = 0; i < numCameras; ++i)
    {
      if (parameters.nCameraSystem.isVirtual(i))
      {
        continue; // do nothing if the current camera is virtual camera
      }

      cv::Mat filtered = cv::imread(
          path + "/cam" + std::to_string(i) + "/data/" + *cam_iterators.at(i),
          cv::IMREAD_GRAYSCALE);
      // std::cout << "RGB: " << path + "/cam" + std::to_string(i) + "/data/" + *cam_iterators.at(i) << std::endl;
      cv::Mat depth;
      if (parameters.nCameraSystem.isDepthCamera(i))
      {
        if (dataType == OkvisDatasetType::realsense)
        {
          depth = cv::imread(
              path + "/cam" + std::to_string(parameters.nCameraSystem.virtualCameraIdx(i)) +
                  "/data/" + *cam_iterators.at(i),
              cv::IMREAD_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
        }
        else if (dataType == OkvisDatasetType::viSim)
        {
          depth = cv::imread(
              path + "/depth0/data/" + *cam_iterators.at(i), cv::IMREAD_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
          depth.convertTo(depth, CV_16UC1, 1.0f);
          // depthTransformation(depth, 5.0);
        }
        else
        {
          std::cerr << "OKVIS Error: wrong dataset format!" << std::endl;
          return false;
        }
      }
      cv::imshow("depth", depth);
      cv::waitKey(0);
      std::string nanoseconds = cam_iterators.at(i)->substr(
          cam_iterators.at(i)->size() - 13, 9);
      std::string seconds = cam_iterators.at(i)->substr(
          0, cam_iterators.at(i)->size() - 13);
      t = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));
      std::cout << "Current Time: " << t << std::endl;
      if (start == okvis::Time(0.0))
      {
        start = t;
      }

      // get all IMU measurements till then
      okvis::Time t_imu = start;
      do
      {
        if (!std::getline(imu_file, line))
        {
          std::cout << std::endl
                    << "Finished. Press any key to exit." << std::endl
                    << std::flush;
          cv::waitKey();
          break;
        }

        std::stringstream stream(line);
        std::string s;
        std::getline(stream, s, ',');
        std::string nanoseconds = s.substr(s.size() - 9, 9);
        std::string seconds = s.substr(0, s.size() - 9);

        Eigen::Vector3d gyr;
        for (int j = 0; j < 3; ++j)
        {
          std::getline(stream, s, ',');
          gyr[j] = std::stof(s);
        }

        Eigen::Vector3d acc;
        for (int j = 0; j < 3; ++j)
        {
          std::getline(stream, s, ',');
          acc[j] = std::stof(s);
        }

        t_imu = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));

        // add the IMU measurement for (blocking) processing
        if (t_imu - start + okvis::Duration(1.0) > deltaT)
        {
          okvis_estimator->addImuMeasurement(t_imu, acc, gyr);
        }
      } while (t_imu <= t);

      // add the image to the frontend for (blocking) processing
      if (t - start > deltaT)
      {
        if (parameters.nCameraSystem.isDepthCamera(i))
        {
          okvis_estimator->addImage(t, i, filtered, depth);
          okvis_estimator->addImage(t, parameters.nCameraSystem.virtualCameraIdx(i),
                                    cv::Mat::zeros(filtered.rows, filtered.cols, CV_8U));
          std::cout << "Num: " << counter << std::endl;
          // std::cout << "depth camera index: " << parameters.nCameraSystem.virtualCameraIdx(i) << std::endl;
        }
        else
        {
          okvis_estimator->addImage(t, i, filtered);
          // std::cout << "RGB camera index  : " << i << std::endl;
        }
      }
      cam_iterators[i]++;
    }
    ++counter;

    //   display progress
    if (counter % 20 == 0)
    {
      std::cout << "\rProgress: "
                << int(double(counter) / double(num_camera_images) * 100) << "%  \n"
                << std::flush;
    }
  }

  // ==========     DUMP VOLUME      =========

  if (config_->dump_volume_file != "")
  {
    kfusion_->dumpVolume(config_->dump_volume_file);
  }

  if (powerMonitor_ && powerMonitor_->isActive())
  {
    std::ofstream powerStream("power.rpt");
    powerMonitor_->powerStats.print_all_data(powerStream);
    powerStream.close();
  }
  std::cout << "{";
  Stats_.print_all_data(std::cout, false);
  std::cout << "}" << std::endl;

  return true;
}

/// \brief Run okvis + mid-fusion live
bool vimidAction::runOkvisLive()
{
  // check whether the config file is valid
  if (config_->okvis_config_file.empty() || !file_exists(config_->okvis_config_file))
  {
    std::cerr << "OKVIS Error: improper config file input!" << std::endl;
    return false;
  }

  // read configuration file
  okvis::VioParametersReader vio_parameters_reader(config_->okvis_config_file);
  okvis::VioParameters parameters;
  vio_parameters_reader.getParameters(parameters);
  okvis::ThreadedKFVio *okvis_estimator = new okvis::ThreadedKFVio(parameters);

  PoseViewer poseViewer;
  okvis_estimator->setDenseStereoCallback(
      std::bind(&vimidAction::procOkvisAFrameCallback, this,
                std::placeholders::_1));
  okvis_estimator->setBlocking(false);

  rsReader_ = new okvis::ZR300Reader(width_, height_, 60, okvis_estimator);
  rsReader_->getColorIntrins(rsFx_, rsFy_, rsCx_, rsCy_);
  rsReader_->setTimeShift(26.856921);

  while (true)
  {
    if (!config_->no_gui)
    {
      poseViewer.display(T_WC_);
      okvis_estimator->display();
      drawthem(extraRender_, depthRender_, trackRender_, volumeRender_,
               segmentRender_, raycastRender_, computationSize_, computationSize_,
               computationSize_, computationSize_, computationSize_, computationSize_);
    }
  }

  // ==========     DUMP VOLUME      =========

  if (config_->dump_volume_file != "")
  {
    kfusion_->dumpVolume(config_->dump_volume_file);
  }

  if (powerMonitor_ && powerMonitor_->isActive())
  {
    std::ofstream powerStream("power.rpt");
    powerMonitor_->powerStats.print_all_data(powerStream);
    powerStream.close();
  }
  std::cout << "{";
  Stats_.print_all_data(std::cout, false);
  std::cout << "}" << std::endl;

  return true;
}

/// \brief Run ground truth + mid-fusion
bool vimidAction::runGtVimid()
{
  // initialize data loader
  loader_ = new dataLoader(config_->okvis_dataset_path);

  if (!config_->no_gui)
  {
    while (procGtAFrame() == 0)
    {
      std::cout << "==========================================" << std::endl;
      drawthem(extraRender_, depthRender_, trackRender_, volumeRender_,
               segmentRender_, raycastRender_, computationSize_, computationSize_,
               computationSize_, computationSize_, computationSize_, computationSize_);
      if (config_->pause)
        getchar();
    }
  }
  else
  {
    while (procGtAFrame() == 0)
    {
    }
    std::cout << __LINE__ << std::endl;
  }
  // ==========     DUMP VOLUME      =========

  if (config_->dump_volume_file != "")
  {
    double start = tock();
    kfusion_->dump_mesh(config_->dump_volume_file);
    double end = tock();
    Stats_.sample("meshing", end - start, PerfStats::TIME);
  }

  if (powerMonitor_ && powerMonitor_->isActive())
  {
    std::ofstream powerStream("power.rpt");
    powerMonitor_->powerStats.print_all_data(powerStream);
    powerStream.close();
  }
  std::cout << "{";
  Stats_.print_all_data(std::cout, false);
  std::cout << "}" << std::endl;

  return true;
}

/// \brief Run VI-mid-fusion
bool vimidAction::runMIDFusion()
{
  // initialize data loader
  loader_ = new dataLoader(config_);
  PoseViewer poseViewer;

  if (!config_->no_gui)
  {
    while (procMIDAFrame() == 0)
    {
      std::cout << "==========================================" << std::endl;
      // poseViewer.display(T_WC_);
      drawthem(extraRender_, depthRender_, trackRender_, volumeRender_,
               segmentRender_, mainViewRender_, computationSize_, computationSize_,
               computationSize_, computationSize_, computationSize_, computationSize_);
      // drawit(mainViewRender_, computationSize_);
      // drawNineScene(extraRender_, maskRender_, mainViewRender_,
      //               trackRender_, volumeRender_, topViewRender_,
      //               segmentRender_, raycastRender_, depthRender_,
      //               computationSize_, computationSize_, computationSize_,
      //               computationSize_, computationSize_, computationSize_,
      //               computationSize_, computationSize_, computationSize_);
      if (config_->pause)
        getchar();
    }
  }
  else
  {
    while (procMIDAFrame() == 0)
    {
    }
    std::cout << __LINE__ << std::endl;
  }
  // ==========     DUMP VOLUME      =========

  if (config_->dump_volume_file != "")
  {
    double start = tock();
    kfusion_->dump_mesh(config_->dump_volume_file);

    double end = tock();
    Stats_.sample("meshing", end - start, PerfStats::TIME);
  }

  if (powerMonitor_ && powerMonitor_->isActive())
  {
    std::ofstream powerStream("power.rpt");
    powerMonitor_->powerStats.print_all_data(powerStream);
    powerStream.close();
  }
  std::cout << "{";
  Stats_.print_all_data(std::cout, false);
  std::cout << "}" << std::endl;

  return true;
}

bool vimidAction::runVimid()
{
  // initialize data loader
  loader_ = new dataLoader(config_);
  PoseViewer poseViewer;
  int idx = 0;

  if (!config_->no_gui)
  {
    while (procVimidAFrame() == 0)
    {
      std::cout << "==========================================" << std::endl;
      poseViewer.display(T_WC_);
      // drawthem(extraRender_, depthRender_, trackRender_, volumeRender_,
      //          segmentRender_, mainViewRender_, computationSize_, computationSize_,
      //          computationSize_, computationSize_, computationSize_, computationSize_);
      // drawit(mainViewRender_, computationSize_);
      drawNineScene(extraRender_, maskRender_, mainViewRender_,
                    trackRender_, volumeRender_, topViewRender_,
                    segmentRender_, raycastRender_, depthRender_,
                    computationSize_, computationSize_, computationSize_,
                    computationSize_, computationSize_, computationSize_,
                    computationSize_, computationSize_, computationSize_);

      if (config_->in_debug & config_->dump_volume_file != "")
      {
        std::string mesh_path = config_->dump_volume_file + std::to_string(idx);
        std::cout << "Saving mesh at " << mesh_path << std::endl;
        double start = tock();
        kfusion_->dump_mesh(mesh_path);
        double end = tock();
        Stats_.sample("meshing", end - start, PerfStats::TIME);
      }

      idx++;

      if (config_->pause)
        getchar();
    }
  }
  else
  {
    while (procVimidAFrame() == 0)
    {
      if (config_->in_debug & config_->dump_volume_file != "")
      {
        std::string mesh_path = config_->dump_volume_file + std::to_string(idx);
        std::cout << "Saving mesh at " << mesh_path << std::endl;
        double start = tock();
        kfusion_->dump_mesh(mesh_path);
        double end = tock();
        Stats_.sample("meshing", end - start, PerfStats::TIME);
      }

      idx++;
    }
    std::cout << __LINE__ << std::endl;
  }
  // ==========     DUMP VOLUME      =========

  if (config_->dump_volume_file != "")
  {
    double start = tock();
    kfusion_->dump_mesh(config_->dump_volume_file);

    double end = tock();
    // Stats_.sample("meshing", end-start, PerfStats::TIME);
  }

  // if (powerMonitor_ && powerMonitor_->isActive()) {
  //   std::ofstream powerStream("power.rpt");
  //   powerMonitor_->powerStats.print_all_data(powerStream);
  //   powerStream.close();
  // }
  // std::cout << "{";
  // Stats_.print_all_data(std::cout, false);
  // std::cout << "}" << std::endl;

  return true;
}

/// \brief Run main function
bool vimidAction::run()
{
  bool gt = false;
  bool vimid = true;
  if (gt)
  {
    return runGtVimid();
  }
  if (vimid)
  {
    // return runMIDFusion();
    return runVimid();
  }
  if (!config_->okvis_config_file.empty() && !config_->okvis_dataset_path.empty())
  {
    runType_ = RunType::okvisDataset;
    return runOkvisDataset();
  }
  else if (!config_->okvis_config_file.empty() && config_->okvis_dataset_path.empty())
  {
    runType_ = RunType::okvisLive;
    return runOkvisLive();
  }
  else
  {
    runType_ = RunType::naive;
    return runNaive();
  }
}

/// \brief Transform the T-Matrix from OKVIS type to Mid-Fusion type
Matrix4 vimidAction::fromOkvisTransformation(okvis::kinematics::Transformation &T_OKVIS)
{
  Eigen::Matrix4d T = T_OKVIS.T();
  Matrix4 result;
  for (int i = 0; i < 4; i++)
  {
    result.data[i].x = T(i, 0);
    result.data[i].y = T(i, 1);
    result.data[i].z = T(i, 2);
    result.data[i].w = T(i, 3);
  }
  return result;
}

/// \brief Check if the RGB image is valid and change it to uchar3
bool vimidAction::rgbToUchar3Image(const cv::Mat &mat, uchar3 *image)
{
  unsigned char *pixelPtr = (unsigned char *)mat.data;
  int cn = mat.channels();
  int nRows = mat.rows;
  int nCols = mat.cols;

  if (!nRows || !nCols)
  {
    std::cerr << "Input Error: RGB image is invalid!" << std::endl;
    return false;
  }

  for (int i = 0; i < nRows; ++i)
  {
    for (int j = 0; j < nCols; ++j)
    {
      unsigned char b = pixelPtr[i * nCols * cn + j * cn + 0]; // B
      unsigned char g = pixelPtr[i * nCols * cn + j * cn + 1]; // G
      unsigned char r = pixelPtr[i * nCols * cn + j * cn + 2]; // R
      image[i * nCols + j] = make_uchar3(r, g, b);
    }
  }
  return true;
}

/// \brief Convert the depth format from InteriorNet format to OKVIS format
bool vimidAction::depthTransformation(cv::Mat &mat, double scale)
{
  if (scale <= 0)
  {
    std::cerr << "Input Error: Improper scale input." << std::endl;
    return false;
  }

  int cn = mat.channels();
  if (cn != 1)
  {
    std::cerr << "Input Error: Depth image is invalid." << std::endl;
    return false;
  }

  int nRows = mat.rows;
  int nCols = mat.cols;
  for (int v = 0; v < nRows; ++v)
  {
    for (int u = 0; u < nCols; ++u)
    {
      double d = static_cast<double>(mat.at<uint16_t>(v, u)) * scale;
      // double u_norm = (u - rsCx_) / rsFx_;
      // double v_norm = (v - rsCy_) / rsFy_;
      // mat.at<uint16_t>(v, u) = d / std::sqrt(u_norm * u_norm + v_norm * v_norm + 1);
      mat.at<uint16_t>(v, u) = d;
    }
  }
  return true;
}

/// \brief Check if the depth image is valid and transfer distance from center to normal plane
bool vimidAction::disToUshortDepth(const cv::Mat &mat, uint16_t *depth, double scale)
{
  if (scale <= 0)
  {
    std::cerr << "Input Error: Improper scale input." << std::endl;
    return false;
  }

  int cn = mat.channels();
  if (cn != 1)
  {
    std::cerr << "Input Error: Depth image is invalid." << std::endl;
    return false;
  }

  int nRows = mat.rows;
  int nCols = mat.cols;
  for (int v = 0; v < nRows; ++v)
  {
    for (int u = 0; u < nCols; ++u)
    {
      double d = static_cast<double>(mat.at<uint16_t>(v, u)) * scale;
      depth[v * nCols + u] = d;
    }
  }
  return true;
}

/// \brief Check if the file exists
bool vimidAction::file_exists(const std::string &path)
{
  std::ifstream file(path);
  return file.good();
}

/// \brief Check if the directory exists
bool vimidAction::directory_exists(const std::string &path)
{
  struct stat sb;
  return stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
}

/// \brief Make a new directory
void vimidAction::make_directory(const std::string &path)
{
  mkdir(path.c_str(), 0755);
}

/// \brief Check if the OKVIS dataset valid and return (rgb + imu) files
bool vimidAction::checkOkvisDataset(std::string &path, vimidAction::OkvisDatasetType &type)
{
  std::string depthVISimList = path + "/depth0/data.csv";
  std::string depthRSList = path + "/cam1/data.csv";
  std::string rgbList = path + "/cam0/data.csv";
  std::string imuList = path + "/imu0/data.csv";

  if (file_exists(depthVISimList))
  {
    type = vimidAction::OkvisDatasetType::viSim;
  }
  else if (file_exists(depthRSList))
  {
    type = vimidAction::OkvisDatasetType::realsense;
  }
  else
  {
    return false;
  }

  return (file_exists(rgbList) && file_exists(imuList));
}
