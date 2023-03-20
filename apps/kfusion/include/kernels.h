/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2020-2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2023 Binbin Xu
 * SPDX-FileCopyrightText: 2021-2023 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef _KERNELS_
#define _KERNELS_

#include <cstdlib>
#include <commons.h>
#include <perfstats.h>
#include <timings.h>
#include <config.h>
#include "segmentation.h"
#include "object.h"
#include "tracking.h"
#include <okvis/Estimator.hpp>
#include "../KeyFrame/KeyFrame.h"

/// OBJ ///

class Kfusion {
 private:
  // object lists
  ObjectList objectlist_;     // also contains static environment (first model)
  ObjectPointer static_map_;  // static background
  TrackingPointer rgbdTracker_;

  // segmentation
  std::shared_ptr<Segmentation> segmenter_;
  // TODO: modify to real-time running
  std::vector<SegmentationResult> mask_rcnn_result_;
  SegmentationResult volume2mask(float4 k, float mu);
  int min_object_size_;


  // create new objects based on mask (SegmentationResult)
  void generate_new_objects(const SegmentationResult &masks, float4 k, const uint frame);

  // generate new object parameters, i.e. pose and size, based on mask and depth
  void spawnNewObjectKernel(Matrix4 &T_w_o, float &volume_size, int &volume_resol,
                            const cv::Mat &mask, const Matrix4 &T_w_c,
                            const float4 &k);

  void add_object_into_list(ObjectPointer &f_objectpoint,
                            ObjectList &f_objectlist);

  // for old kfusion global model parameters
  // float3 volumeDimensions;
  // uint3 volumeResolution;
  // int voxel_block_size;

  // for rendering
  float _mu;
  bool shouldRender = false;

  uint2 computationSize;
//  float step;
  Matrix4 camera_pose;  //camera pose=> same as background volume pose
  Matrix4 init_camera_pose;
  long unsigned int kfId = 0;
  // Matrix4 preTrackingPose;
  Eigen::Matrix4d preTrackingPose;
  Matrix4 *viewPose;  //for raycasting

  std::vector<int> iterations;
  bool _tracked;
  bool _integrated;
  float3 _init_camera_Pose;  //for reading ground-truth pose
  float3 initial_pos_factor_; //for reading ground-truth pose
  bool using_2d2d_ = false;
  bool using_3d3d_ = true;
  int erase_num_ = 0;

  Configuration config;

//  void raycast(uint frame, const float4& k, float mu);

//  const float min_object_ratio_ = 0.002f;//currently allow any objects
//  const float min_object_ratio_ = 0.9f;//currently allow any objects
  float min_object_ratio_;//currently allow any objects

  //for tracking:
  const bool use_icp_tracking_ = true;
  bool use_rgb_tracking_;
  const bool use_live_depth_only = false;
  bool geom_refine_human = false;
  const bool do_edge_refinement = true;

  bool in_debug_;
  bool render_output;
  std::string render_folders;
  bool use_GT_segment;
  std::vector<std::string> GT_mask_files_;

  float3 * vertex_before_fusion;
  float3 * normal_before_fusion;

  const float residual_threshold_ = 1.5f; // 1.5f
  const float obj_move_out_threshold_ = 0.05f;  // 0.05f 0.2f
  float obj_moved_threshold_ = 0.8f;  // 0.7f or 0.8f
  std::set<int> objects_in_view_;
  std::set<int> move_obj_set;

  double time_obj_num_ = 0;
  double total_ = 0;
  double init_time_ = 0;
  double tracking_ = 0;
  double obj_tracking_ = 0;
  double segmentation_ = 0;
  double integration_ = 0;
  double raycasting_ = 0;
  double computation_time_ = 0;
  double same_obj_frame = 0;
  size_t mov_obj_num_ = 0;
  size_t in_view_obj_num_ = 0;

  // for okvis
  Matrix4 pose;
  float3 _initPose;
  float3 **icp_cov_pyramid;
  float sigma_bright;
  float sigma_disparity;
  float sigma_xy;
  float baseline;
  float focal;
  uint raycast_frame = 2;
  LocalizationState state_;
  bool firstRun = true;
  Eigen::Matrix4d currPose;
  okvis::Time prevTimeStamp;
  okvis::Time currTimeStamp;

  // Loop closure
  std::vector<KeyFrame*> vpKFs_;
  KeyFrame *currentKF_;


 public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Kfusion(uint2 inputSize, uint3 volumeResolution, float3 volumeSize,
          float3 initPose, std::vector<int> &pyramid, Configuration config)
      : computationSize(make_uint2(inputSize.x, inputSize.y)),
        geo2mask_result(inputSize),
        rendered_mask_(inputSize),
        geo_mask(inputSize),
        mask_rcnn(inputSize),
        human_out_(inputSize),
        raycast_mask_(inputSize)
  {

    this->_init_camera_Pose = initPose;
    this->preTrackingPose = Eigen::Matrix4d::Identity();
    this->infoMat_ = Eigen::Matrix<double,6,6>::Identity();
		// this->volumeDimensions = volumeDimensions;  //size=dimensions
		// this->volumeResolution = volumeResolution;
    // this->voxel_block_size = config.voxel_block_size;
    this->_mu = config.mu;
    this->config = config;

    camera_pose.data[0] = {1.f, 0.f, 0.f, initPose.x};
    camera_pose.data[1] = {0.f, 1.f, 0.f, initPose.y};
    camera_pose.data[2] = {0.f, 0.f, 1.f, initPose.z};
    camera_pose.data[3] = {0.f, 0.f, 0.f, 1.f};

    this->iterations.clear();
    for (std::vector<int>::iterator it = pyramid.begin(); it != pyramid.end(); it++) {
      this->iterations.push_back(*it);
    }

    // class id 0: background
    const int bg_class_id = 0;
    const Matrix4 bg_pose = Identity();
    // const Matrix4 bg_pose = toMatrix4(make_float4(1,1,1,0), make_float3(1, 1,1));
    // const Matrix4 bg_pose = camera_pose;
    All_Prob_Vect bg_all_prob = All_Prob_Vect::Zero();
    // this->static_map_ = std::make_shared<Object>(config.voxel_block_size,
    //                                              volumeSize, volumeResolution,
    //                                              bg_pose, bg_pose, bg_class_id,
    //                                              bg_all_prob, inputSize);
    this->static_map_ = std::allocate_shared<Object>(Eigen::aligned_allocator<Object>(),
                                                     config.voxel_block_size,
                                                     volumeSize, volumeResolution,
                                                     bg_pose, bg_pose, bg_class_id,
                                                     bg_all_prob, inputSize);
    add_object_into_list(this->static_map_, this->objectlist_);

    use_rgb_tracking_ = (!config.disable_rgb);
    // this->rgbdTracker_ = std::make_shared<Tracking>(inputSize, iterations,
    //                                                 this->use_live_depth_only,
    //                                                 this->use_icp_tracking_,
    //                                                 this->use_rgb_tracking_);
    this->rgbdTracker_ = std::allocate_shared<Tracking>(Eigen::aligned_allocator<Tracking>(),
                                                        inputSize, iterations,
                                                        this->use_live_depth_only,
                                                        this->use_icp_tracking_,
                                                        this->use_rgb_tracking_);
    // in_debug_ = false;
    segmenter_ = std::make_shared<Segmentation>();
    segmenter_->refine_human_boundary_ = geom_refine_human;

    frame_masks_ = std::make_shared<SegmentationResult>(computationSize.y,
                                                        computationSize.x);
    viewPose = &camera_pose;

    currentKF_ = new KeyFrame();
    this->languageSpecificConstructor();
  }
//Allow a kfusion object to be created with a pose which include orientation as well as position
  Kfusion(uint2 inputSize, uint3 volumeResolution, float3 volumeSize,
          Matrix4 initPose, std::vector<int> & pyramid, Configuration config)
      : computationSize(make_uint2(inputSize.x, inputSize.y)),
        geo2mask_result(inputSize),
        rendered_mask_(inputSize),
        geo_mask(inputSize),
        mask_rcnn(inputSize),
        human_out_(inputSize),
        raycast_mask_(inputSize)
  {
    this->_init_camera_Pose = getPosition();
//		this->volumeDimensions = volumeDimensions;
//		this->volumeResolution = volumeResolution;
//        this->voxel_block_size = config.voxel_block_size;
    this->_mu = config.mu;
    camera_pose = initPose;

    this->iterations.clear();
    for (std::vector<int>::iterator it = pyramid.begin();
         it != pyramid.end(); it++) {
      this->iterations.push_back(*it);

    }

    //class id 0: background
    const int bg_class_id = 0;
    const Matrix4 bg_pose = Identity();
    All_Prob_Vect bg_all_prob = All_Prob_Vect::Zero();
    this->static_map_ = std::make_shared<Object>(config.voxel_block_size,
                                                 volumeSize, volumeResolution,
                                                 bg_pose, bg_pose, bg_class_id,
                                                 bg_all_prob, inputSize);
    add_object_into_list(this->static_map_, this->objectlist_);
    use_rgb_tracking_ = (!config.disable_rgb);
    this->rgbdTracker_ = std::make_shared<Tracking>(inputSize, iterations,
                                                    this->use_live_depth_only,
                                                    this->use_icp_tracking_,
                                                    this->use_rgb_tracking_);
    segmenter_ = std::make_shared<Segmentation>();
    segmenter_->refine_human_boundary_ = geom_refine_human;
    frame_masks_ = std::make_shared<SegmentationResult>(computationSize.x,
                                                        computationSize.y);
//    in_debug_ = false;
    viewPose = &camera_pose;
    this->languageSpecificConstructor();
  }

  void languageSpecificConstructor();
  ~Kfusion();

  void reset();
  bool getTracked() {
    return (_tracked);
  }
  bool getIntegrated() {
    return (_integrated);
  }
  float3 getPosition() {
    //std::cerr << "InitPose =" << _initPose.x << "," << _initPose.y  <<"," << _initPose.z << "    ";
    //std::cerr << "pose =" << pose.data[0].w << "," << pose.data[1].w  <<"," << pose.data[2].w << "    ";
    float xt = camera_pose.data[0].w - _init_camera_Pose.x;
    float yt = camera_pose.data[1].w - _init_camera_Pose.y;
    float zt = camera_pose.data[2].w - _init_camera_Pose.z;
    return (make_float3(xt, yt, zt));
  }

  float3 getInitPos(){
    return _init_camera_Pose;
  }

  Matrix4 getInitPose(){
    return init_camera_pose;
  }

  bool preprocessing(const ushort * inputDepth, const uint2 inputSize,
                     const bool filterInput);

  bool preprocessing(const ushort * inputDepth, const uchar3 * inputRGB,
                     const uint2 inputSize, const bool filterInput);

  bool tracking(float4 k, uint tracking_rate, uint frame);
  bool raycasting(float4 k, float mu, uint frame);
  bool raycasting(float4 k, float mu, uint frame, KeyFrame *kf);
  bool integration(float4 k, uint integration_rate, float mu, uint frame);

  void dumpVolume(std::string filename);
  void printStats();

  void getPointCloudFromVolume();

  void renderVolume(uchar4 * out, const uint2 outputSize, int frame, int rate,
                    float4 k, float mu, bool render_color);
  void renderMainView(uchar4 * out, const uint2 outputSize, int frame, int rate,
                      float4 k, float mu, bool render_color, Matrix4 poseToSee);
  void renderTrack(uchar4 * out, const uint2 outputSize);
  void renderTrack(uchar4 * out, const uint2 outputSize, int type, int frame);
  void renderDepth(uchar4* out, uint2 outputSize);
  void renderDepth(uchar4* out, uint2 outputSize, int frame);
  void save_input(const uchar3 * input_RGB, uint2 outputSize, int frame);

  void renderIntensity(uchar4* out, uint2 outputSize);

  void renderClass(uchar4 * out, uint2 outputSize,
                   SegmentationResult segmentationResult);
  void renderInstance(uchar4 * out, uint2 outputSize,
                      const SegmentationResult& segmentationResult);
  void renderInstance(uchar4 * out, uint2 outputSize,
                      const SegmentationResult& segmentationResult,
                      int frame, std::string labelSource);
  void renderMaskWithImage(uchar4 * out, uint2 outputSize,
                           const SegmentationResult& segmentationResult);
  void renderMaskWithImage(uchar4 * out, uint2 outputSize,
                           const SegmentationResult& segmentationResult,
                           int frame, std::string labelSource);
  void renderMaskMotionWithImage(uchar4 * out, uint2 outputSize,
                                 const SegmentationResult& segmentationResult,
                                 int frame);
  Matrix4 getPose() {
    return camera_pose;
  }
  
  void setViewPose(Matrix4 *value = NULL) {
    if (value == NULL){
      viewPose = &camera_pose;
      shouldRender = false;
    }
    else {
      viewPose = value;
      shouldRender = true;
    }
  }

  void setPoseScale(const float3 initial_pos_factor){
    initial_pos_factor_ = initial_pos_factor;
  }
  Matrix4 *getViewPose() {
    return (viewPose);
  }
  float3 getModelSize() {
    return (static_map_->get_volume_size());
  }
  uint3 getModelResolution() {
    return (static_map_->get_volume_resol());
  }
  uint2 getComputationResolution() {
    return (computationSize);
  }

  float * getDepth();

  void dump_mesh(const std::string filename);

  void save_poses(const std::string filename, const int frame);

  void save_poses(const std::string filename, const okvis::Time &timestamp);

  void save_times(const std::string filename, const int frame, double* timings);
  /**
 * a function called to set parameters for balancing ICP and RGB tracker
 * @param [in] test_condition using which sensor and in what kind enviorment
 * 0: zr300
 * 1; asus
 * 2: ICL-NUIM datasets
 * 3: TUM RGB-D datasets
 */
  void obtainErrorParameters(int test_condition);


  //segmentation
  std::vector<bool *> masks;
  std::vector<int> class_ids;

  //color intergration
  bool render_color_ = true;

  SegmentationResult get_global_masks(){
    return *(this->frame_masks_);
  }

  void split_labels(SegmentationResult& segmentationResult,
                             const ObjectList& objectList);
  void split_labels_and_delete(SegmentationResult& segmentationResult,
                               ObjectList& objectList);
  void split_labels(SegmentationResult& segmentationResult,
                    std::set<int>& object_in_view,
                    const ObjectList& objectList);
  // void split_labels(SegmentationResult& segmentationResult,
  //                   std::set<int>& object_in_view,
  //                   ObjectList& objectList);

  //perform segmentation
  bool segment(float4 k, uint frame, std::string segFolder, bool hasMaskRCNN);
  bool readMaskRCNN(float4 k, uint frame, std::string segFolder);
  bool MaskRCNN_next_frame(uint frame, std::string segFolder);

  SegmentationResult geo2mask_result;
  SegmentationResult rendered_mask_;
  SegmentationResult geo_mask;
  SegmentationResult mask_rcnn;
  SegmentationResult human_out_;
  std::set<int> matched_model_index_;
  std::shared_ptr<SegmentationResult> frame_masks_; //masks genrated in the
  // segmentation
  SegmentationResult raycast_mask_;
  uint segment_startFrame_ = 0;

  // For okvis
  bool adaptPose(Matrix4 p, float3 &default_pos, uint tracking_rate, uint frame);
  bool calc_track_residual(const float4 k, const uint tracking_rate, const uint frame);
  void calc_icp_cov(float3* conv_icp, const uint2 size,
                    const float* depth_image, const Matrix4& K,
                    const float sigma_xy, const float sigma_disp,
                    const float baseline);
  bool okvisTracking(float4 k, uint tracking_rate, uint frame, Matrix4 p);
  bool okvisTrackingTwo(float4 k, uint tracking_rate, uint frame, Matrix4 p);
  bool vimidTracking(float4 k, 
                     uint tracking_rate, 
                     uint frame, 
                     Matrix4 p, 
                     okvis::VioParameters &params,
                     okvis::ImuMeasurementDeque &imuData,
                     okvis::Time &prevTime,
                     okvis::Time &currTime,
                     bool use_imu,
                     double *timings);
  void sparsePreTracking(float4 k,
                         uint tracking_rate,
                         uint frame,
                         cv::Mat prevImg,
                         cv::Mat currImg,
                         cv::Mat prevDepth,
                         cv::Mat currDepth);

  void delete_wrong_obj();
  void calc_unmatched_time();

  void saveFieldSlice(const std::string &file_path,
                      const Eigen::Vector3f &point_W,
                      const std::string &num) const;

  cv::Mat prevResizeRGB_;
  cv::Mat currResizeRGB_;

  // KeyFrames
  Eigen::Matrix<double,6,6> infoMat_;

  bool insertKeyFrame(uint frame, KeyFrame *kf);
};

void delete_outside_view_obj(SegmentationResult &masks, ObjectList &objectlis);

void synchroniseDevices(); // Synchronise CPU and GPU


#endif
