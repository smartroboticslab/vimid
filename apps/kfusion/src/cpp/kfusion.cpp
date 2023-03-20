/*

 SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 SPDX-FileCopyrightText: 2023 Binbin Xu
 SPDX-FileCopyrightText: 2023 Yifei Ren
 SPDX-License-Identifier: BSD-3-Clause

*/

#include <kernels.h>
#include "timings.h"
#include <perfstats.h>
#include <vtk-io.h>
#include <memory>
#include <octree.hpp>
#include "continuous/volume_instance.hpp"
#include "algorithms/meshing.hpp"
#include "geometry/octree_collision.hpp"
#include "preprocessing.h"
#include "rendering/rendering.h"
#include "frontEnd/pose_estimation.h"
// #include "tracking.cpp"
// #include "bfusion/mapping_impl.hpp"
// #include "kfusion/mapping_impl.hpp"
// #include "bfusion/alloc_impl.hpp"
// #include "kfusion/alloc_impl.hpp"

extern PerfStats Stats;

// input once
float *gaussian;

// inter-frame
// Volume<FieldType> volume;
float3 *vertex;
float3 *normal;

// float3 bbox_min;
// float3 bbox_max;

// intra-frame

/// Coordinate frames:
///   _l:

float *floatDepth;
float3 *inputRGB;   // rgb input image
float *g_inputGrey; // grey input image

Matrix4 T_w_r;
Matrix4 raycastPose;

// For okvis
float **l_D; // depth pyramid on the live image (D_1)
float **r_D; // depth pyramid on the reference image (D_0)
float **I_l; // live image (I_1)
float **I_r; // reference image (I_0)
float **l_gradx;
float **l_grady;
Matrix4 lastPose;
float3 **l_vertex; // vertex seen from live frame
float3 **r_vertex; // vertex seen from reference frame
float3 **l_normal; // normal seen from live frame
float *distance_to_vertex;

Matrix4 camera_pose_ori;
Matrix4 T_w_r_ori;
float3 vertex_ori;
float3 normal_ori;
float3 vertex_before_fusion_ori;
float3 normal_before_fusion_ori;
LocalizationState state_ori;
Eigen::Matrix<double, 15, 1> linearizationPoint_ori;

TrackData **trackingResult;

bool bayesian = false;

// For debugging purposes, will be deleted once done.
std::vector<Matrix4> poses;

void Kfusion::languageSpecificConstructor()
{

  if (getenv("KERNEL_TIMINGS"))
    print_kernel_timing = true;

  // internal buffers to initialize

  floatDepth = (float *)calloc(
      sizeof(float) * computationSize.x * computationSize.y, 1);
  inputRGB = (float3 *)calloc(
      sizeof(float3) * computationSize.x * computationSize.y, 1);
  g_inputGrey = (float *)calloc(
      sizeof(float) * computationSize.x * computationSize.y, 1);
  vertex = (float3 *)calloc(
      sizeof(float3) * computationSize.x * computationSize.y, 1);
  normal = (float3 *)calloc(
      sizeof(float3) * computationSize.x * computationSize.y, 1);
  vertex_before_fusion = (float3 *)calloc(
      sizeof(float3) * computationSize.x * computationSize.y, 1);
  normal_before_fusion = (float3 *)calloc(
      sizeof(float3) * computationSize.x * computationSize.y, 1);
  //	trackingResult = (TrackData*) calloc(
  //			2 * sizeof(TrackData) * computationSize.x * computationSize.y, 1);

  // ********* BEGIN : Generate the gaussian *************
  size_t gaussianS = radius * 2 + 1;
  gaussian = (float *)calloc(gaussianS * sizeof(float), 1);
  int x;
  for (unsigned int i = 0; i < gaussianS; i++)
  {
    x = i - 2;
    gaussian[i] = expf(-(x * x) / (2 * delta * delta));
  }
  // ********* END : Generate the gaussian *************

  if (config.groundtruth_file != "")
  {
    parseGTFile(config.groundtruth_file, poses);
    std::cout << "Parsed " << poses.size() << " poses" << std::endl;
  }

  bayesian = config.bayesian;

  std::string segFolder = config.maskrcnn_folder;
  std::string maskFolder = segFolder + "/mask/";
  std::string classFolder = segFolder + "/class_id/";
  std::string probFolder = segFolder + "/all_prob/";
  std::vector<std::string> mask_files = segmenter_->readFiles(maskFolder);
  std::vector<std::string> class_files = segmenter_->readFiles(classFolder);
  std::vector<std::string> prob_files = segmenter_->readFiles(probFolder);
  mask_rcnn_result_.reserve(mask_files.size());
  for (size_t i = 0; i < mask_files.size(); ++i)
  {
    mask_rcnn_result_.push_back(segmenter_->load_mask_rcnn(class_files[i],
                                                           mask_files[i],
                                                           prob_files[i],
                                                           computationSize.y,
                                                           computationSize.x));
  }

  if (config.gt_mask_folder == "")
    use_GT_segment = false;
  else
  {
    use_GT_segment = true;
    GT_mask_files_ = segmenter_->readFiles(config.gt_mask_folder);
  }

  objects_in_view_.insert(0);
  in_debug_ = config.in_debug;
  if (config.output_images != "")
  {
    render_output = true;
  }
  else
  {
    render_output = false;
  }
  //  volume.init(volumeResolution.x, volumeDimensions.x);
  min_object_ratio_ = config.min_obj_ratio;
  min_object_size_ = computationSize.x * computationSize.y * min_object_ratio_;
  Object::absorb_outlier_bg = config.absorb_outlier_bg;
  obj_moved_threshold_ = config.obj_moved;
  geom_refine_human = config.geom_refine_human;
  if (geom_refine_human)
    segmenter_->geom_dilute_size = 12;
  segment_startFrame_ = config.init_frame;
  reset();
}

Kfusion::~Kfusion()
{
  free(gaussian);
  free(floatDepth);
  free(g_inputGrey);
  free(inputRGB);
  free(vertex);
  free(normal);
  free(vertex_before_fusion);
  free(normal_before_fusion);
}

void Kfusion::reset()
{
}
void init(){};
// stub
void clean(){};

float *Kfusion::getDepth()
{
  return floatDepth;
}
bool Kfusion::preprocessing(const ushort *inputDepth, const uint2 inputSize,
                            const bool filterInput)
{

  mm2metersKernel(floatDepth, computationSize, inputDepth, inputSize);
  if (filterInput)
  {
    bilateralFilterKernel(floatDepth, floatDepth, computationSize, gaussian,
                          e_delta, radius, config.depth_max);
  }
  else
  {
    /*
          unsigned int y;
    #pragma omp parallel for \
          shared(l_D), private(y)
          for (y = 0; y < computationSize.y; y++) {
            for (unsigned int x = 0; x < computationSize.x; x++) {
              l_D[0][x + y*computationSize.x] = floatDepth[x + y*computationSize.x];
            }
          }
          */
  }
  return true;
}

void Kfusion::save_input(const uchar3 *input_RGB, uint2 inputSize, int frame)
{
  if (render_output)
  {
    std::string rgb_file;
    rgb_file = config.output_images + "rgb";
    int ratio = inputSize.x / computationSize.x;
    cv::Mat renderImg = cv::Mat::zeros(cv::Size(computationSize.x, computationSize.y), CV_8UC3);
    unsigned int y;
    for (y = 0; y < computationSize.y; y++)
      for (unsigned int x = 0; x < computationSize.x; x++)
      {
        uint pos = x * ratio + inputSize.x * y * ratio;
        renderImg.at<cv::Vec3b>(y, x)[2] = input_RGB[pos].x; // R
        renderImg.at<cv::Vec3b>(y, x)[1] = input_RGB[pos].y; // G
        renderImg.at<cv::Vec3b>(y, x)[0] = input_RGB[pos].z; // B
      }

    std::ostringstream name;
    name << rgb_file + "_" << std::setfill('0') << std::setw(5) << std::to_string(frame) << ".png";
    cv::imwrite(name.str(), renderImg);
  }
}

bool Kfusion::preprocessing(const ushort *inputDepth, const uchar3 *_inputRGB,
                            const uint2 inputSize, const bool filterInput)
{

  //  rgb2intensity(I_l[0], computationSize, _inputRGB, inputSize);
  currResizeRGB_ = resizeMat(_inputRGB, inputSize, computationSize);
  rgb2intensity(g_inputGrey, inputRGB, computationSize, _inputRGB, inputSize);

  mm2metersKernel(floatDepth, computationSize, inputDepth, inputSize);
  if (filterInput)
  {
    bilateralFilterKernel(floatDepth, floatDepth, computationSize, gaussian,
                          e_delta, radius, config.depth_max);
  }
  else
  {
    /*
    unsigned int y;
#pragma omp parallel for \
    shared(l_D), private(y)
    for (y = 0; y < computationSize.y; y++) {
      for (unsigned int x = 0; x < computationSize.x; x++) {
        l_D[0][x + y*computationSize.x] = floatDepth[x + y*computationSize.x];
      }
    }
  */
  }
  return true;
}

bool Kfusion::tracking(float4 k, uint tracking_rate, uint frame)
{

  if (frame % tracking_rate != 0)
    return false;

  bool camera_tracked;

  T_w_r = camera_pose; // get old pose T_w_(c-1)

  // camera tracking against all pixels&vertices(excluding human)
  if (human_out_.pair_instance_seg_.find(INVALID) == human_out_.pair_instance_seg_.end())
  {
    rgbdTracker_->set_params_frame(k, g_inputGrey, floatDepth);
  }
  else
  {
    if (in_debug_)
    {
      human_out_.output(frame, "human_out_for_tracking");
    }
    rgbdTracker_->set_params_frame(k, g_inputGrey, floatDepth, human_out_.pair_instance_seg_.at(INVALID).instance_mask_);
  }

  if (!poses.empty())
  {
    printMatrix4("Camera Pose", camera_pose);
    this->camera_pose = poses[frame];
    this->camera_pose.data[0].w = (this->camera_pose.data[0].w - poses[0].data[0].w) + this->_init_camera_Pose.x;
    this->camera_pose.data[1].w = (this->camera_pose.data[1].w - poses[0].data[1].w) + this->_init_camera_Pose.y;
    this->camera_pose.data[2].w = (this->camera_pose.data[2].w - poses[0].data[2].w) + this->_init_camera_Pose.z;
    printMatrix4("use ground truth pose", camera_pose);
    camera_tracked = true;
  }
  else
  {

    if (frame == segment_startFrame_)
    {
      // put reference frame information to current frame memory
      rgbdTracker_->setRefImgFromCurr();
      return false;
    }
    // track against all objects except people
    // estimate static/dynamic objects => only track existing&dynamic objects
    camera_tracked = rgbdTracker_->trackLiveFrame(camera_pose,
                                                  T_w_r, k,
                                                  //                                                  objectlist_[0]->m_vertex,
                                                  //                                                  objectlist_[0]->m_normal);
                                                  vertex, normal);

    objectlist_[0]->virtual_camera_pose_ = camera_pose;

    // for later motion segmentation
    memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
           sizeof(TrackData) * computationSize.x * computationSize.y * 2);
  }

  // use ICP&RGB residual to evaluate moving (possibility) for each object

  // raycast again to obtain rendered mask and vertices under CURRENT camera
  //  pose and LAST object poses.=>for segmentation and motion residuals
  move_obj_set.clear();
  if (raycast_mask_.pair_instance_seg_.size() > 1)
  {
    rendered_mask_.reset();
    objects_in_view_.clear();
    raycastObjectList(objectlist_, vertex_before_fusion, normal_before_fusion,
                      rendered_mask_.labelImg, objects_in_view_, camera_pose,
                      k, computationSize, nearPlane, farPlane, config.mu, false);

    split_labels(rendered_mask_, objects_in_view_, objectlist_);
    rendered_mask_.set_render_label(true);
    if (in_debug_)
    {
      rendered_mask_.output(frame, "render_initial_track");
    }

    // calculate ICP and RGB residuals on all vertices in current camera pose
    rgbdTracker_->compute_residuals(camera_pose, T_w_r,
                                    vertex, normal,
                                    vertex_before_fusion,
                                    use_icp_tracking_, use_rgb_tracking_,
                                    residual_threshold_);

    // check mask with tracking result
    // find moving objects
    for (auto object_render = rendered_mask_.pair_instance_seg_.begin();
         object_render != rendered_mask_.pair_instance_seg_.end(); ++object_render)
    {
      int obj_id = object_render->first;
      // skip background
      if (obj_id == 0)
        continue;

      const cv::Mat &object_mask = object_render->second.instance_mask_;
      cv::Mat object_inlier_mask = object_mask.clone();
      int obj_mask_num = cv::countNonZero(object_mask);
      bool isSmall = (obj_mask_num < 500);
      check_static_state(object_inlier_mask, object_mask,
                         rgbdTracker_->getTrackingResult(), computationSize,
                         use_icp_tracking_, use_rgb_tracking_);
      float inlier_ratio = static_cast<float>(cv::countNonZero(object_inlier_mask)) / (cv::countNonZero(object_mask));

      bool approach = check_truncated(object_mask, computationSize);
      if (approach)
      {
        objectlist_.at(obj_id)->approach_boundary = true;
      }
      else
      {
        objectlist_.at(obj_id)->approach_boundary = false;
      }
      // connect objectlist with raycastmask
      if ((inlier_ratio < obj_moved_threshold_) &&
          (inlier_ratio > obj_move_out_threshold_) &&
          !isSmall &&
          !approach &&
          (Object::static_object.find(object_render->second.class_id_) == Object::static_object.end()))
      {
        objectlist_.at(obj_id)->set_static_state(false);
        move_obj_set.insert(obj_id);
        //        std::cout<<"object "<<obj_id<<" moving threshold: "<<inlier_ratio
        //                 <<" class id: "<<objectlist_.at(obj_id)->class_id_<<std::endl;
      }
      else
      {
        objectlist_.at(obj_id)->set_static_state(true);
        //        std::cout<<"object "<<obj_id<<" static threshold: "<<inlier_ratio
        //                 <<" class id: "<<objectlist_.at(obj_id)->class_id_<<std::endl;
      }

      if (in_debug_)
      {
        std::ostringstream name;
        name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
             << "inlier"
             << "_frame_" << frame << "_object_id_" << obj_id << ".png";
        cv::imwrite(name.str(), object_inlier_mask);

        std::ostringstream name2;
        name2 << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build"
                 "-debug/debug/"
              << "all"
              << "_frame_" << frame << "_object_id_" << obj_id << ".png";
        cv::imwrite(name2.str(), object_mask);
      }
    }

    // refine camera motion using all static vertice, if there is moving
    //  object
    if ((move_obj_set.size() != 0) &&
        (rendered_mask_.pair_instance_seg_.size() != 0))
    {
      // build dynamic object mask
      cv::Size imgSize = raycast_mask_.pair_instance_seg_.begin()->second.instance_mask_.size();
      cv::Mat dynamic_obj_mask = cv::Mat::zeros(imgSize, CV_8UC1);
      for (auto object_raycast = raycast_mask_.pair_instance_seg_.begin();
           object_raycast != raycast_mask_.pair_instance_seg_.end(); ++object_raycast)
      {
        int obj_id = object_raycast->first;
        if (move_obj_set.find(obj_id) != move_obj_set.end())
        {
          cv::bitwise_or(object_raycast->second.instance_mask_,
                         dynamic_obj_mask, dynamic_obj_mask);
        }
      }
      if (in_debug_)
      {
        std::ostringstream name;
        name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
             << "dynamic_object"
             << "_frame_" << frame << ".png";
        cv::imwrite(name.str(), dynamic_obj_mask);
      }

      // rgbdTracker_->enable_RGB_tracker(false);
      if (human_out_.pair_instance_seg_.find(INVALID) != human_out_.pair_instance_seg_.end())
      {
        cv::bitwise_or(dynamic_obj_mask, human_out_.pair_instance_seg_.at(INVALID).instance_mask_, dynamic_obj_mask);
      }

      camera_tracked = rgbdTracker_->trackLiveFrame(camera_pose,
                                                    T_w_r, k,
                                                    vertex,
                                                    normal, dynamic_obj_mask);
      // rgbdTracker_->enable_RGB_tracker(true);
      objectlist_[0]->virtual_camera_pose_ = camera_pose;
      // for later motion segmentation
      memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
             sizeof(TrackData) * computationSize.x * computationSize.y * 2);
    }

    // refine poses of dynamic objects
    ///  multiple objects tracking
    for (auto exist_object = objectlist_.begin() + 1;
         exist_object != objectlist_.end(); ++exist_object)
    {
      if ((*exist_object)->is_static())
      {
        // for later motion segmentation
        memcpy((*exist_object)->trackresult_, objectlist_.at(0)->trackresult_,
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
        continue;
      }
      else
      {
        rgbdTracker_->trackEachObject(*exist_object, k, camera_pose, T_w_r,
                                      g_inputGrey, floatDepth);
        //        (*exist_object)->set_static_state(true);

        memcpy((*exist_object)->trackresult_, rgbdTracker_->getTrackingResult(),
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
      }
    }
  }

  //
  ////perform motion refinement accordingly
  if (!move_obj_set.empty())
  {
    // calculate ICP and RGB residuals on all vertices in current camera pose
    rgbdTracker_->compute_residuals(camera_pose, T_w_r,
                                    vertex, normal,
                                    vertex_before_fusion,
                                    use_icp_tracking_, use_rgb_tracking_,
                                    residual_threshold_);
    // for later motion segmentation
    memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
           sizeof(TrackData) * computationSize.x * computationSize.y * 2);

    // raycast again to obtain rendered mask and vertices under current camera
    //  pose and object poses.=>for segmentation and motion residuals
    rendered_mask_.reset();
    raycastObjectList(objectlist_, vertex_before_fusion, normal_before_fusion,
                      rendered_mask_.labelImg, objects_in_view_, camera_pose,
                      k, computationSize, nearPlane, farPlane, config.mu, false);

    // find the objects on this frame =>only (track and) integrate them
    split_labels(rendered_mask_, objectlist_);
    rendered_mask_.set_render_label(true);
    if (in_debug_)
    {
      rendered_mask_.output(frame, "volume2mask_new");
    }
  }

  if (((rendered_mask_.pair_instance_seg_.size() > 1) && (use_rgb_tracking_)) ||
      (!move_obj_set.empty()))
  {

    for (auto exist_object = objectlist_.begin() + 1;
         exist_object != objectlist_.end(); ++exist_object)
    {
      if ((*exist_object)->is_static())
      {
        //        for later motion segmentation
        memcpy((*exist_object)->trackresult_, objectlist_.at(0)->trackresult_,
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
        continue;
      }
      else
      {
        rgbdTracker_->compute_residuals((*exist_object)->virtual_camera_pose_, T_w_r,
                                        (*exist_object)->m_vertex, (*exist_object)->m_normal,
                                        (*exist_object)->m_vertex_bef_integ,
                                        use_icp_tracking_, use_rgb_tracking_,
                                        residual_threshold_);

        memcpy((*exist_object)->trackresult_, rgbdTracker_->getTrackingResult(),
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
      }
    }
  }

  // put reference frame information to current frame memory
  rgbdTracker_->setRefImgFromCurr();

  if (in_debug_)
  {
    printMatrix4("tracking: camera pose", camera_pose);

    for (auto object = objectlist_.begin(); object != objectlist_.end();
         ++object)
    {
      int class_id = (*object)->class_id_;
      std::cout << "tracking: object id: " << (*object)->instance_label_
                << " ,class id: " << class_id << std::endl;
      printMatrix4("tracking: object pose ", (*object)->volume_pose_);
    }
  }

  return camera_tracked;
}

// void mask_semantic_output(const SegmentationResult& masks,
//                           const ObjectList& objectlist,
//                           const uint&frame, std::string str ) {
//   for (auto object = objectlist.begin(); object != objectlist.end();
//        ++object) {
//     const int class_id = (*object)->class_id_;
//     const int instance_id = (*object)->instance_label_;
//     cv::Mat labelMask;
////    if (frame>3){//because raycasting starts from frame 2
//    auto object_mask = masks.instance_id_mask.find(class_id);
//    if ( object_mask!= masks.instance_id_mask.end()) {
//      labelMask = object_mask->second;
//      std::ostringstream name;
//      name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
//           <<str<<"_frame_"<< frame << "_object_id_" <<instance_id
//           <<"_class_id_" <<class_id<<".png";
//      cv::imwrite(name.str(), labelMask);
//    } else {
//      std::cout<<"frame_"<< frame<<"_"<< str<< "_mask missing: " << class_id << std::endl;
//      continue;
//    }
//  }
//}

void mask_instance_output(const SegmentationResult &masks,
                          const ObjectList &objectlist,
                          const uint &frame, std::string str)
{
  for (auto object = objectlist.begin(); object != objectlist.end();
       ++object)
  {
    const int class_id = (*object)->class_id_;
    const int instance_id = (*object)->instance_label_;
    cv::Mat labelMask;
    // modifify the label mask from segmentation
    //    if (frame>3){//because raycasting starts from frame 2 (seems)
    auto object_mask = masks.pair_instance_seg_.find(instance_id);
    if (object_mask != masks.pair_instance_seg_.end())
    {
      labelMask = object_mask->second.instance_mask_;
      std::ostringstream name;
      name << "/home/ryf/slam/vimid/apps/kfusion/cmake-build-debug/debug/"
           << str << "/frame_" << frame << "_object_id_" << instance_id
           << "_class_id_" << class_id << ".png";
      cv::imwrite(name.str(), labelMask);
    }
    else
    {
      std::cout << "frame_" << frame << "_" << str << "_mask missing: " << instance_id << std::endl;
      continue;
    }
  }
}

// Delete the wrong object
void Kfusion::delete_wrong_obj()
{
  // Delete first
  std::cout << "size: " << objectlist_.size() << std::endl;
  int count = 0;
  int erase_num = 0;
  for (auto object = objectlist_.begin(); object != objectlist_.end(); ++object)
  {
    // Bring instance id forward
    (*object)->instance_label_ -= erase_num;

    // Continue for bg and exist objects for certain
    if ((*object)->instance_label_ == 0 || (*object)->isExist_)
      continue;

    const int instance_id = (*object)->instance_label_;
    const int class_id = (*object)->class_id_;

    if ((*object)->exist_prob_ < 0.1)
    {
      object = objectlist_.erase(object);
      object--;
      erase_num++;
      std::cout << "Model with instance id " << instance_id
                << " class id " << class_id
                << " is deleted." << std::endl;
      // cv::waitKey(0);
    }
    count++;
  }
  erase_num_ += erase_num;
  // std::cout << "count: " << count << std::endl;
}

void Kfusion::calc_unmatched_time()
{
  // Calculate the model unmatched time
  for (auto object = objectlist_.begin() + 1; object != objectlist_.end(); ++object)
  {

    // If the object is approach the boundary and does not
    // in the objects_in_view_, set the out_boundary flag
    const int instance_id = (*object)->instance_label_;
    if ((*object)->approach_boundary &&
        (objects_in_view_.find(instance_id) == objects_in_view_.end()))
    {
      (*object)->out_boundary = true;
    }

    // std::cout << "Calculate object unmatched time" << std::endl;
    // std::cout << "size: " << objectlist_.size() << std::endl;
    if ((*object)->isExist_)
      continue;

    // If the object is out of the view, reset the outside_field_time
    auto obj_in_view = objects_in_view_.find(instance_id);
    if (obj_in_view == objects_in_view_.end())
    {
      (*object)->outside_field_time_ = 0;
      continue;
    }
    else
    {
      auto find_matched_obj = matched_model_index_.find(instance_id);
      if (find_matched_obj != matched_model_index_.end())
      {
        // Current object model can be matched in frame
        (*object)->outside_field_time_ = 0;
        (*object)->update_exist_prob(0.5);
      }
      else
      {
        // Current object model can not be matched in frame
        (*object)->update_exist_prob(-1);

        // auto object_mask = raycast_mask_.pair_instance_seg_.find(instance_id);
        // if (object_mask != raycast_mask_.pair_instance_seg_.end()) {
        //   // Current object model is in the view
        //   (*object)->outside_field_time_++;
        // } else {
        //   // Current object model is not in the view
        //   (*object)->outside_field_time_ = 0;
        // }
      }
    }
  }
}

bool Kfusion::raycasting(float4 k, float mu, uint frame)
{

  bool doRaycast = false;

  std::cout << "objects (id) in this view: ";
  for (const auto &id : objects_in_view_)
  {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  // if(frame > 2) {
  raycastPose = camera_pose;
  // single frame raycasting
  // raycastKernel(static_map_->volume_, vertex, normal, computationSize,
  //     raycastPose * getInverseCameraMatrix(k), nearPlane, farPlane, mu,
  //     static_map_->get_volume_step(), static_map_->get_volume_step()*BLOCK_SIDE);
  raycast_mask_.reset();
  // std::cout<<"labelImg in raycasting type: "<<frame_masks_->labelImg.type()<<std::endl;
  raycastObjectList(objectlist_, vertex, normal, raycast_mask_.labelImg, objects_in_view_,
                    raycastPose, k, computationSize, nearPlane, farPlane,
                    mu, true);
  split_labels(raycast_mask_, objectlist_);

  if (in_debug_)
  {
    mask_instance_output(raycast_mask_, objectlist_, frame, "raycast");
  }
  // delete_outside_view_obj(raycast_mask_, objectlist_);
  calc_unmatched_time();
  doRaycast = true;
  //  }
  return doRaycast;
}

bool Kfusion::raycasting(float4 k, float mu, uint frame, KeyFrame *kf)
{

  bool doRaycast = false;

  std::cout << "objects (id) in this view: ";
  for (const auto &id : objects_in_view_)
  {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  // if(frame > 2) {
  raycastPose = camera_pose;
  // single frame raycasting
  // raycastKernel(static_map_->volume_, vertex, normal, computationSize,
  //     raycastPose * getInverseCameraMatrix(k), nearPlane, farPlane, mu,
  //     static_map_->get_volume_step(), static_map_->get_volume_step()*BLOCK_SIDE);
  raycast_mask_.reset();
  // std::cout<<"labelImg in raycasting type: "<<frame_masks_->labelImg.type()<<std::endl;
  raycastObjectList(objectlist_, vertex, normal, raycast_mask_.labelImg, objects_in_view_,
                    raycastPose, k, computationSize, nearPlane, farPlane,
                    mu, true);
  split_labels(raycast_mask_, objectlist_);

  if (kf != nullptr)
  {
    kf->SetVertex(vertex);
    kf->SetNormal(normal);
    vpKFs_.push_back(kf);
  }

  if (in_debug_)
  {
    mask_instance_output(raycast_mask_, objectlist_, frame, "raycast");
  }
  // delete_outside_view_obj(raycast_mask_, objectlist_);
  calc_unmatched_time();
  doRaycast = true;
  //  }
  return doRaycast;
}

// use objectlist_ instead: integrate each volume separately
bool Kfusion::integration(float4 k, uint integration_rate, float mu,
                          uint frame)
{

  //  //on the first frame, only integrate backgrond
  //  if (frame == 0){
  //    cv::Mat bg_mask;
  //    auto find_instance_mask = frame_masks_->instance_id_mask.find(0);
  //    if ( find_instance_mask != frame_masks_->instance_id_mask.end()){
  //      bg_mask = find_instance_mask->second;
  //    }
  //    else{
  //      bg_mask = cv::Mat::ones(cv::Size(computationSize.y, computationSize.x), CV_8UC1);
  //    }
  //    static_map_->integrate_volume_kernel(floatDepth, inputRGB, bg_mask,
  //                            computationSize, camera_pose, mu, k,
  //                            frame);
  //    return true;
  //  }

  //  bool doIntegrate = poses.empty() ? this->rgbdTracker_->checkPose(camera_pose,
  //                                                                   T_w_r) : true;

  bool doIntegrate = true;

  if ((doIntegrate && ((frame % integration_rate) == 0)) || (frame <= 3))
  {
    // single global volume integration
    //    static_map_->integration_static_kernel(floatDepth, inputRGB,
    //                                           computationSize, camera_pose, mu,
    //                                           k, frame);
    //    if (in_debug_) {
    //      mask_instance_output(*frame_masks_, objectlist_, frame, "segment");
    //    }
    // multiple objects tracking
    // integrate from the frame_mask direction
    // ENSURE: frame_masks share the SAME instance ID as object_list
    for (auto object_mask = frame_masks_->pair_instance_seg_.begin();
         object_mask != frame_masks_->pair_instance_seg_.end(); ++object_mask)
    {
      const int &instance_id = object_mask->first;
      const instance_seg &instance_mask = object_mask->second;
      if (instance_id == INVALID)
        continue;
      //      const cv::Mat &instance_mask = object_mask->second.instance_mask_;
      const ObjectPointer &objectPtr = (objectlist_.at(instance_id));
      if (objectPtr->lost)
      {
        std::cout << "ID: " << instance_id << " label: " << object_mask->second.class_id_ << std::endl;
        // cv::waitKey(0);
        continue;
      }

      // create generalized instance mask
      cv::Mat gener_inst_mask;
      instance_mask.generalize_label(gener_inst_mask);

      // use motion residual to remove outliers in the generalized mask
      if (frame > 1)
      {
        objectPtr->refine_mask_use_motion(gener_inst_mask,
                                          use_icp_tracking_,
                                          use_rgb_tracking_);
      }

      cv::Mat debug;
      gener_inst_mask.convertTo(debug, CV_8U, 255.0);
      // std::ostringstream name;
      // name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
      //      <<"frame_"<< frame << "_object_id_" <<instance_id
      //      <<"_gener_refine_mask.png";
      cv::imshow("gener_refine_mask", debug);
      // cv::waitKey(0);
      // cv::imwrite(name.str(), debug);

      // ensure instance_id is the same order in the objectlist
      // geometric fusion
      objectPtr->integrate_volume_kernel(floatDepth, inputRGB, gener_inst_mask,
                                         computationSize, camera_pose, mu, k,
                                         frame);

      // semantic fusion
      // no need to integrate background
      if (instance_mask.class_id_ == 0)
        continue;

      objectlist_[instance_id]->fuse_semantic_kernel(*frame_masks_, instance_id);
    }

    //  for (auto object = objectlist_.begin(); object != objectlist_.end();
    //       ++object){
    //    const int instance_id = (*object)->instance_label_;
    //    cv::Mat labelMask;
    //    auto find_instance_mask = frame_masks_->instance_id_mask.find(instance_id);
    //      if ( find_instance_mask != frame_masks_->instance_id_mask.end()){
    //        labelMask = find_instance_mask->second;
    ////        std::cout<<"labelMask in intergration type: "<<labelMask.type()<<std::endl;
    //      }
    //      else{
    //        std::cout<<"in integration: mask missing: "<<instance_id<<std::endl;
    //        labelMask = cv::Mat::ones(cv::Size(computationSize.y, computationSize.x), CV_8UC1);
    ////        continue;
    //      }
    //    (*object)->integrate_volume_kernel(floatDepth, inputRGB, labelMask,
    //                                       computationSize, camera_pose, mu, k,
    //                                       frame);
    //  }

    doIntegrate = true;
  }
  else
  {
    doIntegrate = false;
  }
  return doIntegrate;
}

SegmentationResult Kfusion::volume2mask(float4 k, float mu)
{
  SegmentationResult renderedMask(computationSize.y, computationSize.x);
  volume2MaskKernel(objectlist_, renderedMask.labelImg, camera_pose, k,
                    computationSize, nearPlane, farPlane, mu);

  split_labels(renderedMask, objectlist_);
  // split_labels_and_delete(renderedMask, objectlist_);
  renderedMask.set_render_label(true);
  return renderedMask;
}

bool Kfusion::MaskRCNN_next_frame(uint frame, std::string segFolder)
{
  if (segFolder == "")
    return true;
  if (mask_rcnn_result_.size() > frame)
    return true;
  return false;
}

bool Kfusion::readMaskRCNN(float4 k, uint frame, std::string segFolder)
{
  // Test if do-segment
  if (segFolder == "")
  {
    if (frame == segment_startFrame_)
    {
      cv::Mat bg_mask = cv::Mat::ones(computationSize.y, computationSize.x,
                                      CV_8UC1) *
                        255;
      instance_seg bg(0, bg_mask);
      frame_masks_->pair_instance_seg_.insert(std::make_pair(0, bg));
    }
    return false;
  }

  mask_rcnn = mask_rcnn_result_[frame];
  // Over-segment current frame based on geometric edges
  geo_mask = segmenter_->compute_geom_edges(floatDepth, computationSize, k);

  if (in_debug_)
  {
    std::cout << "maskrcnn type: " << mask_rcnn.pair_instance_seg_.begin()->second.instance_mask_.type() << std::endl;
    mask_rcnn.output(frame, "maskrcnn");
  }

  // If there is no mask-rcnn recognition
  if (mask_rcnn.pair_instance_seg_.size() == 0)
  {
    // if initially
    if (frame == segment_startFrame_)
    {
      std::cout << "mask-rcnn fails on the first frame, skip segmentation on "
                   "this frame"
                << std::endl;
      segment_startFrame_++;
    }
    return false;
  }
  // If there is mask-rcnn recognition
  else
  {
    if (geom_refine_human)
    {
      // combine geometric components based on mask-rcnn
      std::cout << "Mask-RCNN has results, now combine it with geometric "
                   "components"
                << std::endl;
      if (do_edge_refinement)
      {
        geo2mask_result.reset();
        segmenter_->mergeLabels(geo2mask_result, geo_mask, mask_rcnn,
                                segmenter_->geo2mask_threshold);
        // segmenter_->mergeLabels(geo2mask_result, mask_rcnn, geo_mask,
        //                         segmenter_->geo2mask_threshold);
      }
      else
      {
        geo2mask_result = mask_rcnn;
      }

      if (in_debug_)
      {
        geo2mask_result.output(frame, "geo2mask_result");
      }

      // not believe maskrcnn to remove human
      segmenter_->remove_human_and_small(human_out_, geo2mask_result,
                                         min_object_size_);
      // geo2mask_result.inclusion_check();
    }
    else
    {
      // believe maskrcnn to remove human
      segmenter_->remove_human_and_small(human_out_, mask_rcnn,
                                         min_object_size_);
      // mask_rcnn.inclusion_check();
    }

    if (in_debug_)
    {
      human_out_.output(frame, "human_out");
    }

    return true;
  }
}

bool Kfusion::segment(float4 k, uint frame, std::string segFolder,
                      bool hasMaskRCNN)
{

  // Test if do segment
  if (segFolder == "")
  {
    /*if(frame == segment_startFrame_){
      cv::Mat bg_mask = cv::Mat::ones(computationSize.y, computationSize.x,
                                      CV_8UC1)*255;
      instance_seg bg(0, bg_mask);
      frame_masks_->pair_instance_seg_.insert(std::make_pair(0, bg));
    }*/
    return false;
  }

  if (use_GT_segment)
  {
    // cv::Mat GT_mask = cv::imread(GT_mask_files_[frame]);
    // GT_mask = (GT_mask == 0);

    cv::Mat GT_label = cv::imread(GT_mask_files_[frame], CV_16SC1);

    cv::Mat GT_mask = (GT_label == 0);
    cv::cvtColor(GT_mask, GT_mask, CV_BGR2GRAY);
    // GT_mask = (GT_mask == 7);
    // cv::Mat GT_label = cv::imread(GT_label_files_[frame]);
    if ((GT_mask.cols != static_cast<int>(computationSize.x)) ||
        (GT_mask.rows != static_cast<int>(computationSize.y)))
    {
      cv::Size imgSize = cv::Size(computationSize.x, computationSize.y);
      cv::resize(GT_mask, GT_mask, imgSize);
    }
    SegmentationResult GT_segmentation(computationSize);
    instance_seg one_instance(58, GT_mask);
    GT_segmentation.pair_instance_seg_.insert(std::make_pair(1, one_instance));

    cv::Mat GT_mask1 = (GT_label == 2);
    cv::cvtColor(GT_mask1, GT_mask1, CV_BGR2GRAY);

    // GT_mask = (GT_mask == 7);
    // cv::Mat GT_label = cv::imread(GT_label_files_[frame]);
    if ((GT_mask1.cols != static_cast<int>(computationSize.x)) ||
        (GT_mask1.rows != static_cast<int>(computationSize.y)))
    {
      cv::Size imgSize = cv::Size(computationSize.x, computationSize.y);
      cv::resize(GT_mask1, GT_mask1, imgSize);
    }
    SegmentationResult GT_segmentation1(computationSize);
    instance_seg one_instance1(57, GT_mask1);
    GT_segmentation.pair_instance_seg_.insert(std::make_pair(2, one_instance1));

    GT_segmentation.generate_bgLabels();
    frame_masks_ = std::make_shared<SegmentationResult>(GT_segmentation);

    if (frame == segment_startFrame_)
    {
      generate_new_objects(GT_segmentation, k, frame);
    }

    return true;
  }

  if (hasMaskRCNN && (!geom_refine_human))
  {
    // combine geometric components based on mask-rcnn
    std::cout << "Mask-RCNN has results, now combine it with geometric "
                 "components"
              << std::endl;
    geo2mask_result = human_out_;

    if (in_debug_)
    {
      geo2mask_result.output(frame, "geo2mask_before");
    }

    if (do_edge_refinement)
    {
      segmenter_->mergeLabels(geo2mask_result, geo_mask, human_out_,
                              segmenter_->geo2mask_threshold);
    }

    if (in_debug_)
    {
      geo2mask_result.output(frame, "geo2mask_after");
    }
  }
  else
  {
    geo2mask_result = human_out_;
  }
  geo2mask_result.inclusion_check();

  // For the first frame, no extra object generated yet,
  // no need to project volumes to masks
  if (frame == segment_startFrame_)
  {
    if (!hasMaskRCNN)
    {
      std::cout << "mask-rcnn fails on the first frame, skip segmentation on "
                   "this frame"
                << std::endl;
      segment_startFrame_++;
      return false;
    }

    // if there is mask-rcnn on the first frame, combine it with geometric
    //  over-segmentation to create segmentation mask
    else
    {
      if (!geom_refine_human)
      {
        geo2mask_result.exclude_human();
        // geo2mask_result.inclusion_check();
      }

      // on the first frame, only background model has already been generated
      SegmentationResult first_frame_mask(computationSize);
      const instance_seg &bg = geo2mask_result.pair_instance_seg_.at(0);
      first_frame_mask.pair_instance_seg_.insert(std::make_pair(0, bg));
      frame_masks_ = std::make_shared<SegmentationResult>(first_frame_mask);

      if (in_debug_)
      {
        frame_masks_->output(frame, "frame_masks");
      }

      // if there is extra mask other than background generated on the first
      // frame if(frame_masks_->instance_id_mask.size()>1){
      generate_new_objects(geo2mask_result, k, frame);

      return true;
    }
  }

  if (frame > segment_startFrame_)
  {
    bool hasnewModel = false;

    if (hasMaskRCNN)
    {
      if (raycast_mask_.pair_instance_seg_.size() <= 1)
      {
        rendered_mask_ = volume2mask(k, config.mu);
        if (in_debug_)
        {
          rendered_mask_.output(frame, "volume2mask_old");
        }
      }

      SegmentationResult newModelMask(computationSize);
      SegmentationResult mask2model_result(computationSize);
      std::map<int, cv::Mat> inclusion_map; // used for inclusion check
      matched_model_index_.clear();         // used for wrong model elimination
      uint relative_frame = frame - segment_startFrame_;

      if (in_debug_)
      {
        rendered_mask_.output(frame, "rendered_after_matching");
      }
      // Assign the local segmentation to the projection from global objects
      hasnewModel = segmenter_->local2global(mask2model_result, newModelMask,
                                             geo2mask_result, rendered_mask_,
                                             min_object_size_,
                                             segmenter_->mask2model_threshold,
                                             segmenter_->new_model_threshold,
                                             inclusion_map,
                                             relative_frame,
                                             matched_model_index_);

      // For the remaining that cannot match mask-rcnn (possibly due to wrong
      // recognition), try to match with the projected labels
      // std::cout << "start final merge" << std::endl;
      if (in_debug_)
      {
        mask2model_result.output(frame, "mask2model");
        newModelMask.output(frame, "newModelMask");
      }

      if (do_edge_refinement)
      {
        SegmentationResult final_result = segmenter_->finalMerge(geo_mask, mask2model_result,
                                                                 segmenter_->geo2mask_threshold);
        final_result.exclude_human();
        frame_masks_ = std::make_shared<SegmentationResult>(final_result);
      }
      else
      {
        frame_masks_ = std::make_shared<SegmentationResult>(mask2model_result);
      }
      // frame_masks_ = std::make_shared<SegmentationResult>(mask2model_result);
      frame_masks_->combine_labels();

      if (in_debug_)
      {
        frame_masks_->output(frame, "frame_masks");
      }

      if (hasnewModel)
      {
        generate_new_objects(newModelMask, k, frame);
      }
      // frame_masks_->inclusion_check();

      return true;
    }
    else
    {
      if (in_debug_)
      {
        std::cout << "mask-rcnn fails on the frame " << frame
                  << ", now combine geometric components with projected labels"
                  << std::endl;
      }
      /*rendered_mask = volume2mask(k, mu);
      if (in_debug_) {
        mask_instance_output(rendered_mask, objectlist_, frame, "volume2mask");
      }*/

      SegmentationResult final_result = segmenter_->finalMerge(geo_mask,
                                                               rendered_mask_,
                                                               segmenter_->geo2mask_threshold);

      final_result.generate_bgLabels();
      frame_masks_ = std::make_shared<SegmentationResult>(final_result);
      // frame_masks_->inclusion_check();

      // since there is no mask-rcnn, cannot judge if there is new object
      return true;
    }
  }

  if (in_debug_)
  {
    for (auto object = objectlist_.begin(); object != objectlist_.end();
         ++object)
    {
      int class_id = (*object)->class_id_;
      std::cout << "after segment: class id: " << class_id << std::endl;
      printMatrix4("after segment: object pose ", (*object)->volume_pose_);
    }
  }

  return true;
}

void Kfusion::generate_new_objects(const SegmentationResult &masks,
                                   const float4 k, const uint frame)
{
  double start = tock();
  // Has new label
  for (auto object_mask = masks.pair_instance_seg_.begin();
       object_mask != masks.pair_instance_seg_.end(); ++object_mask)
  {

    // If the mask is approach boundary, stop generating the object
    const cv::Mat &mask = object_mask->second.instance_mask_;
    // uint2 maskSize;
    // maskSize.x = mask.cols;
    // maskSize.y = mask.rows;
    // if (check_truncated(mask, maskSize)) continue;
    /*
        //threshold the size of mask to determine if a new object needs to be
            // generated
        int mask_size = cv::countNonZero(mask);
        if (mask_size< min_object_size_) {
          std::cout<<"object size "<<mask_size<<" smaller than threshold "
                                                <<min_object_size_<<std::endl;
          cv::bitwise_or(mask, masks.instance_id_mask[0], masks.instance_id_mask[0]);
          masks.instance_id_mask.erase(object_mask);
          continue;
        }
    */
    const int &instance_id = object_mask->first;
    const int &class_id = object_mask->second.class_id_;
    const All_Prob_Vect &class_all_prob = object_mask->second.all_prob_;

    if ((class_id == 0) || (class_id == 255))
      continue;

    // ignore person:
    // if(class_id == 1) continue;
    // if(class_id == 57) continue;

    // Determine the the volume size and pose
    float volume_size;
    Matrix4 T_w_o;
    int volume_resol;
    spawnNewObjectKernel(T_w_o, volume_size, volume_resol, mask, camera_pose, k);

    if (volume_size == 0)
      return;

    if (in_debug_)
    {
      std::ostringstream name;
      name << "/home/ryf/slam/vimid/apps/kfusion/cmake-build-debug/debug/"
           << "new_generate/"
           << "frame_" << frame << "_object_id_" << instance_id
           << "_class_id_" << class_id << ".png";
      cv::imwrite(name.str(), mask);
    }

    // Matrix4 fake_pose = Identity();
    ObjectPointer new_object(new Object(config.voxel_block_size,
                                        make_float3(volume_size),
                                        make_uint3(volume_resol),
                                        T_w_o, camera_pose, class_id,
                                        class_all_prob,
                                        computationSize));

    // Create generalized instance mask
    cv::Mat gener_inst_mask;
    object_mask->second.generalize_label(gener_inst_mask);

    new_object->integrate_volume_kernel(floatDepth, inputRGB, gener_inst_mask,
                                        computationSize, camera_pose, _mu, k, frame);

    // Create fg_outlier_mask
    cv::Mat fg_outlier_mask;
    cv::bitwise_not(mask, fg_outlier_mask);
    new_object->prev_fg_outlier_mask_ = fg_outlier_mask;

    std::cout << "New object generated at frame " << frame
              << ", with volume size: " << new_object->get_volume_size().x
              << ", with volume resol: " << new_object->get_volume_resol().x
              << ", with volume step: " << new_object->get_volume_step()
              << ", class id: " << class_id << std::endl;
    printMatrix4(", with pose", T_w_o);
    add_object_into_list(new_object, this->objectlist_);
    objects_in_view_.insert(new_object->instance_label_);
    // if (instance_id>(int)objectlist_.size())
    //   new_object->instance_label_ = instance_id;
  }

  init_time_ = tock() - start;
}

void Kfusion::spawnNewObjectKernel(Matrix4 &T_w_o, float &volume_size,
                                   int &volume_resol,
                                   const cv::Mat &mask, const Matrix4 &T_w_c,
                                   const float4 &k)
{
  float3 *c_vertex;
  c_vertex = (float3 *)calloc(sizeof(float3) * computationSize.x *
                                  computationSize.y,
                              1);
  depth2vertexKernel(c_vertex, floatDepth, computationSize,
                     getInverseCameraMatrix(k));

  float x_min = INFINITY;
  float y_min = INFINITY;
  float z_min = INFINITY;
  float x_max = -INFINITY;
  float y_max = -INFINITY;
  float z_max = -INFINITY;
  float x_avg = 0;
  float y_avg = 0;
  float z_avg = 0;
  int count = 0;

  std::cout << computationSize.x << " " << computationSize.y << std::endl;
  std::cout << mask.size() << std::endl;
  //  std::cout<<"spawn mask type: "<<mask.type()<<std::endl;
  // TODO: openmp
  for (uint pixely = 0; pixely < computationSize.y; pixely++)
  {
    for (uint pixelx = 0; pixelx < computationSize.x; pixelx++)
    {
      if (mask.at<uchar>(pixely, pixelx) == 0)
        continue;
      //      std::cout<<pixelx<<" "<<pixely<<std::endl;
      int id = pixelx + pixely * computationSize.x;
      float3 w_vertex = T_w_c * c_vertex[id];
      if (w_vertex.x > x_max)
        x_max = w_vertex.x;
      if (w_vertex.x < x_min)
        x_min = w_vertex.x;
      if (w_vertex.y > y_max)
        y_max = w_vertex.y;
      if (w_vertex.y < y_min)
        y_min = w_vertex.y;
      if (w_vertex.z > z_max)
        z_max = w_vertex.z;
      if (w_vertex.z < z_min)
        z_min = w_vertex.z;
      x_avg += w_vertex.x;
      y_avg += w_vertex.y;
      z_avg += w_vertex.z;
      count++;
    }
  }
  x_avg = x_avg / count;
  y_avg = y_avg / count;
  z_avg = z_avg / count;

  //  x_avg = (x_max+x_min)/2.0f;
  //  y_avg = (y_max+y_min)/2.0f;
  //  z_avg = (z_max+z_min)/2.0f;

  std::cout << "max/min x/y/z: " << x_min << " " << x_max << " " << y_min << " " << y_max << " "
            << z_min << " " << z_max << std::endl;
  float max_size = max(make_float3(x_max - x_min, y_max - y_min, (z_max - z_min) / 2));
  std::cout << "average of vertex: " << x_avg << " " << y_avg << " " << z_avg << ", with the max size " << max_size << std::endl;
  volume_size = fminf(2.5 * max_size, 5.0);

  // control one volume larger than 0.01cm
  if (volume_size < 0.64)
    volume_resol = 256;
  if ((0.64 < volume_size) && (volume_size < 1.28))
    volume_resol = 512;
  if ((1.28 < volume_size) && (volume_size < 2.56))
    volume_resol = 1024;
  if ((2.56 < volume_size) && (volume_size < 5.12))
    volume_resol = 2048;
  //  if ((5.12 < volume_size) && ( volume_size< 10.24)) volume_resol = 512;

  // shift the T_w_o from center to left side corner
  T_w_o = Identity();
  T_w_o.data[0].w = x_avg - volume_size / 2;
  T_w_o.data[1].w = y_avg - volume_size / 2;
  T_w_o.data[2].w = z_avg - volume_size / 2;
  free(c_vertex);
}

void Kfusion::add_object_into_list(ObjectPointer &f_objectpoint,
                                   ObjectList &f_objectlist)
{
  size_t current_object_nums = f_objectlist.size();
  f_objectpoint->instance_label_ = current_object_nums; // labels starting from 0
  f_objectpoint->output_id_ = current_object_nums + erase_num_;
  f_objectlist.push_back(f_objectpoint);
};

void Kfusion::dumpVolume(std::string)
{
}

void Kfusion::printStats()
{
  int occupiedVoxels = 0;
  for (unsigned int x = 0; x < static_map_->volume_._resol; x++)
  {
    for (unsigned int y = 0; y < static_map_->volume_._resol; y++)
    {
      for (unsigned int z = 0; z < static_map_->volume_._resol; z++)
      {
        if (static_map_->volume_[make_uint3(x, y, z)].x < 1.f)
        {
          occupiedVoxels++;
        }
      }
    }
  }
  std::cout << "The number of non-empty voxel is: " << occupiedVoxels << std::endl;
}

template <typename FieldType>
void raycastOrthogonal(Volume<FieldType> &volume, std::vector<float4> &points, const float3 origin, const float3 direction,
                       const float farPlane, const float step)
{

  // first walk with largesteps until we found a hit
  auto select_depth = [](const auto &val)
  { return val.x; };
  float t = 0;
  float stepsize = step;
  float f_t = volume.interp(origin + direction * t, select_depth);
  t += step;
  float f_tt = 1.f;

  for (; t < farPlane; t += stepsize)
  {
    f_tt = volume.interp(origin + direction * t, select_depth);
    if ((std::signbit(f_tt) != std::signbit(f_t)))
    { // got it, jump out of inner loop
      if (f_t == 1.0 || f_tt == 1.0)
      {
        f_t = f_tt;
        continue;
      }
      t = t + stepsize * f_tt / (f_t - f_tt);
      points.push_back(make_float4(origin + direction * t, 1));
    }
    if (f_tt < std::abs(0.8f)) // coming closer, reduce stepsize
      stepsize = step;
    f_t = f_tt;
  }
}

void Kfusion::getPointCloudFromVolume()
{

  std::vector<float4> points;

  float x = 0, y = 0, z = 0;

  int3 resolution = make_int3(static_map_->volume_._resol);
  float3 incr = make_float3(
      this->static_map_->volume_._size / resolution.x,
      this->static_map_->volume_._size / resolution.y,
      this->static_map_->volume_._size / resolution.z);

  // XY plane

  std::cout << "Raycasting from XY plane.. " << std::endl;
  for (y = 0; y < this->static_map_->volume_._size; y += incr.y)
  {
    for (x = 0; x < this->static_map_->volume_._size; x += incr.x)
    {
      raycastOrthogonal(static_map_->volume_, points, make_float3(x, y, 0), make_float3(0, 0, 1),
                        this->static_map_->volume_._size, static_map_->get_volume_step());
    }
  }

  // ZY PLANE
  std::cout << "Raycasting from ZY plane.. " << std::endl;
  for (z = 0; z < this->static_map_->volume_._size; z += incr.z)
  {
    for (y = 0; y < this->static_map_->volume_._size; y += incr.y)
    {
      raycastOrthogonal(static_map_->volume_, points, make_float3(0, y, z), make_float3(1, 0, 0),
                        this->static_map_->volume_._size, static_map_->get_volume_step());
    }
  }

  // ZX plane

  for (z = 0; z < this->static_map_->volume_._size; z += incr.z)
  {
    for (x = 0; x < this->static_map_->volume_._size; x += incr.x)
    {
      raycastOrthogonal(static_map_->volume_, points, make_float3(x, 0, z), make_float3(0, 1, 0),
                        this->static_map_->volume_._size, static_map_->get_volume_step());
    }
  }

  int num_points = points.size();
  std::cout << "Total number of ray-casted points : " << num_points << std::endl;

  if (!getenv("TRAJ"))
  {
    std::cout << "Can't output the model point-cloud, unknown trajectory" << std::endl;
    return;
  }

  int trajectory = std::atoi(getenv("TRAJ"));
  std::stringstream filename;

  filename << "./pointcloud-vanilla-traj" << trajectory << "-" << static_map_->volume_._resol << ".ply";

  // Matrix4 flipped = toMatrix4( TooN::SE3<float>(TooN::makeVector(0,0,0,0,0,0)));

  // flipped.data[0].w =  (-1 * this->_initPose.x);
  // flipped.data[1].w =  (-1 * this->_initPose.y);
  // flipped.data[2].w =  (-1 * this->_initPose.z);

  // std::cout << "Generating point-cloud.. " << std::endl;
  // for(std::vector<float4>::iterator it = points.begin(); it != points.end(); ++it){
  //         float4 vertex = flipped * (*it);
  //     }
}
std::vector<uchar4> colors = random_color(91);

void Kfusion::renderVolume(uchar4 *out, uint2 outputSize, int frame,
                           int raycast_rendering_rate, float4 k, float largestep, bool render_color)
{
  if (frame % raycast_rendering_rate == 0)
  {

    //
    //    renderVolumeKernel(static_map_->volume_, out, outputSize,
    //                       *(this->viewPose) * getInverseCameraMatrix(k), nearPlane,
    //                       farPlane * 2.0f, _mu, static_map_->get_volume_step(), largestep,
    //                       get_translation(*(this->viewPose)), ambient,
    //                       false, render_color_,
    //                       vertex, normal);

    //    Matrix4 poseTosee = camera_pose;
    //  poseTosee.data[0].w -= 0.1f;
    //    poseTosee.data[1].w -= 0.1f;
    //    poseTosee.data[2].w -= 0.1f;
    //        setViewPose(&poseTosee);
    renderVolume_many_Kernel(objectlist_, out, outputSize, *(this->viewPose), k,
                             nearPlane, farPlane * 2.0f, _mu, largestep, ambient,
                             (!compareMatrix4(*(this->viewPose), raycastPose) || (computationSize.x != outputSize.x) ||
                              (computationSize.y != outputSize.y)),
                             render_color, vertex, normal,
                             raycast_mask_.labelImg, colors);

    if (render_output)
    {
      std::string volume_file;
      if (render_color)
      {
        volume_file = config.output_images + "color_volume";
      }
      else
      {
        volume_file = config.output_images + "label_volume";
      }
      opengl2opencv(out, outputSize, frame - segment_startFrame_, volume_file);
    }
    // renderVolume_many_Kernel(objectlist_, out, outputSize, *(this->viewPose), k,
    //                          nearPlane, farPlane * 2.0f, _mu, largestep, ambient,
    //                          true,
    //                          render_color_, vertex, normal,
    //                          frame_masks_->labelImg, colors);
  }
}

void Kfusion::renderMainView(uchar4 *out, uint2 outputSize, int frame,
                             int raycast_rendering_rate, float4 k, float largestep, bool render_color, Matrix4 poseToSee)
{
  if (frame % raycast_rendering_rate == 0)
  {

    // Matrix4 poseToSee = this->init_camera_pose;
    // poseToSee.data[2].w += 0.5f;
    // setViewPose(&poseTosee);
    renderVolume_many_Kernel(objectlist_, out, outputSize, poseToSee, k,
                             nearPlane, farPlane * 2.0f, _mu, largestep, ambient,
                             (!compareMatrix4(poseToSee, raycastPose) || (computationSize.x != outputSize.x) ||
                              (computationSize.y != outputSize.y)),
                             render_color, vertex, normal,
                             raycast_mask_.labelImg, colors);

    if (render_output)
    {
      std::string volume_file;
      if (render_color)
      {
        volume_file = config.output_images + "color_volume_main";
      }
      else
      {
        volume_file = config.output_images + "label_volume_main";
      }
      opengl2opencv(out, outputSize, frame, volume_file);
    }
  }
}

void Kfusion::renderTrack(uchar4 *out, uint2 outputSize)
{

  if ((use_icp_tracking_) && (!use_rgb_tracking_))
  {
    if (raycast_mask_.pair_instance_seg_.size() > 1)
    {
      renderTrackKernel(out, objectlist_.at(1)->trackresult_, outputSize);
    }
    else
    {
      renderTrackKernel(out, objectlist_.at(0)->trackresult_, outputSize);
    }
  }
  if ((!use_icp_tracking_) && (use_rgb_tracking_))
  {
    if (raycast_mask_.pair_instance_seg_.size() > 1)
    {
      renderRGBTrackKernel(out, objectlist_.at(1)->trackresult_ + computationSize.x * computationSize.y, outputSize);
    }
    else
    {
      renderRGBTrackKernel(out, objectlist_.at(0)->trackresult_ + computationSize.x * computationSize.y, outputSize);
    }
  }
  if ((use_icp_tracking_) && (use_rgb_tracking_))
  {
    if (render_output)
    {
      //      renderTrackKernel(out, objectlist_.at(0)->trackresult_, outputSize);
      //      cv::Mat icp_track = opengl2opencv(out, outputSize);
      //      renderRGBTrackKernel(out, objectlist_.at(1)->trackresult_
      //          + computationSize.x * computationSize.y, outputSize);
    }
    if (raycast_mask_.pair_instance_seg_.size() > 1)
    {
      //      renderRGBTrackKernel(out, objectlist_.at(1)->trackresult_
      //          + computationSize.x * computationSize.y, outputSize);
      //      renderTrackKernel(out, objectlist_.at(1)->trackresult_, outputSize);

      render_RGBD_TrackKernel(out, objectlist_.at(1)->trackresult_,
                              objectlist_.at(1)->trackresult_ + computationSize
                                                                        .x *
                                                                    computationSize.y,
                              outputSize);
    }
    else
    {
      render_RGBD_TrackKernel(out, objectlist_.at(0)->trackresult_,
                              objectlist_.at(0)->trackresult_ + computationSize
                                                                        .x *
                                                                    computationSize.y,
                              outputSize);
    }
  }
}

void Kfusion::renderTrack(uchar4 *out, uint2 outputSize, int type, int frame)
{

  if (type == 0)
  {
    if (raycast_mask_.pair_instance_seg_.size() > 1)
    {
      renderTrackKernel(out, objectlist_.at(0)->trackresult_, outputSize);
    }
    else
    {
      renderTrackKernel(out, objectlist_.at(0)->trackresult_, outputSize);
    }
    if (render_output)
    {
      std::string motion_file;
      motion_file = config.output_images + "motion_icp";
      opengl2opencv(out, outputSize, frame, motion_file);
    }
  }
  if (type == 1)
  {
    if (raycast_mask_.pair_instance_seg_.size() > 1)
    {
      renderRGBTrackKernel(out, objectlist_.at(1)->trackresult_ + computationSize.x * computationSize.y, outputSize);
    }
    else
    {
      renderRGBTrackKernel(out, objectlist_.at(0)->trackresult_ + computationSize.x * computationSize.y, outputSize);
    }
    if (render_output)
    {
      std::string motion_file;
      motion_file = config.output_images + "motion_rgb";
      opengl2opencv(out, outputSize, frame, motion_file);
    }
  }
  if (type == 2)
  {

    if (raycast_mask_.pair_instance_seg_.size() > 1)
    {
      //      renderRGBTrackKernel(out, objectlist_.at(1)->trackresult_
      //          + computationSize.x * computationSize.y, outputSize);
      //      renderTrackKernel(out, objectlist_.at(1)->trackresult_, outputSize);

      render_RGBD_TrackKernel(out, objectlist_.at(0)->trackresult_,
                              objectlist_.at(0)->trackresult_ +
                                  computationSize.x * computationSize.y,
                              outputSize);
    }
    else
    {
      render_RGBD_TrackKernel(out, objectlist_.at(0)->trackresult_,
                              objectlist_.at(0)->trackresult_ +
                                  computationSize.x * computationSize.y,
                              outputSize);
    }

    if (render_output)
    {
      std::string motion_file;
      motion_file = config.output_images + "motion_joint_bg";
      opengl2opencv(out, outputSize, frame, motion_file);
    }
  }
}

void Kfusion::renderDepth(uchar4 *out, uint2 outputSize)
{
  renderDepthKernel(out, floatDepth, outputSize, nearPlane, farPlane);
}

void Kfusion::renderDepth(uchar4 *out, uint2 outputSize, int frame)
{
  renderDepthKernel(out, floatDepth, outputSize, nearPlane, farPlane);
  if (render_output)
  {
    std::string depth_file;
    depth_file = config.output_images + "depth";
    opengl2opencv(out, outputSize, frame, depth_file);
  }
}

void Kfusion::renderIntensity(uchar4 *out, uint2 outputSize)
{
  renderIntensityKernel(out, g_inputGrey, outputSize);
}

void Kfusion::renderClass(uchar4 *out,
                          uint2 outputSize,
                          SegmentationResult segmentationResult)
{
  //  if (segmentationResult.class_id_mask.size()>0){
  //    renderMaskKernel(out, g_inputGrey, outputSize, segmentationResult.labelImg, colors);
  renderClassMaskKernel(out, outputSize, segmentationResult, colors);
  //  }
  //  else{
  //    std::cout<<"renderIntensity"<<std::endl;
  //    renderIntensityKernel(out, g_inputGrey, outputSize);
  //  }
}

void Kfusion::renderInstance(uchar4 *out, uint2 outputSize,
                             const SegmentationResult &segmentationResult)
{
  //  if (segmentationResult.class_id_mask.size()>0){
  //    renderMaskKernel(out, g_inputGrey, outputSize, segmentationResult.labelImg, colors);
  renderInstanceMaskKernel(out, outputSize, segmentationResult, colors);
  //  }
  //  else{
  //    std::cout<<"renderIntensity"<<std::endl;
  //    renderIntensityKernel(out, g_inputGrey, outputSize);
  //  }
}

void Kfusion::renderInstance(uchar4 *out, uint2 outputSize,
                             const SegmentationResult &segmentationResult,
                             int frame, std::string labelSource)
{
  //  if (segmentationResult.class_id_mask.size()>0){
  //    renderMaskKernel(out, g_inputGrey, outputSize, segmentationResult.labelImg, colors);
  renderInstanceMaskKernel(out, outputSize, segmentationResult, colors);

  if (render_output)
  {
    std::string mask_file;
    mask_file = config.output_images + labelSource;
    opengl2opencv(out, outputSize, frame, mask_file);
  }
}

void Kfusion::renderMaskWithImage(uchar4 *out, uint2 outputSize,
                                  const SegmentationResult &
                                      segmentationResult)
{
  renderMaskWithImageKernel(out, outputSize, inputRGB, segmentationResult,
                            colors);
}

void Kfusion::renderMaskWithImage(uchar4 *out, uint2 outputSize,
                                  const SegmentationResult &segmentationResult,
                                  int frame, std::string labelSource)
{
  renderMaskWithImage(out, outputSize, segmentationResult);

  if (render_output)
  {
    std::string volume_file;
    volume_file = config.output_images + labelSource;
    opengl2opencv(out, outputSize, frame, volume_file);
  }
}

void Kfusion::renderMaskMotionWithImage(uchar4 *out, uint2 outputSize,
                                        const SegmentationResult &segmentationResult,
                                        int frame)
{
  renderMaskMotionWithImageKernel(out, outputSize, inputRGB,
                                  segmentationResult, objectlist_, colors);

  if (render_output)
  {
    std::string volume_file;
    volume_file = config.output_images + "inst_geo_motion";
    opengl2opencv(out, outputSize, frame, volume_file);
  }
}

void Kfusion::split_labels(SegmentationResult &segmentationResult,
                           const ObjectList &objectList)
{
  for (auto object = objectList.begin(); object != objectList.end(); ++object)
  {
    const int &instance_id = (*object)->instance_label_;
    const int &class_id = (*object)->class_id_;
    const cv::Mat &mask = (segmentationResult.labelImg == instance_id);
    if (cv::countNonZero(mask) < 20)
      continue;
    instance_seg splitted(class_id, mask);
    segmentationResult.pair_instance_seg_.insert(std::make_pair(instance_id, splitted));
  }
}

void Kfusion::split_labels_and_delete(SegmentationResult &segmentationResult,
                                      ObjectList &objectList)
{
  int index = 0;
  for (auto object = objectList.begin(); object != objectList.end(); ++object)
  {
    // Delete the object if it is in view but do not match the mask up to 5 times consecutive
    if ((*object)->outside_field_time_ >= 5)
    {
      object = objectList.erase(object);
      object--;
      index++;
      continue;
    }
    const int &instance_id = (*object)->instance_label_;
    const int &class_id = (*object)->class_id_;
    const cv::Mat &mask = (segmentationResult.labelImg == instance_id);
    if (cv::countNonZero(mask) < 1)
      continue;
    instance_seg splitted(class_id, mask);
    segmentationResult.pair_instance_seg_.insert(std::make_pair(instance_id, splitted));
    segmentationResult.correspond_index_ = index;
    index++;
  }
}

void Kfusion::split_labels(SegmentationResult &segmentationResult,
                           std::set<int> &object_in_view,
                           const ObjectList &objectList)
{
  object_in_view.clear();
  for (auto object = objectList.begin(); object != objectList.end(); ++object)
  {
    const int &instance_id = (*object)->instance_label_;
    const int &class_id = (*object)->class_id_;
    const cv::Mat &mask = (segmentationResult.labelImg == instance_id);
    if (cv::countNonZero(mask) < 20)
      continue; // 1
    instance_seg splitted(class_id, mask);
    object_in_view.insert(instance_id);
    segmentationResult.pair_instance_seg_.insert(std::make_pair(instance_id, splitted));
  }
}

// void Kfusion::renderMask(uchar4 * out, uint2 outputSize, SegmentationResult segmentationResult) {
//   renderIntensityKernel(out, I_l[0], outputSize, segmentationResult.class_mask, segmentationResult.class_id, colors);
// }

// use objectlist_ instead: dump all volumes
void Kfusion::dump_mesh(const std::string filename)
{

  auto inside = [](const Volume<FieldType>::compute_type &val)
  {
    // meshing::status code;
    // if(val.y == 0.f)
    //   code = meshing::status::UNKNOWN;
    // else
    //   code = val.x < 0.f ? meshing::status::INSIDE : meshing::status::OUTSIDE;
    // return code;
    // std::cerr << val.x << " ";
    return val.x < 0.f;
  };

  auto select = [](const Volume<FieldType>::compute_type &val)
  {
    return val.x;
  };

  for (const auto &obj : objectlist_)
  {
    std::vector<Triangle> mesh;
    algorithms::marching_cube(obj->volume_._map_index, select, inside, mesh);
    const int obj_id = obj->instance_label_;
    Matrix4 T = obj->volume_pose_;
    Eigen::Matrix4f T_WM = fromMidFusionToEigen(T);
    // const std::string obj_vol_name = filename+"_"+std::to_string(obj_id) + ".vtk";
    // writeVtkMesh(obj_vol_name.c_str(), mesh);
    const std::string obj_vol_name = filename + "_" + std::to_string(obj_id) + ".ply";
    std::cout << obj_vol_name << std::endl;
    save_mesh_ply(mesh, obj_vol_name, T_WM, nullptr, nullptr);
  }
}

// void Kfusion::saveFieldSlice(const std::string &file_path,
//                              const Eigen::Vector3f &point_W,
//                              const std::string &num) const
// {
//   auto get_field_value = [](const Volume<FieldType>::compute_type& val) {
//       return val.x;
//   };
//
//   const std::string file_name_x =
//           (num == std::string("")) ? (file_path + "_x.vtk") : (file_path + "_x_" + num + ".vtk");
//   const std::string file_name_y =
//           (num == std::string("")) ? (file_path + "_y.vtk") : (file_path + "_y_" + num + ".vtk");
//   const std::string file_name_z =
//           (num == std::string("")) ? (file_path + "_z.vtk") : (file_path + "_z_" + num + ".vtk");
//   // save3DSlice()
// }

void Kfusion::save_poses(const std::string filename, const int frame)
{

  for (const auto &obj : objectlist_)
  {
    const int obj_id = obj->instance_label_;
    Matrix4 obj_pose;
    std::string obj_pose_file;
    if (obj_id == 0)
    {
      obj_pose = camera_pose;
      obj_pose_file = filename + "_camera";
    }
    else
    {
      obj_pose = obj->volume_pose_;
      obj_pose_file = filename + "_" + std::to_string(obj_id);
    }
    Eigen::Quaternionf q = getQuaternion(obj_pose);
    std::ofstream ofs;
    if (obj->pose_saved_)
    {
      ofs.open(obj_pose_file, std::ofstream::app);
    }
    else
    {
      ofs.open(obj_pose_file);
      obj->pose_saved_ = true;
    }

    if (ofs.is_open())
    {
      // save in the TUM pose: x-y-z-qx-qy-qz-qw
      ofs << frame << "\t" << camera_pose.data[0].w << "\t" << camera_pose.data[1].w
          << "\t" << camera_pose.data[2].w << "\t" << q.x() << "\t" << q.y()
          << "\t" << q.z() << "\t" << q.w() << "\t" << std::endl;
      ofs.close();
    }
    else
    {
      std::cout << "Error opening file for object " << obj_id << std::endl;
    }
  }
}

void Kfusion::save_poses(const std::string filename, const okvis::Time &timestamp)
{

  for (const auto &obj : objectlist_)
  {
    // const int obj_id = obj->instance_label_;
    const int obj_id = obj->output_id_;
    Matrix4 obj_pose;
    std::string obj_pose_file;
    if (obj_id == 0)
    {
      obj_pose = camera_pose;
      obj_pose_file = filename + "_camera";
    }
    else
    {
      obj_pose = obj->volume_pose_;
      obj_pose_file = filename + "_" + std::to_string(obj_id);
    }
    Eigen::Quaternionf q = getQuaternion(obj_pose);
    std::ofstream ofs;
    if (obj->pose_saved_)
    {
      ofs.open(obj_pose_file, std::ofstream::app);
    }
    else
    {
      ofs.open(obj_pose_file);
      obj->pose_saved_ = true;
    }

    if (ofs.is_open())
    {
      // save in the TUM pose: x-y-z-qx-qy-qz-qw
      ofs << timestamp << " " << obj_pose.data[0].w << " " << obj_pose.data[1].w
          << " " << obj_pose.data[2].w << " " << q.x() << " " << q.y()
          << " " << q.z() << " " << q.w() << std::endl;
    }
    else
    {
      std::cout << "Error opening file for object " << obj_id << std::endl;
    }
  }
}

void Kfusion::save_times(const std::string filename, const int frame,
                         double *timings)
{

  std::string time_file = filename + "_time";
  std::ofstream ofs;
  if (frame == 0)
  {
    ofs.open(time_file);
  }
  else
  {
    ofs.open(time_file, std::ofstream::app);
  }

  size_t obj_num = objectlist_.size();
  size_t mov_obj_num = move_obj_set.size();
  size_t in_view_obj_num = objects_in_view_.size();
  double total = timings[8] - timings[0];
  double tracking = timings[4] - timings[3];
  double segmentation = timings[3] - timings[2] + timings[5] - timings[4];
  double integration = timings[6] - timings[5];
  double raycasting = timings[7] - timings[6];
  double computation = timings[7] - timings[2];
  double obj_tracking = timings[5] - timings[4];

  if ((mov_obj_num == mov_obj_num_) && (in_view_obj_num == in_view_obj_num_) && (time_obj_num_ == obj_num))
  {
    // update
    total_ += total;
    tracking_ += tracking;
    obj_tracking_ += obj_tracking;
    segmentation_ += segmentation;
    integration_ += integration;
    raycasting_ += raycasting;
    computation_time_ += computation;
    same_obj_frame++;
  }
  else
  {
    // output
    if (ofs.is_open())
    {
      //      save in the TUM pose: x-y-z-qx-qy-qz-qw
      ofs << time_obj_num_ << "\t" << mov_obj_num_ << "\t" << in_view_obj_num_ << "\t"
          << tracking_ / same_obj_frame << "\t"
          << obj_tracking_ / same_obj_frame << "\t"
          << segmentation_ / same_obj_frame << "\t"
          << init_time_ << "\t"
          << integration_ / same_obj_frame << "\t"
          << raycasting_ / same_obj_frame << "\t"
          << computation_time_ / same_obj_frame << "\t" << std::endl;
      //      ofs.close();
    }
    else
    {
      std::cout << "Error opening file for time logging " << std::endl;
    }

    // start new
    total_ = total;
    tracking_ = tracking;
    obj_tracking_ = obj_tracking;
    segmentation_ = segmentation;
    integration_ = integration;
    raycasting_ = raycasting;
    computation_time_ = computation;
    same_obj_frame = 1;
    init_time_ = 0;
    time_obj_num_ = obj_num;
    mov_obj_num_ = mov_obj_num;
    in_view_obj_num_ = in_view_obj_num;
  }
  // ofs.close();
}

// For okvis
bool Kfusion::adaptPose(Matrix4 p, float3 &default_pos, uint tracking_rate, uint frame)
{
  if (!config.okvis_mapKeyFrameOnly && frame % tracking_rate != 0)
    return false;

  default_pos = default_pos;
  lastPose = this->camera_pose;

  if (!poses.empty())
  {
    this->camera_pose = poses[frame];
    this->camera_pose.data[0].w = (this->camera_pose.data[0].w - poses[0].data[0].w) + this->_init_camera_Pose.x;
    this->camera_pose.data[1].w = (this->camera_pose.data[1].w - poses[0].data[1].w) + this->_init_camera_Pose.y;
    this->camera_pose.data[2].w = (this->camera_pose.data[2].w - poses[0].data[2].w) + this->_init_camera_Pose.z;
  }
  else
  {
    this->camera_pose = p;
    this->camera_pose.data[0].w = (this->camera_pose.data[0].w - poses[0].data[0].w) + this->_init_camera_Pose.x;
    this->camera_pose.data[1].w = (this->camera_pose.data[1].w - poses[0].data[0].w) + this->_init_camera_Pose.y;
    this->camera_pose.data[2].w = (this->camera_pose.data[2].w - poses[0].data[0].w) + this->_init_camera_Pose.z;
  }
  // std::cout << this->pose <<std::endl;
  return true;
}

void synchroniseDevices()
{
  // Nothing to do in the C++ implementation
}

/// \brief Tracking for OKVIS pose
bool Kfusion::okvisTracking(float4 k, uint tracking_rate, uint frame, Matrix4 p)
{

  if (frame % tracking_rate != 0)
    return false;

  bool camera_tracked;

  T_w_r = camera_pose; // get old pose T_w_(c-1)

  // camera tracking against all pixels&vertices(excluding human)
  if (human_out_.pair_instance_seg_.find(INVALID) == human_out_.pair_instance_seg_.end())
  {
    rgbdTracker_->set_params_frame(k, g_inputGrey, floatDepth);
  }
  else
  {
    if (in_debug_)
    {
      human_out_.output(frame, "human_out_for_tracking");
    }
    rgbdTracker_->set_params_frame(k, g_inputGrey, floatDepth, human_out_.pair_instance_seg_.at(INVALID).instance_mask_);
  }

  if (!poses.empty())
  {
    printMatrix4("Camera Pose", p);
    // printMatrix4("2", poses[frame]);
    this->camera_pose = p;
    // this->camera_pose = poses[frame];
    this->camera_pose.data[0].w = (this->camera_pose.data[0].w - poses[0].data[0].w) + this->_init_camera_Pose.x;
    this->camera_pose.data[1].w = (this->camera_pose.data[1].w - poses[0].data[1].w) + this->_init_camera_Pose.y;
    this->camera_pose.data[2].w = (this->camera_pose.data[2].w - poses[0].data[2].w) + this->_init_camera_Pose.z;
    printMatrix4("use ground truth pose", camera_pose);
    camera_tracked = true;
  }
  else
  {
    if (frame == segment_startFrame_)
    {
      // put reference frame information to current frame memory
      rgbdTracker_->setRefImgFromCurr();
      return false;
    }
    // track against all objects except people
    // estimate static/dynamic objects => only track existing&dynamic objects
    camera_tracked = rgbdTracker_->trackLiveFrame(camera_pose,
                                                  T_w_r, k,
                                                  vertex, normal);

    objectlist_[0]->virtual_camera_pose_ = camera_pose;

    // for later motion segmentation
    memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
           sizeof(TrackData) * computationSize.x * computationSize.y * 2);
  }

  // use ICP&RGB residual to evaluate moving (possibility) for each object

  // raycast again to obtain rendered mask and vertices under CURRENT camera
  // pose and LAST object poses.=>for segmentation and motion residuals
  move_obj_set.clear();
  if (raycast_mask_.pair_instance_seg_.size() > 1)
  {
    rendered_mask_.reset();
    objects_in_view_.clear();
    raycastObjectList(objectlist_, vertex_before_fusion, normal_before_fusion,
                      rendered_mask_.labelImg, objects_in_view_, camera_pose,
                      k, computationSize, nearPlane, farPlane, config.mu, false);

    split_labels(rendered_mask_, objects_in_view_, objectlist_);
    rendered_mask_.set_render_label(true);
    if (in_debug_)
    {
      rendered_mask_.output(frame, "render_initial_track");
    }

    // calculate ICP and RGB residuals on all vertices in current camera pose
    rgbdTracker_->compute_residuals(camera_pose, T_w_r,
                                    vertex, normal,
                                    vertex_before_fusion,
                                    use_icp_tracking_, use_rgb_tracking_,
                                    residual_threshold_);

    // check mask with tracking result
    // find moving objects
    for (auto object_render = rendered_mask_.pair_instance_seg_.begin();
         object_render != rendered_mask_.pair_instance_seg_.end(); ++object_render)
    {
      int obj_id = object_render->first;
      // skip background
      if (obj_id == 0)
        continue;

      const cv::Mat &object_mask = object_render->second.instance_mask_;
      cv::Mat object_inlier_mask = object_mask.clone();
      check_static_state(object_inlier_mask, object_mask,
                         rgbdTracker_->getTrackingResult(), computationSize,
                         use_icp_tracking_, use_rgb_tracking_);
      float inlier_ratio = static_cast<float>(cv::countNonZero(object_inlier_mask)) / (cv::countNonZero(object_mask));
      // connect objectlist with raycastmask
      if ((inlier_ratio < obj_moved_threshold_) &&
          (inlier_ratio > obj_move_out_threshold_) &&
          (Object::static_object.find(object_render->second.class_id_) == Object::static_object.end()))
      {
        objectlist_.at(obj_id)->set_static_state(false);
        move_obj_set.insert(obj_id);
        //        std::cout<<"object "<<obj_id<<" moving threshold: "<<inlier_ratio
        //                 <<" class id: "<<objectlist_.at(obj_id)->class_id_<<std::endl;
      }
      else
      {
        objectlist_.at(obj_id)->set_static_state(true);
        cv::imshow("object", object_mask);
        if (frame > 400)
          cv::waitKey(0);
        //        std::cout<<"object "<<obj_id<<" static threshold: "<<inlier_ratio
        //                 <<" class id: "<<objectlist_.at(obj_id)->class_id_<<std::endl;
      }

      if (in_debug_)
      {
        std::ostringstream name;
        name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
             << "inlier"
             << "_frame_" << frame << "_object_id_" << obj_id << ".png";
        cv::imwrite(name.str(), object_inlier_mask);

        std::ostringstream name2;
        name2 << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build"
                 "-debug/debug/"
              << "all"
              << "_frame_" << frame << "_object_id_" << obj_id << ".png";
        cv::imwrite(name2.str(), object_mask);
      }
    }

    // refine camera motion using all static vertice, if there is moving object
    if (move_obj_set.size() != 0 && rendered_mask_.pair_instance_seg_.size() != 0)
    {
      // build dynamic object mask
      cv::Size imgSize = raycast_mask_.pair_instance_seg_.begin()->second.instance_mask_.size();
      cv::Mat dynamic_obj_mask = cv::Mat::zeros(imgSize, CV_8UC1);
      for (auto object_raycast = raycast_mask_.pair_instance_seg_.begin();
           object_raycast != raycast_mask_.pair_instance_seg_.end(); ++object_raycast)
      {
        int obj_id = object_raycast->first;
        if (move_obj_set.find(obj_id) != move_obj_set.end())
        {
          cv::bitwise_or(object_raycast->second.instance_mask_,
                         dynamic_obj_mask, dynamic_obj_mask);
        }
      }
      if (in_debug_)
      {
        std::ostringstream name;
        name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
             << "dynamic_object"
             << "_frame_" << frame << ".png";
        cv::imwrite(name.str(), dynamic_obj_mask);
      }

      // rgbdTracker_->enable_RGB_tracker(false);
      if (human_out_.pair_instance_seg_.find(INVALID) != human_out_.pair_instance_seg_.end())
      {
        cv::bitwise_or(dynamic_obj_mask, human_out_.pair_instance_seg_.at(INVALID).instance_mask_, dynamic_obj_mask);
      }

      camera_tracked = rgbdTracker_->trackLiveFrame(camera_pose,
                                                    T_w_r, k,
                                                    vertex,
                                                    normal, dynamic_obj_mask);
      // rgbdTracker_->enable_RGB_tracker(true);
      objectlist_[0]->virtual_camera_pose_ = camera_pose;
      // for later motion segmentation
      memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
             sizeof(TrackData) * computationSize.x * computationSize.y * 2);
    }

    // refine poses of dynamic objects
    ///  multiple objects tracking
    for (auto exist_object = objectlist_.begin() + 1;
         exist_object != objectlist_.end(); ++exist_object)
    {
      if ((*exist_object)->is_static())
      {
        // for later motion segmentation
        memcpy((*exist_object)->trackresult_, objectlist_.at(0)->trackresult_,
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
        continue;
      }
      else
      {
        rgbdTracker_->trackEachObject(*exist_object, k, camera_pose, T_w_r,
                                      g_inputGrey, floatDepth);
        //        (*exist_object)->set_static_state(true);

        memcpy((*exist_object)->trackresult_, rgbdTracker_->getTrackingResult(),
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
      }
    }
  }

  //
  ////perform motion refinement accordingly
  if (!move_obj_set.empty())
  {
    // calculate ICP and RGB residuals on all vertices in current camera pose
    rgbdTracker_->compute_residuals(camera_pose, T_w_r,
                                    vertex, normal,
                                    vertex_before_fusion,
                                    use_icp_tracking_, use_rgb_tracking_,
                                    residual_threshold_);
    // for later motion segmentation
    memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
           sizeof(TrackData) * computationSize.x * computationSize.y * 2);

    // raycast again to obtain rendered mask and vertices under current camera
    //  pose and object poses.=>for segmentation and motion residuals
    rendered_mask_.reset();
    raycastObjectList(objectlist_, vertex_before_fusion, normal_before_fusion,
                      rendered_mask_.labelImg, objects_in_view_, camera_pose,
                      k, computationSize, nearPlane, farPlane, config.mu, false);

    // find the objects on this frame =>only (track and) integrate them
    split_labels(rendered_mask_, objectlist_);
    rendered_mask_.set_render_label(true);
    if (in_debug_)
    {
      rendered_mask_.output(frame, "volume2mask_new");
    }
  }

  if (((rendered_mask_.pair_instance_seg_.size() > 1) && (use_rgb_tracking_)) ||
      (!move_obj_set.empty()))
  {

    for (auto exist_object = objectlist_.begin() + 1;
         exist_object != objectlist_.end(); ++exist_object)
    {
      if ((*exist_object)->is_static())
      {
        //        for later motion segmentation
        memcpy((*exist_object)->trackresult_, objectlist_.at(0)->trackresult_,
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
        continue;
      }
      else
      {
        rgbdTracker_->compute_residuals((*exist_object)->virtual_camera_pose_, T_w_r,
                                        (*exist_object)->m_vertex, (*exist_object)->m_normal,
                                        (*exist_object)->m_vertex_bef_integ,
                                        use_icp_tracking_, use_rgb_tracking_,
                                        residual_threshold_);

        memcpy((*exist_object)->trackresult_, rgbdTracker_->getTrackingResult(),
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
      }
    }
  }

  // put reference frame information to current frame memory
  rgbdTracker_->setRefImgFromCurr();

  // if (in_debug_){
  //   printMatrix4("tracking: camera pose", camera_pose);
  //
  //
  //   for (auto object = objectlist_.begin(); object != objectlist_.end();
  //        ++object){
  //     int class_id = (*object)->class_id_;
  //     std::cout<<"tracking: object id: "<<(*object)->instance_label_
  //              <<" ,class id: "<<class_id<<std::endl;
  //     printMatrix4("tracking: object pose ", (*object)->volume_pose_);
  //
  //   }
  // }

  return camera_tracked;
}

/// \brief Tracking for OKVIS pose 2
bool Kfusion::okvisTrackingTwo(float4 k, uint tracking_rate, uint frame, Matrix4 p)
{

  if (frame % tracking_rate != 0)
    return false;

  bool camera_tracked;

  T_w_r = camera_pose; // get old pose T_w_(c-1)

  // camera tracking against all pixels&vertices(excluding human)
  if (human_out_.pair_instance_seg_.find(INVALID) == human_out_.pair_instance_seg_.end())
  {
    rgbdTracker_->set_params_frame(k, g_inputGrey, floatDepth);
  }
  else
  {
    if (in_debug_)
    {
      human_out_.output(frame, "human_out_for_tracking");
    }
    rgbdTracker_->set_params_frame(k, g_inputGrey, floatDepth, human_out_.pair_instance_seg_.at(INVALID).instance_mask_);
  }

  // use OKVIS as tracking result
  if (!poses.empty())
  {
    this->camera_pose = p;
    this->camera_pose.data[0].w = (this->camera_pose.data[0].w - poses[0].data[0].w) + this->_init_camera_Pose.x;
    this->camera_pose.data[1].w = (this->camera_pose.data[1].w - poses[0].data[1].w) + this->_init_camera_Pose.y;
    this->camera_pose.data[2].w = (this->camera_pose.data[2].w - poses[0].data[2].w) + this->_init_camera_Pose.z;
    printMatrix4("use ground truth pose", camera_pose);
    camera_tracked = true;
  }
  else
  {
    this->camera_pose = p;
    this->camera_pose.data[0].w = this->camera_pose.data[0].w + this->_init_camera_Pose.x;
    this->camera_pose.data[1].w = this->camera_pose.data[1].w + this->_init_camera_Pose.y;
    this->camera_pose.data[2].w = this->camera_pose.data[2].w + this->_init_camera_Pose.z;
    printMatrix4("use OKVIS pose", camera_pose);
    camera_tracked = true;
  }

  // use ICP&RGB residual to evaluate moving (possibility) for each object

  // raycast again to obtain rendered mask and vertices under CURRENT camera
  // pose and LAST object poses.=>for segmentation and motion residuals
  move_obj_set.clear();
  if (raycast_mask_.pair_instance_seg_.size() > 1)
  {
    rendered_mask_.reset();
    objects_in_view_.clear();
    raycastObjectList(objectlist_, vertex_before_fusion, normal_before_fusion,
                      rendered_mask_.labelImg, objects_in_view_, camera_pose,
                      k, computationSize, nearPlane, farPlane, config.mu, false);

    split_labels(rendered_mask_, objects_in_view_, objectlist_);
    rendered_mask_.set_render_label(true);
    if (in_debug_)
    {
      rendered_mask_.output(frame, "render_initial_track");
    }

    // calculate ICP and RGB residuals on all vertices in current camera pose
    rgbdTracker_->compute_residuals(camera_pose, T_w_r,
                                    vertex, normal,
                                    vertex_before_fusion,
                                    use_icp_tracking_, use_rgb_tracking_,
                                    residual_threshold_);

    // check mask with tracking result
    // find moving objects
    for (auto object_render = rendered_mask_.pair_instance_seg_.begin();
         object_render != rendered_mask_.pair_instance_seg_.end(); ++object_render)
    {
      int obj_id = object_render->first;
      // skip background
      if (obj_id == 0)
        continue;

      const cv::Mat &object_mask = object_render->second.instance_mask_;
      cv::Mat object_inlier_mask = object_mask.clone();
      check_static_state(object_inlier_mask, object_mask,
                         rgbdTracker_->getTrackingResult(), computationSize,
                         use_icp_tracking_, use_rgb_tracking_);
      float inlier_ratio = static_cast<float>(cv::countNonZero(object_inlier_mask)) / (cv::countNonZero(object_mask));
      // connect objectlist with raycastmask
      if ((inlier_ratio < obj_moved_threshold_) &&
          (inlier_ratio > obj_move_out_threshold_) &&
          (Object::static_object.find(object_render->second.class_id_) == Object::static_object.end()))
      {
        objectlist_.at(obj_id)->set_static_state(false);
        move_obj_set.insert(obj_id);
        //        std::cout<<"object "<<obj_id<<" moving threshold: "<<inlier_ratio
        //                 <<" class id: "<<objectlist_.at(obj_id)->class_id_<<std::endl;
      }
      else
      {
        objectlist_.at(obj_id)->set_static_state(true);
        //        std::cout<<"object "<<obj_id<<" static threshold: "<<inlier_ratio
        //                 <<" class id: "<<objectlist_.at(obj_id)->class_id_<<std::endl;
      }

      if (in_debug_)
      {
        std::ostringstream name;
        name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
             << "inlier"
             << "_frame_" << frame << "_object_id_" << obj_id << ".png";
        cv::imwrite(name.str(), object_inlier_mask);

        std::ostringstream name2;
        name2 << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build"
                 "-debug/debug/"
              << "all"
              << "_frame_" << frame << "_object_id_" << obj_id << ".png";
        cv::imwrite(name2.str(), object_mask);
      }
    }

    // refine camera motion using all static vertice, if there is moving object
    if (move_obj_set.size() != 0 && rendered_mask_.pair_instance_seg_.size() != 0)
    {
      // build dynamic object mask
      cv::Size imgSize = raycast_mask_.pair_instance_seg_.begin()->second.instance_mask_.size();
      cv::Mat dynamic_obj_mask = cv::Mat::zeros(imgSize, CV_8UC1);
      for (auto object_raycast = raycast_mask_.pair_instance_seg_.begin();
           object_raycast != raycast_mask_.pair_instance_seg_.end(); ++object_raycast)
      {
        int obj_id = object_raycast->first;
        if (move_obj_set.find(obj_id) != move_obj_set.end())
        {
          cv::bitwise_or(object_raycast->second.instance_mask_,
                         dynamic_obj_mask, dynamic_obj_mask);
        }
      }
      if (in_debug_)
      {
        std::ostringstream name;
        name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
             << "dynamic_object"
             << "_frame_" << frame << ".png";
        cv::imwrite(name.str(), dynamic_obj_mask);
      }

      // rgbdTracker_->enable_RGB_tracker(false);
      if (human_out_.pair_instance_seg_.find(INVALID) != human_out_.pair_instance_seg_.end())
      {
        cv::bitwise_or(dynamic_obj_mask, human_out_.pair_instance_seg_.at(INVALID).instance_mask_, dynamic_obj_mask);
      }

      camera_tracked = rgbdTracker_->trackLiveFrame(camera_pose,
                                                    T_w_r, k,
                                                    vertex,
                                                    normal, dynamic_obj_mask);
      // rgbdTracker_->enable_RGB_tracker(true);
      objectlist_[0]->virtual_camera_pose_ = camera_pose;
      // for later motion segmentation
      memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
             sizeof(TrackData) * computationSize.x * computationSize.y * 2);
    }

    // refine poses of dynamic objects
    ///  multiple objects tracking
    for (auto exist_object = objectlist_.begin() + 1;
         exist_object != objectlist_.end(); ++exist_object)
    {
      if ((*exist_object)->is_static())
      {
        // for later motion segmentation
        memcpy((*exist_object)->trackresult_, objectlist_.at(0)->trackresult_,
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
        continue;
      }
      else
      {
        rgbdTracker_->trackEachObject(*exist_object, k, camera_pose, T_w_r,
                                      g_inputGrey, floatDepth);
        //        (*exist_object)->set_static_state(true);

        memcpy((*exist_object)->trackresult_, rgbdTracker_->getTrackingResult(),
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
      }
    }
  }

  //
  ////perform motion refinement accordingly
  if (!move_obj_set.empty())
  {
    // calculate ICP and RGB residuals on all vertices in current camera pose
    rgbdTracker_->compute_residuals(camera_pose, T_w_r,
                                    vertex, normal,
                                    vertex_before_fusion,
                                    use_icp_tracking_, use_rgb_tracking_,
                                    residual_threshold_);
    // for later motion segmentation
    memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
           sizeof(TrackData) * computationSize.x * computationSize.y * 2);

    // raycast again to obtain rendered mask and vertices under current camera
    //  pose and object poses.=>for segmentation and motion residuals
    rendered_mask_.reset();
    raycastObjectList(objectlist_, vertex_before_fusion, normal_before_fusion,
                      rendered_mask_.labelImg, objects_in_view_, camera_pose,
                      k, computationSize, nearPlane, farPlane, config.mu, false);

    // find the objects on this frame =>only (track and) integrate them
    split_labels(rendered_mask_, objectlist_);
    rendered_mask_.set_render_label(true);
    if (in_debug_)
    {
      rendered_mask_.output(frame, "volume2mask_new");
    }
  }

  if (((rendered_mask_.pair_instance_seg_.size() > 1) && (use_rgb_tracking_)) ||
      (!move_obj_set.empty()))
  {

    for (auto exist_object = objectlist_.begin() + 1;
         exist_object != objectlist_.end(); ++exist_object)
    {
      if ((*exist_object)->is_static())
      {
        //        for later motion segmentation
        memcpy((*exist_object)->trackresult_, objectlist_.at(0)->trackresult_,
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
        continue;
      }
      else
      {
        rgbdTracker_->compute_residuals((*exist_object)->virtual_camera_pose_, T_w_r,
                                        (*exist_object)->m_vertex, (*exist_object)->m_normal,
                                        (*exist_object)->m_vertex_bef_integ,
                                        use_icp_tracking_, use_rgb_tracking_,
                                        residual_threshold_);

        memcpy((*exist_object)->trackresult_, rgbdTracker_->getTrackingResult(),
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
      }
    }
  }

  // put reference frame information to current frame memory
  rgbdTracker_->setRefImgFromCurr();

  if (in_debug_)
  {
    printMatrix4("tracking: camera pose", camera_pose);

    for (auto object = objectlist_.begin(); object != objectlist_.end();
         ++object)
    {
      int class_id = (*object)->class_id_;
      std::cout << "tracking: object id: " << (*object)->instance_label_
                << " ,class id: " << class_id << std::endl;
      printMatrix4("tracking: object pose ", (*object)->volume_pose_);
    }
  }

  return camera_tracked;
}

/// \brief Tracking for VIMID pose
bool Kfusion::vimidTracking(float4 k,
                            uint tracking_rate,
                            uint frame,
                            Matrix4 p,
                            okvis::VioParameters &params,
                            okvis::ImuMeasurementDeque &imuData,
                            okvis::Time &prevTime,
                            okvis::Time &currTime,
                            bool use_imu,
                            double *timings)
{
  if (frame % tracking_rate != 0)
    return false;
  bool camera_tracked;

  T_w_r = camera_pose; // get old pose T_w_(c-1)

  // Save states used for re-optimization
  camera_pose_ori = camera_pose;
  T_w_r_ori = T_w_r;
  state_ori = state_;
  vertex_ori = *vertex;
  normal_ori = *normal;
  vertex_before_fusion_ori = *vertex_before_fusion;
  normal_before_fusion_ori = *normal_before_fusion;

  // Get pure object masks
  cv::Size imgSize = human_out_.pair_instance_seg_.begin()->second.instance_mask_.size();
  cv::Mat obj_mask = cv::Mat::zeros(imgSize, CV_8UC1);
  unsigned long long mask_num = 0;
  rgbdTracker_->set_obj_mask(human_out_, obj_mask, mask_num);

  // Camera tracking against all pixels & vertices (excluding human)
  if (human_out_.pair_instance_seg_.find(INVALID) == human_out_.pair_instance_seg_.end())
  {
    rgbdTracker_->set_params_frame_new(k, g_inputGrey, floatDepth, obj_mask);
  }
  else
  {
    if (in_debug_)
    {
      human_out_.output(frame, "human_out_for_tracking");
    }
    rgbdTracker_->set_params_frame(k, g_inputGrey, floatDepth,
                                   human_out_.pair_instance_seg_.at(INVALID).instance_mask_, obj_mask);
  }

  if (!poses.empty())
  {
    printMatrix4("Camera Pose", p);
    this->camera_pose = p;
    this->camera_pose = poses[frame];
    printMatrix4("use ground truth pose", camera_pose);
    this->camera_pose.data[0].w = (this->camera_pose.data[0].w - poses[0].data[0].w) + this->_init_camera_Pose.x;
    this->camera_pose.data[1].w = (this->camera_pose.data[1].w - poses[0].data[1].w) + this->_init_camera_Pose.y;
    this->camera_pose.data[2].w = (this->camera_pose.data[2].w - poses[0].data[2].w) + this->_init_camera_Pose.z;
    camera_tracked = true;
  }
  else
  {
    if (frame == segment_startFrame_)
    {
      // put reference frame information to current frame memory
      rgbdTracker_->setRefImgFromCurr();

      std::cout << "First run..." << std::endl;
      std::cout << "Size: " << imuData.size() << std::endl;
      currPose = Eigen::Matrix4d::Identity();

      if (use_imu)
      {
        // Initialize the pose from IMU
        okvis::kinematics::Transformation init_pose;
        okvis::Estimator::initPoseFromImu(imuData, init_pose);
        currPose.block<3, 3>(0, 0) = init_pose.C();
      }

      currPose.block<3, 1>(0, 3) = Eigen::Vector3d(this->_init_camera_Pose.x, this->_init_camera_Pose.y, this->_init_camera_Pose.z);
      rgbdTracker_->setLinearizationPoint(currPose);
      camera_pose = fromOkvisToMidFusion(okvis::kinematics::Transformation(currPose.eval()));
      lastPose = camera_pose;
      init_camera_pose = camera_pose;
      Eigen::Vector3d trans = currPose.topRightCorner(3, 1);
      Eigen::Matrix3d rot = currPose.topLeftCorner(3, 3);
      state_.W_r_WS_ = trans;
      state_.C_WS_ = rot;
      std::cout << "Current pose: " << currPose << std::endl;
      std::cout << "End first run." << std::endl;

      return false;
    }

    // Set sparse tracking result to camera_pose
    camera_tracked = rgbdTracker_->trackLiveFrameWithImu(camera_pose,
                                                         T_w_r,
                                                         this->preTrackingPose,
                                                         state_, k,
                                                         vertex, normal,
                                                         obj_mask,
                                                         false,
                                                         imuData,
                                                         prevTime,
                                                         currTime,
                                                         params.imu,
                                                         use_imu);

    printMatrix4("Current camera pose", camera_pose);

    objectlist_[0]->virtual_camera_pose_ = camera_pose;

    // for later motion segmentation
    memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
           sizeof(TrackData) * computationSize.x * computationSize.y * 2);
  }

  // use ICP&RGB residual to evaluate moving (possibility) for each object

  // raycast again to obtain rendered mask and vertices under CURRENT camera
  // pose and LAST object poses.=>for segmentation and motion residuals
  move_obj_set.clear();
  if (raycast_mask_.pair_instance_seg_.size() > 1)
  {
    rendered_mask_.reset();
    objects_in_view_.clear();
    raycastObjectList(objectlist_, vertex_before_fusion, normal_before_fusion,
                      rendered_mask_.labelImg, objects_in_view_, camera_pose,
                      k, computationSize, nearPlane, farPlane, config.mu, false);
    split_labels(rendered_mask_, objects_in_view_, objectlist_);

    // If rendered masks are more than Mask R-CNN masks, re-estimate the camera pose,
    // using bg_masks added by the extra rendered masks
    if (rendered_mask_.pair_instance_seg_.size() - 1 > mask_num)
    {
      // std::cout << "Rendered size: " << rendered_mask_.pair_instance_seg_.size()-1
      //           << " Mask R-CNN size: " << mask_num << std::endl
      //           << "Start to re-estimate the camera pose." << std::endl;

      // Use the original pose and state
      camera_pose = camera_pose_ori;
      T_w_r = T_w_r_ori;
      state_ = state_ori;
      (*vertex) = vertex_ori;
      (*normal) = normal_ori;
      (*vertex_before_fusion) = vertex_before_fusion_ori;
      (*normal_before_fusion) = normal_before_fusion_ori;
      rgbdTracker_->maskClear();

      // Refine obj_mask using rendered masks
      for (auto objPtr = rendered_mask_.pair_instance_seg_.begin();
           objPtr != rendered_mask_.pair_instance_seg_.end(); ++objPtr)
      {
        if (objPtr->second.class_id_ == 0)
          continue;
        cv::bitwise_or(objPtr->second.instance_mask_, obj_mask, obj_mask);
      }

      if (human_out_.pair_instance_seg_.find(INVALID) == human_out_.pair_instance_seg_.end())
      {
        rgbdTracker_->set_params_frame_new(k, g_inputGrey, floatDepth, obj_mask);
      }
      else
      {
        rgbdTracker_->set_params_frame(k, g_inputGrey, floatDepth,
                                       human_out_.pair_instance_seg_.at(INVALID).instance_mask_, obj_mask);
      }

      camera_tracked = rgbdTracker_->trackLiveFrameWithImu(camera_pose,
                                                           T_w_r,
                                                           this->preTrackingPose,
                                                           state_, k,
                                                           vertex, normal,
                                                           obj_mask,
                                                           true,
                                                           imuData,
                                                           prevTime,
                                                           currTime,
                                                           params.imu,
                                                           use_imu);

      objectlist_[0]->virtual_camera_pose_ = camera_pose;

      // for later motion segmentation
      memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
             sizeof(TrackData) * computationSize.x * computationSize.y * 2);

      move_obj_set.clear();
      if (raycast_mask_.pair_instance_seg_.size() > 1)
      {
        rendered_mask_.reset();
        objects_in_view_.clear();
        raycastObjectList(objectlist_, vertex_before_fusion, normal_before_fusion,
                          rendered_mask_.labelImg, objects_in_view_, camera_pose,
                          k, computationSize, nearPlane, farPlane, config.mu, false);
        split_labels(rendered_mask_, objects_in_view_, objectlist_);
      }
    }
    timings[4] = tock();

    rendered_mask_.set_render_label(true);

    if (in_debug_)
    {
      rendered_mask_.output(frame, "render_initial_track");
    }

    // Calculate ICP and RGB residuals on all vertices in current camera pose
    rgbdTracker_->compute_residuals(camera_pose, T_w_r,
                                    vertex, normal,
                                    vertex_before_fusion,
                                    use_icp_tracking_, use_rgb_tracking_,
                                    residual_threshold_);

    // Check mask with tracking result
    // Find moving objects
    for (auto object_render = rendered_mask_.pair_instance_seg_.begin();
         object_render != rendered_mask_.pair_instance_seg_.end(); ++object_render)
    {
      int obj_id = object_render->first;

      // Skip background
      if (obj_id == 0)
        continue;

      const cv::Mat &object_mask = object_render->second.instance_mask_;
      cv::Mat object_inlier_mask = object_mask.clone();
      check_static_state(object_inlier_mask, object_mask,
                         rgbdTracker_->getTrackingResult(), computationSize,
                         use_icp_tracking_, use_rgb_tracking_);

      int obj_mask_num = cv::countNonZero(object_mask);
      bool isSmall = (obj_mask_num < 500);
      bool isBig = (obj_mask_num > 2000);
      objectlist_.at(obj_id)->is_small = isSmall;
      objectlist_.at(obj_id)->is_big = isBig;
      objectlist_.at(obj_id)->current_size = obj_mask_num;
      if (obj_mask_num > objectlist_.at(obj_id)->max_size)
      {
        objectlist_.at(obj_id)->max_size = obj_mask_num;
      }

      float inlier_ratio = static_cast<float>(cv::countNonZero(object_inlier_mask)) / obj_mask_num;

      // If the mask is approach boundary, set approach_boundary to true
      bool approach = check_truncated(object_mask, computationSize);
      if (approach)
      {
        objectlist_.at(obj_id)->approach_boundary = true;
      }
      else
      {
        objectlist_.at(obj_id)->approach_boundary = false;
      }

      // Update bounding volume and check whether it's truncated
      Matrix4 T_wo = objectlist_.at(obj_id)->volume_pose_;
      Eigen::Matrix4f T_oc = fromMidFusionToEigen(inverse(T_wo) * camera_pose);
      objectlist_.at(obj_id)->bounding_volume.merge(rgbdTracker_->get_live_vertex(),
                                                    T_oc,
                                                    object_mask.clone());
      cv::Mat bb_pic = currResizeRGB_.clone();
      objectlist_.at(obj_id)->bounding_volume.draw(bb_pic, computationSize, T_oc, k, 0.0);

      // Connect object list with raycast mask
      if ((inlier_ratio < obj_moved_threshold_) &&
          (inlier_ratio > obj_move_out_threshold_) &&
          !isSmall &&
          !approach &&
          (Object::static_object.find(object_render->second.class_id_) == Object::static_object.end()))
      {

        // if truncated large, set the object to static
        bool is_truncated = objectlist_.at(obj_id)->bounding_volume.isTruncated(T_oc, k, computationSize);
        is_truncated = false;
        if (is_truncated)
        {
          objectlist_.at(obj_id)->set_static_state(true);
        }
        else
        {
          objectlist_.at(obj_id)->set_static_state(false);
          move_obj_set.insert(obj_id);
        }
      }
      else
      {
        objectlist_.at(obj_id)->set_static_state(true);
        // std::cout << "Static object " << obj_id << "  inlier ratio: " << inlier_ratio
        //           << ", class id: " << objectlist_.at(obj_id)->class_id_ << std::endl;
      }

      cv::Mat fg_outlier_mask;
      cv::Mat temp = object_mask.clone();
      for (auto objPtr = human_out_.pair_instance_seg_.begin();
           objPtr != human_out_.pair_instance_seg_.end(); objPtr++)
      {
        if (objPtr->first == 0 || objPtr->first == INVALID)
          continue;
        cv::Mat or_mask, and_mask;
        cv::bitwise_or(objPtr->second.instance_mask_, object_mask, or_mask);
        cv::bitwise_and(objPtr->second.instance_mask_, object_mask, and_mask);
        float iou = static_cast<float>(cv::countNonZero(and_mask)) / cv::countNonZero(or_mask);
        if (iou > 0.5)
        {
          cv::bitwise_or(objPtr->second.instance_mask_, object_mask, temp);
          break;
        }
      }
      cv::bitwise_not(temp, fg_outlier_mask);
      objectlist_.at(obj_id)->fg_outlier_mask_ = fg_outlier_mask;

      if (in_debug_)
      {
        std::ostringstream name;
        name << "/home/ryf/slam/vimid/apps/kfusion/cmake-build-debug/debug/"
             << "inlier/"
             << "inlier_frame_" << frame << "_obj_id_" << obj_id << ".png";
        cv::imwrite(name.str(), object_inlier_mask);

        std::ostringstream name2;
        name2 << "/home/ryf/slam/vimid/apps/kfusion/cmake-build-debug/debug/inlier/"
              << "all"
              << "_frame_" << frame << "_obj_id_" << obj_id << ".png";
        cv::imwrite(name2.str(), object_mask);
      }
    }

    // Refine poses of dynamic objects
    // Multiple objects tracking
    timings[5] = tock();
    for (auto exist_object = objectlist_.begin() + 1; exist_object != objectlist_.end(); ++exist_object)
    {

      std::cout << "**********************" << std::endl;
      std::cout << "Object ID: " << (*exist_object)->instance_label_ << std::endl;
      // std::cout << "Object Class: " << (*exist_object)->class_id_ << std::endl;
      // std::cout << "Object Approach: " << (*exist_object)->approach_boundary << std::endl;
      // std::cout << "Object Out: " << (*exist_object)->out_boundary << std::endl;

      float in_view_ratio = static_cast<float>((*exist_object)->current_size) / (*exist_object)->max_size;

      const bool do_reloc = true;
      if (do_reloc)
      {
        // 400, 50
        if ((*exist_object)->approach_boundary && (*exist_object)->out_boundary)
        {
          (*exist_object)->lost = true;
          (*exist_object)->out_boundary = false;
        }
        if (/*!(*exist_object)->approach_boundary &&*/ (*exist_object)->lost &&
            in_view_ratio > 0.7)
        {
          // if the object's in_view_ratio is smaller than 0.7, do not do relocalisation
          (*exist_object)->relocalize = true;
        }
        if ((*exist_object)->relocalize)
        {
          Matrix4 volume_pose_before = (*exist_object)->volume_pose_;
          Matrix4 virtual_camera_before = (*exist_object)->virtual_camera_pose_;
          bool reloc_success = (*exist_object)->relocalisation(currResizeRGB_.clone(), camera_pose, rgbdTracker_->get_no_outlier_mask(), rgbdTracker_->get_l_vertex()[0], rgbdTracker_->get_l_depth(), rgbdTracker_->getImgSize(), k, frame);
          if (!reloc_success)
            continue;

          Matrix4 temp_volume_pose = (*exist_object)->volume_pose_;
          Matrix4 temp_virtual_camera = (*exist_object)->virtual_camera_pose_;

          // render again
          SegmentationResult rendered_mask(rendered_mask_.width_, rendered_mask_.height_);
          std::set<int> objects_in_view;
          raycastObjectList(objectlist_, vertex_before_fusion, normal_before_fusion,
                            rendered_mask.labelImg, objects_in_view, camera_pose,
                            k, computationSize, nearPlane, farPlane, config.mu, false);
          split_labels(rendered_mask, objects_in_view, objectlist_);

          (*exist_object)->prev_fg_outlier_mask_ = rgbdTracker_->get_no_outlier_mask()[0].clone();
          if (rendered_mask.pair_instance_seg_.find((*exist_object)->instance_label_) != rendered_mask.pair_instance_seg_.end())
          {
            (*exist_object)->fg_outlier_mask_ = rendered_mask.pair_instance_seg_.at((*exist_object)->instance_label_).instance_mask_.clone();
            cv::imshow("curr", (*exist_object)->fg_outlier_mask_);
          }
          else
          {
            (*exist_object)->fg_outlier_mask_ = rgbdTracker_->get_no_outlier_mask()[0].clone();
          }

          rgbdTracker_->trackEachObjectFg(*exist_object, k, camera_pose,
                                          T_w_r, g_inputGrey, floatDepth);

          float **reductionOutput = rgbdTracker_->getReductionOutput();
          float curr_res = reductionOutput[0][0] / reductionOutput[0][28];
          std::cout << "Current object residual is: " << curr_res << std::endl;
          if (curr_res > 0.6)
          {
            (*exist_object)->volume_pose_ = volume_pose_before;
            (*exist_object)->virtual_camera_pose_ = virtual_camera_before;
            continue;
          }
          else if (curr_res > 0.3)
          {
            (*exist_object)->volume_pose_ = temp_volume_pose;
            (*exist_object)->virtual_camera_pose_ = temp_virtual_camera;
            continue;
          }

          // delete the duplicate
          for (auto object_render = rendered_mask_.pair_instance_seg_.begin();
               object_render != rendered_mask_.pair_instance_seg_.end(); ++object_render)
          {
            int obj_id = object_render->first;

            cv::Mat object_mask = object_render->second.instance_mask_;
            if (obj_id == 0 || obj_id == (*exist_object)->instance_label_)
              continue;

            cv::Mat curr_mask = (*exist_object)->fg_outlier_mask_.clone();
            cv::Mat and_mask, or_mask;
            cv::bitwise_or(curr_mask, object_mask, or_mask);
            cv::bitwise_and(curr_mask, object_mask, and_mask);
            int and_size = cv::countNonZero(and_mask);
            int or_size = cv::countNonZero(or_mask);
            float iou = static_cast<float>(and_size) / static_cast<float>(or_size);

            if (iou > 0.5)
            {
              objectlist_.at(obj_id)->isExist_ = false;
              objectlist_.at(obj_id)->exist_prob_ = 0;

              // cv::imwrite("/home/ryf/Videos/object_relocalisation/paper/relocalisation/obj_reloc.png", curr_mask);
              // cv::imwrite("/home/ryf/Videos/object_relocalisation/paper/relocalisation/obj_new.png", object_mask);
              // cv::waitKey(0);
            }
          }

          rendered_mask_ = rendered_mask;
          rendered_mask_.set_render_label(true);
          objects_in_view_ = objects_in_view;
          (*exist_object)->relocalize = false;
          (*exist_object)->lost = false;
          continue;
          // timings[6] = tock();
        }
        if ((*exist_object)->lost)
          continue;
      }

      if ((*exist_object)->is_static())
      {
        // Object keyframe set
        (*exist_object)->set_keyframe(currResizeRGB_.clone(), camera_pose, rgbdTracker_->get_no_outlier_mask(), rgbdTracker_->get_l_vertex()[0], rgbdTracker_->getImgSize());
      }
      else
      {
        // VI-MID version
        rgbdTracker_->trackEachObjectFg(*exist_object, k, camera_pose,
                                        T_w_r, g_inputGrey, floatDepth);

        // check whether need to do object relocalisation
        float **reductionOutput = rgbdTracker_->getReductionOutput();
        float curr_res = reductionOutput[0][0] / reductionOutput[0][28];
        // Object keyframe set
        if (curr_res <= 0.3 && (in_view_ratio > 0.2))
        {
          (*exist_object)->set_keyframe(currResizeRGB_.clone(), camera_pose, rgbdTracker_->get_no_outlier_mask(), rgbdTracker_->get_l_vertex()[0], rgbdTracker_->getImgSize());
        }
      }
      // for later motion segmentation
      memcpy((*exist_object)->trackresult_, rgbdTracker_->getTrackingResult(),
             sizeof(TrackData) * computationSize.x * computationSize.y * 2);
    }
  }

  // Perform motion refinement accordingly
  if (!move_obj_set.empty())
  {
    // calculate ICP and RGB residuals on all vertices in current camera pose
    rgbdTracker_->compute_residuals(camera_pose, T_w_r,
                                    vertex, normal,
                                    vertex_before_fusion,
                                    use_icp_tracking_, use_rgb_tracking_,
                                    residual_threshold_);
    // for later motion segmentation
    memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
           sizeof(TrackData) * computationSize.x * computationSize.y * 2);

    // raycast again to obtain rendered mask and vertices under current camera
    // pose and object poses.=>for segmentation and motion residuals
    rendered_mask_.reset();
    raycastObjectList(objectlist_, vertex_before_fusion, normal_before_fusion,
                      rendered_mask_.labelImg, objects_in_view_, camera_pose,
                      k, computationSize, nearPlane, farPlane, config.mu, false);

    // find the objects on this frame =>only (track and) integrate them
    split_labels(rendered_mask_, objectlist_);
    rendered_mask_.set_render_label(true);
    if (in_debug_)
    {
      rendered_mask_.output(frame, "volume2mask_new");
    }
  }

  if (((rendered_mask_.pair_instance_seg_.size() > 1) && (use_rgb_tracking_)) ||
      (!move_obj_set.empty()))
  {

    for (auto exist_object = objectlist_.begin() + 1;
         exist_object != objectlist_.end(); ++exist_object)
    {
      if ((*exist_object)->is_static())
      {
        // for later motion segmentation
        memcpy((*exist_object)->trackresult_, rgbdTracker_->getTrackingResult(),
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
        continue;
      }
      else
      {
        rgbdTracker_->compute_residuals((*exist_object)->virtual_camera_pose_, T_w_r,
                                        (*exist_object)->m_vertex, (*exist_object)->m_normal,
                                        (*exist_object)->m_vertex_bef_integ,
                                        use_icp_tracking_, use_rgb_tracking_,
                                        residual_threshold_);

        memcpy((*exist_object)->trackresult_, rgbdTracker_->getTrackingResult(),
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
      }
    }
  }

  // Marginalization
  if (use_imu)
  {
    rgbdTracker_->marginalization();
  }

  // Put reference frame information to current frame memory
  // rgbdTracker_->setRefImgFromCurr(use_imu);
  rgbdTracker_->setRefImgFromCurr();
  // Set for the prev_fg_outlier_mask
  for (auto exist_object = objectlist_.begin() + 1; exist_object != objectlist_.end(); ++exist_object)
  {
    (*exist_object)->prev_fg_outlier_mask_ = (*exist_object)->fg_outlier_mask_;
  }

  // if (in_debug_){
  //   printMatrix4("tracking: camera pose", camera_pose);
  //
  //
  //   for (auto object = objectlist_.begin(); object != objectlist_.end();
  //        ++object){
  //     int class_id = (*object)->class_id_;
  //     std::cout<<"tracking: object id: "<<(*object)->instance_label_
  //              <<" ,class id: "<<class_id<<std::endl;
  //     printMatrix4("tracking: object pose ", (*object)->volume_pose_);
  //
  //   }
  // }
  return camera_tracked;
}

void Kfusion::sparsePreTracking(float4 k,
                                uint tracking_rate,
                                uint frame,
                                cv::Mat prevImg,
                                cv::Mat currImg,
                                cv::Mat prevDepth,
                                cv::Mat currDepth)
{
  return;
  if (frame % tracking_rate != 0)
    return;
  if (!poses.empty())
    return;

  if (frame != segment_startFrame_)
  {
    // Detect and match the key points using ORB features
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(prevImg, currImg, keypoints_1, keypoints_2, matches, true, false);

    // Set camera intrinsics
    cv::Mat K = (cv::Mat_<double>(3, 3) << k.x, 0, k.z, 0, k.y, k.w, 0, 0, 1);

    if (using_2d2d_)
    {
      // Using 2d-2d alignment
      std::vector<cv::Point2f> points_1;
      std::vector<cv::Point2f> points_2;
      for (int i = 0; i < (int)matches.size(); i++)
      {
        points_1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points_2.push_back(keypoints_2[matches[i].trainIdx].pt);
      }

      // cv::Mat fundamental_matrix;
      // fundamental_matrix = cv::findFundamentalMat(points_1, points_2, CV_FM_RANSAC);

      cv::Point2d principle_point(k.z, k.w);
      cv::Mat essential_matrix;
      essential_matrix = cv::findEssentialMat(points_1, points_2, k.x, principle_point);

      // Third: recover the transformation from essential matrix
      cv::Mat R, t;
      cv::recoverPose(essential_matrix, points_1, points_2, R, t, k.x, principle_point);

      // Fourth: initialize the current camera pose
      Matrix4 delta_T;
      for (int i = 0; i < 3; i++)
      {
        delta_T.data[i].x = R.at<double>(i, 0);
        delta_T.data[i].y = R.at<double>(i, 1);
        delta_T.data[i].z = R.at<double>(i, 2);
        delta_T.data[i].w = t.at<double>(i);
      }
      delta_T.data[3].x = 0;
      delta_T.data[3].y = 0;
      delta_T.data[3].z = 0;
      delta_T.data[3].w = 1;

      printMatrix4("delta_T", delta_T);

      this->preTrackingPose = fromMidFusionToOkvis(camera_pose * inverse(delta_T)).T();
    }
    // Using 3d-3d alignment
    else if (using_3d3d_)
    {
      // Create the 3D poins
      std::vector<cv::Point3f> pts1, pts2;
      for (cv::DMatch m : matches)
      {
        ushort d1 = prevDepth.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = currDepth.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0)
          continue;

        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        float dd1 = float(d1) / 5000.0;
        float dd2 = float(d2) / 5000.0;
        pts1.push_back(cv::Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(cv::Point3f(p2.x * dd2, p2.y * dd2, dd2));
      }

      // SVD pose estimation
      Eigen::Matrix3d R;
      Eigen::Vector3d t;
      pose_estimation_3d3d(pts1, pts2, R, t);

      // Initialize the current camera pose
      Matrix4 delta_T;
      for (int i = 0; i < 3; i++)
      {
        delta_T.data[i].x = R(i, 0);
        delta_T.data[i].y = R(i, 1);
        delta_T.data[i].z = R(i, 2);
        delta_T.data[i].w = t(i);
      }
      delta_T.data[3].x = 0;
      delta_T.data[3].y = 0;
      delta_T.data[3].z = 0;
      delta_T.data[3].w = 1;

      this->preTrackingPose = fromMidFusionToOkvis(camera_pose * inverse(delta_T)).T();
    }
  }
}
