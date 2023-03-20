/*
 * SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2023 Binbin Xu
 * SPDX-FileCopyrightText: 2023 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/

#include "object.h"
//#include "sophus/so3.hpp"
#include <brisk/brisk.h>

//Object::Object(const Matrix4& pose, const int id, const uint2 imgSize)
//    : volume_pose_(pose), class_id_(id){
//  volume_.init(volume_resol_.x, volume_size_.x);
//  m_vertex = (float3*) calloc(
//      sizeof(float3) * imgSize.x * imgSize.y, 1);
//  m_normalmal = (float3*) calloc(
//      sizeof(float3) * imgSize.x * imgSize.y, 1);
//};

Object::Object(const int voxel_block_size, const float3& volume_size, const uint3&
volume_resol, const Matrix4& pose, const Matrix4& virtual_T_w_c, const int
class_id, const All_Prob_Vect&
all_prob, const uint2 imgSize)
    : voxel_block_size_(voxel_block_size), volume_size_(volume_size),
      volume_resol_(volume_resol), isStatic_(true), volume_pose_(pose),
      virtual_camera_pose_(virtual_T_w_c), class_id_(class_id),
      semanticfusion_(all_prob)
{
  volume_.init(volume_resol.x, volume_size.x);
  volume_step = min(volume_size_) / max(volume_resol_);
  m_vertex = (float3*) calloc(
      sizeof(float3) * imgSize.x * imgSize.y, 1);
  m_normal = (float3*) calloc(
      sizeof(float3) * imgSize.x * imgSize.y, 1);

  m_vertex_bef_integ = (float3*) calloc(
      sizeof(float3) * imgSize.x * imgSize.y, 1);
  m_normal_bef_integ = (float3*) calloc(
      sizeof(float3) * imgSize.x * imgSize.y, 1);

  //semanticfusion
  fused_time_ = 1;

  //tracking result
  trackresult_ = (TrackData*) calloc(sizeof(TrackData) * imgSize.x * imgSize.y * 2, 1);
  pose_saved_ = false;

  // relative_pose initialization
  setIdentity(delta_pose);

  // bounding volume
  bounding_volume = AABB();

  // last_T_CO
  setIdentity(last_T_OC_);
}

bool Object::absorb_outlier_bg = false;

Object::~Object(){
  this->volume_.release();
  if(allocationList_) delete(allocationList_);
  free(m_vertex);
  free(m_normal);
  free(m_vertex_bef_integ);
  free(m_normal_bef_integ);
  free(trackresult_);
};

void Object::integration_static_kernel(const float * depthImage,
                                       const float3*rgbImage,
                                       const uint2& imgSize, const Matrix4& T_w_c,
                                       const float mu, const float4& k,
                                       const uint frame){
  const float &voxelsize =  this->get_voxel_size();
  int num_vox_per_pix = volume_._size/((VoxelBlock<FieldType>::side) *voxelsize);
  size_t total = num_vox_per_pix * imgSize.x * imgSize.y;
  if(!reserved_) {
    allocationList_ = (octlib::key_t* ) calloc(sizeof(octlib::key_t) * total, 1);
    reserved_ = total;
  }
  unsigned int allocated = 0;
  if(std::is_same<FieldType, SDF>::value) {
    allocated  = buildAllocationList(allocationList_, reserved_,
                                     volume_._map_index, T_w_c,
                                     getCameraMatrix(k), depthImage, imgSize,
                                     volume_._resol, voxelsize, 2*mu);
  } else if(std::is_same<FieldType, BFusion>::value) {
    allocated = buildOctantList(allocationList_, reserved_, volume_._map_index,
                                T_w_c, getCameraMatrix(k), depthImage,
                                imgSize, voxelsize,
                                compute_stepsize, step_to_depth, 6*mu);
  }

  volume_._map_index.alloc_update(allocationList_, allocated);

  if(std::is_same<FieldType, SDF>::value) {
//      if (!render_color_){
//        struct sdf_update funct(floatDepth, computationSize, mu, 100);
//        iterators::projective_functor<FieldType, INDEX_STRUCTURE, struct sdf_update>
//          it(volume._map_index, funct, inverse(pose), getCameraMatrix(k),
//             make_int2(computationSize));
//        it.apply();
//      }
//      else{
    struct sdf_update funct(depthImage, rgbImage, imgSize, mu, 100);
    iterators::projective_functor<FieldType, INDEX_STRUCTURE, struct sdf_update>
        it(volume_._map_index, funct, inverse(T_w_c), getCameraMatrix(k),
           make_int2(imgSize));
    it.apply();
//      }

  } else if(std::is_same<FieldType, BFusion>::value) {
    float timestamp = (1.f / 30.f) * frame;
//      if (!render_color_) {
//        struct bfusion_update funct(floatDepth, computationSize, mu, timestamp);
//        iterators::projective_functor<FieldType,
//                                      INDEX_STRUCTURE,
//                                      struct bfusion_update>
//          it(volume._map_index,
//             funct,
//             inverse(pose),
//             getCameraMatrix(k),
//             make_int2(computationSize));
//        it.apply();
//      }
//      else{
    struct bfusion_update funct(depthImage, rgbImage, imgSize, mu, timestamp);
    iterators::projective_functor<FieldType, INDEX_STRUCTURE, struct bfusion_update>
        it(volume_._map_index, funct, inverse(T_w_c), getCameraMatrix(k),
           make_int2(imgSize));
    it.apply();
  }
//    }
}


void Object::integrate_volume_kernel(const float * depthImage,
                                     const float3*rgbImage,
                                     const cv::Mat& mask, const uint2& imgSize,
                                     const Matrix4& T_w_c, const float mu,
                                     const float4& k, const uint frame){

  const float &voxelsize =  this->get_voxel_size();
  int num_vox_per_pix = volume_._size/((VoxelBlock<FieldType>::side) *voxelsize);
  size_t total = num_vox_per_pix * imgSize.x * imgSize.y;
  if(!reserved_) {
    allocationList_ = (octlib::key_t* ) calloc(sizeof(octlib::key_t) * total, 1);
    reserved_ = total;
  }
  unsigned int allocated = 0;
  if(std::is_same<FieldType, SDF>::value) {
    allocated  = buildVolumeAllocationList(allocationList_, reserved_,
                                     volume_._map_index, T_w_c, volume_pose_,
                                     getCameraMatrix(k), depthImage, mask, imgSize,
                                     volume_._resol, voxelsize, 2*mu);
  } else if(std::is_same<FieldType, BFusion>::value) {
    allocated = buildVolumeOctantList(allocationList_, reserved_, volume_._map_index,
                                      T_w_c, volume_pose_, getCameraMatrix(k),
                                      depthImage, mask, imgSize, voxelsize,
                                      compute_stepsize, step_to_depth, 6*mu);
  }

  volume_._map_index.alloc_update(allocationList_, allocated);

  if(std::is_same<FieldType, SDF>::value) {
//      if (!render_color_){
//        struct sdf_update funct(floatDepth, computationSize, mu, 100);
//        iterators::projective_functor<FieldType, INDEX_STRUCTURE, struct sdf_update>
//          it(volume._map_index, funct, inverse(pose), getCameraMatrix(k),
//             make_int2(computationSize));
//        it.apply();
//      }
//      else{
    struct sdf_update funct(depthImage, rgbImage, mask, imgSize, mu, 100);
    iterators::projective_functor<FieldType, INDEX_STRUCTURE, struct sdf_update>
        it(volume_._map_index, funct, inverse(T_w_c)*volume_pose_, getCameraMatrix(k),
           make_int2(imgSize));
    it.apply();
//      }

  } else if(std::is_same<FieldType, BFusion>::value) {
    float timestamp = (1.f / 30.f) * frame;
//      if (!render_color_) {
//        struct bfusion_update funct(floatDepth, computationSize, mu, timestamp);
//        iterators::projective_functor<FieldType,
//                                      INDEX_STRUCTURE,
//                                      struct bfusion_update>
//          it(volume._map_index,
//             funct,
//             inverse(pose),
//             getCameraMatrix(k),
//             make_int2(computationSize));
//        it.apply();
//      }
//      else{
    struct bfusion_update funct(depthImage, rgbImage, mask, imgSize, mu,
        timestamp);
    iterators::projective_functor<FieldType, INDEX_STRUCTURE, struct bfusion_update>
        it(volume_._map_index, funct, inverse(T_w_c)*volume_pose_, getCameraMatrix(k),
           make_int2(imgSize));
    it.apply();
  }
//    }
}

void Object::fuse_semantic_kernel(const SegmentationResult& segmentationResult,
                                  const int instance_id){
  const instance_seg& seg = segmentationResult.pair_instance_seg_.at(instance_id);
  const int class_id = seg.class_id_;

  if ((class_id != 0)) {
    fuseSemanticLabel(seg);
  }

  //semantic fusion
//  if (this->fused_time_ == 0){
//    std::cout<<"first time semanticfusion: input->"<<class_id<<" "
//                                                               "fused->"<<this->class_id_<<std::endl;
//  }
//  else
  if (!seg.rendered_){
//    std::cout<<"semanticfusion: id->"<<instance_id<<" input->"<<class_id
//             <<" fused->"<<this->class_id_<<std::endl;
  }
//  else{
//    std::cout<<"this class not seen on this frame: id->"<<instance_id
//             <<" before->"<<class_id<<" after->"<<this->class_id_<<std::endl;
//  }

}

void Object::fuseSemanticLabel(const instance_seg& input_seg){
  if(label_size_ != input_seg.all_prob_.size()) {
    std::cerr<<"label number is incorrect"<<std::endl;
    exit (EXIT_FAILURE);
  }

   // semanticfusion_ = All_Prob_Vect::Constant(1.4564);
//    std::cout<<input_labels<<std::endl;
//    std::cout<<input_labels.size()<<" "<<input_labels.rows()<<std::endl;
//    std::cout<<semanticfusion_.size()<<" "<<semanticfusion_.rows()<<std::endl;
//    std::cout<<semanticfusion_<<std::endl;
//  semanticfusion_ = input_labels;
//    semanticfusion_ *= fused_time_;
//    semanticfusion_ = semanticfusion_+ input_labels; // /
//        semanticfusion_ /= (fused_time_ + 1);

  const All_Prob_Vect& input_all_prob = input_seg.all_prob_;
  const int recog_times = input_seg.recog_time_;
  semanticfusion_ = (semanticfusion_ * static_cast<float>(fused_time_) +
      input_all_prob * recog_times)
      / static_cast<float>(fused_time_ + recog_times);

  All_Prob_Vect::Index max_id;
  float max_prob = semanticfusion_.maxCoeff(&max_id);

  //get fused class id
  int fused_id = max_id + 1;  // +1 shift due to bg has no probablity
  this->class_id_ = fused_id;
  this->fused_time_ = this->fused_time_ + recog_times;
}


void Object::refine_mask_use_motion(cv::Mat& mask, const bool use_icp,
                                    const bool use_rgb){
  unsigned int y;
/*#pragma omp parallel for \
        shared(mask), private(y)*/
  for (y = 0; y < mask.rows; y++) {
    for (unsigned int x = 0; x < mask.cols; x++) {
      const uint icp_pos = x + y * mask.cols;
      const uint rgb_pos = icp_pos + mask.cols * mask.rows;

      // if within of the band => recognised as objects
      if (mask.at<float>(y,x) >= 0){
        // -4/-5: depth distance or normal is wrong => allocated
        // => now refine/remove high residual areas => don't fuse tsdf
        if (use_icp){
          if ((this->trackresult_[icp_pos].result == -4) ||
              (this->trackresult_[icp_pos].result == -6)) {
//          if ((this->trackresult_[icp_pos].result < -3)) {
            if (absorb_outlier_bg){
//              mask.at<float>(y, x) = 0;
            }
            else{
              mask.at<float>(y, x) = -1;
            }
            continue;
          }
        }

        if (use_rgb){
          if (this->trackresult_[rgb_pos].result < -3) {
            if (absorb_outlier_bg){
//              mask.at<float>(y, x) = 0;
            }
            else{
              mask.at<float>(y, x) = -1;
            }
            continue;
          }
        }
      }


      // if outside of the band => recognised as background
/*      if (mask.at<float>(y,x) < 0){
        //-3: no correspondence in model => not allocated => don't fuse tsdf
//        if (this->trackresult_[pos].result == -3) {
//          mask.at<float>(y,x) = -1;
//          continue;
//        }

        //-4/-5: depth distance or normal is wrong => allocated
        // => now update this area as background
        if (this->trackresult_[rgb_pos].result < -2) {
          mask.at<float>(y,x) = 0;
          continue;
        }
      }*/

    }
  }
}

void Object::set_static_state(const bool state){
  if (static_object.find(this->class_id_) == static_object.end())
    isStatic_ = state;
}

bool Object::is_static() const{
  return this->isStatic_;
}

const std::set<int> Object::static_object({/*50 /* orange,*/ 63 /*tv*/, 69, 70});

void Object::update_exist_prob(float d) {
  log_odds_ += d;
  double ratio = pow(2, log_odds_);
  exist_prob_ = ratio / (1 + ratio);
  if (exist_prob_ > 0.95){
    isExist_ = true;
    std::cout << "Current object " << instance_label_ << " has fixed!" << std::endl;
  }
}

Matrix4 Object::update_pose(okvis::Time &prevTime, okvis::Time &currTime) {
  double dt = (currTime - prevTime).toSec();
  Eigen::Matrix4d delta_T = Eigen::Matrix4d::Identity();
  delta_T.topLeftCorner(3,3) = rodrigues(w_ * dt);
  delta_T.topRightCorner(3,1) = v_ * dt;
  std::cout << delta_T <<std::endl;
  return volume_pose_ * fromOkvisToMidFusion(okvis::kinematics::Transformation(delta_T));
}

//void Object::update_speed(Matrix4 prev_pose, okvis::Time &prevTime, okvis::Time &currTime) {
//  double dt = (currTime - prevTime).toSec();
//  Eigen::Matrix4d pose_0 = fromMidFusionToEigen(prev_pose).cast<double>();
//  Eigen::Matrix4d pose_1 = fromMidFusionToEigen(volume_pose_).cast<double>();
//  v_ = pose_0.topLeftCorner(3,3).transpose() * (pose_1.topRightCorner(3,1) - pose_0.topRightCorner(3,1)) / dt;
//  Eigen::Matrix3d R = pose_0.topLeftCorner(3,3).transpose() * pose_1.topLeftCorner(3,3);
//  Sophus::SO3d R_0(pose_0.topLeftCorner(3,3));
//  Sophus::SO3d R_1(pose_1.topLeftCorner(3,3));
//  // std::cout << R << std::endl;
//  // std::cout << R * R.transpose() << std::endl;
//  // Sophus::SO3d SO3_R(pose_0.topLeftCorner(3,3).transpose() * pose_1.topLeftCorner(3,3));
//  w_ = (R_0.inverse() * R_1).log() / dt;
//  // w_ = SO3_R.log() / dt;
//}

std::vector<cv::KeyPoint> Object::extract_kps(cv::Mat kf_rgb,
                                              std::vector<cv::Mat> no_outlier_mask){

  // Delete small points in kf_obj_mask
  cv::Mat kf_obj_rgb;
  cv::Mat kf_obj_mask;
  if (fg_outlier_mask_.empty()) {
    cv::bitwise_not(no_outlier_mask[0], kf_obj_mask);
  } else {
    cv::bitwise_not(fg_outlier_mask_, kf_obj_mask);
  }

  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
  cv::morphologyEx(kf_obj_mask, kf_obj_mask, cv::MORPH_OPEN, element);

  kf_rgb.copyTo(kf_obj_rgb, kf_obj_mask);
  cv::imshow("curr_obj_mask", kf_obj_rgb);

  // Extract the KeyPoints of the object
  std::vector<cv::KeyPoint> kps_temp;
  std::vector<cv::KeyPoint> kps;
  // cv::Ptr<cv::Feature2D> detector = cv::BRISK::create(20);
  cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
  cv::Mat dp;
  detector->detectAndCompute(kf_rgb, cv::Mat(), kps_temp, dp);

  cv::Mat temp = kf_rgb.clone();
  cv::drawKeypoints(kf_rgb, kps_temp, temp, cv::Scalar(0, 0, 255));
  cv::imshow("BRISK_temp", temp);

  for (auto kp : kps_temp) {
    if (kf_obj_mask.at<uchar>(kp.pt.y, kp.pt.x) != 0) {
      kps.push_back(kp);
    }
  }

  cv::Mat kp_img = kf_rgb.clone();
  cv::drawKeypoints(kf_rgb, kps, kp_img, cv::Scalar(0, 0, 255));
  cv::imshow("BRISK", kp_img);
  // cv::waitKey(0);

  return kps;
}

std::vector<cv::KeyPoint> Object::extractAndDescribe(cv::Mat kf_rgb,
                                                     std::vector<cv::Mat> no_outlier_mask,
                                                     std::vector<cv::KeyPoint> &kps,
                                                     cv::Mat &des,
                                                     bool use_mask){

  // Delete small points in kf_obj_mask
  cv::Mat kf_obj_rgb;
  cv::Mat kf_obj_mask;
  if (fg_outlier_mask_.empty() || !use_mask) {
    cv::bitwise_not(no_outlier_mask[0], kf_obj_mask);
  } else {
    cv::bitwise_not(fg_outlier_mask_, kf_obj_mask);
    // cv::bitwise_not(no_outlier_mask[0], kf_obj_mask);
  }

  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
  cv::morphologyEx(kf_obj_mask, kf_obj_mask, cv::MORPH_OPEN, element);
  // cv::morphologyEx(kf_obj_mask, kf_obj_mask, cv::MORPH_CLOSE, element);

  // if (use_mask) {
  //   kf_rgb.copyTo(kf_obj_rgb, kf_obj_mask);
  // } else {
  //   kf_obj_rgb = kf_rgb.clone();
  // }

  // if (!fg_outlier_mask_.empty()) {
  //   cv::imshow("curr_obj_mask", fg_outlier_mask_);
  // }

  // Extract the KeyPoints of the object
  std::vector<cv::KeyPoint> kps_temp;
  // cv::Ptr<cv::Feature2D> detector = cv::BRISK::create();
  // // cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
  // detector->detect(kf_rgb, kps_temp);
  // brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> briskDetector(30, 0, 100, 400);
  brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> briskDetector(20, 0, 200, 400);
  brisk::BriskDescriptorExtractor briskDescriptorExtractor(true, true); // check if you need rotation invariance etc!
  cv::Mat kf_gray;
  // cv::cvtColor(kf_obj_rgb, kf_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(kf_rgb, kf_gray, cv::COLOR_BGR2GRAY);
  briskDetector.detect(kf_gray, kps_temp);


  cv::Mat temp = kf_rgb.clone();
  cv::drawKeypoints(kf_rgb, kps_temp, temp, cv::Scalar(0, 0, 255));
  cv::imshow("BRISK_temp", temp);

  for (auto kp : kps_temp) {
    if (kf_obj_mask.at<uchar>(kp.pt.y, kp.pt.x) != 0) {
      kps.push_back(kp);
    }
  }

  briskDescriptorExtractor.compute(kf_gray, kps, des);
  // detector->compute(kf_rgb, kps, des);

  cv::Mat kp_img = kf_rgb.clone();
  cv::drawKeypoints(kf_rgb, kps, kp_img, cv::Scalar(0, 0, 255));
  cv::imshow("BRISK", kp_img);
  // cv::waitKey(0);

  return kps;
}

void Object::set_keyframe(cv::Mat kf_rgb, Matrix4 kf_pose,
                          std::vector<cv::Mat> no_outlier_mask,
                          float3 *l_vertex, uint2 imgSize) {
  return;
  // Check whether need to insert a new KeyFrame using view angle difference
  // Matrix4 T_CO = inverse(kf_pose) * volume_pose_;
  Matrix4 T_OC = inverse(volume_pose_) * kf_pose;

  if (!kf_first_) {
    Eigen::Vector3d t_OC = fromMidFusionToEigenD(T_OC).block<3, 1>(0, 3).eval();
    Eigen::Vector3d last_t_OC = fromMidFusionToEigenD(last_T_OC_).block<3, 1>(0, 3).eval();
    Eigen::Matrix3d delta_R = fromMidFusionToEigenD((inverse(last_T_OC_) * T_OC)).block<3, 3>(0, 0);
    Eigen::Vector3d eulerAngle = delta_R.eulerAngles(2, 1, 0);

    double norm_1 = t_OC.norm();
    double norm_2 = last_t_OC.norm();
    double cosVal = t_OC.dot(last_t_OC) / (norm_1 * norm_2);
    double angleDiff = acos(cosVal) * 180 / M_PI;
    double normDiff = abs(norm_1 - norm_2);
    std::cout << "cosVal: " << cosVal << std::endl;
    std::cout << "Angle Diff: " << angleDiff << " Norm Diff: " << normDiff << std::endl;
    std::cout << "AD * ND = " << angleDiff * normDiff << std::endl;
    std::cout << "Euler Angle: " << eulerAngle << std::endl;
    if (angleDiff * normDiff < 0.05 && abs(eulerAngle(0)) < 5 &&
        abs(eulerAngle(1)) < 5 && abs(eulerAngle(2)) < 5) {
    // if (angleDiff < 1) {
      // do not set the KeyFrame
      return;
    }
  } else {
    kf_first_ = false;
  }

  std::cout << "Start setting the KeyFrame" << std::endl;
  // cv::waitKey(0);


  // Extract and describe
  std::vector<cv::KeyPoint> kps_temp;
  cv::Mat des;
  extractAndDescribe(kf_rgb, no_outlier_mask, kps_temp, des);
  // opengv::points_t kps_gv;
  std::vector<cv::Point3d> kps_3d;

  // Delete kps with wrong depth
  std::vector<cv::KeyPoint> kps_2d;
  for (const auto &kp : kps_temp) {
    const int index = int(kp.pt.x + kp.pt.y * imgSize.x);
    float3 v = l_vertex[index];
    // if (v.z == 0) continue;
    kps_2d.push_back(kp);
    // float3 v_W = rotate(kf_pose, make_float3(v.x, v.y, v.z));
    kps_3d.push_back(cv::Point3d(v.x, v.y, v.z));
  }


  if (kps_3d.size() > 2){
    // kf_kps_list_.push_back(kps_gv);
    kf_kps_2d_.push_back(kps_2d);
    kf_kps_3d_.push_back(kps_3d);
    kf_des_.push_back(des);
    kf_rgb_list_.push_back(kf_rgb.clone());

    // Save T_CO
    kf_pose_list_.push_back(inverse(kf_pose) * volume_pose_);

    // Save T_WO
    kf_T_WO_list_.push_back(volume_pose_);

    // Save the current T_OC as the reference
    last_T_OC_ = T_OC;
    // cv::waitKey(0);
  } else {
    kf_first_ = true;
  }
}

bool Object::relocalisation(cv::Mat kf_rgb, Matrix4 kf_pose,
                            std::vector<cv::Mat> no_outlier_mask,
                            float3 *l_vertex, float *l_depth,
                            uint2 imgSize, float4 k, uint frame) {

  std::cout << "Start object relocalisation..." << std::endl;

  std::vector<cv::KeyPoint> kps_temp;
  std::vector<cv::KeyPoint> kps_2d;
  cv::Mat des;
  extractAndDescribe(kf_rgb, no_outlier_mask, kps_temp, des, false);

  // Delete kps with wrong depth
  for (auto kp : kps_temp) {
    const int index = int(kp.pt.x + kp.pt.y * imgSize.x);
    float3 v = l_vertex[index];
    // if (v.z == 0) continue;
    kps_2d.push_back(kp);
  }

  if (kps_2d.size() <= 3) {
    std::cout << "Current kps size <= 3!" << std::endl;
    return false;
  }

  // Choose the most resonable KeyFrame
  for (int i = kf_pose_list_.size() - 1; i >= 0; i--) {
    // match
    std::vector<cv::KeyPoint> ref_kps_2d = kf_kps_2d_[i];
    std::vector<cv::Point3d> ref_kps_3d = kf_kps_3d_[i];
    assert(ref_kps_2d.size() == ref_kps_3d.size());
    cv::Mat ref_des = kf_des_[i];

    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> matches_temp;
    std::vector<std::vector<cv::DMatch>> matchess;

    brisk::BruteForceMatcher matcher;
    // matcher.radiusMatch(ref_des, des, matchess, 56);
    matcher.match(ref_des, des, matches_temp);
    // matches = matchess[1];
    // matches[0].distance

    // for (auto matches : matchess) {

      std::cout << "brisk size " << matchess.size() << std::endl;

      cv::Mat img_match;
      for (cv::DMatch m : matches_temp) {
        if (m.distance < 100) {
          matches.push_back(m);
        }
      }
      std::cout << "number of matches: " << matches.size() << std::endl;
      cv::drawMatches(kf_rgb_list_[i].clone(), ref_kps_2d, kf_rgb.clone(), kps_2d, matches, img_match);
      cv::imshow("Matches", img_match);
      // if (instance_label_ == 1) cv::waitKey(0);
      // if (frame > 520) cv::waitKey(0);
      // cv::waitKey(0);
      if (matches.size() < 6) { //6
        continue;
      }

      // cv::waitKey(0);
      // if (matches.size() > 0) {
      //   // choose good matches
      //   auto min_max = std::minmax_element(matches.begin(), matches.end(),
      //                                      [](const cv::DMatch &m1, const cv::DMatch &m2)
      //                                      {return m1.distance < m2.distance;});
      //   double min_dist = min_max.first->distance;
      //   std::cout << "Min Dist: " << min_dist << std::endl;
      //
      //   std::vector<cv::DMatch> goodMatches;
      //   for (int j = 0; j < ref_des.rows; j++) {
      //     if (matches[j].distance <= std::max(1.5 * min_dist, 0.0)) {
      //       goodMatches.push_back(matches[j]);
      //     }
      //   }
      //   matches = goodMatches;
      //
      //   std::cout << "number of matches: " << matches.size() << std::endl;
      //   if (matches.size() < 6) {
      //     continue;
      //   }
      //
      //   cv::drawMatches(kf_rgb_list_[i].clone(), ref_kps_2d, kf_rgb.clone(), kps_2d, matches, img_match);
      //   cv::imshow("Matches", img_match);
      // }

      // re-order the key points
      std::vector<cv::Point3d> pts1;
      std::vector<cv::Point2d> pts2;
      cv::Mat K = (cv::Mat_<double>(3, 3) << k.x, 0, k.z, 0, k.y, k.w, 0, 0, 1);
      for (cv::DMatch m : matches) {
        cv::KeyPoint ref_kp = ref_kps_2d[m.queryIdx];
        cv::Point2d pt_2d = pixel2cam(ref_kp.pt, K);
        const int index = int(ref_kp.pt.x + ref_kp.pt.y * imgSize.x);
        float d = l_depth[index];
        pts1.push_back(cv::Point3d(pt_2d.x * d, pt_2d.y * d, d));
        pts2.push_back(kps_2d[m.trainIdx].pt);
      }

      // solve PnP ransac
      cv::Mat rVec, tVec, R;
      cv::Mat inliers;
      bool success;
      success = cv::solvePnPRansac(pts1, pts2, K, cv::Mat(), rVec, tVec,
                                   true, 1000, 3, 0.99, inliers);
      // check for PnP inlier ratio
      float inlier_ratio = static_cast<float>(inliers.rows) / pts1.size();
      std::cout << "PnP inlier ratio: " << inlier_ratio << std::endl;
      if (inlier_ratio < 0.6 /*|| inliers.rows < 6*/) { // 0.7 for 200
        continue;
      }

      if (!success) {
        continue;
      }

      // project the ref points to current image
      std::vector<cv::Point2d> proj_2d;
      std::vector<cv::KeyPoint> proj_kp;
      // cv::projectPoints(ref_kps_3d, rVec, tVec, K, cv::Mat()   , proj_2d);
      cv::projectPoints(pts1, rVec, tVec, K, cv::Mat(), proj_2d);
      for (auto pt : proj_2d) {
        cv::KeyPoint kp;
        kp.pt = pt;
        proj_kp.push_back(kp);
      }
      cv::Mat kp_1 = kf_rgb_list_[i].clone();
      cv::Mat kp_2 = kf_rgb.clone();
      cv::drawKeypoints(kf_rgb_list_[i].clone(), ref_kps_2d, kp_1, cv::Scalar(0, 0, 255));
      cv::drawKeypoints(kf_rgb, proj_kp, kp_2, cv::Scalar(0, 0, 255));
      cv::imshow("Projection_1", kp_2);

      cv::Rodrigues(rVec, R);
      std::cout << "Rotation: " << R << std::endl;
      std::cout << "Tranlation: " << tVec << std::endl;

      // update T_WO (volume_pose_)
      Matrix4 delta_T;
      for (int j = 0; j < 3; j++) {
        delta_T.data[j].x = R.at<double>(j, 0);
        delta_T.data[j].y = R.at<double>(j, 1);
        delta_T.data[j].z = R.at<double>(j, 2);
        delta_T.data[j].w = tVec.at<double>(j);
      }
      delta_T.data[3].x = 0;
      delta_T.data[3].y = 0;
      delta_T.data[3].z = 0;
      delta_T.data[3].w = 1;

      // re-calculate the intensity residuals
      float residuals = 0;
      cv::Mat kf_gray;
      cv::cvtColor(kf_rgb, kf_gray, cv::COLOR_BGR2GRAY);
      std::vector<cv::KeyPoint> kps_dst;
      for (int j = 0; j < pts1.size(); j++) {
        cv::Point3d kps = pts1[j];
        cv::Point2d pt_2d = pts2[j];
        float3 p = delta_T * make_float3(kps.x, kps.y, kps.z);
        Matrix4 intrinsic = getCameraMatrix(k);
        float3 p_p = rotate(intrinsic, p);
        cv::KeyPoint kp;
        kp.pt.x = round(p_p.x / p_p.z);
        kp.pt.y = round(p_p.y / p_p.z);
        kps_dst.push_back(kp);

        uchar intensity_0 = kf_gray.at<uchar>(kp.pt.y, kp.pt.x);
        uchar intensity_1 = kf_gray.at<uchar>(pt_2d.y, pt_2d.x);
        std::cout << int(intensity_0) << " " << int(intensity_1) << std::endl;
        residuals += pow(intensity_0 - intensity_1, 2);
      }
      residuals /= pts1.size();
      std::cout << "Current residual: " << residuals << std::endl;
      // if (residuals > 1000) {
      //   // cv::waitKey(0);
      //   // return false;
      //   continue;
      // }

      cv::Mat kp_img = kf_rgb.clone();
      cv::drawKeypoints(kf_rgb, kps_dst, kp_img, cv::Scalar(0, 0, 255));
      cv::imshow("Projection_2", kp_img);

      // update T_WO (volume_pose_)
      Matrix4 prev_T_CO = kf_pose_list_[i];
      Matrix4 curr_T_CO = delta_T * prev_T_CO;
      Matrix4 temp = volume_pose_;
      volume_pose_ = kf_pose * curr_T_CO;
      virtual_camera_pose_ = temp * inverse(curr_T_CO);
      // cv::waitKey(0);

    cv::imwrite("/home/ryf/Videos/object_relocalisation/paper/relocalisation/current_kf.png", kf_rgb);
    cv::imwrite("/home/ryf/Videos/object_relocalisation/paper/relocalisation/ref_kf.png", kf_rgb_list_[i]);
    cv::imwrite("/home/ryf/Videos/object_relocalisation/paper/relocalisation/match.png", img_match);

    return true;
    }
  // }
  return false;
}


inline bool pointInFrustumInf(const float4 &k, const uint2 &size, const Eigen::Vector3f &vertex) {
  // If Z is negative or smaller than 1, means the point is behind the image plane
  if (vertex(2) <= 1) return false;
  Eigen::Vector3f vertex_unit_plane = vertex / vertex(2);
  Eigen::Matrix3f camera = fromMidFusionToEigen(getCameraMatrix(k)).topLeftCorner<3,3>();
  const Eigen::Vector2f proj_pixel = (camera * vertex_unit_plane).head<2>();
  if (proj_pixel(0) >= 0 && proj_pixel(0) <= size.x &&
      proj_pixel(1) >= 0 && proj_pixel(1) <= size.y) {
    return true;
  } else {
    return false;
  }
}

/// @brief Bounding Volume part1: AABB
AABB::AABB() :
    min_(Eigen::Vector3f::Constant(INFINITY)),
    max_(Eigen::Vector3f::Constant(-INFINITY)) {
  updateVertices();
}

AABB::AABB(const Eigen::Vector3f &min, const Eigen::Vector3f &max) :
    min_(min), max_(max) {
  updateVertices();
}

AABB::AABB(const float3 *vertex_map,
           const Eigen::Matrix4f &T_vm,
           const cv::Mat &mask) {
  vertexMapMinMax(vertex_map, mask, T_vm, min_, max_);
  updateVertices();
}

bool AABB::isVisible(const Eigen::Matrix4f &T_VC,
                     const float4 &k,
                     const uint2 &size) const {
  const Eigen::Matrix4f T_CV = T_VC.inverse();

  // Convert the AABB vertices from world to camera coordinates and test for
  // visibility one by one.
  for (int i = 0; i < vertices_.cols(); ++i){
    const Eigen::Vector3f vertex_c = (T_CV * vertices_.col(i).homogeneous()).head<3>();
    const bool visible = pointInFrustumInf(k, size, vertex_c);
    if (visible){
      return true;
    }
  }
  return false;
}

bool AABB::isTruncated(const Eigen::Matrix4f &T_VC,
                       const float4 &k,
                       const uint2 &size) const {
  const Eigen::Matrix4f T_CV = T_VC.inverse();
  int visible_num = 0;

  // Convert the AABB vertices from world to camera coordinates and test for
  // visibility one by one.
  for (int i = 0; i < vertices_.cols(); ++i){
    const Eigen::Vector3f vertex_c = (T_CV * vertices_.col(i).homogeneous()).head<3>();
    const bool visible = pointInFrustumInf(k, size, vertex_c);
    if (visible){
      visible_num++;
    }
  }

  if (visible_num < 6) {
    return true;
  } else {
    return false;
  }
}

void AABB::merge(const AABB &other) {
  // Min.
  if (other.min_.x() < min_.x()) {
    min_.x() = other.min_.x();
  }
  if (other.min_.y() < min_.y()) {
    min_.y() = other.min_.y();
  }
  if (other.min_.z() < min_.z()) {
    min_.z() = other.min_.z();
  }
  // Max.
  if (other.max_.x() > max_.x()) {
    max_.x() = other.max_.x();
  }
  if (other.max_.y() > max_.y()) {
    max_.y() = other.max_.y();
  }
  if (other.max_.z() > max_.z()) {
    max_.z() = other.max_.z();
  }

  updateVertices();
}

void AABB::merge(const float3 *vertex_map,
                 const Eigen::Matrix4f &T_vm,
                 const cv::Mat &mask) {
  const AABB tmp_bounding_volume(vertex_map, T_vm, mask);
  merge(tmp_bounding_volume);
  updateVertices();
}

void AABB::overlay(uint32_t *out,
                   const uint2 &output_size,
                   const Eigen::Matrix4f &T_VC,
                   const float4 &k,
                   const float) const {
  // TODO: Use opacity parameter

  // Project the AABB vertices on the image plane.
  std::vector<cv::Point2f> projected_vertices = projectAABB(T_VC, k, output_size);

  // Create a cv::Mat header for the output image so that OpenCV drawing
  // functions can be used. No data is copied or deallocated by OpenCV so this
  // operation is fast.
  cv::Mat out_mat(cv::Size(output_size.x, output_size.y), CV_8UC4, out);

  // Draw the bottom box edges.
  cv::line(out_mat, projected_vertices[0], projected_vertices[1], _overlay_color);
  cv::line(out_mat, projected_vertices[1], projected_vertices[3], _overlay_color);
  cv::line(out_mat, projected_vertices[3], projected_vertices[2], _overlay_color);
  cv::line(out_mat, projected_vertices[2], projected_vertices[0], _overlay_color);

  // Draw the top box edges.
  cv::line(out_mat, projected_vertices[4], projected_vertices[5], _overlay_color);
  cv::line(out_mat, projected_vertices[5], projected_vertices[7], _overlay_color);
  cv::line(out_mat, projected_vertices[7], projected_vertices[6], _overlay_color);
  cv::line(out_mat, projected_vertices[6], projected_vertices[4], _overlay_color);

  // Draw the vertical box edges.
  cv::line(out_mat, projected_vertices[0], projected_vertices[4], _overlay_color);
  cv::line(out_mat, projected_vertices[1], projected_vertices[5], _overlay_color);
  cv::line(out_mat, projected_vertices[2], projected_vertices[6], _overlay_color);
  cv::line(out_mat, projected_vertices[3], projected_vertices[7], _overlay_color);
}

void AABB::draw(cv::Mat &out_mat,
                const uint2 &output_size,
                const Eigen::Matrix4f &T_VC,
                const float4 &k,
                const float) const {
  // TODO: Use opacity parameter

  // Project the AABB vertices on the image plane.
  std::vector<cv::Point2f> projected_vertices = projectAABB(T_VC, k, output_size);

  // Draw the bottom box edges.
  cv::line(out_mat, projected_vertices[0], projected_vertices[1], _overlay_color);
  cv::line(out_mat, projected_vertices[1], projected_vertices[3], _overlay_color);
  cv::line(out_mat, projected_vertices[3], projected_vertices[2], _overlay_color);
  cv::line(out_mat, projected_vertices[2], projected_vertices[0], _overlay_color);

  // Draw the top box edges.
  cv::line(out_mat, projected_vertices[4], projected_vertices[5], _overlay_color);
  cv::line(out_mat, projected_vertices[5], projected_vertices[7], _overlay_color);
  cv::line(out_mat, projected_vertices[7], projected_vertices[6], _overlay_color);
  cv::line(out_mat, projected_vertices[6], projected_vertices[4], _overlay_color);

  // Draw the vertical box edges.
  cv::line(out_mat, projected_vertices[0], projected_vertices[4], _overlay_color);
  cv::line(out_mat, projected_vertices[1], projected_vertices[5], _overlay_color);
  cv::line(out_mat, projected_vertices[2], projected_vertices[6], _overlay_color);
  cv::line(out_mat, projected_vertices[3], projected_vertices[7], _overlay_color);
}

std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> AABB::edges() const
{
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> e;
  e.reserve(24);
  // Add the bottom AABB edges.
  e.push_back(vertices_.col(0));
  e.push_back(vertices_.col(1));
  e.push_back(vertices_.col(1));
  e.push_back(vertices_.col(3));
  e.push_back(vertices_.col(3));
  e.push_back(vertices_.col(2));
  e.push_back(vertices_.col(2));
  e.push_back(vertices_.col(0));
  // Add the top AABB edges.
  e.push_back(vertices_.col(4));
  e.push_back(vertices_.col(5));
  e.push_back(vertices_.col(5));
  e.push_back(vertices_.col(7));
  e.push_back(vertices_.col(7));
  e.push_back(vertices_.col(6));
  e.push_back(vertices_.col(6));
  e.push_back(vertices_.col(4));
  // Add the vertical AABB edges.
  e.push_back(vertices_.col(0));
  e.push_back(vertices_.col(4));
  e.push_back(vertices_.col(1));
  e.push_back(vertices_.col(5));
  e.push_back(vertices_.col(2));
  e.push_back(vertices_.col(6));
  e.push_back(vertices_.col(3));
  e.push_back(vertices_.col(7));
  return e;
}

void AABB::updateVertices() {
  // Recompute the AABB vertices.
  vertices_.col(0) = Eigen::Vector3f(min_.x(), min_.y(), min_.z());
  vertices_.col(1) = Eigen::Vector3f(max_.x(), min_.y(), min_.z());
  vertices_.col(2) = Eigen::Vector3f(min_.x(), max_.y(), min_.z());
  vertices_.col(3) = Eigen::Vector3f(max_.x(), max_.y(), min_.z());
  vertices_.col(4) = Eigen::Vector3f(min_.x(), min_.y(), max_.z());
  vertices_.col(5) = Eigen::Vector3f(max_.x(), min_.y(), max_.z());
  vertices_.col(6) = Eigen::Vector3f(min_.x(), max_.y(), max_.z());
  vertices_.col(7) = Eigen::Vector3f(max_.x(), max_.y(), max_.z());
}

std::vector<cv::Point2f> AABB::projectAABB(const Eigen::Matrix4f &T_VC,
                                           const float4 &k,
                                           const uint2 &size) const {
  std::vector<cv::Point2f> projected_vertices(8, cv::Point2f(-1.f, -1.f));
  const Eigen::Matrix4f T_CV = T_VC.inverse();

  // Project the AABB vertices on the image plane. Vertices outside the image
  // plane will have coordinates [-1, -1]^T.
  for (size_t i = 0; i < 8; ++i){
    const Eigen::Vector3f vertex_c = (T_CV * vertices_.col(i).homogeneous()).head<3>();

    if (vertex_c(2) <= 1) continue;

    Eigen::Vector3f vertex_unit_plane = vertex_c / vertex_c(2);
    Eigen::Matrix3f camera = fromMidFusionToEigen(getCameraMatrix(k)).topLeftCorner<3,3>();
    const Eigen::Vector2f proj_pixel = (camera * vertex_unit_plane).head<2>();

    projected_vertices[i] = cv::Point2f(proj_pixel.x(), proj_pixel.y());
  }

  return projected_vertices;
}

int vertexMapStats(const float3 *point_cloud_C,
                   const cv::Mat &mask,
                   const Eigen::Matrix4f &T_MC,
                   Eigen::Vector3f &vertex_min,
                   Eigen::Vector3f &vertex_max,
                   Eigen::Vector3f &vertex_mean)
{
  // Initialize min, max and mean vertex elements.
  vertex_min = Eigen::Vector3f::Constant(INFINITY);
  vertex_max = Eigen::Vector3f::Constant(-INFINITY);
  vertex_mean = Eigen::Vector3f::Zero();
  int count = 0;

  //TODO: parallelize
  for (int pixely = 0; pixely < mask.rows; ++pixely) {
    for (int pixelx = 0; pixelx < mask.cols; ++pixelx) {
      if (mask.at<uchar>(pixely, pixelx) != 0) {
        const int pixel_ind = pixelx + pixely * mask.cols;

        // Skip vertices whose coordinates are all zero as they are invalid.
        if (point_cloud_C[pixel_ind].z <= 0) {
          continue;
        }

        Eigen::Vector3f pc;
        pc << point_cloud_C[pixel_ind].x, point_cloud_C[pixel_ind].y, point_cloud_C[pixel_ind].z;
        const Eigen::Vector4f vertex = T_MC * pc.homogeneous();

        if (vertex.x() > vertex_max.x())
          vertex_max.x() = vertex.x();
        if (vertex.x() < vertex_min.x())
          vertex_min.x() = vertex.x();
        if (vertex.y() > vertex_max.y())
          vertex_max.y() = vertex.y();
        if (vertex.y() < vertex_min.y())
          vertex_min.y() = vertex.y();
        if (vertex.z() > vertex_max.z())
          vertex_max.z() = vertex.z();
        if (vertex.z() < vertex_min.z())
          vertex_min.z() = vertex.z();

        vertex_mean.x() += vertex.x();
        vertex_mean.y() += vertex.y();
        vertex_mean.z() += vertex.z();
        count++;
      }
    }
  }
  // Is the average needed for the center or will the midpoint do?
  vertex_mean.x() /= count;
  vertex_mean.y() /= count;
  vertex_mean.z() /= count;

  return count;
}

void vertexMapMinMax(const float3 *point_cloud_C,
                     const cv::Mat &mask,
                     const Eigen::Matrix4f &T_MC,
                     Eigen::Vector3f &vertex_min,
                     Eigen::Vector3f &vertex_max)
{
  // Initialize min and max vertex elements.
  vertex_min = Eigen::Vector3f::Constant(INFINITY);
  vertex_max = Eigen::Vector3f::Constant(-INFINITY);

  //TODO: parallelize
  for (int pixely = 0; pixely < mask.rows; ++pixely) {
    for (int pixelx = 0; pixelx < mask.cols; ++pixelx) {
      if (mask.at<uchar>(pixely, pixelx) != 0) {
        const int pixel_ind = pixelx + pixely * mask.cols;

        // Skip vertices whose coordinates are all zero as they are invalid.
        if (point_cloud_C[pixel_ind].z <= 0) {
          continue;
        }

        Eigen::Vector3f pc;
        pc << point_cloud_C[pixel_ind].x, point_cloud_C[pixel_ind].y, point_cloud_C[pixel_ind].z;
        const Eigen::Vector4f vertex = T_MC * pc.homogeneous();

        if (vertex.x() > vertex_max.x())
          vertex_max.x() = vertex.x();
        if (vertex.x() < vertex_min.x())
          vertex_min.x() = vertex.x();
        if (vertex.y() > vertex_max.y())
          vertex_max.y() = vertex.y();
        if (vertex.y() < vertex_min.y())
          vertex_min.y() = vertex.y();
        if (vertex.z() > vertex_max.z())
          vertex_max.z() = vertex.z();
        if (vertex.z() < vertex_min.z())
          vertex_min.z() = vertex.z();
      }
    }
  }
}