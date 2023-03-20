/*
 * SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2023 Binbin Xu
 * SPDX-FileCopyrightText: 2023 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef OFUSION_OBJECT_H
#define OFUSION_OBJECT_H

#include "vector_types.h"
#include <commons.h>
#include "math_utils.h"
#include <map>
#include <vector>
#include <memory>
#include <segmentation.h>
#include "../continuous/volume_instance.hpp"

#include "../bfusion/mapping_impl.hpp"
#include "../kfusion/mapping_impl.hpp"
#include "../bfusion/alloc_impl.hpp"
#include "../kfusion/alloc_impl.hpp"
#include "okvis/Time.hpp"

#include "opencv2/opencv.hpp"

class Object;
class BoundingVolume;
class AABB;
static cv::Scalar _overlay_color = cv::Scalar(0, 255, 0);
typedef std::shared_ptr<Object> ObjectPointer;
typedef std::vector<ObjectPointer> ObjectList;
typedef ObjectList::iterator ObjectListIterator;


/// \brief Bounding Volume
class BoundingVolume{
public:
    virtual ~BoundingVolume(){};

    virtual bool isVisible(const Eigen::Matrix4f &T_VC,
                           const float4 &k,
                           const uint2 &size) const = 0;

    virtual bool isTruncated(const Eigen::Matrix4f &T_VC,
                             const float4 &k,
                             const uint2 &size) const = 0;

    virtual void merge(const float3 *vertex_map,
                       const Eigen::Matrix4f &T_vm,
                       const cv::Mat &mask) = 0;

    // virtual cv::Mat raycastingMask(const uint2 &mask_size,
    //                                const Eigen::Matrix4f &T_VC,
    //                                const float4 &k) = 0;

    virtual void overlay(uint32_t *out,
                         const uint2 &output_size,
                         const Eigen::Matrix4f &T_VC,
                         const float4 &k,
                         const float opacity) const = 0;

    virtual std::vector<Eigen::Vector3f,
            Eigen::aligned_allocator<Eigen::Vector3f>> edges() const = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class AABB : public BoundingVolume {
public:
    Eigen::Vector3f min_;
    Eigen::Vector3f max_;

    AABB();

    AABB(const Eigen::Vector3f &min, const Eigen::Vector3f &max);

    AABB(const float3 *vertex_map,
         const Eigen::Matrix4f &T_vm,
         const cv::Mat &mask);

    bool isVisible(const Eigen::Matrix4f &T_VC,
                   const float4 &k,
                   const uint2 &size) const;

    bool isTruncated(const Eigen::Matrix4f &T_VC,
                     const float4 &k,
                     const uint2 &size) const;

    void merge(const AABB &other);

    void merge(const float3 *vertex_map,
               const Eigen::Matrix4f &T_vm,
               const cv::Mat &mask);

    // cv::Mat raycastingMask(const uint2 &mask_size,
    //                        const Eigen::Matrix4f &T_VC,
    //                        const float4 &k);

    void overlay(uint32_t *out,
                 const uint2 &output_size,
                 const Eigen::Matrix4f &T_VC,
                 const float4 &k,
                 const float opacity) const;

    void draw(cv::Mat &out_mat,
              const uint2 &output_size,
              const Eigen::Matrix4f &T_VC,
              const float4 &k,
              const float) const;

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> edges() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    Eigen::Matrix<float, 3, 8> vertices_;

    void updateVertices();

    std::vector<cv::Point2f> projectAABB(const Eigen::Matrix4f &T_VC,
                                         const float4 &k,
                                         const uint2 &size) const;
};


/**
   * Given a vertex map and a binary mask, compute the minimum, maximum and
   * mean vertex coordinates after transforming them to the Map frame using
   * T_MC. Returns the number of vertices processed.
   */
int vertexMapStats(const float3 *point_cloud_C,
                   const cv::Mat &mask,
                   const Eigen::Matrix4f &T_MC,
                   Eigen::Vector3f &vertex_min,
                   Eigen::Vector3f &vertex_max,
                   Eigen::Vector3f &vertex_mean);



/**
   * Given a vertex map and a binary mask, compute the minimum and maximum
   * vertex coordinates after transforming them to the Map frame using T_MC.
   */
void vertexMapMinMax(const float3 *point_cloud_C,
                     const cv::Mat &mask,
                     const Eigen::Matrix4f &T_MC,
                     Eigen::Vector3f &vertex_min,
                     Eigen::Vector3f &vertex_max);



class Object{
 private:
  // volume
  int voxel_block_size_ = 8;
  float3 volume_size_ = make_float3(10);  // dimensions = size
  uint3 volume_resol_ = make_uint3(1024); // resolutions
  float volume_step = min(volume_size_) / max(volume_resol_);
  bool isStatic_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  // Initilisation
  Object(const int voxel_block_size, const float3 &volume_size, 
         const uint3 &volume_resol, const Matrix4 &pose, 
         const Matrix4 &virtual_T_w_c, const int class_id, 
         const All_Prob_Vect &all_prob, const uint2 imgSize);

  void set_volume_size(const float3& volume_size){
    this->volume_size_ = volume_size;
  };

  float3 get_volume_size(){
    return this->volume_size_;
  };

  uint3 get_volume_resol(){
    return this->volume_resol_;
  };

  float get_voxel_size(){
    return volume_size_.x/volume_resol_.x;
  }

  float get_volume_step(){
    return volume_step;
  }

  /**
 * @brief integrate a depth image into volume, based on camera pose T_w_c
 * @param depthImage [in] input depth image
 * @param rgbImage [in] input rgb image
 * @param imgSize [in] image size
 * @param T_w_c [in] camera pose
 * @param mu [in] TSDF mu
 * @param k [in] intrinsic matrix
 * @param frame [in] frame number for bfusion integration
 */
  void integration_static_kernel(const float * depthImage, const float3*rgbImage,
                                 const uint2& imgSize, const Matrix4& T_w_c,
                                 const float mu, const float4& k, const uint frame);

  /**
* @brief integrate a depth image into volume, based on camera pose T_w_c, and
   * the estimated volume pose T_w_o
* @param depthImage [in] input depth image
* @param rgbImage [in] input rgb image
* @param mask [in] segmentation mask corresponding this volume instance
* @param imgSize [in] image size
* @param T_w_c [in] camera pose
* @param mu [in] TSDF mu
* @param k [in] intrinsic matrix
* @param frame [in] frame number for bfusion integration
*/
  void integrate_volume_kernel(const float * depthImage, const float3*rgbImage,
                               const cv::Mat& mask, const uint2& imgSize,
                               const Matrix4& T_w_c, const float mu,
                               const float4& k, const uint frame);

  void fuse_semantic_kernel(const SegmentationResult& segmentationResult,
                            const int instance_id);

  void fuseSemanticLabel(const instance_seg& input_seg);

  void refine_mask_use_motion(cv::Mat& mask, const bool use_icp,
                              const bool use_rgb);

  void set_static_state(const bool state);

  void update_exist_prob(float d);

  bool is_static() const;

  Matrix4 update_pose(okvis::Time &prevTime, okvis::Time &currTime);

//  void update_speed(Matrix4 prev_pose, okvis::Time &prevTime, okvis::Time &currTime);

  std::vector<cv::KeyPoint> extract_kps(cv::Mat kf_rgb, std::vector<cv::Mat> no_outlier_mask);

  std::vector<cv::KeyPoint> extractAndDescribe(cv::Mat kf_rgb, std::vector<cv::Mat> no_outlier_mask,
                                               std::vector<cv::KeyPoint> &kps, cv::Mat &des,
                                               bool use_mask = true);

  /// \brief Project 2d pixels to 3d points in camera coordinate
  static cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K){
    return cv::Point2d(
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
  }

  void set_keyframe(cv::Mat kf_rgb, Matrix4 kf_pose,
                    std::vector<cv::Mat> no_outlier_mask,
                    float3 *l_vertex, uint2 imgSize);

  bool relocalisation(cv::Mat kf_rgb, Matrix4 kf_pose,
                      std::vector<cv::Mat> no_outlier_mask,
                      float3 *l_vertex, float *l_depth,
                      uint2 imgSize, float4 k, uint frame);

  virtual ~Object();

 public:
  //octree-based volume representation
  Volume<FieldType> volume_;
  octlib::key_t* allocationList_ = nullptr;
  size_t reserved_ = 0;

  //vertex and normal belonging to this volumne
  float3 * m_vertex;
  float3 * m_normal;

  //vertex and normal before intergration, belonging to this volumne
  float3 * m_vertex_bef_integ;
  float3 * m_normal_bef_integ;

  //pose
  Matrix4 volume_pose_;
  // speed and angular velocity
  Eigen::Vector3d v_ = Eigen::Vector3d(0.0, 0.0, 0.0);
  Eigen::Vector3d w_ = Eigen::Vector3d(0.0, 0.0, 0.0);
  // relative transformation
  Matrix4 delta_pose;
  // object tracking using RGB only
  bool rgb_only = false;

  // object relocalisation
  bool is_small = false;
  bool is_big = false;
  bool approach_boundary = false;
  bool out_boundary = false;
  bool relocalize = false;
  bool lost = false;
  int max_size = 0;
  int current_size = 0;
  float last_res_ = 0;

  //virtual camera pose
  Matrix4 virtual_camera_pose_;

  //labels
  int class_id_;
  int instance_label_;
  int output_id_;

  //semanticfusion
  All_Prob_Vect semanticfusion_ = All_Prob_Vect::Zero();
  int fused_time_;
  static const uint label_size_ = 80;
  int outside_field_time_ = 0;
  bool isExist_ = false;
  double exist_prob_ = 0.5;
  float log_odds_ = 0;

  //Tracking result for this frame
  TrackData* trackresult_;

  static const std::set<int> static_object;
  bool pose_saved_;

  static bool absorb_outlier_bg;

  // Dynamic mask
  cv::Mat fg_outlier_mask_;
  cv::Mat prev_fg_outlier_mask_;

  // Keyframes
  std::vector<Matrix4> kf_pose_list_;
  std::vector<Matrix4> kf_T_WO_list_;
  std::vector<std::vector<cv::KeyPoint>> kf_kps_2d_;
  std::vector<std::vector<cv::Point3d>> kf_kps_3d_;
  std::vector<cv::Mat> kf_des_;
  std::vector<cv::Mat> kf_rgb_kps_list_;
  std::vector<cv::Mat> kf_rgb_list_;
  Matrix4 last_T_OC_;
  bool kf_first_ = true;

  // Bounding Volume
  AABB bounding_volume;
};


#endif //OFUSION_OBJECT_H
