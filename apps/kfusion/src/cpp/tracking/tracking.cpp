/*

 SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 SPDX-FileCopyrightText: 2023 Binbin Xu
 SPDX-FileCopyrightText: 2023 Yifei Ren
 SPDX-License-Identifier: BSD-3-Clause

*/

#include "tracking.h"

Tracking::Tracking(const uint2 imgSize,
                   const std::vector<int> &GN_opt_iterations,
                   const bool use_live_depth_only,
                   const bool using_ICP,
                   const bool using_RGB)
    : imgSize_(make_uint2(imgSize.x, imgSize.y)),
      GN_opt_iterations_(GN_opt_iterations),
      use_live_depth_only_(use_live_depth_only),
      using_ICP_(using_ICP),
      using_RGB_(using_RGB),
      jtj_v_(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()),
      jtr_v_(Eigen::Matrix<double, 6, 1>::Zero()),
      jtj_v_lifted_(Eigen::Matrix<double, 15, 15, Eigen::RowMajor>::Zero()),
      jtr_v_lifted_(Eigen::Matrix<double, 15, 1>::Zero()),
      j_imu0_(Eigen::Matrix<double, 15, 15, Eigen::RowMajor>::Zero()),
      j_imu1_(Eigen::Matrix<double, 15, 15, Eigen::RowMajor>::Zero()),
      H_star_(Eigen::Matrix<double, 15, 15, Eigen::RowMajor>::Zero()),
      b_star0_(Eigen::Matrix<double, 15, 1>::Zero()),
      linearizationPoint_(Eigen::Matrix<double, 15, 1>::Zero()),
      JtJ_(Eigen::Matrix<double, 30, 30, Eigen::RowMajor>::Zero()),
      JtR_(Eigen::Matrix<double, 30, 1>::Zero())
{
  // Have an initial prior for the speed and biases and pose
  H_star_.block<6, 6>(0, 0) = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Identity(); // pose
  H_star_.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
  H_star_.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 1.0 / (0.03 * 0.03);
  H_star_.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 1.0 / (0.01 * 0.01);
  b_star0_ = Eigen::Matrix<double, 15, 1>::Zero();

  reductionoutput_ = (float **)calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  trackingResult_ =
      (TrackData **)calloc(sizeof(TrackData *) * GN_opt_iterations.size(), 1);

  l_D_ = (float **)calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  l_D_ref_ = (float *)calloc(sizeof(float) * (imgSize.x * imgSize.y), 1);
  r_D_kf_ = (float *)calloc(sizeof(float) * (imgSize.x * imgSize.y), 1);
  r_D_live_ = (float **)calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  r_D_render_ = (float **)calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  l_I_ = (float **)calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  r_I_ = (float **)calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  l_vertex_ = (float3 **)calloc(sizeof(float3 *) * GN_opt_iterations.size(), 1);
  l_vertex_ref_ = (float3 *)calloc(sizeof(float3) * (imgSize.x * imgSize.y), 1);
  r_vertex_kf_ = (float3 *)calloc(sizeof(float3) * (imgSize.x * imgSize.y), 1);
  r_Vertex_live_ = (float3 **)calloc(sizeof(float3 *) * GN_opt_iterations.size(), 1);
  r_Vertex_render_ = (float3 **)calloc(sizeof(float3 *) * GN_opt_iterations.size(), 1);
  l_normal_ = (float3 **)calloc(sizeof(float3 *) * GN_opt_iterations.size(), 1);
  l_gradx_ = (float **)calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  l_grady_ = (float **)calloc(sizeof(float *) * GN_opt_iterations.size(), 1);

  icp_cov_pyramid_ =
      (float3 **)calloc(sizeof(float3 *) * GN_opt_iterations.size(), 1);

  for (unsigned int level = 0; level < GN_opt_iterations.size(); ++level)
  {
    uint2 localimagesize = make_uint2(
        imgSize.x / (int)pow(2, level),
        imgSize.y / (int)pow(2, level));
    localImgSize_.push_back(localimagesize); // from fine to coarse
    reductionoutput_[level] = (float *)calloc(sizeof(float) * 8 * 32, 1);
    trackingResult_[level] = (TrackData *)calloc(
        2 * sizeof(TrackData) * imgSize.x * imgSize.y, 1);
    l_D_[level] = (float *)calloc(sizeof(float) * (imgSize.x * imgSize.y) / (int)pow(2, 2 * level), 1);
    r_D_live_[level] = (float *)calloc(sizeof(float) * (imgSize.x * imgSize.y) / (int)pow(2, level), 1);
    r_D_render_[level] = (float *)calloc(sizeof(float) * (imgSize.x * imgSize.y) / (int)pow(2, level), 1);
    l_I_[level] = (float *)calloc(sizeof(float) * (imgSize.x * imgSize.y) / (int)pow(2, 2 * level), 1);
    r_I_[level] = (float *)calloc(sizeof(float) * (imgSize.x * imgSize.y) / (int)pow(2, 2 * level), 1);
    l_vertex_[level] = (float3 *)calloc(sizeof(float3) * (imgSize.x * imgSize.y) / (int)pow(2, 2 * level), 1);
    r_Vertex_live_[level] = (float3 *)calloc(sizeof(float3) * (imgSize.x * imgSize.y) / (int)pow(2, level), 1);
    r_Vertex_render_[level] = (float3 *)calloc(sizeof(float3) * (imgSize.x * imgSize.y) / (int)pow(2, level), 1);
    l_normal_[level] = (float3 *)calloc(sizeof(float3) * (imgSize.x * imgSize.y) / (int)pow(2, 2 * level), 1);
    l_gradx_[level] = (float *)calloc(sizeof(float) * (imgSize.x * imgSize.y) / (int)pow(2, 2 * level), 1);
    l_grady_[level] = (float *)calloc(sizeof(float) * (imgSize.x * imgSize.y) / (int)pow(2, 2 * level), 1);
    icp_cov_pyramid_[level] = (float3 *)calloc(sizeof(float3) * (imgSize.x * imgSize.y) / (int)pow(2, 2 * level), 1);
  }

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

  if (using_ICP ^ using_RGB)
    stack_ = 1;
  if (using_ICP && using_RGB)
    stack_ = 2;

  //  if (this->robustWeight == RobustW::noweight){
  //    float mini_gradient_magintude[3] = {0.2 / 255., 0.2 / 255., 0.2 / 255.};
  //    float rgb_tracking_threshold[3] = {0.15, 0.15, 0.15};
  //    memcpy(this->mini_gradient_magintude_, mini_gradient_magintude,
  //           sizeof(float) * 3);
  //    memcpy(this->rgb_tracking_threshold_, rgb_tracking_threshold,
  //           sizeof(float) * 3);
  //
  //  }
  for (unsigned int level = 0; level < GN_opt_iterations_.size(); ++level)
  {
    cv::Mat outlier_mask = cv::Mat::zeros(localImgSize_[level].y,
                                          localImgSize_[level].x, CV_8UC1);
    no_outlier_mask_.push_back(outlier_mask);
  }
}

Tracking::~Tracking()
{
  for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i)
  {
    free(l_D_[i]);
    free(r_D_live_[i]);
    free(r_D_render_[i]);
    free(l_I_[i]);
    free(r_I_[i]);
    free(l_vertex_[i]);
    free(r_Vertex_live_[i]);
    free(r_Vertex_render_[i]);
    free(l_normal_[i]);
    free(l_gradx_[i]);
    free(l_grady_[i]);
    free(icp_cov_pyramid_[i]);
    free(trackingResult_[i]);
    free(reductionoutput_[i]);
  }
  free(l_D_);
  free(l_D_ref_);
  free(r_D_live_);
  free(r_D_render_);
  free(l_I_);
  free(r_I_);
  free(r_vertex_kf_);
  free(r_D_kf_);
  free(l_vertex_);
  free(l_vertex_ref_);
  free(r_Vertex_live_);
  free(r_Vertex_render_);
  free(l_normal_);
  free(l_gradx_);
  free(l_grady_);
  free(icp_cov_pyramid_);
  free(gaussian);
  free(trackingResult_);
  free(reductionoutput_);
}

// Get pure object masks excluding human
void Tracking::set_obj_mask(SegmentationResult human_out, cv::Mat &obj_mask, unsigned long long &mask_num)
{
  for (auto objPtr = human_out.pair_instance_seg_.begin();
       objPtr != human_out.pair_instance_seg_.end(); ++objPtr)
  {
    if (objPtr->second.class_id_ == 0)
      continue;
    cv::bitwise_or(objPtr->second.instance_mask_, obj_mask, obj_mask);
    if (objPtr->second.class_id_ != 255 && objPtr->second.class_id_ != 1)
      mask_num++;
  }
}

void Tracking::set_params_frame(const float4 k,
                                const float *f_l_I,
                                const float *f_l_D)
{

  k_ = k;
  K_ = getCameraMatrix(k);

  //  const float weight_rgb = 0.1f;  unused. use R^-1 instead
  //  const float weight_icp = 1.0f;  unused. use R^-1 instead

  // pre-compute for ICP tracking
  if (k.y < 0)
  { // ICL-NUIM
    obtainErrorParameters(Dataset::icl_nuim);
  }
  else
  { // TUM
    obtainErrorParameters(Dataset::tum_rgbd);
  }

  buildPyramid(f_l_I, f_l_D);

  for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level)
  {
    if (using_ICP_)
    {
      calc_icp_cov(icp_cov_pyramid_[level], localImgSize_[level], l_D_[level],
                   getCameraMatrix(k_ / (1 << level)), sigma_xy_, sigma_disparity_,
                   baseline_);
    }
  }

  outlier_mask_ = no_outlier_mask_;
  bg_outlier_mask_ = no_outlier_mask_;
}

void Tracking::set_params_frame_new(const float4 k,
                                    const float *f_l_I,
                                    const float *f_l_D,
                                    const cv::Mat &obj_mask)
{

  k_ = k;
  K_ = getCameraMatrix(k);

  // pre-compute for ICP tracking
  if (k.y < 0)
  { // ICL-NUIM
    obtainErrorParameters(Dataset::icl_nuim);
  }
  else
  { // TUM
    obtainErrorParameters(Dataset::tum_rgbd);
  }

  buildPyramid(f_l_I, f_l_D);

  for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level)
  {
    if (using_ICP_)
    {
      calc_icp_cov(icp_cov_pyramid_[level], localImgSize_[level], l_D_[level],
                   getCameraMatrix(k_ / (1 << level)), sigma_xy_, sigma_disparity_,
                   baseline_);
    }
  }

  outlier_mask_ = no_outlier_mask_;
  for (unsigned int level = 0; level < GN_opt_iterations_.size(); ++level)
  {
    // Set bg_outlier_mask
    cv::Mat bg_outlier_mask;
    cv::Size bg_mask_size = obj_mask.size() / ((1 << level));
    cv::resize(obj_mask, bg_outlier_mask, bg_mask_size);
    bg_outlier_mask_.push_back(bg_outlier_mask);
  }
}

void Tracking::set_params_frame(const float4 k, const float *f_l_I,
                                const float *f_l_D, const cv::Mat &human_mask)
{
  k_ = k;
  K_ = getCameraMatrix(k);

  //  const float weight_rgb = 0.1f;  unused. use R^-1 instead
  //  const float weight_icp = 1.0f;  unused. use R^-1 instead

  // pre-compute for ICP tracking
  if (k.y < 0)
  { // ICL-NUIM
    obtainErrorParameters(Dataset::icl_nuim);
  }
  else
  { // TUM
    obtainErrorParameters(Dataset::tum_rgbd);
  }

  buildPyramid(f_l_I, f_l_D);

  for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level)
  {
    if (using_ICP_)
    {
      calc_icp_cov(icp_cov_pyramid_[level], localImgSize_[level], l_D_[level],
                   getCameraMatrix(k_ / (1 << level)), sigma_xy_, sigma_disparity_,
                   baseline_);
    }
  }

  for (unsigned int level = 0; level < GN_opt_iterations_.size(); ++level)
  {
    cv::Mat outlier_mask;
    cv::Size mask_size = human_mask.size() / ((1 << level));
    cv::resize(human_mask, outlier_mask, mask_size);
    outlier_mask_.push_back(outlier_mask);
  }
}

void Tracking::set_params_frame(const float4 k,
                                const float *f_l_I,
                                const float *f_l_D,
                                const cv::Mat &human_mask,
                                const cv::Mat &obj_mask)
{
  k_ = k;
  K_ = getCameraMatrix(k);

  //  const float weight_rgb = 0.1f;  unused. use R^-1 instead
  //  const float weight_icp = 1.0f;  unused. use R^-1 instead

  // pre-compute for ICP tracking
  if (k.y < 0)
  { // ICL-NUIM
    obtainErrorParameters(Dataset::icl_nuim);
  }
  else
  { // TUM
    obtainErrorParameters(Dataset::tum_rgbd);
  }

  buildPyramid(f_l_I, f_l_D);

  for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level)
  {
    if (using_ICP_)
    {
      calc_icp_cov(icp_cov_pyramid_[level], localImgSize_[level], l_D_[level],
                   getCameraMatrix(k_ / (1 << level)), sigma_xy_, sigma_disparity_,
                   baseline_);
    }
  }

  for (unsigned int level = 0; level < GN_opt_iterations_.size(); ++level)
  {
    // Set full_outlier_mask
    cv::Mat outlier_mask;
    cv::Size mask_size = human_mask.size() / ((1 << level));
    cv::resize(human_mask, outlier_mask, mask_size);
    outlier_mask_.push_back(outlier_mask);

    // Set bg_outlier_mask
    cv::Mat bg_outlier_mask;
    cv::Size bg_mask_size = human_mask.size() / ((1 << level));
    cv::bitwise_or(human_mask, obj_mask, bg_outlier_mask);
    cv::resize(bg_outlier_mask, bg_outlier_mask, bg_mask_size);
    bg_outlier_mask_.push_back(bg_outlier_mask);
  }
}

bool Tracking::trackLiveFrame(Matrix4 &T_w_l,
                              const Matrix4 &T_w_r,
                              const float4 k,
                              const float3 *model_vertex,
                              const float3 *model_normal)
{
  // rendering the reference depth information from model
  if (using_RGB_ && (!use_live_depth_only_))
  {
    for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i)
    {
      Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
      if (i == 0)
      {
        vertex2depth(r_D_render_[0], model_vertex, imgSize_, T_w_r);
        depth2vertexKernel(r_Vertex_render_[0], r_D_render_[0],
                           localImgSize_[0], invK);
      }
      else
      {
        // using the rendered depth
        halfSampleRobustImageKernel(r_D_render_[i], r_D_render_[i - 1],
                                    localImgSize_[i - 1], e_delta * 3, 1);
        depth2vertexKernel(r_Vertex_render_[i], r_D_render_[i],
                           localImgSize_[i], invK);
      }
    }
  }

  Matrix4 pose_update;
  Matrix4 previous_pose = T_w_l;

  // coarse-to-fine iteration
  for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level)
  {
    float previous_error = INFINITY;
    for (int i = 0; i < GN_opt_iterations_[level]; ++i)
    {
      if (using_ICP_)
      {
        const Matrix4 projectReference = getCameraMatrix(k_) * inverse(T_w_r);
        track_ICP_Kernel(trackingResult_[level], imgSize_, l_vertex_[level],
                         l_normal_[level], localImgSize_[level], model_vertex,
                         model_normal, imgSize_, T_w_l, projectReference,
                         dist_threshold, normal_threshold,
                         icp_cov_pyramid_[level], outlier_mask_[level]);
      }
      if (using_RGB_)
      {
        // render reference image to live image -- opposite to the original function call
        if (use_live_depth_only_)
        {
          trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
                   imgSize_, r_Vertex_live_[level], r_Vertex_live_[level],
                   r_I_[level], localImgSize_[level],
                   l_I_[level], localImgSize_[level], l_gradx_[level],
                   l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
                   rgb_tracking_threshold_[level],
                   mini_gradient_magintude_[level], sigma_bright_,
                   prev_outlier_mask_[level]);
        }
        else
        {
          trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
                   imgSize_, r_Vertex_render_[level], r_Vertex_live_[level],
                   r_I_[level], localImgSize_[level],
                   l_I_[level], localImgSize_[level], l_gradx_[level],
                   l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
                   rgb_tracking_threshold_[level],
                   mini_gradient_magintude_[level], sigma_bright_,
                   prev_outlier_mask_[level]);
        }
      }

      if (using_ICP_ && (!using_RGB_))
      {
        reduceKernel(reductionoutput_[level], trackingResult_[level],
                     imgSize_, localImgSize_[level]);
      }

      if ((!using_ICP_) && using_RGB_)
      {
        reduceKernel(reductionoutput_[level],
                     trackingResult_[level] + imgSize_.x * imgSize_.y,
                     imgSize_, localImgSize_[level]);
      }

      if (using_ICP_ && using_RGB_)
      {
        reduceKernel(reductionoutput_[level], trackingResult_[level],
                     imgSize_, localImgSize_[level]);
      }

      const float current_error = reductionoutput_[level][0] / reductionoutput_[level][28];
      // std::cout<< "Level " << level << ", Iteration " << i << ", Error: " << current_error<<std::endl;

      if (current_error > (previous_error /* + 1e-1f*/))
      {
        // std::cout<< "Level " << level << " Iteration " << i << " - Cost increase from " << previous_error << " to " <<
        //         current_error<<std::endl;
        if (step_back_in_GN_)
        {
          T_w_l = previous_pose;
        }

        /*const Matrix4 projectReference = getCameraMatrix(k_) * inverse(T_w_r);
        track_ICP_Kernel(trackingResult_[level], imgSize_, l_vertex_[level],
                         l_normal_[level], localImgSize_[level], model_vertex,
                         model_normal, imgSize_, T_w_l, projectReference,
                         dist_threshold, normal_threshold, icp_cov_pyramid_[level]);
        reduceKernel(reductionoutput_[level], trackingResult_[level],
                     imgSize_, localImgSize_[level]);
        step_back_error = reductionoutput_[level][0]/reductionoutput_[level][28];
//        std::cout<< "Level " << level << " step back " << ", Error: " <<
//                 step_back_error<<std::endl;
        assert(step_back_error == previous_error);*/
        break;
      }
      previous_error = current_error;

      if (solvePoseKernel(pose_update, reductionoutput_[level],
                          icp_threshold))
      {
        // previous_pose = T_w_l;
        // T_w_l = pose_update * previous_pose;
        break;
      }

      previous_pose = T_w_l;
      T_w_l = pose_update * previous_pose;
      // printMatrix4("updated live pose", T_w_l);
    }
  }
  // check the pose issue
  //  bool tracked = checkPoseKernel(T_w_l, T_w_r, reductionoutput_, imgSize_,
  //                                 track_threshold);
  bool tracked = true;

  /*  const Matrix4 projectReference = getCameraMatrix(k_) * inverse(T_w_r);
    track_ICP_Kernel(trackingResult_[0], imgSize_, l_vertex_[0],
                     l_normal_[0], localImgSize_[0], model_vertex,
                     model_normal, imgSize_, T_w_l, projectReference,
                     dist_threshold, normal_threshold, icp_cov_pyramid_[0]);
    reduceKernel(reductionoutput_[0], trackingResult_[0],
                 imgSize_, localImgSize_[0]);
    final_error = reductionoutput_[0][0]/reductionoutput_[0][28];
  //  std::cout<< "Final Level " << ", Error: " << final_error<<std::endl;
    assert(step_back_error == final_error);*/
  return tracked;
}

float2 Tracking::obj_warp(float3 &c1_vertex_o1, float3 &o2_vertex,
                          const Matrix4 &T_c1_o1, const Matrix4 &T_c2_o2,
                          const Matrix4 &K, const float3 &c2_vertex)
{
  o2_vertex = inverse(T_c2_o2) * c2_vertex;
  c1_vertex_o1 = T_c1_o1 * o2_vertex;
  const float3 proj_vertex = rotate(K, c1_vertex_o1);
  return make_float2(proj_vertex.x / proj_vertex.z,
                     proj_vertex.y / proj_vertex.z);
}

bool Tracking::obj_icp_residual(float &residual, float3 &o1_refNormal_o1,
                                float3 &diff, const Matrix4 &T_w_o1,
                                const float3 o2_vertex, const float3 *c1_vertice_render,
                                const float3 *c1_Normals_render,
                                const uint2 &inSize, const float2 &proj_pixel)
{
  float3 w_refVertex_o1 = bilinear_interp(c1_vertice_render, inSize,
                                          proj_pixel);
  float3 w_refNormal_o1 = bilinear_interp(c1_Normals_render, inSize,
                                          proj_pixel);
  //  const uint2 refPixel = make_uint2(proj_pixel.x, proj_pixel.y);
  //  float3 w_refVertex_o1 =c1_vertice_render[refPixel.x + refPixel.y * inSize.x];
  //  float3 w_refNormal_o1 =c1_Normals_render[refPixel.x + refPixel.y * inSize.x];

  if (w_refNormal_o1.x == INVALID)
    return false;
  float3 o1_refVertex_o1 = inverse(T_w_o1) * w_refVertex_o1;
  o1_refNormal_o1 = rotate(inverse(T_w_o1), w_refNormal_o1);
  diff = o2_vertex - o1_refVertex_o1;
  residual = dot(o1_refNormal_o1, diff);
  return true;
}

void Tracking::track_Obj_ICP_Kernel(TrackData *output,
                                    const uint2 jacobian_size,
                                    const float3 *c2_vertice_live,
                                    const float3 *c2_normals_live,
                                    uint2 inSize,
                                    const float3 *c1_vertice_render,
                                    const float3 *c1_Normals_render,
                                    uint2 refSize,
                                    const Matrix4 &T_c2_o2, // to be estimated
                                    const Matrix4 &T_c1_o1,
                                    const Matrix4 &T_w_o1,
                                    const float4 k,
                                    const float dist_threshold,
                                    const float normal_threshold,
                                    const float3 *icp_cov_layer,
                                    const cv::Mat &outlier_mask)
{
  TICK();
  uint2 pixel = make_uint2(0, 0);
  unsigned int pixely, pixelx;
#pragma omp parallel for shared(output), private(pixel, pixelx, pixely)
  for (pixely = 0; pixely < inSize.y; pixely++)
  {
    for (pixelx = 0; pixelx < inSize.x; pixelx++)
    {

      pixel.x = pixelx;
      pixel.y = pixely;
      const unsigned idx = pixel.x + pixel.y * jacobian_size.x;
      TrackData &row = output[idx];

      if (outlier_mask.at<uchar>(pixely, pixelx) != 0)
      {
        row.result = -6;
        continue;
      }
      if (c2_normals_live[pixel.x + pixel.y * inSize.x].x == INVALID)
      {
        row.result = -1;
        continue;
      }

      const float3 c2_vertex = c2_vertice_live[pixel.x + pixel.y * inSize.x];
      if ((c2_vertex.x == 0) && (c2_vertex.y == 0) && (c2_vertex.z == 0))
      {
        row.result = -1;
        continue;
      }

      float3 o2_vertex, c1_vertex_o1;
      const float2 projpixel = obj_warp(c1_vertex_o1, o2_vertex, T_c1_o1,
                                        T_c2_o2, getCameraMatrix(k), c2_vertex);

      if (projpixel.x < 1 || projpixel.x > refSize.x - 2 ||
          projpixel.y < 1 || projpixel.y > refSize.y - 2 ||
          std::isnan(projpixel.x) || std::isnan(projpixel.y))
      {
        row.result = -2;
        continue;
      }

      const float3 c_normal = c2_normals_live[pixel.x + pixel.y * inSize.x];
      const float3 o2_normal = rotate(inverse(T_c2_o2), c_normal);

      float residual;
      float3 o1_refNormal_o1, diff;
      bool has_residual = obj_icp_residual(residual, o1_refNormal_o1, diff,
                                           T_w_o1, o2_vertex, c1_vertice_render, c1_Normals_render, refSize, projpixel);

      if (!has_residual)
      {
        row.result = -3;
        continue;
      }

      if (length(diff) > dist_threshold)
      {
        row.result = -4;
        continue;
      }
      if (dot(o2_normal, o1_refNormal_o1) < normal_threshold)
      {
        row.result = -5;
        continue;
      }

      // calculate the inverse of covariance as weights
      const float3 P = icp_cov_layer[pixel.x + pixel.y * inSize.x];
      const float sigma_icp = o1_refNormal_o1.x * o1_refNormal_o1.x * P.x + o1_refNormal_o1.y * o1_refNormal_o1.y * P.y + o1_refNormal_o1.z * o1_refNormal_o1.z * P.z;
      const float inv_cov = sqrtf(1.0 / sigma_icp);

      row.error = inv_cov * residual;

      //      float3 Jtrans = rotate(o1_refNormal_o1, transpose(T_c2_o2));
      float3 Jtrans = rotate(T_c2_o2, o1_refNormal_o1);
      ((float3 *)row.J)[0] = -1.0f * inv_cov * -1.0f * Jtrans;
      ((float3 *)row.J)[1] = /*-1.0f */ inv_cov * cross(c2_vertex, Jtrans);

      row.result = 1;
    }
  }
}

void Tracking::track_obj_RGB_kernel(TrackData *output, const uint2 jacobian_size,
                                    const float3 *r_vertices_render,
                                    const float3 *r_vertices_live, const float *r_image,
                                    uint2 r_size, const float *l_image, uint2 l_size,
                                    const float *l_gradx, const float *l_grady,
                                    const Matrix4 &T_c2_o2, // to be estimated
                                    const Matrix4 &T_c1_o1,
                                    const Matrix4 &K, const float residual_criteria,
                                    const float grad_threshold, const float sigma_bright,
                                    const cv::Mat &outlier_mask)
{
  TICK();
  uint2 r_pixel = make_uint2(0, 0);
  unsigned int r_pixely, r_pixelx;
#pragma omp parallel for shared(output), private(r_pixel, r_pixelx, r_pixely)
  for (r_pixely = 0; r_pixely < r_size.y; r_pixely++)
  {
    for (r_pixelx = 0; r_pixelx < r_size.x; r_pixelx++)
    {
      r_pixel.x = r_pixelx;
      r_pixel.y = r_pixely;

      TrackData &row = output[r_pixel.x + r_pixel.y * jacobian_size.x];

      if (outlier_mask.at<uchar>(r_pixely, r_pixelx) != 0)
      {
        row.result = -6;
        continue;
      }

      const int r_index = r_pixel.x + r_pixel.y * r_size.x;
      float3 r_vertex_render = r_vertices_render[r_index];
      const float3 r_vertex_live = r_vertices_live[r_index];

      // if rendered depth is not available
      if ((r_vertex_render.z <= 0.f) || (r_vertex_render.z == INVALID))
      {
        // if live depth is not availvle too =>depth error
        //        if (r_vertex_live.z <= 0.f ||r_vertex_live.z == INVALID) {
        row.result = -1;
        continue;
        //        }

        /*else{
//          if live depth is availvle, use live depth instead
//          would introduce occlusion however
          r_vertex_render = r_vertex_live;
        }*/
      }

      // if the difference between rendered and live depth is too large =>occlude
      if (length(r_vertex_render - r_vertex_live) > occluded_depth_diff_)
      {
        // not in the case that no live depth
        if (r_vertex_live.z > 0.f)
        {
          row.result = -3;
          continue;
        }
      }

      float3 o1_vertex, c2_vertex_o1;
      const float2 projpixel = obj_warp(c2_vertex_o1, o1_vertex,
                                        T_c2_o2, T_c1_o1, K, r_vertex_render);

      if (projpixel.x < 1 || projpixel.x > l_size.x - 2 || projpixel.y < 1 || projpixel.y > l_size.y - 2)
      {
        row.result = -2;
        continue;
      }

      const float residual = rgb_residual(r_image, r_pixel, r_size, l_image, projpixel, l_size);
      const float inv_cov = 1.0 / sigma_bright;
      bool gradValid = obj_rgb_jacobian(row.J, c2_vertex_o1, projpixel,
                                        l_gradx, l_grady, l_size, K, grad_threshold, inv_cov);
      // threshold small gradients
      //      if (gradValid == false) {
      //        row.result = -5;
      //        continue;
      //      }

      row.error = inv_cov * residual;

      //      if (row.error  * row.error > residual_criteria){
      ////        std::cout<<row.error<<std::endl;
      //        row.result = -4;
      //        continue;
      //      }

      row.result = 1;
    }
  }
}

void Tracking::trackEachObject(ObjectPointer &objectptr,
                               const float4 k,
                               const Matrix4 &T_w_c2,
                               const Matrix4 &T_w_c1,
                               const float *f_l_I,
                               const float *f_l_D)
{

  bool tracked = false;
  // Object pose in the last frame
  const Matrix4 T_wo1 = objectptr->volume_pose_;
  const float3 *w_V_m0 = objectptr->m_vertex;
  const float3 *w_N_m0 = objectptr->m_normal;

  Matrix4 T_c1_o1 = inverse(T_w_c1) * T_wo1;
  Matrix4 T_c2_o2 = inverse(T_w_c2) * T_wo1;

  if (!use_virtual_camera_)
  {

    if (using_RGB_ && (!use_live_depth_only_))
    {
      for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i)
      {
        Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
        if (i == 0)
        {
          vertex2depth(r_D_render_[0], w_V_m0, imgSize_, T_w_c1);
          depth2vertexKernel(r_Vertex_render_[0], r_D_render_[0],
                             localImgSize_[0], invK);
          // memcpy(r_Vertex_[0], model_vertex,
          //        sizeof(float3) * localImgSize_[0].x * localImgSize_[0].y);
        }
        else
        {
          // using the rendered depth
          halfSampleRobustImageKernel(r_D_render_[i], r_D_render_[i - 1],
                                      localImgSize_[i - 1], e_delta * 3, 1);
          depth2vertexKernel(r_Vertex_render_[i], r_D_render_[i],
                             localImgSize_[i], invK);
        }
      }
    }

    Matrix4 pose_update;
    Matrix4 previous_pose = T_c2_o2;

    // coarse-to-fine iteration
    for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level)
    {
      float previous_error = INFINITY;

      for (int i = 0; i < GN_opt_iterations_[level]; ++i)
      {
        if (using_ICP_)
        {
          track_Obj_ICP_Kernel(trackingResult_[level], imgSize_,
                               l_vertex_[level], l_normal_[level],
                               localImgSize_[level], w_V_m0, w_N_m0, imgSize_,
                               T_c2_o2, T_c1_o1, T_wo1, k,
                               dist_threshold,
                               normal_threshold, icp_cov_pyramid_[level],
                               outlier_mask_[level]);
        }

        if (using_RGB_)
        {
          track_obj_RGB_kernel(trackingResult_[level] + imgSize_.x * imgSize_.y,
                               imgSize_, r_Vertex_render_[level],
                               r_Vertex_live_[level], r_I_[level],
                               localImgSize_[level], l_I_[level], localImgSize_[level],
                               l_gradx_[level], l_grady_[level],
                               T_c2_o2, T_c1_o1,
                               getCameraMatrix(k_ / (1 << level)),
                               rgb_tracking_threshold_[level],
                               mini_gradient_magintude_[level],
                               sigma_bright_, prev_outlier_mask_[level]);
        }

        if (using_ICP_ && (!using_RGB_))
        {
          reduceKernel(reductionoutput_[level], trackingResult_[level],
                       imgSize_, localImgSize_[level]);
        }

        if ((!using_ICP_) && using_RGB_)
        {
          reduceKernel(reductionoutput_[level],
                       trackingResult_[level] + imgSize_.x * imgSize_.y,
                       imgSize_, localImgSize_[level]);
        }

        if (using_ICP_ && using_RGB_)
        {
          reduceKernel(reductionoutput_[level], trackingResult_[level],
                       imgSize_, localImgSize_[level]);
        }

        const float current_error =
            reductionoutput_[level][0] / reductionoutput_[level][28];
        //        std::cout<< "Level " << level << ", Iteration " << i << ", Error: " << current_error<<std::endl;

        if (reductionoutput_[level][28] == 0)
        {
          break;
        }

        if (current_error > (previous_error))
        {
          T_c2_o2 = previous_pose;
          //          std::cout<< "Level " << level << " Iteration " << i << " - Cost increase from " << previous_error << " to " <<
          //                   current_error<<std::endl;
          break;
        }
        previous_error = current_error;

        if (solvePoseKernel(pose_update, reductionoutput_[level],
                            icp_threshold))
        {
          break;
        }

        previous_pose = T_c2_o2;
        T_c2_o2 = pose_update * previous_pose;
      }
    }
    objectptr->volume_pose_ = T_w_c2 * T_c2_o2;
    objectptr->virtual_camera_pose_ = T_wo1 * inverse(T_c2_o2);
  }
  else
  {

    Matrix4 T_wc1_v = T_w_c1;
    trackLiveFrame(T_wc1_v, T_w_c1, k, w_V_m0, w_N_m0);

    objectptr->volume_pose_ = T_w_c1 * inverse(T_wc1_v) * T_wo1;
    objectptr->virtual_camera_pose_ = T_wc1_v;
  }
}

void Tracking::trackEachObjectFg(ObjectPointer &objectptr,
                                 const float4 k,
                                 const Matrix4 &T_w_c2,
                                 const Matrix4 &T_w_c1,
                                 const float *f_l_I,
                                 const float *f_l_D)
{

  bool tracked = false;
  // Object pose in the last frame
  const Matrix4 T_wo1 = objectptr->volume_pose_;
  const float3 *w_V_m0 = objectptr->m_vertex;
  const float3 *w_N_m0 = objectptr->m_normal;

  Matrix4 T_c1_o1 = inverse(T_w_c1) * T_wo1;
  Matrix4 T_c2_o2 = inverse(T_w_c2) * T_wo1;

  if (!use_virtual_camera_)
  {
    if (using_RGB_ && (!use_live_depth_only_))
    {
      for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i)
      {
        Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
        if (i == 0)
        {
          vertex2depth(r_D_render_[0], w_V_m0, imgSize_, T_w_c1);
          depth2vertexKernel(r_Vertex_render_[0], r_D_render_[0],
                             localImgSize_[0], invK);
          // memcpy(r_Vertex_[0], model_vertex,
          //        sizeof(float3) * localImgSize_[0].x * localImgSize_[0].y);
        }
        else
        {
          // using the rendered depth
          halfSampleRobustImageKernel(r_D_render_[i], r_D_render_[i - 1],
                                      localImgSize_[i - 1], e_delta * 3, 1);
          depth2vertexKernel(r_Vertex_render_[i], r_D_render_[i],
                             localImgSize_[i], invK);
        }
      }
    }

    Matrix4 pose_update;
    Matrix4 previous_pose = T_c2_o2;

    // Create outlier masks
    std::vector<cv::Mat> fg_outlier_masks;
    std::vector<cv::Mat> prev_fg_outlier_masks;

    // if (!objectptr->fg_outlier_mask_.empty()){
    //   cv::imshow("fg", objectptr->fg_outlier_mask_);
    // }
    // if (!objectptr->prev_fg_outlier_mask_.empty()){
    //   cv::imshow("prev_fg", objectptr->prev_fg_outlier_mask_);
    // }
    // cv::waitKey(0);

    for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i)
    {
      // Add mask to fg_outlier_masks
      cv::Mat outlier_mask;
      cv::Size mask_size = objectptr->fg_outlier_mask_.size() / ((1 << i));
      cv::resize(objectptr->fg_outlier_mask_, outlier_mask, mask_size);
      fg_outlier_masks.push_back(outlier_mask);

      // Add mask to prev_fg_outlier_masks
      cv::Mat prev_outlier_mask;
      cv::Size prev_mask_size = objectptr->prev_fg_outlier_mask_.size() / ((1 << i));
      if (objectptr->prev_fg_outlier_mask_.empty())
      {
        // cv::Mat prev_mask = cv::Mat::zeros(prev_mask_size[0], prev_mask_size[1], CV_8UC1);
        // no_outlier_mask_.push_back(prev_mask );
        prev_fg_outlier_masks = no_outlier_mask_;
      }
      else
      {
        cv::resize(objectptr->prev_fg_outlier_mask_, prev_outlier_mask, prev_mask_size);
        prev_fg_outlier_masks.push_back(prev_outlier_mask);
      }
    }

    bool using_icp = using_ICP_;
    bool using_rgb = using_RGB_;
    if (objectptr->rgb_only)
    {
      using_icp = false;
    }
    // std::cout << "using icp: " << using_icp << std::endl;

    // coarse-to-fine iteration
    for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level)
    {
      float previous_error = INFINITY;

      for (int i = 0; i < GN_opt_iterations_[level]; ++i)
      {
        if (using_icp)
        {
          track_Obj_ICP_Kernel(trackingResult_[level], imgSize_,
                               l_vertex_[level], l_normal_[level],
                               localImgSize_[level], w_V_m0, w_N_m0, imgSize_,
                               T_c2_o2, T_c1_o1, T_wo1, k,
                               dist_threshold,
                               normal_threshold, icp_cov_pyramid_[level],
                               fg_outlier_masks[level]);
        }

        if (using_rgb)
        {
          track_obj_RGB_kernel(trackingResult_[level] + imgSize_.x * imgSize_.y,
                               imgSize_, r_Vertex_render_[level],
                               r_Vertex_live_[level], r_I_[level],
                               localImgSize_[level], l_I_[level], localImgSize_[level],
                               l_gradx_[level], l_grady_[level],
                               T_c2_o2, T_c1_o1,
                               getCameraMatrix(k_ / (1 << level)),
                               rgb_tracking_threshold_[level],
                               mini_gradient_magintude_[level],
                               sigma_bright_, prev_fg_outlier_masks[level]);
        }

        if (using_icp && (!using_rgb))
        {
          reduceKernel(reductionoutput_[level], trackingResult_[level],
                       imgSize_, localImgSize_[level]);
        }

        if ((!using_icp) && using_rgb)
        {
          reduceKernel(reductionoutput_[level],
                       trackingResult_[level] + imgSize_.x * imgSize_.y,
                       imgSize_, localImgSize_[level]);
        }

        if (using_icp && using_rgb)
        {
          reduceKernel(reductionoutput_[level], trackingResult_[level],
                       imgSize_, localImgSize_[level]);
        }

        const float current_error =
            reductionoutput_[level][0] / reductionoutput_[level][28];

        // std::cout << "Level " << level << ", Iteration "
        //           << i << ", Error: " << current_error << std::endl;

        if (reductionoutput_[level][28] == 0)
        {
          break;
        }

        if (current_error > (previous_error))
        {
          // T_c2_o2 = previous_pose;
          // std::cout << "Level " << level << " Iteration " << i
          //           <<" - Cost increase from " << previous_error
          //           << " to " << current_error << std::endl;
          break;
        }
        previous_error = current_error;

        if (solvePoseKernel(pose_update, reductionoutput_[level], icp_threshold))
        {
          // previous_pose = T_c2_o2;
          // T_c2_o2 = pose_update * previous_pose;
          break;
        }

        previous_pose = T_c2_o2;
        T_c2_o2 = pose_update * previous_pose;
      }
    }
    objectptr->volume_pose_ = T_w_c2 * T_c2_o2;
    objectptr->virtual_camera_pose_ = T_wo1 * inverse(T_c2_o2);
  }
  else
  {

    Matrix4 T_wc1_v = T_w_c1;
    trackLiveFrame(T_wc1_v, T_w_c1, k, w_V_m0, w_N_m0);

    objectptr->volume_pose_ = T_w_c1 * inverse(T_wc1_v) * T_wo1;
    objectptr->virtual_camera_pose_ = T_wc1_v;
  }
}

bool Tracking::checkPose(Matrix4 &pose, const Matrix4 &oldPose)
{
  bool checked = checkPoseKernel(pose, oldPose, reductionoutput_[0], imgSize_,
                                 track_threshold);
  if (checked == false)
  {
    std::cout << "pose tracking is wrong, getting back to old pose" << std::endl;
  }
  return checked;
  //  return true;
}

TrackData *Tracking::getTrackingResult()
{
  // if (!using_ICP_) return trackingResult_[0] + imgSize_.x * imgSize_.y;
  // else
  return trackingResult_[0];
}

float **Tracking::getReductionOutput()
{
  return reductionoutput_;
}

void Tracking::obtainErrorParameters(const Dataset &dataset)
{

  // read from yaml file
  //   try {
  //     cv::FileStorage fNode(filename, cv::FileStorage::READ);
  //   }
  //   catch (...) {
  //     assert(0 && "YAML file not parsed correctly.");
  //   }
  //
  //   cv::FileStorage fNode(filename, cv::FileStorage::READ);
  //
  //   if (!fNode.isOpened()) {
  //     assert(0 && "YAML file was not opened.");
  //   }
  //
  //   sigma_b = fNode["sigma_brightness"];
  //   sigma_disp = fNode["sigma_disparity"];
  //   sigma_xy = fNode["sigma_xy"];
  //   baseline = fNode["baseline"];
  //   focal = fNode["focal_length"];

  // manual setting
  sigma_disparity_ = 1.0; // 5.5
  sigma_xy_ = 1.0;        // 5.5

  if (dataset == Dataset::zr300)
  { // zr300
    sigma_bright_ = 1.0f;
    baseline_ = 0.07f;
    focal_ = 617.164;
  }
  if (dataset == Dataset::asus)
  { // asus
    baseline_ = 0.6;
    focal_ = 580.0;
  }
  if (dataset == Dataset::icl_nuim)
  { // ICL-NUIM
    sigma_bright_ = sqrtf(100.0) / 255.0;
    baseline_ = 0.075;
    focal_ = 481.2;
  }
  if (dataset == Dataset::tum_rgbd)
  { // TUM datasets
    sigma_bright_ = sqrtf(100.0) / 255.0;
    // baseline_ = 0.075;
    // focal_ = 525.0;
    baseline_ = 0.1;
    focal_ = 383.432;
  }
}

void Tracking::buildPyramid(const float *l_I, const float *l_D)
{

  // half sample the coarse layers for input/reference rgb-d frames
  for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i)
  {
    if (i == 0)
    {
      // memory copy the first layer
      //      memcpy(l_I_[0], l_I, sizeof(float)*imgSize_.x*imgSize_.y);
      //      memcpy(l_D_[0], l_D, sizeof(float)*imgSize_.x*imgSize_.y);
      // bilateral filtering the input depth
      bilateralFilterKernel(l_I_[0], l_I, imgSize_, gaussian, 0.1f, 2);
      bilateralFilterKernel(l_D_[0], l_D, imgSize_, gaussian, 0.1f, 2);
    }
    else
    {
      halfSampleRobustImageKernel(l_D_[i], l_D_[i - 1], localImgSize_[i - 1],
                                  e_delta * 3, 1);
      halfSampleRobustImageKernel(l_I_[i], l_I_[i - 1], localImgSize_[i - 1],
                                  e_delta * 3, 1);
      // using the rendered depth
      /*
      if (using_RGB_ && use_rendered_depth_){
        halfSampleRobustImageKernel(r_D_[i], r_D_[i - 1], localImgSize_[i-1],
                                    e_delta * 3, 1);
      }
       */
    }

    // prepare the 3D information from the input depth maps
    Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
    depth2vertexKernel(l_vertex_[i], l_D_[i], localImgSize_[i], invK);
    if (k_.y < 0)
      vertex2normalKernel<FieldType, true>(l_normal_[i], l_vertex_[i],
                                           localImgSize_[i]);
    else
      vertex2normalKernel<FieldType, false>(l_normal_[i], l_vertex_[i],
                                            localImgSize_[i]);
    if (using_RGB_)
    {
      gradientsKernel(l_gradx_[i], l_grady_[i], l_I_[i], localImgSize_[i]);
      // depth2vertexKernel(r_Vertex_[i], r_D_[i], localImgSize_[i], invK);
      // comment out: rendered case: done in trackLive frame;
      // direct depth case, performed in last memcpy
    }
  }
}

/*
void Tracking::buildPreviousPyramid(const float* r_I, const float* r_D){

  //half sample the coarse layers for input/reference rgb-d frames
  for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i) {
    if (i == 0){
      // bilateral filtering the input depth
      bilateralFilterKernel(r_I_[0], r_I, imgSize_, gaussian, 0.1f, 2);
      bilateralFilterKernel(r_D_[0], r_D, imgSize_, gaussian, 0.1f, 2);
    }
    else{
      halfSampleRobustImageKernel(r_D_[i], r_D_[i - 1], localImgSize_[i-1],
                                  e_delta * 3, 1);
      halfSampleRobustImageKernel(r_I_[i], r_I_[i - 1],localImgSize_[i-1],
                                  e_delta * 3, 1);
    }

    // prepare the 3D information from the input depth maps
    Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
    depth2vertexKernel(r_Vertex_[i], r_D_[i], localImgSize_[i], invK);

  }
}
*/

void Tracking::calc_icp_cov(float3 *conv_icp, const uint2 size,
                            const float *depth_image, const Matrix4 &K,
                            const float sigma_xy, const float sigma_disp,
                            const float baseline)
{
  unsigned int pixely, pixelx;
  const float focal = K.data[0].x;
#pragma omp parallel for shared(conv_icp), private(pixelx, pixely)
  for (pixely = 0; pixely < size.y; pixely++)
  {
    for (pixelx = 0; pixelx < size.x; pixelx++)
    {
      float3 &cov = conv_icp[pixelx + pixely * size.x];

      // get depth
      float depth = depth_image[pixelx + pixely * size.x];

      // standard deviations
      cov.x = (depth / focal) * sigma_xy;
      cov.y = (depth / focal) * sigma_xy;
      cov.z = (depth * depth * sigma_disp) / (focal * baseline);

      // square to get the variances
      cov.x = cov.x * cov.x;
      cov.y = cov.y * cov.y;
      cov.z = cov.z * cov.z;
    }
  }
}

void Tracking::track_ICP_Kernel(TrackData *output,
                                const uint2 jacobian_size,
                                const float3 *inVertex,
                                const float3 *inNormal,
                                uint2 inSize,
                                const float3 *refVertex,
                                const float3 *refNormal,
                                uint2 refSize,
                                const Matrix4 &Ttrack,
                                const Matrix4 &view,
                                const float dist_threshold,
                                const float normal_threshold,
                                const float3 *icp_cov_layer)
{

  cv::Mat outlier_mask(cv::Size(inSize.x, inSize.y), CV_8UC1, cv::Scalar(0));
  //  deprecated, using the same name function below to enable using mask
  track_ICP_Kernel(output, jacobian_size,
                   inVertex, inNormal, inSize,
                   refVertex, refNormal, refSize,
                   Ttrack, view,
                   dist_threshold, normal_threshold,
                   icp_cov_layer,
                   outlier_mask);
}

void Tracking::trackRGB(TrackData *output, const uint2 jacobian_size,
                        const float3 *r_vertices_render,
                        const float3 *r_vertices_live, const float *r_image,
                        uint2 r_size, const float *l_image, uint2 l_size,
                        const float *l_gradx, const float *l_grady,
                        const Matrix4 &T_w_r, const Matrix4 &T_w_l,
                        const Matrix4 &K, const float residual_criteria,
                        const float grad_threshold, const float sigma_bright)
{

  cv::Mat outlier_mask(cv::Size(r_size.x, r_size.y), CV_8UC1, cv::Scalar(0));

  //  deprecated, using the same name function below to enable using mask
  trackRGB(output, jacobian_size,
           r_vertices_render, r_vertices_live, r_image, r_size,
           l_image, l_size,
           l_gradx, l_grady,
           T_w_r, T_w_l, K,
           residual_criteria, grad_threshold,
           sigma_bright, outlier_mask);
}

void Tracking::track_ICP_Kernel(TrackData *output,
                                const uint2 jacobian_size,
                                const float3 *inVertex,
                                const float3 *inNormal,
                                uint2 inSize,
                                const float3 *refVertex,
                                const float3 *refNormal,
                                uint2 refSize,
                                const Matrix4 &Ttrack,
                                const Matrix4 &view,
                                const float dist_threshold,
                                const float normal_threshold,
                                const float3 *icp_cov_layer,
                                const cv::Mat &outlier_mask)
{
  TICK();
  uint2 pixel = make_uint2(0, 0);
  unsigned int pixely, pixelx;
#pragma omp parallel for shared(output), private(pixel, pixelx, pixely)
  for (pixely = 0; pixely < inSize.y; pixely++)
  {
    for (pixelx = 0; pixelx < inSize.x; pixelx++)
    {
      pixel.x = pixelx;
      pixel.y = pixely;

      TrackData &row = output[pixel.x + pixel.y * jacobian_size.x];

      if (outlier_mask.at<uchar>(pixely, pixelx) != 0)
      {
        row.result = -6;
        continue;
      }

      if (inNormal[pixel.x + pixel.y * inSize.x].x == INVALID)
      {
        row.result = -1;
        continue;
      }

      const float3 projectedVertex = Ttrack * inVertex[pixel.x + pixel.y * inSize.x];
      const float3 projectedPos = view * projectedVertex;
      //      const float2 projPixel = make_float2(
      //          projectedPos.x / projectedPos.z + 0.5f,
      //          projectedPos.y / projectedPos.z + 0.5f);
      //      if (projPixel.x < 0 || projPixel.x > refSize.x - 1
      //          || projPixel.y < 0 || projPixel.y > refSize.y - 1) {
      //        row.result = -2;
      //        continue;
      //      }

      const float2 projPixel = make_float2(
          projectedPos.x / projectedPos.z,
          projectedPos.y / projectedPos.z);

      if (projPixel.x < 1 || projPixel.x > refSize.x - 2 || projPixel.y < 1 || projPixel.y > refSize.y - 2 ||
          std::isnan(projPixel.x) || std::isnan(projPixel.y))
      {
        row.result = -2;
        continue;
      }

      const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
      const float3 referenceNormal = refNormal[refPixel.x + refPixel.y * refSize.x];
      //      const float3 referenceNormal = bilinear_interp(refNormal, refSize, projPixel);
      if (referenceNormal.x == INVALID)
      {
        row.result = -3;
        continue;
      }

      const float3 diff = refVertex[refPixel.x + refPixel.y * refSize.x] - projectedVertex;
      //      const float3 diff = bilinear_interp(refVertex, refSize, projPixel) - projectedVertex;
      const float3 projectedNormal = rotate(Ttrack, inNormal[pixel.x + pixel.y * inSize.x]);

      if (length(diff) > dist_threshold)
      {
        row.result = -4;
        continue;
      }
      if (dot(projectedNormal, referenceNormal) < normal_threshold)
      {
        row.result = -5;
        continue;
      }
      row.result = 1;

      // calculate the inverse of covariance as weights
      const float3 P = icp_cov_layer[pixel.x + pixel.y * inSize.x];
      const float sigma_icp = referenceNormal.x * referenceNormal.x * P.x + referenceNormal.y * referenceNormal.y * P.y + referenceNormal.z * referenceNormal.z * P.z;
      const float inv_cov = sqrtf(1.0 / sigma_icp);

      row.error = inv_cov * dot(referenceNormal, diff);
      ((float3 *)row.J)[0] = inv_cov * referenceNormal;
      ((float3 *)row.J)[1] =
          inv_cov * cross(projectedVertex, referenceNormal);
    }
  }
  TOCK("trackKernel", inSize.x * inSize.y);
}

void Tracking::trackRGB(TrackData *output, const uint2 jacobian_size,
                        const float3 *r_vertices_render,
                        const float3 *r_vertices_live, const float *r_image,
                        uint2 r_size, const float *l_image, uint2 l_size,
                        const float *l_gradx, const float *l_grady,
                        const Matrix4 &T_w_r, const Matrix4 &T_w_l,
                        const Matrix4 &K, const float residual_criteria,
                        const float grad_threshold, const float sigma_bright,
                        const cv::Mat &outlier_mask)
{
  TICK();
  uint2 r_pixel = make_uint2(0, 0);
  unsigned int r_pixely, r_pixelx;
//  static const float sqrt_w = sqrtf(w);
#pragma omp parallel for shared(output), private(r_pixel, r_pixelx, r_pixely)
  for (r_pixely = 0; r_pixely < r_size.y; r_pixely++)
  {
    for (r_pixelx = 0; r_pixelx < r_size.x; r_pixelx++)
    {
      r_pixel.x = r_pixelx;
      r_pixel.y = r_pixely;

      TrackData &row = output[r_pixel.x + r_pixel.y * jacobian_size.x];
      if (outlier_mask.at<uchar>(r_pixely, r_pixelx) != 0)
      {
        row.result = -6;
        continue;
      }

      const int r_index = r_pixel.x + r_pixel.y * r_size.x;
      float3 r_vertex_render = r_vertices_render[r_index];
      const float3 r_vertex_live = r_vertices_live[r_index];

      // if rendered depth is not available
      if ((r_vertex_render.z <= 0.f) || (r_vertex_render.z == INVALID))
      {
        // if live depth is not availvle too =>depth error
        //        if (r_vertex_live.z <= 0.f ||r_vertex_live.z == INVALID) {
        row.result = -1;
        continue;
        //        }

        /*else{
//          if live depth is availvle, use live depth instead
//          would introduce occlusion however
          r_vertex_render = r_vertex_live;
        }*/
      }

      // if the difference between rendered and live depth is too large =>occlude
      if (length(r_vertex_render - r_vertex_live) > occluded_depth_diff_)
      {
        // not in the case that no live depth
        if (r_vertex_live.z > 0.f)
        {
          row.result = -3;
          //          std::cout<<r_vertex_render.z <<" "<<r_vertex_live.z<<std::endl;
          continue;
        }
      }

      float3 l_vertex, w_vertex;
      float2 proj_pixel = warp(l_vertex, w_vertex, T_w_r, T_w_l, K, r_vertex_render);
      if (proj_pixel.x < 1 || proj_pixel.x > l_size.x - 2 || proj_pixel.y < 1 || proj_pixel.y > l_size.y - 2 ||
          std::isnan(proj_pixel.x) || std::isnan(proj_pixel.y))
      {
        row.result = -2;
        continue;
      }

      const float residual = rgb_residual(r_image, r_pixel, r_size, l_image, proj_pixel, l_size);

      const float inv_cov = 1.0 / sigma_bright;

      bool gradValid = rgb_jacobian(row.J, l_vertex, w_vertex, T_w_l, proj_pixel, l_gradx, l_grady, l_size, K,
                                    grad_threshold, inv_cov);
      // threshold small gradients
      //      if (gradValid == false) {
      //        row.result = -5;
      //        continue;
      //      }
      row.error = inv_cov * residual;

      //      if (row.error  * row.error > residual_criteria){
      ////        std::cout<<row.error<<std::endl;
      //        row.result = -4;
      //        continue;
      //      }

      row.result = 1;
    }
  }
}

float2 Tracking::warp(float3 &l_vertex, float3 &w_vertex, const Matrix4 &T_w_r,
                      const Matrix4 &T_w_l, const Matrix4 &K, const float3 &r_vertex)
{
  w_vertex = T_w_r * r_vertex;
  l_vertex = inverse(T_w_l) * w_vertex;
  const float3 proj_vertex = rotate(K, l_vertex);
  return make_float2(proj_vertex.x / proj_vertex.z,
                     proj_vertex.y / proj_vertex.z);
}

bool Tracking::rgb_jacobian(float J[6], const float3 &l_vertex,
                            const float3 &w_vertex, const Matrix4 &T_w_l,
                            const float2 &proj_pixel, const float *l_gradx,
                            const float *l_grady, const uint2 &l_size,
                            const Matrix4 &K, const float grad_threshold, const float weight)
{

  const float gradx = bilinear_interp(l_gradx, l_size, proj_pixel);
  const float grady = bilinear_interp(l_grady, l_size, proj_pixel);
  const float grad_mag = length(make_float2(gradx, grady));

  if (grad_mag < grad_threshold)
    return false;

  const float fx = K.data[0].x;
  const float fy = K.data[1].y;

  float3 Jtrans = (1.f / l_vertex.z) * make_float3(gradx * fx, grady * fy,
                                                   -(gradx * l_vertex.x * fx + grady * l_vertex.y * fy) / l_vertex.z);

  Jtrans = -1.f * rotate(Jtrans, transpose(T_w_l));
  float3 Jrot = cross(w_vertex, Jtrans);

  /* Omitting the -1.f factor because in reduceKernel JTe is not
   * multiplied by -1.f. */
  ((float3 *)J)[0] = /* -1.f */ weight * Jtrans;
  ((float3 *)J)[1] = /* -1.f */ weight * Jrot;

  return true;
}

bool Tracking::obj_rgb_jacobian(float J[6], const float3 &l_vertex,
                                const float2 &proj_pixel, const float *l_gradx,
                                const float *l_grady, const uint2 &l_size,
                                const Matrix4 &K, const float grad_threshold,
                                const float weight)
{

  const float gradx = bilinear_interp(l_gradx, l_size, proj_pixel);
  const float grady = bilinear_interp(l_grady, l_size, proj_pixel);
  const float grad_mag = length(make_float2(gradx, grady));

  if (grad_mag < grad_threshold)
    return false;

  const float fx = K.data[0].x;
  const float fy = K.data[1].y;

  float3 Jtrans = (1.f / l_vertex.z) * make_float3(gradx * fx, grady * fy,
                                                   -(gradx * l_vertex.x * fx + grady * l_vertex.y * fy) / l_vertex.z);

  float3 Jrot = cross(l_vertex, Jtrans);

  /* Omitting the -1.f factor because in reduceKernel JTe is not
   * multiplied by -1.f. */
  ((float3 *)J)[0] = /* -1.f */ weight * Jtrans;
  ((float3 *)J)[1] = /* -1.f */ weight * Jrot;

  return true;
}

float Tracking::rgb_residual(const float *r_image, const uint2 &r_pixel,
                             const uint2 &r_size, const float *l_image,
                             const float2 &proj_pixel, const uint2 &l_size)
{
  float l_interpolated = bilinear_interp(l_image, l_size, proj_pixel);
  return (r_image[r_pixel.x + r_pixel.y * r_size.x] - l_interpolated);
}

void Tracking::reduceKernel(float *out, TrackData *J, const uint2 Jsize,
                            const uint2 size)
{
  TICK();
  int blockIndex;
  for (blockIndex = 0; blockIndex < 8; ++blockIndex)
    reduce(blockIndex, out, J, Jsize, size, this->stack_);

  TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out);
  for (int j = 1; j < 8; ++j)
  {
    values[0] += values[j];
    // std::cerr << "REDUCE ";for(int ii = 0; ii < 32;ii++)
    // std::cerr << values[0][ii] << " ";
    // std::cerr << "\n";
  }
  TOCK("reduceKernel", 512);
}

void Tracking::reduce(int blockIndex, float *out, TrackData *J, const uint2 Jsize, const uint2 size, const int stack)
{
  float *sums = out + blockIndex * 32;

  for (uint i = 0; i < 32; ++i)
    sums[i] = 0;
  float sums0, sums1, sums2, sums3, sums4, sums5, sums6, sums7, sums8, sums9,
      sums10, sums11, sums12, sums13, sums14, sums15, sums16, sums17,
      sums18, sums19, sums20, sums21, sums22, sums23, sums24, sums25,
      sums26, sums27, sums28, sums29, sums30, sums31;
  sums0 = 0.0f;
  sums1 = 0.0f;
  sums2 = 0.0f;
  sums3 = 0.0f;
  sums4 = 0.0f;
  sums5 = 0.0f;
  sums6 = 0.0f;
  sums7 = 0.0f;
  sums8 = 0.0f;
  sums9 = 0.0f;
  sums10 = 0.0f;
  sums11 = 0.0f;
  sums12 = 0.0f;
  sums13 = 0.0f;
  sums14 = 0.0f;
  sums15 = 0.0f;
  sums16 = 0.0f;
  sums17 = 0.0f;
  sums18 = 0.0f;
  sums19 = 0.0f;
  sums20 = 0.0f;
  sums21 = 0.0f;
  sums22 = 0.0f;
  sums23 = 0.0f;
  sums24 = 0.0f;
  sums25 = 0.0f;
  sums26 = 0.0f;
  sums27 = 0.0f;
  sums28 = 0.0f;
  sums29 = 0.0f;
  sums30 = 0.0f;
  sums31 = 0.0f;

// comment me out to try coarse grain parallelism
#pragma omp parallel for reduction(+ \
                                   : sums0, sums1, sums2, sums3, sums4, sums5, sums6, sums7, sums8, sums9, sums10, sums11, sums12, sums13, sums14, sums15, sums16, sums17, sums18, sums19, sums20, sums21, sums22, sums23, sums24, sums25, sums26, sums27, sums28, sums29, sums30, sums31)
  for (uint y = blockIndex; y < size.y; y += 8)
  {
    for (uint x = 0; x < size.x; x++)
    {
      for (int k = 0; k < stack; ++k)
      {

        const unsigned stacked_offset = k * (Jsize.x * Jsize.y);
        const TrackData &row = J[stacked_offset + (x + y * Jsize.x)]; // ...
        if (row.result < 1)
        {
          // accesses sums[28..31]
          /*(sums+28)[1]*/ sums29 += row.result == -4 ? 1 : 0;
          /*(sums+28)[2]*/ sums30 += row.result == -5 ? 1 : 0;
          /*(sums+28)[3]*/ sums31 += row.result > -4 ? 1 : 0;

          continue;
        }

        float irls_weight = calc_robust_weight(row.error, robustWeight_);

        // Error part
        /*sums[0]*/ sums0 += calc_robust_residual(row.error, robustWeight_);

        // JTe part
        /*for(int i = 0; i < 6; ++i)
          sums[i+1] += row.error * row.J[i];*/
        sums1 += irls_weight * row.error * row.J[0];
        sums2 += irls_weight * row.error * row.J[1];
        sums3 += irls_weight * row.error * row.J[2];
        sums4 += irls_weight * row.error * row.J[3];
        sums5 += irls_weight * row.error * row.J[4];
        sums6 += irls_weight * row.error * row.J[5];

        // JTJ part, unfortunatly the double loop is not unrolled well...
        /*(sums+7)[0]*/ sums7 += irls_weight * row.J[0] * row.J[0];
        /*(sums+7)[1]*/ sums8 += irls_weight * row.J[0] * row.J[1];
        /*(sums+7)[2]*/ sums9 += irls_weight * row.J[0] * row.J[2];
        /*(sums+7)[3]*/ sums10 += irls_weight * row.J[0] * row.J[3];

        /*(sums+7)[4]*/ sums11 += irls_weight * row.J[0] * row.J[4];
        /*(sums+7)[5]*/ sums12 += irls_weight * row.J[0] * row.J[5];

        /*(sums+7)[6]*/ sums13 += irls_weight * row.J[1] * row.J[1];
        /*(sums+7)[7]*/ sums14 += irls_weight * row.J[1] * row.J[2];
        /*(sums+7)[8]*/ sums15 += irls_weight * row.J[1] * row.J[3];
        /*(sums+7)[9]*/ sums16 += irls_weight * row.J[1] * row.J[4];

        /*(sums+7)[10]*/ sums17 += irls_weight * row.J[1] * row.J[5];

        /*(sums+7)[11]*/ sums18 += irls_weight * row.J[2] * row.J[2];
        /*(sums+7)[12]*/ sums19 += irls_weight * row.J[2] * row.J[3];
        /*(sums+7)[13]*/ sums20 += irls_weight * row.J[2] * row.J[4];
        /*(sums+7)[14]*/ sums21 += irls_weight * row.J[2] * row.J[5];

        /*(sums+7)[15]*/ sums22 += irls_weight * row.J[3] * row.J[3];
        /*(sums+7)[16]*/ sums23 += irls_weight * row.J[3] * row.J[4];
        /*(sums+7)[17]*/ sums24 += irls_weight * row.J[3] * row.J[5];

        /*(sums+7)[18]*/ sums25 += irls_weight * row.J[4] * row.J[4];
        /*(sums+7)[19]*/ sums26 += irls_weight * row.J[4] * row.J[5];

        /*(sums+7)[20]*/ sums27 += irls_weight * row.J[5] * row.J[5];

        // extra info here
        /*(sums+28)[0]*/ sums28 += 1;
      }
    }
  }
  sums[0] = sums0;
  sums[1] = sums1;
  sums[2] = sums2;
  sums[3] = sums3;
  sums[4] = sums4;
  sums[5] = sums5;
  sums[6] = sums6;
  sums[7] = sums7;
  sums[8] = sums8;
  sums[9] = sums9;
  sums[10] = sums10;
  sums[11] = sums11;
  sums[12] = sums12;
  sums[13] = sums13;
  sums[14] = sums14;
  sums[15] = sums15;
  sums[16] = sums16;
  sums[17] = sums17;
  sums[18] = sums18;
  sums[19] = sums19;
  sums[20] = sums20;
  sums[21] = sums21;
  sums[22] = sums22;
  sums[23] = sums23;
  sums[24] = sums24;
  sums[25] = sums25;
  sums[26] = sums26;
  sums[27] = sums27;
  sums[28] = sums28;
  sums[29] = sums29;
  sums[30] = sums30;
  sums[31] = sums31;
}

bool Tracking::solvePoseKernel(Matrix4 &pose_update, const float *output,
                               float icp_threshold)
{
  bool res = false;
  TICK();
  // Update the pose regarding the tracking result
  TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
  TooN::Vector<6> x = solve(values[0].slice<1, 27>());
  TooN::SE3<> delta(x);
  pose_update = toMatrix4(delta);

  // Return validity test result of the tracking
  if ((norm(x) < icp_threshold) && (std::sqrt(output[0] / output[28]) < 2e-3))
  {
    //    std::cout<<"updating done, jump of normal equation solving iteration"<<std::endl;
    res = true;
  }

  //    std::cout<<"updating pose, with pertubation: "<<norm(x)<<" residual: "<<(std::sqrt(output[0]/output[28]))
  //             <<std::endl;

  TOCK("updatePoseKernel", 1);
  return res;
}

bool Tracking::checkPoseKernel(Matrix4 &pose, const Matrix4 &oldPose,
                               const float *output, const uint2 imageSize,
                               const float track_threshold)
{

  // Check the tracking result, and go back to the previous camera position if necessary

  TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);

  bool is_residual_high = (std::sqrt(values(0, 0) / values(0, 28)) > 2e-1);
  bool is_trackPoints_few = (values(0, 28) / (imageSize.x * imageSize.y) < track_threshold);

  if (is_residual_high || is_trackPoints_few)
  {
    pose = oldPose;

    //      std::cerr<<"tracking fails. residual: "<<(std::sqrt(values(0, 0) / values(0, 28) ))<<"pass? "<<is_residual_high<<" tracking "
    //          "points: "<<(values(0, 28) / (imageSize.x * imageSize.y))<<"pass? "<<is_trackPoints_few<<std::endl;
    return false;
  }
  else
  {
    //      std::cout<<"pose updating checked"<<std::endl;
    return true;
  }
}

void Tracking::setRefImgFromCurr()
{
  if (using_RGB_)
  {
    for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level)
    {
      memcpy(r_I_[level], l_I_[level],
             sizeof(float) * localImgSize_[level].x * localImgSize_[level].y);
      memcpy(r_D_live_[level], l_D_[level],
             sizeof(float) * localImgSize_[level].x * localImgSize_[level].y);
      memcpy(r_Vertex_live_[level], l_vertex_[level],
             sizeof(float3) * localImgSize_[level].x * localImgSize_[level].y);
    }
    prev_outlier_mask_ = outlier_mask_;
    prev_bg_outlier_mask_ = bg_outlier_mask_;
  }
  outlier_mask_.clear();
  bg_outlier_mask_.clear();
}

void Tracking::maskClear()
{
  outlier_mask_.clear();
  bg_outlier_mask_.clear();
}

/*
 * backward rendering
 * warp the live image to the ref image position based on the T_w_l and T_w_r
 * and then calculate the residual image between warped image and ref image
 */

void Tracking::warp_to_residual(cv::Mat &warped_image, cv::Mat &residual,
                                const float *l_image, const float *ref_image,
                                const float3 *r_vertex, const Matrix4 &T_w_l,
                                const Matrix4 &T_w_r, const Matrix4 &K, const uint2 outSize)
{

  residual = cv::Mat::zeros(cv::Size(outSize.x, outSize.y), CV_8UC1);
  warped_image = cv::Mat::zeros(cv::Size(outSize.x, outSize.y), CV_8UC1);

  unsigned int y;
#pragma omp parallel for shared(warped_image, residual), private(y)
  for (y = 0; y < outSize.y; y++)
    for (unsigned int x = 0; x < outSize.x; x++)
    {
      const float3 vertex = r_vertex[x + y * outSize.x];
      if (vertex.z == 0.f || vertex.z == INVALID)
      {
        continue;
      }

      float3 l_vertex, w_vertex;
      float2 projpixel = warp(l_vertex, w_vertex, T_w_r, T_w_l, K,
                              vertex);
      if (projpixel.x < 1 || projpixel.x > outSize.x - 2 || projpixel.y < 1 || projpixel.y > outSize.y - 2)
      {
        continue;
      }

      // the corresponding pixel of (x, y) on the warped_image is the
      //  projpixel on the input_image
      float interpolated = bilinear_interp(l_image, outSize, projpixel);
      warped_image.at<uchar>(y, x) = 255.0 * interpolated;

      float ref_intensity = ref_image[x + y * outSize.x];
      residual.at<uchar>(y, x) = 255.0 * fabs(ref_intensity - interpolated);
    }
}

void Tracking::dump_residual(TrackData *res_data, const uint2 inSize)
{
  std::ofstream residual_file;
  residual_file.open("residual.csv");
  unsigned int pixely, pixelx;
  for (pixely = 0; pixely < inSize.y; pixely++)
  {
    for (pixelx = 0; pixelx < inSize.x; pixelx++)
    {
      int id = pixelx + pixely * inSize.x;
      if (res_data[id].result < 1)
        continue;
      residual_file << sq(res_data[id].error) << ",\n";
    }
  }
  residual_file.close();
}

float Tracking::calc_robust_residual(const float residual,
                                     const RobustW &robustfunction)
{
  double loss = 0.f;
  switch (robustfunction)
  {
  case RobustW::noweight:
    loss = 0.5 * residual * residual;
    break;
  case RobustW::Huber:
    if (fabsf(residual) <= huber_k)
    {
      loss = 0.5 * residual * residual;
      break;
    }
    else
    {
      loss = huber_k * (fabsf(residual) - 0.5 * huber_k);
      break;
    }
  case RobustW::Tukey:
    if (fabsf(residual) <= tukey_b)
    {
      loss = tukey_b * tukey_b / 6.0 * (1.0 - pow(1.0 - sq(residual / tukey_b), 3));
      break;
    }
    else
    {
      loss = tukey_b * tukey_b / 6.0;
      break;
    }
  case RobustW::Cauchy:
    loss = 0.5 * cauchy_k * cauchy_k * log(1.0 + sq(residual / cauchy_k));
    break;
  }
  return static_cast<float>(loss);
}

float Tracking::calc_robust_weight(const float residual,
                                   const RobustW &robustfunction)
{
  double irls_weight = 1.0f;
  switch (robustfunction)
  {
  case RobustW::noweight:
    irls_weight = 1.0f;
    break;
  case RobustW::Huber:
    if (fabsf(residual) <= huber_k)
    {
      irls_weight = 1.0f;
      break;
    }
    else
    {
      irls_weight = huber_k / fabsf(residual);
      break;
    }
  case RobustW::Tukey:
    if (fabsf(residual) <= tukey_b)
    {
      irls_weight = sq(1.0 - sq(residual) / sq(tukey_b));
      break;
    }
    else
    {
      irls_weight = 0;
      break;
    }
  case RobustW::Cauchy:
    irls_weight = 1 / (1 + sq(residual / cauchy_k));
    break;
  }
  return static_cast<float>(irls_weight);
}

void Tracking::compute_residuals(const Matrix4 &T_w_l,
                                 const Matrix4 &T_w_r,
                                 const float3 *w_refVertices_r,
                                 const float3 *w_refNormals_r,
                                 const float3 *w_refVertices_l,
                                 const bool computeICP,
                                 const bool computeRGB,
                                 const float threshold)
{

  // Overuse the r_D_ and r_Vertex_ for rendered live depth and vertices
  Matrix4 invK = getInverseCameraMatrix(k_);
  vertex2depth(l_D_ref_, w_refVertices_l, imgSize_, T_w_l);
  // vertex2depth(l_D_ref_, w_refVertices_l, imgSize_, T_w_r);
  depth2vertexKernel(l_vertex_ref_, l_D_ref_, localImgSize_[0], invK);

  compute_residual_kernel(trackingResult_[0], imgSize_, T_w_r, T_w_l, k_,
                          r_I_[0], l_I_[0], l_vertex_[0], l_normal_[0],
                          w_refVertices_r, w_refNormals_r, l_vertex_ref_,
                          computeICP, computeRGB, icp_cov_pyramid_[0],
                          sigma_bright_, threshold);
}

void Tracking::compute_residual_kernel(TrackData *res_data,
                                       const uint2 img_size,
                                       const Matrix4 &T_w_r,
                                       const Matrix4 &T_w_l,
                                       const float4 k,
                                       const float *r_image,
                                       const float *l_image,
                                       const float3 *l_vertices,
                                       const float3 *l_normals,
                                       const float3 *w_refVertices_r,
                                       const float3 *w_refNormals_r,
                                       const float3 *l_refVertices_l,
                                       const bool computeICP,
                                       const bool computeRGB,
                                       const float3 *icp_cov_layer,
                                       const float sigma_bright,
                                       const float threshold)
{
  TICK();
  uint2 pixel = make_uint2(0, 0);
  const Matrix4 K = getCameraMatrix(k);
  unsigned int pixely, pixelx;
#pragma omp parallel for shared(res_data), private(pixel, pixelx, pixely)
  for (pixely = 0; pixely < img_size.y; pixely++)
  {
    for (pixelx = 0; pixelx < img_size.x; pixelx++)
    {
      pixel.x = pixelx;
      pixel.y = pixely;

      const int in_index = pixel.x + pixel.y * img_size.x;

      if (outlier_mask_[0].at<uchar>(pixel.y, pixel.x) != 0)
      {
        res_data[in_index].result = -6;
        res_data[in_index + img_size.x * img_size.y].result = -6;
        continue;
      }
      if (prev_outlier_mask_[0].at<uchar>(pixel.y, pixel.x) != 0)
      {
        res_data[in_index].result = -6;
        res_data[in_index + img_size.x * img_size.y].result = -6;
        continue;
      }

      // ICP
      if (computeICP)
      {
        TrackData &icp_row = res_data[in_index];
        compute_icp_residual_kernel(icp_row, in_index, img_size, T_w_l,
                                    T_w_r, K, l_vertices, l_normals,
                                    w_refVertices_r, w_refNormals_r,
                                    icp_cov_layer, threshold);
      }

      // RGB residual
      if (computeRGB)
      {
        TrackData &rgb_row = res_data[in_index + img_size.x * img_size.y];
        compute_rgb_residual_kernel(rgb_row, pixel, img_size, T_w_l, T_w_r, K,
                                    r_image, l_image, l_refVertices_l,
                                    l_vertices, sigma_bright, threshold);
      }
    }
  }
}

void Tracking::compute_icp_residual_kernel(TrackData &icp_row,
                                           const int in_index,
                                           const uint2 img_size,
                                           const Matrix4 &T_w_l,
                                           const Matrix4 &T_w_r,
                                           const Matrix4 &K,
                                           const float3 *l_vertices,
                                           const float3 *l_normals,
                                           const float3 *w_refVertices_r,
                                           const float3 *w_refNormals_r,
                                           const float3 *icp_cov_layer,
                                           const float threshold)
{
  if (l_normals[in_index].x == INVALID)
  {
    icp_row.result = -1;
    return;
  }

  float3 w_vertex, r_vertex;
  const float3 l_vertex = l_vertices[in_index];
  float2 proj_pixel = warp(r_vertex, w_vertex, T_w_l, T_w_r, K, l_vertex);
  if (proj_pixel.x < 1 || proj_pixel.x > img_size.x - 2 || proj_pixel.y < 1 || proj_pixel.y > img_size.y - 2)
  {
    icp_row.result = -2;
    return;
  }

  const uint2 ref_pixel = make_uint2(proj_pixel.x, proj_pixel.y);
  const int ref_index = ref_pixel.x + ref_pixel.y * img_size.x;
  const float3 w_refnormal_r = w_refNormals_r[ref_index];
  const float3 w_refvertex_r = w_refVertices_r[ref_index];

  if (w_refnormal_r.x == INVALID)
  {
    icp_row.result = -3;
    return;
  }

  const float3 diff = w_refvertex_r - w_vertex;
  const float3 w_normal = rotate(T_w_l, l_normals[in_index]);

  if (length(diff) > dist_threshold)
  {
    icp_row.result = -4;
    return;
  }
  if (dot(w_normal, w_refnormal_r) < normal_threshold)
  {
    icp_row.result = -5;
    return;
  }

  // calculate the inverse of covariance as weights
  const float3 P = icp_cov_layer[in_index];
  const double sigma_icp = w_refnormal_r.x * w_refnormal_r.x * P.x + w_refnormal_r.y * w_refnormal_r.y * P.y + w_refnormal_r.z * w_refnormal_r.z * P.z;
  const double inv_cov = sqrt(1.0 / sigma_icp);

  const double icp_error = inv_cov * dot(w_refnormal_r, diff);

  if (fabs(icp_error) > threshold)
  {
    icp_row.result = -4;
    return;
  }

  icp_row.error = static_cast<float>(icp_error);
  icp_row.result = 1;
}

void Tracking::compute_rgb_residual_kernel(TrackData &rgb_row,
                                           const uint2 in_pixel,
                                           const uint2 img_size,
                                           const Matrix4 &T_w_l,
                                           const Matrix4 &T_w_r,
                                           const Matrix4 &K,
                                           const float *r_image,
                                           const float *l_image,
                                           const float3 *l_refVertices_l,
                                           const float3 *l_Vertices_l,
                                           const float sigma_bright,
                                           const float threshold)
{

  const int in_index = in_pixel.x + in_pixel.y * img_size.x;
  float3 l_vertex_ref = l_refVertices_l[in_index];
  const float3 l_vertex_live = l_Vertices_l[in_index];

  if (l_vertex_ref.z <= 0.f || l_vertex_ref.z == INVALID)
  {
    if (fill_in_missing_depth_rgb_residual_checking_)
    {
      if (l_vertex_live.z <= 0.f || l_vertex_live.z == INVALID)
      {
        rgb_row.result = -1;
        return;
      }
      else
      {
        l_vertex_ref = l_vertex_live;
      }
    }
    else
    {
      rgb_row.result = -1;
      return;
    }
  }

  if (length(l_vertex_ref - l_vertex_live) > occluded_depth_diff_)
  {
    // not in the case that no live depth
    if (l_vertex_live.z > 0.f)
    {
      rgb_row.result = -5;
      return;
    }
  }

  float3 r_vertex, w_vertex;
  float2 proj_pixel = warp(r_vertex, w_vertex, T_w_l, T_w_r, K, l_vertex_ref);
  if (proj_pixel.x < 1 || proj_pixel.x > img_size.x - 2 || proj_pixel.y < 1 || proj_pixel.y > img_size.y - 2)
  {
    rgb_row.result = -2;
    return;
  }

  rgb_row.result = 1;
  const float residual = rgb_residual(l_image, in_pixel, img_size,
                                      r_image, proj_pixel, img_size);

  const double inv_cov = 1.0 / sigma_bright;
  const double rgb_error = inv_cov * residual;

  if (fabs(rgb_error) > threshold)
  {
    rgb_row.result = -4;
    return;
  }

  rgb_row.error = static_cast<float>(rgb_error);
  rgb_row.result = 1;
}

bool Tracking::trackLiveFrame(Matrix4 &T_w_l,
                              const Matrix4 &T_w_r,
                              const float4 k,
                              const float3 *model_vertex,
                              const float3 *model_normal,
                              const cv::Mat &mask)
{

  //  T_w_l = T_w_r;  //initilize the new live pose

  //  //rendering the reference depth information from model
  if (using_RGB_ && (!use_live_depth_only_))
  {
    for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i)
    {
      Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
      if (i == 0)
      {
        vertex2depth(r_D_render_[0], model_vertex, imgSize_, T_w_r);
        depth2vertexKernel(r_Vertex_render_[0], r_D_render_[0],
                           localImgSize_[0], invK);

        // memcpy(r_Vertex_[0], model_vertex,
        //        sizeof(float3) * localImgSize_[0].x * localImgSize_[0].y);
      }
      else
      {
        // using the rendered depth
        halfSampleRobustImageKernel(r_D_render_[i], r_D_render_[i - 1],
                                    localImgSize_[i - 1], e_delta * 3, 1);
        depth2vertexKernel(r_Vertex_render_[i], r_D_render_[i],
                           localImgSize_[i], invK);
      }
    }
  }

  std::vector<cv::Mat> outlier_masks;
  for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i)
  {
    cv::Mat outlier_mask;
    cv::Size mask_size = mask.size() / ((1 << i));
    cv::resize(mask, outlier_mask, mask_size);
    outlier_masks.push_back(outlier_mask);
  }

  Matrix4 pose_update;

  Matrix4 previous_pose = T_w_l;

  // coarse-to-fine iteration
  for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level)
  {
    float previous_error = INFINITY;
    for (int i = 0; i < GN_opt_iterations_[level]; ++i)
    {
      if (using_ICP_)
      {
        const Matrix4 projectReference = getCameraMatrix(k_) * inverse(T_w_r);
        // Original
        track_ICP_Kernel(trackingResult_[level], imgSize_, l_vertex_[level],
                         l_normal_[level], localImgSize_[level], model_vertex,
                         model_normal, imgSize_, T_w_l, projectReference,
                         dist_threshold, normal_threshold,
                         icp_cov_pyramid_[level], outlier_masks[level]);

        // Used for bg tracking
        // track_ICP_Kernel(trackingResult_[level], imgSize_, l_vertex_[level],
        //                  l_normal_[level], localImgSize_[level], model_vertex,
        //                  model_normal, imgSize_, T_w_l, projectReference,
        //                  dist_threshold, normal_threshold,
        //                  icp_cov_pyramid_[level], dynamic_obj_mask_[level]);
      }

      if (using_RGB_)
      {
        // render reference image to live image -- opposite to the original function call
        if (use_live_depth_only_)
        {
          trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
                   imgSize_, r_Vertex_live_[level], r_Vertex_live_[level],
                   r_I_[level], localImgSize_[level],
                   l_I_[level], localImgSize_[level], l_gradx_[level],
                   l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
                   rgb_tracking_threshold_[level],
                   mini_gradient_magintude_[level], sigma_bright_,
                   prev_outlier_mask_[level]);

          // trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
          //          imgSize_, r_Vertex_live_[level],r_Vertex_live_[level],
          //          r_I_[level], localImgSize_[level],
          //          l_I_[level], localImgSize_[level], l_gradx_[level],
          //          l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
          //          rgb_tracking_threshold_[level],
          //          mini_gradient_magintude_[level], sigma_bright_,
          //          prev_dynamic_obj_mask_[level]);
        }
        else
        {
          trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
                   imgSize_, r_Vertex_render_[level], r_Vertex_live_[level],
                   r_I_[level], localImgSize_[level],
                   l_I_[level], localImgSize_[level], l_gradx_[level],
                   l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
                   rgb_tracking_threshold_[level],
                   mini_gradient_magintude_[level], sigma_bright_,
                   prev_outlier_mask_[level]);

          // trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
          //          imgSize_, r_Vertex_render_[level],r_Vertex_live_[level],
          //          r_I_[level], localImgSize_[level],
          //          l_I_[level], localImgSize_[level], l_gradx_[level],
          //          l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
          //          rgb_tracking_threshold_[level],
          //          mini_gradient_magintude_[level], sigma_bright_,
          //          prev_dynamic_obj_mask_[level]);
        }
      }

      if (using_ICP_ && (!using_RGB_))
      {
        reduceKernel(reductionoutput_[level], trackingResult_[level],
                     imgSize_, localImgSize_[level]);
      }

      if ((!using_ICP_) && using_RGB_)
      {
        reduceKernel(reductionoutput_[level],
                     trackingResult_[level] + imgSize_.x * imgSize_.y,
                     imgSize_, localImgSize_[level]);
      }

      if (using_ICP_ && using_RGB_)
      {
        reduceKernel(reductionoutput_[level], trackingResult_[level],
                     imgSize_, localImgSize_[level]);
      }

      const float current_error =
          reductionoutput_[level][0] / reductionoutput_[level][28];
      //      std::cout<< "Level " << level << ", Iteration " << i << ", Error: " << current_error<<std::endl;

      if (current_error > (previous_error /*+ 1e-5f*/))
      {
        //        std::cout<< "Level " << level << " Iteration " << i << " - Cost increase from " << previous_error << " to " <<
        //                 current_error<<std::endl;
        if (step_back_in_GN_)
        {
          T_w_l = previous_pose;
        }
        break;
      }
      previous_error = current_error;

      if (solvePoseKernel(pose_update, reductionoutput_[level],
                          icp_threshold))
      {
        //        previous_pose = T_w_l;
        //        T_w_l = pose_update * previous_pose;
        break;
      }

      previous_pose = T_w_l;
      T_w_l = pose_update * previous_pose;
      //      printMatrix4("updated live pose", T_w_l);
    }
  }

  //  check the pose issue
  //  bool tracked = checkPoseKernel(T_w_l, T_w_r, reductionoutput_[0], imgSize_,
  //                                 track_threshold);
  bool tracked = true;

  return tracked;
}

// deprecated
// bool Tracking::trackLiveFrameWithImu(Matrix4& T_w_l,
//                                      Matrix4& T_w_r,
//                                      LocalizationState &state,
//                                      const float4 k,
//                                      const float3* model_vertex,
//                                      const float3* model_normal,
//                                      okvis::ImuMeasurementDeque &imuData,
//                                      okvis::Time &prevTime,
//                                      okvis::Time &currTime,
//                                      okvis::ImuParameters &imuParameters,
//                                      bool useImu){

//   std::cout << "Tracking live frame with IMU..." << std::endl;

//   // Initialize the incremental rotation and translation to apply to the previous pose
//   Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_W_C0 = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
//   Eigen::Matrix<double, 3, 3, Eigen::RowMajor> C_W_C0 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();
//   Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_C0_W = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
//   Eigen::Matrix<double, 3, 3, Eigen::RowMajor> C_C0_W = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();
//   Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_W_C1 = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
//   Eigen::Matrix<double, 3, 3, Eigen::RowMajor> C_W_C1 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();
//   Eigen::Vector3d W_r_W_C0(0.0, 0.0, 0.0);
//   Eigen::Vector3d C0_r_C0_W(0.0, 0.0, 0.0);
//   Eigen::Vector3d W_r_W_C1(0.0, 0.0, 0.0);

//   // Initialize the intrinsics and inverse intrinsics matrices
//   Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();
//   Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Kinv = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

//   // Obtain the Camera to IMU transformations
//   Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_S_C = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
//   Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_C_S = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
//   // T_S_C.block<3, 1>(0, 3) = Eigen::Vector3d(-0.0302200001, 0.0074000000, 0.0160200000);
//   T_S_C.block<3, 1>(0, 3) = Eigen::Vector3d(-0.0288974, -0.00745063, -0.0155697);
//   T_S_C.block<3, 1>(0, 2) = Eigen::Vector3d(0.00207086, 0.00273531, 0.999994);
//   T_S_C.block<3, 1>(0, 1) = Eigen::Vector3d(-0.00282996, 0.999992, -0.00272945);
//   T_S_C.block<3, 1>(0, 0) = Eigen::Vector3d(0.999994,  0.0028243, -0.00207859);
//   T_C_S = T_S_C.inverse();

//   // Obtain the previous state (initially in the camera frame)
//   T_W_C0.block<3,3>(0,0) = state.C_WS_; // The state's sensor frame is the camera frame... therefore C_W_C0
//   T_W_C0.block<3,1>(0,3) = state.W_r_WS_; // W_r_W_C0
//   C_W_C0 = T_W_C0.block<3,3>(0,0);
//   W_r_W_C0 = T_W_C0.block<3,1>(0,3);
//   C_C0_W = C_W_C0.transpose();
//   C0_r_C0_W = -C_C0_W * W_r_W_C0;
//   T_C0_W.block<3,3>(0,0) = C_C0_W;
//   T_C0_W.block<3,1>(0,3) = C0_r_C0_W;

//   // Initialize the current state with the previous state
//   T_W_C1 = T_W_C0;
//   C_W_C1 = T_W_C1.block<3,3>(0,0);
//   W_r_W_C1 = T_W_C1.block<3,1>(0,3);

//   // Create empty residual and Jacobian arrays
//   double *jacobians[4];
//   Eigen::Matrix<double,15,7,Eigen::RowMajor> J0 = Eigen::Matrix<double,15,7,Eigen::RowMajor>::Zero();
//   Eigen::Matrix<double,15,9,Eigen::RowMajor> J1 = Eigen::Matrix<double,15,9,Eigen::RowMajor>::Zero();
//   Eigen::Matrix<double,15,7,Eigen::RowMajor> J2 = Eigen::Matrix<double,15,7,Eigen::RowMajor>::Zero();
//   Eigen::Matrix<double,15,9,Eigen::RowMajor> J3 = Eigen::Matrix<double,15,9,Eigen::RowMajor>::Zero();
//   jacobians[0]=J0.data();
//   jacobians[1]=J1.data();
//   jacobians[2]=J2.data();
//   jacobians[3]=J3.data();
//   double* jacobiansMinimal[4];
//   Eigen::Matrix<double,15,6,Eigen::RowMajor> J0min = Eigen::Matrix<double,15,6,Eigen::RowMajor>::Zero();
//   Eigen::Matrix<double,15,9,Eigen::RowMajor> J1min = Eigen::Matrix<double,15,9,Eigen::RowMajor>::Zero();
//   Eigen::Matrix<double,15,6,Eigen::RowMajor> J2min = Eigen::Matrix<double,15,6,Eigen::RowMajor>::Zero();
//   Eigen::Matrix<double,15,9,Eigen::RowMajor> J3min = Eigen::Matrix<double,15,9,Eigen::RowMajor>::Zero();
//   jacobiansMinimal[0]=J0min.data();
//   jacobiansMinimal[1]=J1min.data();
//   jacobiansMinimal[2]=J2min.data();
//   jacobiansMinimal[3]=J3min.data();
//   Eigen::Matrix<double,15,1> residuals = Eigen::Matrix<double,15,1>::Zero();

//   // Create okvis variables to hold the previous state
//   okvis::kinematics::Transformation prevState;
//   okvis::SpeedAndBias prevSpeedAndBias;
//   Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_W_S0 = T_W_C0 * T_C_S;
//   prevState.set(T_W_S0);
//   prevSpeedAndBias.segment<3>(0) = state.W_v_WS_; // okvis expects velocity in the world frame
//   // std::cout << "Current speed estimate:" << std::endl << state.W_v_WS_ << std::endl;
//   prevSpeedAndBias.segment<3>(3) = state.bg_;
//   prevSpeedAndBias.segment<3>(6) = state.ba_;

//   // Create okvis variables to hold the perturbation to the state
//   Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_W_S1 = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
//   okvis::kinematics::Transformation currState;
//   okvis::SpeedAndBias currSpeedAndBias;

//   // Initialize perturbed state with the previous state
//   T_W_S1 = T_W_S0;
//   currState = prevState;
//   currSpeedAndBias = prevSpeedAndBias;

//   // Instantiate the IMU cost function
//   okvis::ceres::ImuError *imu_cost_function;
//   if (imuData.size() != 0){
//     imu_cost_function = new okvis::ceres::ImuError(imuData, imuParameters, prevTime, currTime);
//   } else {
//     imu_cost_function = 0;
//   }

//   // Propagate the pose speeds and biases with given IMU measurements if do not need to redo-preintegration
//   // Eigen::Matrix<double, 6, 1> Delta_b;
//   // Delta_b = prevSpeedAndBias.tail<6>() - linearizationPoint_.tail<6>();
//   // std::cout << "prev sab: " << prevSpeedAndBias.tail<6>() << std::endl;
//   // std::cout << "ref sab: " << linearizationPoint_.tail<6>() << std::endl;
//   // std::cout << "Delta Norm: " << Delta_b.head<3>().norm() << std::endl;
//   // redo_ = redo_ || (Delta_b.head<3>().norm() > 0.0003);
//   // if (useImu && !redo_ /*&& propCounter_ < 1*/) {
//   //   std::cout << "Start to propagate IMU." << std::endl;
//   //   propCounter_++;
//   //   imu_cost_function->propagation(imu_cost_function->imuMeasurements(),
//   //                                  imu_cost_function->imuParameters(),
//   //                                  currState,
//   //                                  currSpeedAndBias,
//   //                                  prevTime,
//   //                                  currTime);
//   //   T_W_C1 = currState.T() * T_S_C;
//   //   state.W_r_WS_ = T_W_C1.block<3,1>(0,3);
//   //   state.C_WS_ = T_W_C1.block<3,3>(0,0);
//   //   state.W_v_WS_ = currSpeedAndBias.segment<3>(0); // okvis expects velocity in the world frame
//   //   state.bg_ = currSpeedAndBias.segment<3>(3);
//   //   state.ba_ = currSpeedAndBias.segment<3>(6);
//   //
//   //   // Update the pose in the Mid-Fusion world
//   //   T_w_l = fromOkvisToMidFusion(okvis::kinematics::Transformation(T_W_C1));
//   //
//   //   bool tracked = true;
//   //   return tracked;
//   // } else {
//   //   // redo_ = true;
//   //   // propCounter_ = 0;
//   // }

//   // Rendering the reference depth information from model
//   if (using_RGB_ && (!use_live_depth_only_)){
//     for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i) {
//       Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
//       if (i == 0){
//         vertex2depth(r_D_render_[0], model_vertex, imgSize_, T_w_r);
//         depth2vertexKernel(r_Vertex_render_[0], r_D_render_[0],
//                            localImgSize_[0], invK);
//       }
//       else{
//         //using the rendered depth
//         halfSampleRobustImageKernel(r_D_render_[i], r_D_render_[i - 1],
//                                     localImgSize_[i-1], e_delta * 3, 1);
//         depth2vertexKernel(r_Vertex_render_[i], r_D_render_[i],
//                            localImgSize_[i], invK);
//       }
//     }
//   }

//   Matrix4 pose_update;
//   Matrix4 previous_pose = T_w_l;

//   // For debug
//   // cv::imshow("icp_outlier_mask", outlier_mask_[0]);
//   // cv::imshow("rgb_outlier_mask", prev_outlier_mask_[0]);
//   // cv::waitKey(0);

//   // Coarse-to-fine iteration
//   for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level) {
//     float previous_error = INFINITY;
//     for (int i = 0; i < GN_opt_iterations_[level]; ++i) {
//       if (using_ICP_){
//         const Matrix4 projectReference = getCameraMatrix(k_) * inverse(T_w_r);
//         track_ICP_Kernel(trackingResult_[level], imgSize_, l_vertex_[level],
//                          l_normal_[level], localImgSize_[level], model_vertex,
//                          model_normal, imgSize_, T_w_l, projectReference,
//                          dist_threshold, normal_threshold,
//                          icp_cov_pyramid_[level], outlier_mask_[level]);
//       }

//       if (using_RGB_){
//         // render reference image to live image -- opposite to the original function call
//         if (use_live_depth_only_){
//           trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
//                    imgSize_, r_Vertex_live_[level],r_Vertex_live_[level],
//                    r_I_[level], localImgSize_[level],
//                    l_I_[level], localImgSize_[level], l_gradx_[level],
//                    l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
//                    rgb_tracking_threshold_[level],
//                    mini_gradient_magintude_[level], sigma_bright_,
//                    prev_outlier_mask_[level]);
//         }
//         else{
//           trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
//                    imgSize_, r_Vertex_render_[level],r_Vertex_live_[level],
//                    r_I_[level], localImgSize_[level],
//                    l_I_[level], localImgSize_[level], l_gradx_[level],
//                    l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
//                    rgb_tracking_threshold_[level],
//                    mini_gradient_magintude_[level], sigma_bright_,
//                    prev_outlier_mask_[level]);
//         }
//       }

//       if (using_ICP_ && (!using_RGB_)){
//         reduceKernel(reductionoutput_[level], trackingResult_[level],
//                      imgSize_, localImgSize_[level]);
//       }

//       if ((!using_ICP_) && using_RGB_){
//         reduceKernel(reductionoutput_[level],
//                      trackingResult_[level]+ imgSize_.x * imgSize_.y,
//                      imgSize_, localImgSize_[level]);
//       }

//       if (using_ICP_ && using_RGB_){
//         reduceKernel(reductionoutput_[level], trackingResult_[level],
//                      imgSize_, localImgSize_[level]);
//       }

//       // Feed in the jacobians
//       int shift = 7;
//       for (int i = 0; i < 6; i++) {
//         jtr_v_(i) = -reductionoutput_[level][i + 1];
//         for (int j = i; j < 6; j++){
//           jtj_v_(i, j) = jtj_v_(j, i) = reductionoutput_[level][shift++];
//         }
//       }
//       const float current_error = reductionoutput_[level][0] / reductionoutput_[level][28];

//       // Calculate the IMU jacobians for the current pose estimate
//       if(useImu && imuData.size() != 0){
//         // Set up new current state and speedAndBias
//         T_W_S0 = T_W_C0 * T_C_S;
//         prevState.set(T_W_S0);
//         T_W_S1 = T_W_C1 * T_C_S;
//         currState.set(T_W_S1);

//         // Set up the parameter blocks
//         okvis::ceres::PoseParameterBlock poseParameterBlock_0(prevState, 0, prevTime);
//         okvis::ceres::PoseParameterBlock poseParameterBlock_1(currState, 2, currTime);
//         okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_0(prevSpeedAndBias, 1, prevTime);
//         okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_1(currSpeedAndBias, 3, currTime);
//         double *parameters[4];
//         parameters[0] = poseParameterBlock_0.parameters();
//         parameters[1] = speedAndBiasParameterBlock_0.parameters();
//         parameters[2] = poseParameterBlock_1.parameters();
//         parameters[3] = speedAndBiasParameterBlock_1.parameters();

//         // Call evaluateWithMinimalJacobians on imu_cost function
//         bool check = imu_cost_function->EvaluateWithMinimalJacobians(parameters, residuals.data(), jacobians, jacobiansMinimal);
//         if(!check){
//           std::cout << "Did not obtain Jacobians correctly!" << std::endl;
//         }

//         // Set the Jacobians
//         j_imu0_.block<15,6>(0,0) = J0min;
//         j_imu0_.block<15,9>(0,6) = J1min;
//         j_imu1_.block<15,6>(0,0) = J2min;
//         j_imu1_.block<15,9>(0,6) = J3min;

//         Eigen::Matrix<double, 15, 15> correction = Eigen::Matrix<double, 15, 15>::Identity();
//         Eigen::Matrix<double, 3, 1> W_r_C_S = T_W_C0.block<3,3>(0,0) * T_C_S.block<3,1>(0,3);
//         correction.block<3,3>(0,3) = -skew(W_r_C_S);
//         j_imu0_ = j_imu0_ * correction;

//         correction = Eigen::Matrix<double, 15, 15>::Identity();
//         W_r_C_S = T_W_C1.block<3,3>(0,0) * T_C_S.block<3,1>(0,3);
//         correction.block<3,3>(0,3) = -skew(W_r_C_S);
//         j_imu1_ = j_imu1_ * correction;

//         if (false) {
//           Eigen::Matrix<double,15,1> res_ref = residuals;

//           Eigen::Matrix<double, 30, 1> dir = Eigen::Matrix<double, 30, 1>::Ones();
//           Eigen::Matrix<double,30,1> delta30 = 1e-8*dir;

//           okvis::kinematics::Transformation T_update_old(T_W_C0);
//           T_update_old.oplus(delta30.segment<6>(0));
//           okvis::kinematics::Transformation T_update_new(T_W_C1);
//           T_update_new.oplus(delta30.segment<6>(15));

//           Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_W_C0_test = T_update_old.T();
//           Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_W_C1_test = T_update_new.T();

//           okvis::SpeedAndBias prevSpeedAndBias_test = prevSpeedAndBias + delta30.segment<9>(6);
//           okvis::SpeedAndBias currSpeedAndBias_test = currSpeedAndBias + delta30.segment<9>(21);

//           Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_W_S0_test = T_W_C0_test * T_C_S;
//           Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_W_S1_test = T_W_C1_test * T_C_S;

//           okvis::kinematics::Transformation prevState_test;
//           okvis::kinematics::Transformation currState_test;
//           prevState_test.set(T_W_S0_test);
//           currState_test.set(T_W_S1_test);

//           okvis::ceres::PoseParameterBlock poseParameterBlock_0_test(prevState_test, 0, prevTime);
//           okvis::ceres::PoseParameterBlock poseParameterBlock_1_test(currState_test, 2, currTime);
//           okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_0_test(prevSpeedAndBias_test, 1, prevTime);
//           okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_1_test(currSpeedAndBias_test, 3, currTime);
//           double* parameters_test[4];
//           parameters_test[0]=poseParameterBlock_0_test.parameters();
//           parameters_test[1]=speedAndBiasParameterBlock_0_test.parameters();
//           parameters_test[2]=poseParameterBlock_1_test.parameters();
//           parameters_test[3]=speedAndBiasParameterBlock_1_test.parameters();

//           double* jacobians_test[4];
//           Eigen::Matrix<double,15,7,Eigen::RowMajor> J0_test;
//           Eigen::Matrix<double,15,9,Eigen::RowMajor> J1_test;
//           Eigen::Matrix<double,15,7,Eigen::RowMajor> J2_test;
//           Eigen::Matrix<double,15,9,Eigen::RowMajor> J3_test;
//           jacobians_test[0]=J0_test.data();
//           jacobians_test[1]=J1_test.data();
//           jacobians_test[2]=J2_test.data();
//           jacobians_test[3]=J3_test.data();
//           double* jacobiansMinimal_test[4];
//           Eigen::Matrix<double,15,6,Eigen::RowMajor> J0min_test;
//           Eigen::Matrix<double,15,9,Eigen::RowMajor> J1min_test;
//           Eigen::Matrix<double,15,6,Eigen::RowMajor> J2min_test;
//           Eigen::Matrix<double,15,9,Eigen::RowMajor> J3min_test;
//           jacobiansMinimal_test[0]=J0min_test.data();
//           jacobiansMinimal_test[1]=J1min_test.data();
//           jacobiansMinimal_test[2]=J2min_test.data();
//           jacobiansMinimal_test[3]=J3min_test.data();
//           Eigen::Matrix<double,15,1> residuals_test = Eigen::Matrix<double,15,1>::Zero();

//           // Call evaluateWithMinimalJacobians on imu_cost function
//           check = imu_cost_function->EvaluateWithMinimalJacobians(parameters_test, residuals_test.data(), jacobians_test, jacobiansMinimal_test);
//           if(!check) {
//               std::cout << "Did not obtain Jacobians correctly!" << std::endl;
//           }

//           Eigen::Matrix<double,15,1> res_new = residuals_test;
//           Eigen::Matrix<double,15,1> res_dir = (res_new - res_ref)*1e8;
//           std::cout << "Residual direction num: " << res_dir.transpose() << std::endl;
//           std::cout << "Analytical direction: " << (j_imu0_ * dir.segment<15>(0) + j_imu1_ * dir.segment<15>(15)).transpose() << std::endl;
//           // std::cout << "Analytical direction 2: " << (j_imu0_test * dir.segment<15>(0) + j_imu1_test * dir.segment<15>(15)).transpose() << std::endl;
//           std::cout << "Norm Num = " << (res_dir.segment<3>(0)).norm() << std::endl;
//           std::cout << "Norm Ana = " << ((j_imu0_ * dir.segment<15>(0) + j_imu1_ * dir.segment<15>(15)).segment<3>(0)).norm() << std::endl;
//         }
//       }

//       // Solve for the update
//       // vision jacobians were expressed in the SE(3) - need to transform them into S0(3) used in OKVIS
//       if(useImu && imuData.size() != 0){
//         Eigen::Matrix<double, 6, 6, Eigen::RowMajor> correction = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Identity();
//         correction.block<3,3>(0,3) = skew(T_W_C1.block<3,1>(0,3));
//         jtj_v_ = correction.transpose() * jtj_v_ * correction;
//         jtr_v_ = correction.transpose() * jtr_v_;

//         // Lift the vision Jacobians into the expanded state space
//         jtj_v_lifted_.block<6,6>(0,0) = jtj_v_;
//         jtr_v_lifted_.segment<6>(0) = jtr_v_;

//         // Calculate DeltaChi (diff between prev pose and linearization point)
//         Eigen::Matrix<double, 15, 1> DeltaChi = Eigen::Matrix<double, 15 , 1>::Zero();

//         DeltaChi.segment<3>(0) = T_W_C0.block<3,1>(0,3) - linearizationPoint_.segment<3>(0);
//         Eigen::Quaterniond q_lp;
//         if(linearizationPoint_.segment<3>(3).isZero(1e-8)) {
//           std::cout << "linearizationPoint close to zero..." << std::endl;
//           Eigen::Matrix<double, 3, 3, Eigen::RowMajor> eye = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();
//           Eigen::Quaterniond q_lp_temp(eye);
//           q_lp = q_lp_temp;
//           // std::cout << "inside isZero" << std::endl;
//         }
//         else {
//           Eigen::AngleAxisd aa_lp(linearizationPoint_.segment<3>(3).norm(), linearizationPoint_.segment<3>(3).normalized());
//           Eigen::Quaterniond q_lp_temp(aa_lp);
//           q_lp = q_lp_temp;
//         }
//         Eigen::Quaterniond q(C_W_C0);
//         // DeltaChi.segment<3>(3) = 2 * (q * q_lp.inverse()).coeffs().template head<3>();
//         Eigen::AngleAxisd aa_diff(q * q_lp.inverse());
//         DeltaChi.segment<3>(3) = aa_diff.angle() * aa_diff.axis();
//         DeltaChi.segment<9>(6) = prevSpeedAndBias - linearizationPoint_.segment<9>(6);

//         // Calcualte b_star
//         Eigen::Matrix<double, 15, 1> b_star = b_star0_ + H_star_ * DeltaChi;

//         // Construct the complete Jacobian
//         JtJ_.block<15,15>(0,0) = H_star_ + j_imu0_.transpose().eval() * j_imu0_;
//         JtJ_.block<15,15>(0,15) = j_imu0_.transpose().eval() * j_imu1_;
//         JtJ_.block<15,15>(15,0) = j_imu1_.transpose().eval() * j_imu0_;
//         JtJ_.block<15,15>(15,15) = j_imu1_.transpose().eval() * j_imu1_ + jtj_v_lifted_;
//         JtR_.segment<15>(0) = b_star + j_imu0_.transpose().eval() * residuals;
//         JtR_.segment<15>(15) = j_imu1_.transpose().eval() * residuals + jtr_v_lifted_;

//         // Enforce symmetry, condition the Hessian...
//         JtJ_ = (0.5 * (JtJ_ + JtJ_.transpose().eval()).eval());

//         // Solve the system
//         Eigen::Matrix<double, 30, 1> delta30;
//         Eigen::LDLT<Eigen::Matrix<double, 30, 30,Eigen::RowMajor>> ldlt(JtJ_);
//         if (ldlt.info() != Eigen::Success) {
//           std::cout << "bad30" << std::endl;
//         }
//         else {
//           // nothing yet
//         }
//         delta30 = -ldlt.solve(JtR_);

//         // Compute the update
//         okvis::kinematics::Transformation T_update_old(T_W_C0);
//         T_update_old.oplus(delta30.segment<6>(0));
//         T_W_C0 = T_update_old.T();
//         okvis::kinematics::Transformation T_update_new(T_W_C1);
//         T_update_new.oplus(delta30.segment<6>(15));
//         T_W_C1 = T_update_new.T();

//         // Apply the pose update
//         C_W_C0 = T_W_C0.block<3,3>(0,0);
//         W_r_W_C0 = T_W_C0.block<3,1>(0,3);

//         C_C0_W = C_W_C0.transpose().eval();
//         C0_r_C0_W = - C_C0_W * W_r_W_C0;
//         T_C0_W.block<3,3>(0,0) = C_C0_W;
//         T_C0_W.block<3,1>(0,3) = C0_r_C0_W;

//         C_W_C1 = T_W_C1.block<3,3>(0,0);
//         W_r_W_C1 = T_W_C1.block<3,1>(0,3);

//         // Update the velocity and biases based on the optimization results
//         prevSpeedAndBias.segment<3>(0) += delta30.segment<3>(6);
//         prevSpeedAndBias.segment<3>(3) += delta30.segment<3>(9);
//         prevSpeedAndBias.segment<3>(6) += delta30.segment<3>(12);

//         currSpeedAndBias.segment<3>(0) += delta30.segment<3>(21);
//         currSpeedAndBias.segment<3>(3) += delta30.segment<3>(24);
//         currSpeedAndBias.segment<3>(6) += delta30.segment<3>(27);

//         // Update the pose in the Mid-Fusion world
//         T_w_l = fromOkvisToMidFusion(okvis::kinematics::Transformation(T_W_C1));
//         T_w_r = fromOkvisToMidFusion(okvis::kinematics::Transformation(T_W_C0));
//       } else {
//         if (current_error > (previous_error /*+ 1e-5f*/)){
//           if (step_back_in_GN_) {
//             T_w_l = previous_pose;
//           }
//           break;
//         }
//         previous_error = current_error;

//         // Solve the system
//         Eigen::Matrix<double, 6, 1> delta6;
//         delta6 = -jtj_v_.ldlt().solve(jtr_v_);

//         // Compute the update
//         okvis::kinematics::Transformation T_update;
//         T_update.oplus(delta6);

//         // Apply the update to the current state
//         T_W_C1 = T_update.T() * T_W_C1;
//         C_W_C1 = T_W_C1.block<3,3>(0,0);
//         W_r_W_C1 = T_W_C1.block<3,1>(0,3);

//         double norm = delta6(0)*delta6(0) + delta6(1)*delta6(1) + delta6(2)*delta6(2) +
//                       delta6(3)*delta6(3) + delta6(4)*delta6(4) + delta6(5)*delta6(5);
//         norm = sqrt(norm);
//         if ((norm < icp_threshold) && (std::sqrt(reductionoutput_[level][0]/reductionoutput_[level][28]) < 2e-3) ){
//           break;
//         }
//         previous_pose = T_w_l;
//         T_w_l = fromOkvisToMidFusion(okvis::kinematics::Transformation(T_W_C1));

//         // Original update version
//         // if (current_error > (previous_error /*+ 1e-5f*/)){
//         //   if (step_back_in_GN_) {
//         //     T_w_l = previous_pose;
//         //   }
//         //   break;
//         // }
//         // previous_error = current_error;

//         // if (solvePoseKernel(pose_update, reductionoutput_[level], icp_threshold)) {

//         //   break;
//         // }

//         // previous_pose = T_w_l;
//         // T_w_l = pose_update * previous_pose;
//       }
//     }
//   }

//   // Update the state and marginalization
//   if (useImu && imuData.size() != 0){
//     // // Find the covariance of the new state
//     // Eigen::Matrix<double, 15, 15> V = JtJ_.block<15,15>(0,0);
//     // Eigen::Matrix<double, 15, 15> V1 = 0.5 * (V + V.transpose()); // enforce symmetry
//     //
//     // Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> saes(V1);
//     //
//     // double epsilon = std::numeric_limits<double>::epsilon();
//     // double tol = epsilon * V1.cols() * saes.eigenvalues().array().maxCoeff();
//     //
//     // Eigen::Matrix<double, 15, 15> Vinv =  (saes.eigenvectors()) * Eigen::Matrix<double, 15, 1>((
//     //     saes.eigenvalues().array() > tol).select(
//     //     saes.eigenvalues().array().inverse(), 0)).asDiagonal() * (saes.eigenvectors().transpose());
//     //
//     // H_star_ = JtJ_.block<15,15>(15,15) - JtJ_.block<15,15>(15,0) * Vinv * JtJ_.block<15,15>(0,15);
//     // H_star_ = (0.5 * (H_star_ + H_star_.transpose().eval())).eval(); // enforce symmetry
//     //
//     // b_star0_ = JtR_.segment<15>(15) - JtJ_.block<15,15>(15,0) * Vinv * JtR_.segment<15>(0);

//     // Store the linearization point
//     linearizationPoint_.segment<3>(0) = T_W_C1.block<3,1>(0,3);
//     Eigen::AngleAxisd aa;
//     aa = T_W_C1.block<3,3>(0,0);
//     linearizationPoint_.segment<3>(3) = aa.angle() * aa.axis();
//     linearizationPoint_.segment<9>(6) = currSpeedAndBias;

//     // Output the pose in the camera frame
//     state.W_r_WS_ = T_W_C1.block<3,1>(0,3);
//     state.C_WS_ = T_W_C1.block<3,3>(0,0);
//     state.W_v_WS_ = currSpeedAndBias.segment<3>(0); // okvis expects velocity in the world frame
//     state.bg_ = currSpeedAndBias.segment<3>(3);
//     state.ba_ = currSpeedAndBias.segment<3>(6);
//   } else {
//     // Output the pose in the camera frame
//     state.W_r_WS_ = T_W_C1.block<3,1>(0,3);
//     state.C_WS_ = T_W_C1.block<3,3>(0,0);
//   }

//   bool tracked = true;
//   return tracked;
// }

bool Tracking::trackLiveFrameWithImu(Matrix4 &T_w_l,
                                     Matrix4 &T_w_r,
                                     Eigen::Matrix4d preUpdate,
                                     LocalizationState &state,
                                     const float4 k,
                                     const float3 *model_vertex,
                                     const float3 *model_normal,
                                     const cv::Mat &mask,
                                     bool reEstimate,
                                     okvis::ImuMeasurementDeque &imuData,
                                     okvis::Time &prevTime,
                                     okvis::Time &currTime,
                                     okvis::ImuParameters &imuParameters,
                                     bool useImu)
{

  // Initialize the incremental rotation and translation to apply to the previous pose
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_W_C0 = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> C_W_C0 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_C0_W = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> C_C0_W = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_W_C1 = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> C_W_C1 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();
  Eigen::Vector3d W_r_W_C0(0.0, 0.0, 0.0);
  Eigen::Vector3d C0_r_C0_W(0.0, 0.0, 0.0);
  Eigen::Vector3d W_r_W_C1(0.0, 0.0, 0.0);

  // Initialize the intrinsics and inverse intrinsics matrices
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Kinv = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

  // Obtain the Camera to IMU transformations
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_S_C = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_C_S = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
  // T_S_C.block<3, 1>(0, 3) = Eigen::Vector3d(-0.0302200001, 0.0074000000, 0.0160200000);
  T_S_C.block<3, 1>(0, 3) = Eigen::Vector3d(-0.0288974, -0.00745063, -0.0155697);
  T_S_C.block<3, 1>(0, 2) = Eigen::Vector3d(0.00207086, 0.00273531, 0.999994);
  T_S_C.block<3, 1>(0, 1) = Eigen::Vector3d(-0.00282996, 0.999992, -0.00272945);
  T_S_C.block<3, 1>(0, 0) = Eigen::Vector3d(0.999994, 0.0028243, -0.00207859);
  T_C_S = T_S_C.inverse();

  // Obtain the previous state (initially in the camera frame)
  T_W_C0.block<3, 3>(0, 0) = state.C_WS_;   // The state's sensor frame is the camera frame... therefore C_W_C0
  T_W_C0.block<3, 1>(0, 3) = state.W_r_WS_; // W_r_W_C0
  C_W_C0 = T_W_C0.block<3, 3>(0, 0);
  W_r_W_C0 = T_W_C0.block<3, 1>(0, 3);
  C_C0_W = C_W_C0.transpose();
  C0_r_C0_W = -C_C0_W * W_r_W_C0;
  T_C0_W.block<3, 3>(0, 0) = C_C0_W;
  T_C0_W.block<3, 1>(0, 3) = C0_r_C0_W;

  // Initialize the current state with the previous state or the pre-updated state
  T_W_C1 = T_W_C0;
  // T_W_C1.block<3,3>(0,0) = preUpdate.block<3,3>(0,0);
  C_W_C1 = T_W_C1.block<3, 3>(0, 0);
  W_r_W_C1 = T_W_C1.block<3, 1>(0, 3);

  // Create empty residual and Jacobian arrays
  double *jacobians[4];
  Eigen::Matrix<double, 15, 7, Eigen::RowMajor> J0 = Eigen::Matrix<double, 15, 7, Eigen::RowMajor>::Zero();
  Eigen::Matrix<double, 15, 9, Eigen::RowMajor> J1 = Eigen::Matrix<double, 15, 9, Eigen::RowMajor>::Zero();
  Eigen::Matrix<double, 15, 7, Eigen::RowMajor> J2 = Eigen::Matrix<double, 15, 7, Eigen::RowMajor>::Zero();
  Eigen::Matrix<double, 15, 9, Eigen::RowMajor> J3 = Eigen::Matrix<double, 15, 9, Eigen::RowMajor>::Zero();
  jacobians[0] = J0.data();
  jacobians[1] = J1.data();
  jacobians[2] = J2.data();
  jacobians[3] = J3.data();
  double *jacobiansMinimal[4];
  Eigen::Matrix<double, 15, 6, Eigen::RowMajor> J0min = Eigen::Matrix<double, 15, 6, Eigen::RowMajor>::Zero();
  Eigen::Matrix<double, 15, 9, Eigen::RowMajor> J1min = Eigen::Matrix<double, 15, 9, Eigen::RowMajor>::Zero();
  Eigen::Matrix<double, 15, 6, Eigen::RowMajor> J2min = Eigen::Matrix<double, 15, 6, Eigen::RowMajor>::Zero();
  Eigen::Matrix<double, 15, 9, Eigen::RowMajor> J3min = Eigen::Matrix<double, 15, 9, Eigen::RowMajor>::Zero();
  jacobiansMinimal[0] = J0min.data();
  jacobiansMinimal[1] = J1min.data();
  jacobiansMinimal[2] = J2min.data();
  jacobiansMinimal[3] = J3min.data();
  Eigen::Matrix<double, 15, 1> residuals = Eigen::Matrix<double, 15, 1>::Zero();

  // Create okvis variables to hold the previous state
  okvis::kinematics::Transformation prevState;
  okvis::SpeedAndBias prevSpeedAndBias;
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_W_S0 = T_W_C0 * T_C_S;
  prevState.set(T_W_S0);
  prevSpeedAndBias.segment<3>(0) = state.W_v_WS_; // okvis expects velocity in the world frame
  // std::cout << "Current speed estimate:" << std::endl << state.W_v_WS_ << std::endl;
  prevSpeedAndBias.segment<3>(3) = state.bg_;
  prevSpeedAndBias.segment<3>(6) = state.ba_;

  // Create okvis variables to hold the perturbation to the state
  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T_W_S1 = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();
  okvis::kinematics::Transformation currState;
  okvis::SpeedAndBias currSpeedAndBias;

  // Initialize perturbed state with the previous state
  // T_W_S1 = T_W_S0;
  // currState = prevState;
  // T_W_S1.block<3,3>(0,0) = preUpdate.block<3,3>(0,0) * T_C_S.block<3,3>(0,0);
  T_W_S1 = T_W_C1 * T_C_S;
  currState.set(T_W_S1);
  currSpeedAndBias = prevSpeedAndBias;

  // Instantiate the IMU cost function
  okvis::ceres::ImuError *imu_cost_function;
  if (imuData.size() != 0)
  {
    imu_cost_function = new okvis::ceres::ImuError(imuData, imuParameters, prevTime, currTime);
  }
  else
  {
    imu_cost_function = 0;
  }

  // Save linearization point and use the previous one when re-estimate the camera pose
  if (!reEstimate)
  {
    linearizationPointOrigin_ = linearizationPoint_.eval();
  }
  else
  {
    linearizationPoint_ = linearizationPointOrigin_.eval();
  }

  // Rendering the reference depth information from model
  if (using_RGB_ && (!use_live_depth_only_))
  {
    for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i)
    {
      Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
      if (i == 0)
      {
        vertex2depth(r_D_render_[0], model_vertex, imgSize_, T_w_r);
        depth2vertexKernel(r_Vertex_render_[0], r_D_render_[0],
                           localImgSize_[0], invK);
      }
      else
      {
        // using the rendered depth
        halfSampleRobustImageKernel(r_D_render_[i], r_D_render_[i - 1],
                                    localImgSize_[i - 1], e_delta * 3, 1);
        depth2vertexKernel(r_Vertex_render_[i], r_D_render_[i],
                           localImgSize_[i], invK);
      }
    }
  }

  std::vector<cv::Mat> outlier_masks;
  for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i)
  {
    cv::Mat outlier_mask;
    cv::Size mask_size = mask.size() / ((1 << i));
    cv::resize(mask, outlier_mask, mask_size);
    outlier_masks.push_back(outlier_mask);
  }

  Matrix4 pose_update;
  Matrix4 previous_pose = T_w_l;

  // Change T_w_l using pre-updated pose
  T_w_l = fromOkvisToMidFusion(okvis::kinematics::Transformation(T_W_C1));

  // Coarse-to-fine iteration
  for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level)
  {
    // std::cout << "Level " << level << "------------------" << std::endl;
    float previous_error = INFINITY;
    for (int i = 0; i < GN_opt_iterations_[level]; ++i)
    {
      if (using_ICP_)
      {
        const Matrix4 projectReference = getCameraMatrix(k_) * inverse(T_w_r);
        track_ICP_Kernel(trackingResult_[level], imgSize_, l_vertex_[level],
                         l_normal_[level], localImgSize_[level], model_vertex,
                         model_normal, imgSize_, T_w_l, projectReference,
                         dist_threshold, normal_threshold,
                         icp_cov_pyramid_[level], bg_outlier_mask_[level]);
      }

      if (using_RGB_)
      {
        // render reference image to live image -- opposite to the original function call
        if (use_live_depth_only_)
        {
          trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
                   imgSize_, r_Vertex_live_[level], r_Vertex_live_[level],
                   r_I_[level], localImgSize_[level],
                   l_I_[level], localImgSize_[level], l_gradx_[level],
                   l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
                   rgb_tracking_threshold_[level],
                   mini_gradient_magintude_[level], sigma_bright_,
                   prev_bg_outlier_mask_[level]);
        }
        else
        {
          trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
                   imgSize_, r_Vertex_render_[level], r_Vertex_live_[level],
                   r_I_[level], localImgSize_[level],
                   l_I_[level], localImgSize_[level], l_gradx_[level],
                   l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
                   rgb_tracking_threshold_[level],
                   mini_gradient_magintude_[level], sigma_bright_,
                   prev_bg_outlier_mask_[level]);
        }
      }

      if (using_ICP_ && (!using_RGB_))
      {
        reduceKernel(reductionoutput_[level], trackingResult_[level],
                     imgSize_, localImgSize_[level]);
      }

      if ((!using_ICP_) && using_RGB_)
      {
        reduceKernel(reductionoutput_[level],
                     trackingResult_[level] + imgSize_.x * imgSize_.y,
                     imgSize_, localImgSize_[level]);
      }

      if (using_ICP_ && using_RGB_)
      {
        reduceKernel(reductionoutput_[level], trackingResult_[level],
                     imgSize_, localImgSize_[level]);
      }

      // Feed in the jacobians
      int shift = 7;
      for (int h = 0; h < 6; h++)
      {
        jtr_v_(h) = -reductionoutput_[level][h + 1];
        for (int j = h; j < 6; j++)
        {
          jtj_v_(h, j) = jtj_v_(j, h) = reductionoutput_[level][shift++];
        }
      }
      // float current_error = reductionoutput_[level][0]/reductionoutput_[level][28];
      float current_error = reductionoutput_[level][0];

      // Calculate the IMU jacobians for the current pose estimate
      if (useImu && imuData.size() != 0)
      {
        // Set up new current state and speedAndBias
        T_W_S0 = T_W_C0 * T_C_S;
        prevState.set(T_W_S0);

        T_W_S1 = T_W_C1 * T_C_S;
        currState.set(T_W_S1);

        // Set up the parameter blocks
        okvis::ceres::PoseParameterBlock poseParameterBlock_0(prevState, 0, prevTime);
        okvis::ceres::PoseParameterBlock poseParameterBlock_1(currState, 2, currTime);
        okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_0(prevSpeedAndBias, 1, prevTime);
        okvis::ceres::SpeedAndBiasParameterBlock speedAndBiasParameterBlock_1(currSpeedAndBias, 3, currTime);
        double *parameters[4];
        parameters[0] = poseParameterBlock_0.parameters();
        parameters[1] = speedAndBiasParameterBlock_0.parameters();
        parameters[2] = poseParameterBlock_1.parameters();
        parameters[3] = speedAndBiasParameterBlock_1.parameters();

        // Call evaluateWithMinimalJacobians on imu_cost function
        bool check = imu_cost_function->EvaluateWithMinimalJacobians(parameters, residuals.data(), jacobians, jacobiansMinimal);
        if (!check)
        {
          std::cout << "Did not obtain Jacobians correctly!" << std::endl;
        }

        // Set the Jacobians
        j_imu0_.block<15, 6>(0, 0) = J0min;
        j_imu0_.block<15, 9>(0, 6) = J1min;
        j_imu1_.block<15, 6>(0, 0) = J2min;
        j_imu1_.block<15, 9>(0, 6) = J3min;

        Eigen::Matrix<double, 15, 15> correction = Eigen::Matrix<double, 15, 15>::Identity();
        Eigen::Matrix<double, 3, 1> W_r_C_S = T_W_C0.block<3, 3>(0, 0) * T_C_S.block<3, 1>(0, 3);
        correction.block<3, 3>(0, 3) = -skew(W_r_C_S);
        j_imu0_ = j_imu0_ * correction;

        correction = Eigen::Matrix<double, 15, 15>::Identity();
        W_r_C_S = T_W_C1.block<3, 3>(0, 0) * T_C_S.block<3, 1>(0, 3);
        correction.block<3, 3>(0, 3) = -skew(W_r_C_S);
        j_imu1_ = j_imu1_ * correction;
      }

      // Solve for the update
      // vision jacobians were expressed in the SE(3) - need to transform them into S0(3) used in OKVIS
      if (useImu && imuData.size() != 0)
      {
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> correction = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Identity();
        correction.block<3, 3>(0, 3) = skew(T_W_C1.block<3, 1>(0, 3));
        jtj_v_ = correction.transpose() * jtj_v_ * correction;
        jtr_v_ = correction.transpose() * jtr_v_;

        // Lift the vision Jacobians into the expanded state space
        jtj_v_lifted_.block<6, 6>(0, 0) = jtj_v_;
        jtr_v_lifted_.segment<6>(0) = jtr_v_;

        // Calculate DeltaChi (diff between prev pose and linearization point)
        Eigen::Matrix<double, 15, 1> DeltaChi = Eigen::Matrix<double, 15, 1>::Zero();

        DeltaChi.segment<3>(0) = T_W_C0.block<3, 1>(0, 3) - linearizationPoint_.segment<3>(0);
        Eigen::Quaterniond q_lp;
        if (linearizationPoint_.segment<3>(3).isZero(1e-8))
        {
          std::cout << "linearizationPoint close to zero..." << std::endl;
          Eigen::Matrix<double, 3, 3, Eigen::RowMajor> eye = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();
          Eigen::Quaterniond q_lp_temp(eye);
          q_lp = q_lp_temp;
          // std::cout << "inside isZero" << std::endl;
        }
        else
        {
          Eigen::AngleAxisd aa_lp(linearizationPoint_.segment<3>(3).norm(), linearizationPoint_.segment<3>(3).normalized());
          Eigen::Quaterniond q_lp_temp(aa_lp);
          q_lp = q_lp_temp;
        }
        Eigen::Quaterniond q(C_W_C0);
        // DeltaChi.segment<3>(3) = 2 * (q * q_lp.inverse()).coeffs().template head<3>();
        Eigen::AngleAxisd aa_diff(q * q_lp.inverse());
        DeltaChi.segment<3>(3) = aa_diff.angle() * aa_diff.axis();
        DeltaChi.segment<9>(6) = prevSpeedAndBias - linearizationPoint_.segment<9>(6);

        // Calculate b_star
        Eigen::Matrix<double, 15, 1> b_star = b_star0_ + H_star_ * DeltaChi;

        // Construct the complete Jacobian
        JtJ_.block<15, 15>(0, 0) = H_star_ + j_imu0_.transpose().eval() * j_imu0_;
        JtJ_.block<15, 15>(0, 15) = j_imu0_.transpose().eval() * j_imu1_;
        JtJ_.block<15, 15>(15, 0) = j_imu1_.transpose().eval() * j_imu0_;
        JtJ_.block<15, 15>(15, 15) = j_imu1_.transpose().eval() * j_imu1_ + jtj_v_lifted_;
        JtR_.segment<15>(0) = b_star + j_imu0_.transpose().eval() * residuals;
        JtR_.segment<15>(15) = j_imu1_.transpose().eval() * residuals + jtr_v_lifted_;

        // Enforce symmetry, condition the Hessian...
        JtJ_ = (0.5 * (JtJ_ + JtJ_.transpose().eval()).eval());

        // Save information matrix
        // infoMat = JtJ_.block<6,6>(0,0).eval();

        // Solve the system
        Eigen::Matrix<double, 30, 1> delta30;
        Eigen::LDLT<Eigen::Matrix<double, 30, 30, Eigen::RowMajor>> ldlt(JtJ_);
        if (ldlt.info() != Eigen::Success)
        {
          std::cout << "bad30" << std::endl;
        }
        else
        {
          // nothing yet
        }
        delta30 = -ldlt.solve(JtR_);

        // Stop if the cost function starts to increase
        // current_error += 0.5 * residuals.eval().transpose() * residuals.eval();

        // std::cout << " Loop " << i << ", current rgb+icp: " << current_error << std::endl;

        float imu_error = 0;
        for (int i = 0; i < 15; i++)
        {
          imu_error += 0.5 * cauchy_k * cauchy_k * log(1.0 + sq(residuals.segment<1>(i).value() / cauchy_k));
        }
        current_error += imu_error;

        Eigen::Matrix<double, 15, 1> marginalizationCost = DeltaChi - H_star_.inverse() * b_star;
        current_error += 0.5 * marginalizationCost.eval().transpose() * H_star_ * marginalizationCost.eval();
        // std::cout << " Loop " << i << ", current error: " << current_error << std::endl;
        if (current_error > previous_error)
        {
          if (step_back_in_GN_)
          {
          }
          break;
        }
        previous_error = current_error;

        // Compute the update
        okvis::kinematics::Transformation T_update_old(T_W_C0);
        T_update_old.oplus(delta30.segment<6>(0));
        T_W_C0 = T_update_old.T();
        okvis::kinematics::Transformation T_update_new(T_W_C1);
        T_update_new.oplus(delta30.segment<6>(15));
        T_W_C1 = T_update_new.T();

        // Apply the pose update
        C_W_C0 = T_W_C0.block<3, 3>(0, 0);
        W_r_W_C0 = T_W_C0.block<3, 1>(0, 3);

        C_C0_W = C_W_C0.transpose().eval();
        C0_r_C0_W = -C_C0_W * W_r_W_C0;
        T_C0_W.block<3, 3>(0, 0) = C_C0_W;
        T_C0_W.block<3, 1>(0, 3) = C0_r_C0_W;

        C_W_C1 = T_W_C1.block<3, 3>(0, 0);
        W_r_W_C1 = T_W_C1.block<3, 1>(0, 3);

        // Update the velocity and biases based on the optimization results
        prevSpeedAndBias.segment<3>(0) += delta30.segment<3>(6);
        prevSpeedAndBias.segment<3>(3) += delta30.segment<3>(9);
        prevSpeedAndBias.segment<3>(6) += delta30.segment<3>(12);

        currSpeedAndBias.segment<3>(0) += delta30.segment<3>(21);
        currSpeedAndBias.segment<3>(3) += delta30.segment<3>(24);
        currSpeedAndBias.segment<3>(6) += delta30.segment<3>(27);

        // Update the pose in the Mid-Fusion world
        T_w_l = fromOkvisToMidFusion(okvis::kinematics::Transformation(T_W_C1));
        T_w_r = fromOkvisToMidFusion(okvis::kinematics::Transformation(T_W_C0));
      }
      else
      {
        if (current_error > (previous_error /*+ 1e-5f*/))
        {
          if (step_back_in_GN_)
          {
            T_w_l = previous_pose;
          }
          break;
        }
        previous_error = current_error;

        // Solve the system
        Eigen::Matrix<double, 6, 1> delta6;
        delta6 = -jtj_v_.ldlt().solve(jtr_v_);

        // Compute the update
        okvis::kinematics::Transformation T_update;
        T_update.oplus(delta6);

        // Apply the update to the current state
        T_W_C1 = T_update.T() * T_W_C1;
        C_W_C1 = T_W_C1.block<3, 3>(0, 0);
        W_r_W_C1 = T_W_C1.block<3, 1>(0, 3);

        double norm = delta6(0) * delta6(0) + delta6(1) * delta6(1) + delta6(2) * delta6(2) +
                      delta6(3) * delta6(3) + delta6(4) * delta6(4) + delta6(5) * delta6(5);
        norm = sqrt(norm);
        if ((norm < icp_threshold) && (std::sqrt(reductionoutput_[level][0] / reductionoutput_[level][28]) < 2e-3))
        {
          break;
        }
        previous_pose = T_w_l;
        T_w_l = fromOkvisToMidFusion(okvis::kinematics::Transformation(T_W_C1));

        // Original update version
        // if (current_error > (previous_error /*+ 1e-5f*/)){
        //   if (step_back_in_GN_) {
        //     T_w_l = previous_pose;
        //   }
        //   break;
        // }
        // previous_error = current_error;
        //
        // if (solvePoseKernel(pose_update, reductionoutput_[level], icp_threshold)) {
        //
        //   break;
        // }
        //
        // previous_pose = T_w_l;
        // T_w_l = pose_update * previous_pose;
      }
    }
  }

  // Update the state and marginalization
  if (useImu && imuData.size() != 0)
  {
    // Store the linearization point
    linearizationPoint_.segment<3>(0) = T_W_C1.block<3, 1>(0, 3);
    Eigen::AngleAxisd aa;
    aa = T_W_C1.block<3, 3>(0, 0);
    linearizationPoint_.segment<3>(3) = aa.angle() * aa.axis();
    linearizationPoint_.segment<9>(6) = currSpeedAndBias;

    // Output the pose in the camera frame
    state.W_r_WS_ = T_W_C1.block<3, 1>(0, 3);
    state.C_WS_ = T_W_C1.block<3, 3>(0, 0);
    state.W_v_WS_ = currSpeedAndBias.segment<3>(0); // okvis expects velocity in the world frame
    state.bg_ = currSpeedAndBias.segment<3>(3);
    state.ba_ = currSpeedAndBias.segment<3>(6);
  }
  else
  {
    // Output the pose in the camera frame
    state.W_r_WS_ = T_W_C1.block<3, 1>(0, 3);
    state.C_WS_ = T_W_C1.block<3, 3>(0, 0);
  }

  bool tracked = true;
  return tracked;
}

void Tracking::setLinearizationPoint(const Eigen::Matrix4d currPose)
{
  linearizationPoint_.segment<3>(0) = currPose.block<3, 1>(0, 3);
  Eigen::AngleAxisd aa;
  aa = currPose.block<3, 3>(0, 0);
  linearizationPoint_.segment<3>(3) = aa.angle() * aa.axis();
}

void Tracking::marginalization()
{
  // Find the covariance of the new state
  Eigen::Matrix<double, 15, 15> V = JtJ_.block<15, 15>(0, 0);
  Eigen::Matrix<double, 15, 15> V1 = 0.5 * (V + V.transpose()); // enforce symmetry

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> saes(V1);

  double epsilon = std::numeric_limits<double>::epsilon();
  double tol = epsilon * V1.cols() * saes.eigenvalues().array().maxCoeff();

  Eigen::Matrix<double, 15, 15> Vinv = (saes.eigenvectors()) * Eigen::Matrix<double, 15, 1>((saes.eigenvalues().array() > tol).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * (saes.eigenvectors().transpose());

  H_star_ = JtJ_.block<15, 15>(15, 15) - JtJ_.block<15, 15>(15, 0) * Vinv * JtJ_.block<15, 15>(0, 15);
  H_star_ = (0.5 * (H_star_ + H_star_.transpose().eval())).eval(); // enforce symmetry

  b_star0_ = JtR_.segment<15>(15) - JtJ_.block<15, 15>(15, 0) * Vinv * JtR_.segment<15>(0);
}

// void Tracking::objectRelocalization(ObjectPointer& objectptr,
//                                     const float4 k,
//                                     const Matrix4& T_w_c2,
//                                     const Matrix4& T_w_c1,
//                                     const float* f_l_I,
//                                     const float* f_l_D,
//                                     Matrix4 prev_pose,
//                                     cv::Mat prevResizeRGB,
//                                     cv::Mat currResizeRGB) {
//
//   cv::Mat prev_obj_rgb, curr_obj_rgb;
//   cv::Mat prev_obj_mask, curr_obj_mask;
//   if (objectptr->prev_fg_outlier_mask_.empty()) {
//     cv::bitwise_not(no_outlier_mask_, prev_obj_mask);
//   } else {
//     cv::bitwise_not(objectptr->prev_fg_outlier_mask_, prev_obj_mask);
//   }
//   cv::bitwise_not(objectptr->fg_outlier_mask_, curr_obj_mask);
//
//   cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
//   cv::morphologyEx(prev_obj_mask, prev_obj_mask, cv::MORPH_OPEN, element);
//   cv::morphologyEx(curr_obj_mask, curr_obj_mask, cv::MORPH_OPEN, element);
//   prevResizeRGB.copyTo(prev_obj_rgb, prev_obj_mask);
//   currResizeRGB.copyTo(curr_obj_rgb, curr_obj_mask);
//
//   cv::imshow("prev_obj", prev_obj_rgb);
//   cv::imshow("curr_obj", curr_obj_rgb);
//
//   std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
//   std::vector<cv::DMatch> matches;
//   find_feature_matches(prev_obj_rgb, curr_obj_rgb, keypoints_1, keypoints_2, matches, true, true);
//
//   // Set camera intrinsics
//   cv::Mat K = (cv::Mat_<double>(3, 3) << k.x, 0, k.z, 0, k.y, k.w, 0, 0, 1);
//   std::cout << K << std::endl;
//
//   if (true){
//     std::vector<cv::Point3f> pts1;
//     std::vector<cv::Point2f> pts2;
//     for (cv::DMatch m : matches){
//       const int r_index = int(keypoints_1[m.queryIdx].pt.x) + int(keypoints_1[m.queryIdx].pt.y) * imgSize_.x;
//       const int l_index = int(keypoints_2[m.trainIdx].pt.x) + int(keypoints_2[m.trainIdx].pt.y) * imgSize_.x;
//       float3 v1 = r_Vertex_render_save_[r_index];
//       if (v1.z == 0) continue;
//       v1 = T_w_c1 * v1;
//
//       pts1.push_back(cv::Point3f(v1.x, v1.y, v1.z));
//       pts2.push_back(cv::Point2f(int(keypoints_2[m.trainIdx].pt.x), int(keypoints_2[m.trainIdx].pt.y)));
//     }
//
//     cv::Mat R, t;
//     cv::solvePnPRansac(pts1, pts2, K, cv::Mat(), R, t, false, 100 ,0.1);
//
//     std::cout << "Rotation: " << R << std::endl;
//     std::cout << "Translation: " << t << std::endl;
//     Eigen::Vector3d rotation;
//     rotation << R.at<double>(0), R.at<double>(1), R.at<double>(2);
//
//     Eigen::Matrix3d rot3d = rodrigues(rotation);
//
//     Matrix4 delta_T;
//     for (int i = 0; i < 3; i++){
//       delta_T.data[i].x = rot3d(i, 0);
//       delta_T.data[i].y = rot3d(i, 1);
//       delta_T.data[i].z = rot3d(i, 2);
//       delta_T.data[i].w = t.at<double>(i);
//     }
//     delta_T.data[3].x = 0;
//     delta_T.data[3].y = 0;
//     delta_T.data[3].z = 0;
//     delta_T.data[3].w = 1;
//
//     objectptr->volume_pose_ = T_w_c2 * delta_T * inverse(prev_pose);
//     objectptr->virtual_camera_pose_ = prev_pose * inverse(inverse(T_w_c2) * objectptr->volume_pose_);
//
//     printMatrix4("11111", delta_T);
//     printMatrix4("22222", T_w_c2 * delta_T);
//
//     return;
//   }
//
//   // Matrix4 invK = getInverseCameraMatrix(k_);
//   // vertex2depth(l_D_ref_, w_refVertices_l, imgSize_, T_w_l);
//   // // vertex2depth(l_D_ref_, w_refVertices_l, imgSize_, T_w_r);
//   // depth2vertexKernel(l_vertex_ref_, l_D_ref_, localImgSize_[0], invK);
//
//   // if (using_3d2d_){
//   //   // Using 3d-2d alignment
//   //   std::vector<cv::Point3f> points_1;
//   //   std::vector<cv::Point2f> points_2;
//   //   for (int i = 0; i < (int)matches.size(); i++) {
//   //     points_1.push_back(keypoints_1[matches[i].queryIdx].pt);
//   //     points_2.push_back(keypoints_2[matches[i].trainIdx].pt);
//   //   }
//   //
//   //   // cv::Mat fundamental_matrix;
//   //   // fundamental_matrix = cv::findFundamentalMat(points_1, points_2, CV_FM_RANSAC);
//   //
//   //   cv::Point2d principle_point(k.z, k.w);
//   //   cv::Mat essential_matrix;
//   //   essential_matrix = cv::findEssentialMat(points_1, points_2, k.x, principle_point);
//   //
//   //   // Third: recover the transformation from essential matrix
//   //   cv::Mat R, t;
//   //   cv::recoverPose(essential_matrix, points_1, points_2, R, t, k.x, principle_point);
//   //
//   //   // Fourth: initialize the current camera pose
//   //   Matrix4 delta_T;
//   //   for (int i = 0; i < 3; i++){
//   //     delta_T.data[i].x = R.at<double>(i, 0);
//   //     delta_T.data[i].y = R.at<double>(i, 1);
//   //     delta_T.data[i].z = R.at<double>(i, 2);
//   //     delta_T.data[i].w = t.at<double>(i);
//   //   }
//   //   delta_T.data[3].x = 0;
//   //   delta_T.data[3].y = 0;
//   //   delta_T.data[3].z = 0;
//   //   delta_T.data[3].w = 1;
//   //
//   //   printMatrix4("delta_T", delta_T);
//   //
//   //   this->preTrackingPose = fromMidFusionToOkvis(camera_pose * inverse(delta_T)).T();
//   // }
//
//   // Create the 3D poins
//   std::vector<cv::Point3f> pts1, pts2;
//   for (cv::DMatch m : matches){
//     const int r_index = int(keypoints_1[m.queryIdx].pt.x) + int(keypoints_1[m.queryIdx].pt.y) * imgSize_.x;
//     const int l_index = int(keypoints_2[m.trainIdx].pt.x) + int(keypoints_2[m.trainIdx].pt.y) * imgSize_.x;
//     // float3 v1 = r_Vertex_live_[0][r_index];
//     // float3 v2 = l_vertex_[0][l_index];
//     float3 v1 = r_Vertex_render_save_[r_index];
//     float3 v2 = l_vertex_ref_[l_index];
//     if (v1.z == 0 || v2.z == 0) continue;
//     v1 = T_w_c1 * v1;
//     v2 = T_w_c2 * v2;
//
//     pts1.push_back(cv::Point3f(v1.x, v1.y, v1.z));
//     pts2.push_back(cv::Point3f(v2.x, v2.y, v2.z));
//   }
//
//   // SVD pose estimation
//   Eigen::Matrix3d R;
//   Eigen::Vector3d t;
//   pose_estimation_3d3d(pts1, pts2, R, t);
//
//   // Initialize the current camera pose
//   Matrix4 delta_T;
//   for (int i = 0; i < 3; i++){
//     delta_T.data[i].x = R(i, 0);
//     delta_T.data[i].y = R(i, 1);
//     delta_T.data[i].z = R(i, 2);
//     delta_T.data[i].w = t(i);
//   }
//   delta_T.data[3].x = 0;
//   delta_T.data[3].y = 0;
//   delta_T.data[3].z = 0;
//   delta_T.data[3].w = 1;
//
//   printMatrix4("11111", delta_T);
//   // Matrix4 T_o2_o1 = T_w_c2 * delta_T * inverse(T_w_c1);
//   // objectptr->volume_pose_ = prev_pose * inverse(T_o2_o1);
//   // objectptr->virtual_camera_pose_ = prev_pose * inverse(T_w_c2) * objectptr->volume_pose_;
//
//   objectptr->volume_pose_ = prev_pose * inverse(delta_T);
//   objectptr->virtual_camera_pose_ = prev_pose * inverse(inverse(T_w_c2) * objectptr->volume_pose_);
//   // objectptr->volume_pose_ = T_w_c2 * T_c2_o2;
//   // objectptr->virtual_camera_pose_ = T_wo1 * inverse(T_c2_o2);
// }

void Tracking::objectRelocalization(ObjectPointer &objectptr,
                                    const float4 k,
                                    const Matrix4 &T_w_c2,
                                    const Matrix4 &T_w_c1,
                                    const float *f_l_I,
                                    const float *f_l_D,
                                    Matrix4 prev_pose,
                                    cv::Mat prevResizeRGB,
                                    cv::Mat currResizeRGB,
                                    const float3 *model_vertex)
{

  // cv::Mat kf_obj_rgb, curr_obj_rgb;
  // cv::Mat kf_obj_mask, curr_obj_mask;
  // if (objectptr->kf_mask_list_[0].empty()) {
  //   cv::bitwise_not(no_outlier_mask_, kf_obj_mask);
  // } else {
  //   cv::bitwise_not(objectptr->kf_mask_list_[0], kf_obj_mask);
  // }
  // cv::bitwise_not(objectptr->fg_outlier_mask_, curr_obj_mask);
  //
  // cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
  // cv::morphologyEx(kf_obj_mask, kf_obj_mask, cv::MORPH_OPEN, element);
  // cv::morphologyEx(curr_obj_mask, curr_obj_mask, cv::MORPH_OPEN, element);
  // objectptr->kf_rgb_list_[0].copyTo(kf_obj_rgb, kf_obj_mask);
  // currResizeRGB.copyTo(curr_obj_rgb, curr_obj_mask);
  //
  // cv::imshow("prev_obj", kf_obj_rgb);
  // cv::imshow("curr_obj", curr_obj_rgb);
  //
  // std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
  // std::vector<cv::DMatch> matches;
  // find_feature_matches(kf_obj_rgb, curr_obj_rgb, keypoints_1, keypoints_2, matches, true, true);
  //
  // std::cout << "1111111111" <<std::endl;
  //
  // // Calculate rendered vertex
  // Matrix4 invK = getInverseCameraMatrix(k);
  // vertex2depth(r_D_kf_, model_vertex, imgSize_, objectptr->kf_pose_list_[0]);
  // depth2vertexKernel(r_vertex_kf_, r_D_kf_, localImgSize_[0], invK);
  //
  // std::cout << "2222222" <<std::endl;
  //
  // // Set camera intrinsics
  // cv::Mat K = (cv::Mat_<double>(3, 3) << k.x, 0, k.z, 0, k.y, k.w, 0, 0, 1);
  // std::cout << K << std::endl;
  //
  // if (true){
  //   std::vector<cv::Point3f> pts1;
  //   std::vector<cv::Point2f> pts2;
  //   for (cv::DMatch m : matches){
  //     const int r_index = int(keypoints_1[m.queryIdx].pt.x) + int(keypoints_1[m.queryIdx].pt.y) * imgSize_.x;
  //     float3 v1 = r_vertex_kf_[r_index];
  //     if (v1.z == 0) continue;
  //     v1 = objectptr->kf_pose_list_[0] * v1;
  //
  //     pts1.push_back(cv::Point3f(v1.x, v1.y, v1.z));
  //     pts2.push_back(cv::Point2f(int(keypoints_2[m.trainIdx].pt.x), int(keypoints_2[m.trainIdx].pt.y)));
  //   }
  //   std::cout << "3333333" <<std::endl;
  //
  //   cv::Mat R, t;
  //   cv::solvePnPRansac(pts1, pts2, K, cv::Mat(), R, t, false, 100 ,0.1);
  //
  //   std::cout << "Rotation: " << R << std::endl;
  //   std::cout << "Translation: " << t << std::endl;
  //   Eigen::Vector3d rotation;
  //   rotation << R.at<double>(0), R.at<double>(1), R.at<double>(2);
  //
  //   Eigen::Matrix3d rot3d = rodrigues(rotation);
  //
  //   Matrix4 delta_T;
  //   for (int i = 0; i < 3; i++){
  //     delta_T.data[i].x = rot3d(i, 0);
  //     delta_T.data[i].y = rot3d(i, 1);
  //     delta_T.data[i].z = rot3d(i, 2);
  //     delta_T.data[i].w = t.at<double>(i);
  //   }
  //   delta_T.data[3].x = 0;
  //   delta_T.data[3].y = 0;
  //   delta_T.data[3].z = 0;
  //   delta_T.data[3].w = 1;
  //
  //   objectptr->volume_pose_ = T_w_c2 * delta_T * inverse(prev_pose);
  //   objectptr->virtual_camera_pose_ = prev_pose * inverse(inverse(T_w_c2) * objectptr->volume_pose_);
  //
  //   printMatrix4("11111", delta_T);
  //   printMatrix4("22222", T_w_c2 * delta_T);

  return;
  // }
}

void Tracking::visualize_residuals(std::string name)
{
  cv::Mat img_0 = cv::Mat(cv::Size(320, 240), CV_32FC3, cv::Scalar(0, 0, 0));
  cv::Mat img_1 = cv::Mat(cv::Size(320, 240), CV_32FC3, cv::Scalar(0, 0, 0));
  for (uint y = 0; y < 240; y++)
  {
    for (uint x = 0; x < 320; x++)
    {
      uint pos = x + y * 320;
      TrackData *camera_tracking_result = getTrackingResult();
      const TrackData &icp_error = camera_tracking_result[pos];
      unsigned int rgb_pos = pos + 320 * 240;
      const TrackData &rgb_error = camera_tracking_result[rgb_pos];
      if (rgb_error.result == 1)
      {
        // White
        img_0.at<cv::Vec3f>(y, x)[0] = 255;
        img_0.at<cv::Vec3f>(y, x)[1] = 255;
        img_0.at<cv::Vec3f>(y, x)[2] = 255;
      }
      else if (rgb_error.result == -1)
      {
        // Green
        img_0.at<cv::Vec3f>(y, x)[0] = 0;
        img_0.at<cv::Vec3f>(y, x)[1] = 255;
        img_0.at<cv::Vec3f>(y, x)[2] = 0;
      }
      else if (rgb_error.result == -2)
      {
        // Blue
        img_0.at<cv::Vec3f>(y, x)[0] = 255;
        img_0.at<cv::Vec3f>(y, x)[1] = 0;
        img_0.at<cv::Vec3f>(y, x)[2] = 0;
      }
      else if (icp_error.result == -3)
      {
        // Purple
        img_1.at<cv::Vec3f>(y, x)[0] = 128;
        img_1.at<cv::Vec3f>(y, x)[1] = 0;
        img_1.at<cv::Vec3f>(y, x)[2] = 128;
      }
      else if (rgb_error.result == -4)
      {
        // Red
        img_0.at<cv::Vec3f>(y, x)[0] = 0;
        img_0.at<cv::Vec3f>(y, x)[1] = 0;
        img_0.at<cv::Vec3f>(y, x)[2] = 255;
      }
      else if (rgb_error.result == -5)
      {
        // Yellow
        img_0.at<cv::Vec3f>(y, x)[0] = 0;
        img_0.at<cv::Vec3f>(y, x)[1] = 255;
        img_0.at<cv::Vec3f>(y, x)[2] = 255;
      }
      else if (rgb_error.result == -6)
      {
        // Gray
        img_0.at<cv::Vec3f>(y, x)[0] = 217;
        img_0.at<cv::Vec3f>(y, x)[1] = 217;
        img_0.at<cv::Vec3f>(y, x)[2] = 217;
      }

      if (icp_error.result == 1)
      {
        // White
        img_1.at<cv::Vec3f>(y, x)[0] = 255;
        img_1.at<cv::Vec3f>(y, x)[1] = 255;
        img_1.at<cv::Vec3f>(y, x)[2] = 255;
      }
      else if (icp_error.result == -1)
      {
        // Green
        img_1.at<cv::Vec3f>(y, x)[0] = 0;
        img_1.at<cv::Vec3f>(y, x)[1] = 255;
        img_1.at<cv::Vec3f>(y, x)[2] = 0;
      }
      else if (icp_error.result == -2)
      {
        // Blue
        img_1.at<cv::Vec3f>(y, x)[0] = 255;
        img_1.at<cv::Vec3f>(y, x)[1] = 0;
        img_1.at<cv::Vec3f>(y, x)[2] = 0;
      }
      else if (icp_error.result == -3)
      {
        // Purple
        img_1.at<cv::Vec3f>(y, x)[0] = 128;
        img_1.at<cv::Vec3f>(y, x)[1] = 0;
        img_1.at<cv::Vec3f>(y, x)[2] = 128;
      }
      else if (icp_error.result == -4)
      {
        // Red
        img_1.at<cv::Vec3f>(y, x)[0] = 0;
        img_1.at<cv::Vec3f>(y, x)[1] = 0;
        img_1.at<cv::Vec3f>(y, x)[2] = 255;
      }
      else if (icp_error.result == -5)
      {
        // Yellow
        img_1.at<cv::Vec3f>(y, x)[0] = 0;
        img_1.at<cv::Vec3f>(y, x)[1] = 255;
        img_1.at<cv::Vec3f>(y, x)[2] = 255;
      }
      else if (icp_error.result == -6)
      {
        // Gray
        img_0.at<cv::Vec3f>(y, x)[0] = 217;
        img_0.at<cv::Vec3f>(y, x)[1] = 217;
        img_0.at<cv::Vec3f>(y, x)[2] = 217;
      }
    }
  }
  cv::imshow(name + " rgb", img_0);
  cv::imshow(name + " icp", img_1);
}

float3 *Tracking::get_live_vertex()
{
  return l_vertex_[0];
}

std::vector<cv::Mat> Tracking::getBgOutMask()
{
  return bg_outlier_mask_;
}

std::vector<int> Tracking::getGNOptIter()
{
  return GN_opt_iterations_;
}

uint2 Tracking::getImgSize()
{
  return imgSize_;
}

float Tracking::get_sigma_bright()
{
  return sigma_bright_;
}

float *Tracking::get_l_depth()
{
  return l_D_[0];
}

float3 **Tracking::get_l_vertex()
{
  return l_vertex_;
}

float3 *Tracking::get_l_vertex_render()
{
  return l_vertex_ref_;
}

float3 **Tracking::get_l_normal()
{
  return l_normal_;
}

std::vector<cv::Mat> Tracking::get_no_outlier_mask()
{
  return no_outlier_mask_;
}

float **Tracking::get_l_I()
{
  return l_I_;
}

float **Tracking::get_gradx()
{
  return l_gradx_;
}

float **Tracking::get_grady()
{
  return l_grady_;
}

float3 **Tracking::get_icp_cov_pyramid()
{
  return icp_cov_pyramid_;
}

float4 Tracking::getK()
{
  return k_;
}
