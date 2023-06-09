/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2020-2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2023 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */


#ifndef CONSTANT_PARAMETERS_H_
#define CONSTANT_PARAMETERS_H_

////////////////////////// COMPILATION PARAMETERS //////////////////////

const float e_delta = 0.1f;
const int radius = 2;
const float dist_threshold = 0.1f;
const float normal_threshold = 0.8f;
const float track_threshold = 0.15f;
const float maxweight = 100.0f;
const float nearPlane = 0.1f;
const float farPlane = 4.0f;  // 4.0f
const float delta = 4.0f;

const float3 light = make_float3(1, 1, -1.0);
const float3 ambient = make_float3(0.1, 0.1, 0.1);

#endif /* CONSTANT_PARAMETERS_H_ */
