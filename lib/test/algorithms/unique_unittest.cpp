/*
 *
 * Copyright 2016 Emanuele Vespa, Imperial College London 
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 *
 * */
#include "octree.hpp"
#include "math_utils.h"
#include "utils/morton_utils.hpp"
#include "algorithms/unique.hpp"
#include "gtest/gtest.h"
#include <algorithm>

typedef float testT;
typedef unsigned int MortonType;

template <>
struct voxel_traits<testT> {
  typedef float ComputeType;
  typedef float StoredType;
  static inline ComputeType empty(){ return 0.f; }
  static inline ComputeType initValue(){ return 0.f; }
  static inline StoredType translate(const ComputeType value) {
     return value;
  }
};

class UniqueTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      const int3 blocks[10] = {
        {56, 12, 12}, {56, 12, 15}, 
        {128, 128, 128},
        {128, 128, 125}, {128, 128, 127}, 
        {128, 136, 129}, 
        {128, 136, 127}, 
        {136, 128, 136}, 
        {128, 240, 136}, {128, 241, 136}};
      for(int i = 0; i < 10; ++i) { 
        keys[i] = oct.hash(blocks[i].x, blocks[i].y, blocks[i].z);
      }
    }
    
    MortonType keys[10];
    typedef Octree<testT> OctreeF;
    OctreeF oct;
};

class UniqueMultiscaleTest : public ::testing::Test {
  protected:
    virtual void SetUp() {

      oct.init(256, 10);

      const int3 blocks[10] = {
        {56, 12, 12}, 
        {56, 12, 15}, 
        {128, 128, 128},
        {128, 128, 125}, 
        {128, 128, 127}, 
        {128, 136, 129}, 
        {128, 136, 127}, 
        {136, 128, 136}, 
        {128, 240, 136}, {128, 241, 136}};
      for(int i = 0; i < 10; ++i) { 
        keys[i] = oct.hash(blocks[i].x, blocks[i].y, blocks[i].z, 7);
      }

      keys[2] = oct.hash(blocks[2].x, blocks[2].y, blocks[2].z, 3);
      keys[3] = oct.hash(blocks[3].x, blocks[3].y, blocks[3].z, 5);
      keys[4] = oct.hash(blocks[4].x, blocks[4].y, blocks[4].z, 6);
      std::sort(keys, keys + 10); 
    }
   
    MortonType keys[10];
    typedef Octree<testT> OctreeF;
    OctreeF oct;
};

TEST_F(UniqueTest, FilterDuplicates) {
  std::sort(keys, keys + 10);
  const int last = algorithms::unique(keys, 10);
  for(int i = 1; i < last; ++i) { 
    ASSERT_TRUE(keys[i] != keys[i-1]);
  }
}

TEST_F(UniqueMultiscaleTest, FilterDuplicates) {
  std::sort(keys, keys + 10);
  /*
   * 0x1FFu extracts the last 9 bits of a morton number,
   * corresponding to the edge of a voxel block: 3*log2(VoxelBlock<T>::side)
   */
  const int last = algorithms::unique_multiscale(keys, 10, 0x1FFu);
  ASSERT_EQ(last, 7);
  for(int i = 1; i < last; ++i) { 
    // std::cout << "(Key: " << (keys[i-1] & (~0x1FFu)) << ", Scale: " 
    //           << (keys[i-1] & 0x1FFu) << "), "; 
    ASSERT_TRUE(keys[i] != keys[i-1]);
  }
  std::cout << std::endl;
}

TEST_F(UniqueMultiscaleTest, FilterDuplicatesTillLevel) {
  std::sort(keys, keys + 10);
  /*
   * 0x1FFu extracts the last 9 bits of a morton number,
   * corresponding to the edge of a voxel block: 3*log2(VoxelBlock<T>::side)
   */
  const int last = algorithms::unique_multiscale(keys, 10, 0x1FFu, 6);
  ASSERT_EQ(last, 6);
  for(int i = 1; i <= last; ++i) { 
    // std::cout << "(Key: " << (keys[i-1] & (~0x1FFu)) << ", Scale: " 
    //           << (keys[i-1] & 0x1FFu) << "), "; 
    ASSERT_TRUE(keys[i] != keys[i-1]);
  }
  std::cout << std::endl;
}
