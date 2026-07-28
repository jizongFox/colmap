// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/estimators/bundle_adjustment.h"

#include <filesystem>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colmap {

// Absolute camera pose prior read from the pose_priors table. Components are
// populated according to the enabled CLI constraints.
struct DatabasePosePrior {
  data_t data_id;
  bool has_position = false;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Matrix3d position_covariance = Eigen::Matrix3d::Identity();
  bool has_rotation = false;
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d rotation_covariance = Eigen::Matrix3d::Identity();
};

struct DatabasePosePriorBundleAdjustmentOptions {
  bool use_position_priors = false;
  bool use_rotation_priors = false;
  double prior_position_fallback_stddev = 1.0;
  double prior_rotation_fallback_stddev_rad = 0.08726646259971647;  // 5 deg.
  double prior_position_loss_scale = 2.7955321496988725;
  double prior_rotation_loss_scale = 2.7955321496988725;

  bool Enabled() const { return use_position_priors || use_rotation_priors; }
  bool Check() const;
};

// Reads only the components enabled in options. Position priors are Cartesian
// camera centers. Rotation priors are COLMAP world-to-camera quaternions in
// qw, qx, qy, qz order.
std::vector<DatabasePosePrior> ReadDatabasePosePriors(
    const std::filesystem::path& database_path,
    const DatabasePosePriorBundleAdjustmentOptions& options);

// Creates a Ceres bundle adjuster with independently selectable absolute
// camera-center and quaternion-rotation residuals.
std::unique_ptr<BundleAdjuster> CreateDatabasePosePriorBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const DatabasePosePriorBundleAdjustmentOptions& prior_options,
    BundleAdjustmentConfig config,
    const std::filesystem::path& database_path,
    Reconstruction& reconstruction);

}  // namespace colmap
