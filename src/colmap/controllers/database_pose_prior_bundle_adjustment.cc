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

#include "colmap/controllers/database_pose_prior_bundle_adjustment.h"

#include "colmap/estimators/alignment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/math/math.h"
#include "colmap/scene/image.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <ceres/ceres.h>
#include <sqlite3.h>

namespace colmap {
namespace {

struct SqliteDatabaseDeleter {
  void operator()(sqlite3* database) const {
    if (database != nullptr) {
      sqlite3_close_v2(database);
    }
  }
};

struct SqliteStatementDeleter {
  void operator()(sqlite3_stmt* statement) const {
    if (statement != nullptr) {
      sqlite3_finalize(statement);
    }
  }
};

using SqliteDatabasePtr = std::unique_ptr<sqlite3, SqliteDatabaseDeleter>;
using SqliteStatementPtr =
    std::unique_ptr<sqlite3_stmt, SqliteStatementDeleter>;

template <typename MatrixType>
bool ReadStaticMatrixBlob(sqlite3_stmt* statement,
                          const int column,
                          MatrixType* matrix) {
  THROW_CHECK_NOTNULL(matrix);
  if (sqlite3_column_type(statement, column) == SQLITE_NULL) {
    return false;
  }

  const int num_bytes = sqlite3_column_bytes(statement, column);
  const int expected_num_bytes =
      static_cast<int>(matrix->size() * sizeof(typename MatrixType::Scalar));
  if (num_bytes != expected_num_bytes) {
    return false;
  }

  std::memcpy(
      matrix->data(), sqlite3_column_blob(statement, column), num_bytes);
  return true;
}

std::unordered_set<std::string> ReadPosePriorColumns(sqlite3* database) {
  sqlite3_stmt* raw_statement = nullptr;
  if (sqlite3_prepare_v2(database,
                         "PRAGMA table_info(pose_priors);",
                         -1,
                         &raw_statement,
                         nullptr) != SQLITE_OK) {
    return {};
  }
  SqliteStatementPtr statement(raw_statement);

  std::unordered_set<std::string> columns;
  int step_result = SQLITE_OK;
  while ((step_result = sqlite3_step(statement.get())) == SQLITE_ROW) {
    const auto* text = sqlite3_column_text(statement.get(), 1);
    if (text != nullptr) {
      columns.emplace(reinterpret_cast<const char*>(text));
    }
  }
  if (step_result != SQLITE_DONE) {
    return {};
  }
  return columns;
}

std::optional<std::string> FindColumn(
    const std::unordered_set<std::string>& columns,
    const std::initializer_list<const char*> candidates) {
  for (const char* candidate : candidates) {
    if (columns.count(candidate) > 0) {
      return std::string(candidate);
    }
  }
  return std::nullopt;
}

std::string QuoteIdentifier(const std::string& identifier) {
  return "\"" + identifier + "\"";
}

bool MakeValidCovariance(const double fallback_stddev,
                         Eigen::Matrix3d* covariance) {
  THROW_CHECK_NOTNULL(covariance);
  *covariance = (0.5 * (*covariance + covariance->transpose())).eval();
  if (covariance->allFinite()) {
    const Eigen::LLT<Eigen::Matrix3d> llt(*covariance);
    if (llt.info() == Eigen::Success) {
      return true;
    }
  }

  if (!(fallback_stddev > 0.0) || !std::isfinite(fallback_stddev)) {
    return false;
  }
  *covariance = fallback_stddev * fallback_stddev * Eigen::Matrix3d::Identity();
  return true;
}

template <typename T>
void CanonicalizeQuaternion(Eigen::Quaternion<T>* quaternion) {
  THROW_CHECK_NOTNULL(quaternion);
  if (quaternion->w() < T(0)) {
    quaternion->coeffs() *= T(-1);
  }
}

bool DataIdsMatch(const data_t& data_id1, const data_t& data_id2) {
  return data_id1.id == data_id2.id &&
         data_id1.sensor_id.id == data_id2.sensor_id.id &&
         data_id1.sensor_id.type == data_id2.sensor_id.type;
}

struct AbsolutePoseRotationPriorCostFunctor
    : public AutoDiffCostFunctor<AbsolutePoseRotationPriorCostFunctor, 3, 7> {
 public:
  explicit AbsolutePoseRotationPriorCostFunctor(
      const Eigen::Quaterniond& sensor_from_world_rotation_prior)
      : world_from_sensor_rotation_prior_(
            sensor_from_world_rotation_prior.conjugate()) {}

  template <typename T>
  bool operator()(const T* const sensor_from_world, T* residuals_ptr) const {
    Eigen::Quaternion<T> param_from_prior_rotation =
        EigenQuaternionMap<T>(sensor_from_world) *
        world_from_sensor_rotation_prior_.cast<T>();
    CanonicalizeQuaternion(&param_from_prior_rotation);
    EigenQuaternionToAngleAxis(param_from_prior_rotation.coeffs().data(),
                               residuals_ptr);
    return true;
  }

 private:
  const Eigen::Quaterniond world_from_sensor_rotation_prior_;
};

struct AbsoluteRigPoseRotationPriorCostFunctor
    : public AutoDiffCostFunctor<AbsoluteRigPoseRotationPriorCostFunctor,
                                 3,
                                 7,
                                 7> {
 public:
  explicit AbsoluteRigPoseRotationPriorCostFunctor(
      const Eigen::Quaterniond& sensor_from_world_rotation_prior)
      : world_from_sensor_rotation_prior_(
            sensor_from_world_rotation_prior.conjugate()) {}

  template <typename T>
  bool operator()(const T* const sensor_from_rig,
                  const T* const rig_from_world,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> sensor_from_world_rotation =
        EigenQuaternionMap<T>(sensor_from_rig) *
        EigenQuaternionMap<T>(rig_from_world);
    Eigen::Quaternion<T> param_from_prior_rotation =
        sensor_from_world_rotation *
        world_from_sensor_rotation_prior_.cast<T>();
    CanonicalizeQuaternion(&param_from_prior_rotation);
    EigenQuaternionToAngleAxis(param_from_prior_rotation.coeffs().data(),
                               residuals_ptr);
    return true;
  }

 private:
  const Eigen::Quaterniond world_from_sensor_rotation_prior_;
};

class DatabasePosePriorBundleAdjuster : public BundleAdjuster {
 public:
  DatabasePosePriorBundleAdjuster(
      const BundleAdjustmentOptions& options,
      const DatabasePosePriorBundleAdjustmentOptions& prior_options,
      const BundleAdjustmentConfig& config,
      std::vector<DatabasePosePrior> pose_priors,
      const Sim3d& normalized_from_working,
      Reconstruction& reconstruction)
      : BundleAdjuster(options, config),
        prior_options_(prior_options),
        pose_priors_(std::move(pose_priors)),
        normalized_from_working_(normalized_from_working),
        reconstruction_(reconstruction),
        position_loss_function_(std::make_unique<ceres::CauchyLoss>(
            prior_options_.prior_position_loss_scale)),
        rotation_loss_function_(std::make_unique<ceres::CauchyLoss>(
            prior_options_.prior_rotation_loss_scale)) {
    default_bundle_adjuster_ =
        CreateDefaultCeresBundleAdjuster(options_, config_, reconstruction_);
    if (!prior_options_.use_position_priors &&
        prior_options_.use_rotation_priors && !FixTranslationAndScaleGauge()) {
      throw std::runtime_error(
          "Could not fix translation and scale for rotation-only priors.");
    }
    AddPosePriorsToProblem();
  }

  ~DatabasePosePriorBundleAdjuster() override { RestoreWorkingFrame(); }

  bool HasSufficientPosePriors() const {
    const bool enough_positions = !prior_options_.use_position_priors ||
                                  num_contributing_position_priors_ >= 3;
    const bool enough_rotations = !prior_options_.use_rotation_priors ||
                                  num_contributing_rotation_priors_ >= 1;
    return enough_positions && enough_rotations;
  }

  size_t NumContributingPositionPriors() const {
    return num_contributing_position_priors_;
  }

  size_t NumContributingRotationPriors() const {
    return num_contributing_rotation_priors_;
  }

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    try {
      auto summary = default_bundle_adjuster_->Solve();
      RestoreWorkingFrame();
      return summary;
    } catch (...) {
      RestoreWorkingFrame();
      throw;
    }
  }

 private:
  bool FixTranslationAndScaleGauge() {
    ceres::Problem& problem = *default_bundle_adjuster_->Problem();
    Rigid3d* first_rig_from_world = nullptr;
    Rigid3d* second_rig_from_world = nullptr;
    frame_t first_frame_id = 0;
    int second_fixed_translation_dim = 0;

    for (const image_t image_id : config_.Images()) {
      Image& image = reconstruction_.Image(image_id);
      Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();
      if (!problem.HasParameterBlock(rig_from_world.params.data()) ||
          problem.IsParameterBlockConstant(rig_from_world.params.data())) {
        continue;
      }

      if (first_rig_from_world == nullptr) {
        first_rig_from_world = &rig_from_world;
        first_frame_id = image.FrameId();
        continue;
      }
      if (image.FrameId() == first_frame_id) {
        continue;
      }

      const Eigen::Vector3d baseline =
          (*first_rig_from_world * Inverse(rig_from_world)).translation();
      Eigen::Index max_coeff_idx = 0;
      if (baseline.cwiseAbs().maxCoeff(&max_coeff_idx) > 1e-9) {
        second_rig_from_world = &rig_from_world;
        second_fixed_translation_dim = static_cast<int>(max_coeff_idx);
        break;
      }
    }

    if (first_rig_from_world == nullptr || second_rig_from_world == nullptr) {
      LOG(ERROR) << "Rotation-only pose-prior BA needs two variable frames "
                    "with a non-zero baseline to fix translation and scale.";
      return false;
    }

    // Fix all three translation coordinates of the first rig pose, but leave
    // its quaternion variable so absolute rotation priors control orientation.
    SetManifold(&problem,
                first_rig_from_world->params.data(),
                CreateProductManifold(CreateEigenQuaternionManifold(),
                                      CreateSubsetManifold(3, {0, 1, 2})));

    // Fix one translation coordinate of a second frame to remove scale gauge.
    SetManifold(&problem,
                second_rig_from_world->params.data(),
                CreateProductManifold(
                    CreateEigenQuaternionManifold(),
                    CreateSubsetManifold(3, {second_fixed_translation_dim})));

    LOG(INFO) << "Fixed translation and scale gauge for rotation-only "
                 "pose-prior bundle adjustment while leaving rotation free.";
    return true;
  }

  void AddPosePriorsToProblem() {
    ceres::Problem& problem = *default_bundle_adjuster_->Problem();

    for (const DatabasePosePrior& pose_prior : pose_priors_) {
      const image_t image_id = static_cast<image_t>(pose_prior.data_id.id);
      Image& image = reconstruction_.Image(image_id);
      Frame& frame = *image.FramePtr();
      Rigid3d& rig_from_world = frame.RigFromWorld();

      Eigen::Vector3d normalized_position = Eigen::Vector3d::Zero();
      Eigen::Matrix3d normalized_position_covariance =
          Eigen::Matrix3d::Identity();
      if (prior_options_.use_position_priors) {
        normalized_position = normalized_from_working_ * pose_prior.position;
        const Eigen::Matrix3d scaled_rotation =
            normalized_from_working_.scale() *
            normalized_from_working_.rotation().toRotationMatrix();
        normalized_position_covariance = scaled_rotation *
                                         pose_prior.position_covariance *
                                         scaled_rotation.transpose();
      }

      Eigen::Quaterniond normalized_rotation = Eigen::Quaterniond::Identity();
      if (prior_options_.use_rotation_priors) {
        normalized_rotation = pose_prior.rotation *
                              normalized_from_working_.rotation().conjugate();
      }

      if (image.IsRefInFrame()) {
        if (!problem.HasParameterBlock(rig_from_world.params.data()) ||
            problem.IsParameterBlockConstant(rig_from_world.params.data())) {
          continue;
        }

        if (prior_options_.use_position_priors) {
          problem.AddResidualBlock(
              CovarianceWeightedCostFunctor<
                  AbsolutePosePositionPriorCostFunctor>::
                  Create(normalized_position_covariance, normalized_position),
              position_loss_function_.get(),
              rig_from_world.params.data());
          ++num_contributing_position_priors_;
        }

        if (prior_options_.use_rotation_priors) {
          problem.AddResidualBlock(
              CovarianceWeightedCostFunctor<
                  AbsolutePoseRotationPriorCostFunctor>::
                  Create(pose_prior.rotation_covariance, normalized_rotation),
              rotation_loss_function_.get(),
              rig_from_world.params.data());
          ++num_contributing_rotation_priors_;
        }
      } else {
        Rigid3d& sensor_from_rig =
            frame.RigPtr()->SensorFromRig(image.CameraPtr()->SensorId());
        const bool has_sensor_from_rig =
            problem.HasParameterBlock(sensor_from_rig.params.data());
        const bool has_rig_from_world =
            problem.HasParameterBlock(rig_from_world.params.data());
        if (!has_sensor_from_rig || !has_rig_from_world ||
            (problem.IsParameterBlockConstant(sensor_from_rig.params.data()) &&
             problem.IsParameterBlockConstant(rig_from_world.params.data()))) {
          continue;
        }

        if (prior_options_.use_position_priors) {
          problem.AddResidualBlock(
              CovarianceWeightedCostFunctor<
                  AbsoluteRigPosePositionPriorCostFunctor>::
                  Create(normalized_position_covariance, normalized_position),
              position_loss_function_.get(),
              sensor_from_rig.params.data(),
              rig_from_world.params.data());
          ++num_contributing_position_priors_;
        }

        if (prior_options_.use_rotation_priors) {
          problem.AddResidualBlock(
              CovarianceWeightedCostFunctor<
                  AbsoluteRigPoseRotationPriorCostFunctor>::
                  Create(pose_prior.rotation_covariance, normalized_rotation),
              rotation_loss_function_.get(),
              sensor_from_rig.params.data(),
              rig_from_world.params.data());
          ++num_contributing_rotation_priors_;
        }
      }
    }

    LOG(INFO) << "Added " << num_contributing_position_priors_
              << " position-prior residual blocks and "
              << num_contributing_rotation_priors_
              << " rotation-prior residual blocks.";
  }

  void RestoreWorkingFrame() {
    if (!working_frame_restored_) {
      reconstruction_.Transform(Inverse(normalized_from_working_));
      working_frame_restored_ = true;
    }
  }

  DatabasePosePriorBundleAdjustmentOptions prior_options_;
  std::vector<DatabasePosePrior> pose_priors_;
  Sim3d normalized_from_working_;
  Reconstruction& reconstruction_;
  std::unique_ptr<CeresBundleAdjuster> default_bundle_adjuster_;
  std::unique_ptr<ceres::LossFunction> position_loss_function_;
  std::unique_ptr<ceres::LossFunction> rotation_loss_function_;
  size_t num_contributing_position_priors_ = 0;
  size_t num_contributing_rotation_priors_ = 0;
  bool working_frame_restored_ = false;
};

}  // namespace

bool DatabasePosePriorBundleAdjustmentOptions::Check() const {
  if (!Enabled()) {
    return true;
  }
  if (use_position_priors) {
    CHECK_OPTION_GT(prior_position_fallback_stddev, 0);
    CHECK_OPTION_GT(prior_position_loss_scale, 0);
  }
  if (use_rotation_priors) {
    CHECK_OPTION_GT(prior_rotation_fallback_stddev_rad, 0);
    CHECK_OPTION_GT(prior_rotation_loss_scale, 0);
  }
  return true;
}

std::vector<DatabasePosePrior> ReadDatabasePosePriors(
    const std::filesystem::path& database_path,
    const DatabasePosePriorBundleAdjustmentOptions& options) {
  if (!options.Enabled() || !options.Check()) {
    return {};
  }

  sqlite3* raw_database = nullptr;
  const int open_result = sqlite3_open_v2(database_path.string().c_str(),
                                          &raw_database,
                                          SQLITE_OPEN_READONLY,
                                          nullptr);
  SqliteDatabasePtr database(raw_database);
  if (open_result != SQLITE_OK || database == nullptr) {
    LOG(ERROR) << "Cannot open pose-prior database: " << database_path;
    return {};
  }

  const std::unordered_set<std::string> columns =
      ReadPosePriorColumns(database.get());
  if (columns.empty()) {
    LOG(ERROR) << "Database has no readable pose_priors table: "
               << database_path;
    return {};
  }

  if (columns.count("corr_data_id") == 0 ||
      columns.count("corr_sensor_id") == 0 ||
      columns.count("corr_sensor_type") == 0) {
    LOG(ERROR) << "pose_priors is missing required correspondence columns.";
    return {};
  }

  if (options.use_position_priors &&
      (columns.count("position") == 0 ||
       columns.count("coordinate_system") == 0)) {
    LOG(ERROR) << "Position priors are enabled, but pose_priors is missing "
                  "position or coordinate_system.";
    return {};
  }

  const auto rotation_column = FindColumn(columns,
                                          {"rotation",
                                           "rotation_quaternion",
                                           "rotation_prior",
                                           "prior_qvec",
                                           "qvec"});
  const auto rotation_covariance_column =
      FindColumn(columns,
                 {"rotation_covariance",
                  "rotation_prior_covariance",
                  "prior_qvec_covariance",
                  "qvec_covariance"});
  if (options.use_rotation_priors && !rotation_column.has_value()) {
    LOG(ERROR) << "Rotation priors are enabled, but pose_priors does not "
                  "contain a supported quaternion column.";
    return {};
  }

  if (options.use_position_priors &&
      columns.count("position_covariance") == 0) {
    LOG(WARNING) << "pose_priors has no position covariance column; using "
                    "the fallback position standard deviation.";
  }
  if (options.use_rotation_priors && !rotation_covariance_column.has_value()) {
    LOG(WARNING) << "pose_priors has no rotation covariance column; using "
                    "the fallback rotation standard deviation.";
  }

  const std::string position_expression =
      options.use_position_priors ? QuoteIdentifier("position") : "NULL";
  const std::string position_covariance_expression =
      options.use_position_priors && columns.count("position_covariance") > 0
          ? QuoteIdentifier("position_covariance")
          : "NULL";
  const std::string coordinate_system_expression =
      options.use_position_priors ? QuoteIdentifier("coordinate_system")
                                  : "NULL";
  const std::string rotation_expression =
      options.use_rotation_priors ? QuoteIdentifier(*rotation_column) : "NULL";
  const std::string rotation_covariance_expression =
      options.use_rotation_priors && rotation_covariance_column.has_value()
          ? QuoteIdentifier(*rotation_covariance_column)
          : "NULL";

  const std::string query =
      "SELECT corr_data_id, corr_sensor_id, corr_sensor_type, " +
      position_expression + ", " + position_covariance_expression + ", " +
      coordinate_system_expression + ", " + rotation_expression + ", " +
      rotation_covariance_expression + " FROM pose_priors;";

  sqlite3_stmt* raw_statement = nullptr;
  if (sqlite3_prepare_v2(
          database.get(), query.c_str(), -1, &raw_statement, nullptr) !=
      SQLITE_OK) {
    LOG(ERROR) << "Failed to prepare pose-prior query: "
               << sqlite3_errmsg(database.get());
    return {};
  }
  SqliteStatementPtr statement(raw_statement);

  std::vector<DatabasePosePrior> pose_priors;
  int step_result = SQLITE_OK;
  while ((step_result = sqlite3_step(statement.get())) == SQLITE_ROW) {
    const SensorType sensor_type =
        static_cast<SensorType>(sqlite3_column_int(statement.get(), 2));
    if (sensor_type != SensorType::CAMERA) {
      continue;
    }

    DatabasePosePrior pose_prior;
    pose_prior.data_id.id = sqlite3_column_int64(statement.get(), 0);
    pose_prior.data_id.sensor_id.id = sqlite3_column_int64(statement.get(), 1);
    pose_prior.data_id.sensor_id.type = sensor_type;

    if (options.use_position_priors) {
      const int coordinate_system = sqlite3_column_int(statement.get(), 5);
      if (coordinate_system !=
          static_cast<int>(PosePrior::CoordinateSystem::CARTESIAN)) {
        LOG_FIRST_N(WARNING, 1)
            << "Ignoring non-Cartesian pose priors. Convert them to Cartesian "
               "coordinates before bundle adjustment.";
        continue;
      }
      if (!ReadStaticMatrixBlob(statement.get(), 3, &pose_prior.position) ||
          !pose_prior.position.allFinite()) {
        continue;
      }
      if (!ReadStaticMatrixBlob(
              statement.get(), 4, &pose_prior.position_covariance)) {
        pose_prior.position_covariance =
            Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
      }
      if (!MakeValidCovariance(options.prior_position_fallback_stddev,
                               &pose_prior.position_covariance)) {
        continue;
      }
      pose_prior.has_position = true;
    }

    if (options.use_rotation_priors) {
      Eigen::Vector4d quaternion_wxyz;
      if (!ReadStaticMatrixBlob(statement.get(), 6, &quaternion_wxyz) ||
          !quaternion_wxyz.allFinite()) {
        continue;
      }
      pose_prior.rotation = Eigen::Quaterniond(quaternion_wxyz(0),
                                               quaternion_wxyz(1),
                                               quaternion_wxyz(2),
                                               quaternion_wxyz(3));
      const double rotation_norm = pose_prior.rotation.norm();
      if (!(rotation_norm > 0.0) || !std::isfinite(rotation_norm)) {
        continue;
      }
      pose_prior.rotation.normalize();

      if (!ReadStaticMatrixBlob(
              statement.get(), 7, &pose_prior.rotation_covariance)) {
        pose_prior.rotation_covariance =
            Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
      }
      if (!MakeValidCovariance(options.prior_rotation_fallback_stddev_rad,
                               &pose_prior.rotation_covariance)) {
        continue;
      }
      pose_prior.has_rotation = true;
    }

    pose_priors.push_back(std::move(pose_prior));
  }

  if (step_result != SQLITE_DONE) {
    LOG(ERROR) << "Failed while reading pose priors: "
               << sqlite3_errmsg(database.get());
    return {};
  }

  LOG(INFO) << "Read " << pose_priors.size() << " pose priors from "
            << database_path << " (position=" << options.use_position_priors
            << ", rotation=" << options.use_rotation_priors << ")";
  return pose_priors;
}

std::unique_ptr<BundleAdjuster> CreateDatabasePosePriorBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const DatabasePosePriorBundleAdjustmentOptions& prior_options,
    BundleAdjustmentConfig config,
    const std::filesystem::path& database_path,
    Reconstruction& reconstruction) {
  if (!prior_options.Enabled() || !prior_options.Check()) {
    LOG(ERROR) << "At least one pose-prior constraint must be enabled.";
    return nullptr;
  }
  if (options.backend != BundleAdjustmentBackend::CERES || !options.ceres) {
    LOG(ERROR) << "Database pose-prior bundle adjustment requires the Ceres "
                  "backend.";
    return nullptr;
  }
  if (!options.refine_rig_from_world) {
    LOG(ERROR) << "Pose-prior bundle adjustment requires "
                  "BundleAdjustment.refine_rig_from_world=1.";
    return nullptr;
  }
  if (prior_options.use_rotation_priors &&
      options.constant_rig_from_world_rotation) {
    LOG(ERROR) << "Rotation priors cannot be enabled together with "
                  "BundleAdjustment.constant_rig_from_world_rotation=1.";
    return nullptr;
  }

  std::vector<DatabasePosePrior> pose_priors =
      ReadDatabasePosePriors(database_path, prior_options);
  std::vector<DatabasePosePrior> filtered_pose_priors;
  filtered_pose_priors.reserve(pose_priors.size());
  std::unordered_set<image_t> seen_image_ids;
  for (DatabasePosePrior& prior : pose_priors) {
    const image_t image_id = static_cast<image_t>(prior.data_id.id);
    if (!reconstruction.ExistsImage(image_id) || !config.HasImage(image_id)) {
      continue;
    }

    const Image& image = reconstruction.Image(image_id);
    if (!image.HasPose() || !DataIdsMatch(prior.data_id, image.DataId())) {
      LOG_FIRST_N(WARNING, 1)
          << "Ignoring pose priors whose complete data identifier does not "
             "match the input reconstruction.";
      continue;
    }

    if (!seen_image_ids.insert(image_id).second) {
      LOG_FIRST_N(WARNING, 1)
          << "Ignoring duplicate pose-prior rows for the same image.";
      continue;
    }
    filtered_pose_priors.push_back(std::move(prior));
  }
  pose_priors = std::move(filtered_pose_priors);

  if (prior_options.use_position_priors && pose_priors.size() < 3) {
    LOG(ERROR) << "Position-prior BA requires at least three distinct, "
                  "registered, matching images; found "
               << pose_priors.size() << ".";
    return nullptr;
  }
  if (prior_options.use_rotation_priors && pose_priors.empty()) {
    LOG(ERROR) << "Rotation-prior BA requires at least one distinct, "
                  "registered, matching image.";
    return nullptr;
  }

  std::optional<Sim3d> metric_from_original;
  if (prior_options.use_position_priors) {
    std::vector<PosePrior> position_priors;
    position_priors.reserve(pose_priors.size());
    for (const DatabasePosePrior& database_prior : pose_priors) {
      PosePrior position_prior;
      position_prior.corr_data_id = database_prior.data_id;
      position_prior.position = database_prior.position;
      position_prior.position_covariance = database_prior.position_covariance;
      position_prior.coordinate_system = PosePrior::CoordinateSystem::CARTESIAN;
      position_priors.push_back(std::move(position_prior));
    }

    RANSACOptions alignment_options;
    std::vector<double> rms_variances;
    rms_variances.reserve(position_priors.size());
    for (const PosePrior& position_prior : position_priors) {
      rms_variances.push_back(position_prior.position_covariance.trace() / 3.0);
    }
    alignment_options.max_error =
        std::sqrt(kChiSquare95ThreeDof * Median(std::move(rms_variances)));

    Sim3d estimated_metric_from_original;
    if (!AlignReconstructionToPosePriors(reconstruction,
                                         position_priors,
                                         alignment_options,
                                         &estimated_metric_from_original)) {
      LOG(ERROR) << "Failed to align reconstruction to database position "
                    "priors.";
      return nullptr;
    }
    reconstruction.Transform(estimated_metric_from_original);
    metric_from_original = estimated_metric_from_original;
  }

  // Absolute position priors remove the complete similarity gauge. In
  // rotation-only mode the custom adjuster fixes translation and scale while
  // intentionally leaving global orientation free.
  config.FixGauge(BundleAdjustmentGauge::UNSPECIFIED);
  const Sim3d normalized_from_working =
      reconstruction.Normalize(/*fixed_scale=*/true);

  try {
    auto bundle_adjuster = std::make_unique<DatabasePosePriorBundleAdjuster>(
        options,
        prior_options,
        config,
        std::move(pose_priors),
        normalized_from_working,
        reconstruction);
    if (!bundle_adjuster->HasSufficientPosePriors()) {
      LOG(ERROR) << "Insufficient priors entered the Ceres problem: position="
                 << bundle_adjuster->NumContributingPositionPriors()
                 << ", rotation="
                 << bundle_adjuster->NumContributingRotationPriors() << ".";
      bundle_adjuster.reset();
      if (metric_from_original.has_value()) {
        reconstruction.Transform(Inverse(*metric_from_original));
      }
      return nullptr;
    }
    return bundle_adjuster;
  } catch (...) {
    reconstruction.Transform(Inverse(normalized_from_working));
    if (metric_from_original.has_value()) {
      reconstruction.Transform(Inverse(*metric_from_original));
    }
    throw;
  }
}

}  // namespace colmap
