# Conda CUDA installation guide for COLMAP 4.1.1 with full pose priors

This guide builds and installs the custom COLMAP 4.1.1 branch that adds
covariance-weighted position and quaternion rotation priors to the standalone
bundle adjuster.

The primary configuration in this guide is a **CUDA-enabled, headless build**
for an NVIDIA GeForce RTX 5080 on Linux. The RTX 5080 has CUDA compute
capability **12.0**, so the build explicitly targets `sm_120` using CUDA 12.9.
The resulting COLMAP executable includes CUDA support for feature extraction,
feature matching, and CUDA-enabled dense reconstruction.

Repository:

```text
https://github.com/jizongFox/colmap
```

Feature branch:

```text
agent/bundle-adjuster-pose-priors
```

The recommended sequence is:

1. Verify the NVIDIA driver and RTX 5080.
2. Create a dedicated Conda environment.
3. Install the full CUDA 12.9 toolkit from NVIDIA's Conda channel.
4. Verify that `nvcc` can generate and execute native `sm_120` code.
5. Configure COLMAP with `CUDA_ENABLED=ON`, `MVS_ENABLED=ON`, and
   `CMAKE_CUDA_ARCHITECTURES=120`.
6. Build, test, and install COLMAP into the Conda environment.
7. Run a real COLMAP GPU feature-extraction and matching smoke test.
8. Validate the pose-prior database and run the native bundle_adjuster CLI.

The custom pose-prior residuals work with the normal Ceres bundle adjuster.
This CUDA build does **not by itself guarantee GPU sparse bundle adjustment**.
GPU sparse BA additionally requires Ceres to be built with CUDA and NVIDIA
cuDSS. Without cuDSS, COLMAP still uses the GPU for feature extraction,
matching, and dense MVS, while sparse BA falls back to a CPU sparse solver.

## 1. Host requirements

The commands below target Ubuntu 22.04 or Ubuntu 24.04 on x86-64 Linux.

Install the NVIDIA driver on the host operating system. Do not install a Linux
kernel driver inside the Conda environment. For the full CUDA 12.9 Update 1
driver/toolkit pairing, NVIDIA lists Linux driver `575.57.08` or newer. A newer
compatible driver is also acceptable.

Verify the GPU and driver:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

The GPU name should include `GeForce RTX 5080`.

Install Git and a host compiler:

```bash
sudo apt update
sudo apt install -y build-essential git
```

Verify the host tools:

```bash
git --version
gcc --version
g++ --version
```

CUDA 12.9 requires a supported host compiler. Ubuntu 22.04's default GCC 11
and Ubuntu 24.04's default GCC 13 are suitable choices. This guide deliberately
uses the Ubuntu host compiler instead of a Conda-packaged GCC toolchain.

Install Miniconda, Anaconda, or Miniforge before continuing. `mamba` can be
used in place of `conda` for faster dependency resolution.

## 2. Create the CUDA Conda environment

Do not install the stock `colmap` Conda package in this environment. The custom
executable will be compiled from this repository and installed directly into
`$CONDA_PREFIX`.

Create the environment with COLMAP's non-CUDA dependencies from conda-forge:

```bash
conda create -n colmap_4.1.1_priors_cuda \
    --override-channels \
    --strict-channel-priority \
    -c conda-forge \
    python=3.12 \
    "cmake>=3.24" \
    ninja \
    ccache \
    pkg-config \
    boost \
    eigen \
    openimageio \
    curl \
    metis \
    glog \
    gflags \
    gtest \
    ceres-solver \
    suitesparse \
    sqlite \
    cgal-cpp \
    freeimage \
    flann
```

Activate the environment:

```bash
conda activate colmap_4.1.1_priors_cuda
```

Install the complete CUDA 12.9 toolkit from NVIDIA's official Conda channel:

```bash
conda install \
    --override-channels \
    --strict-channel-priority \
    -c nvidia \
    -c conda-forge \
    "cuda=12.9.*"
```

Using NVIDIA's `cuda` meta-package is intentional. It supplies `nvcc`, CUDA
headers, the CUDA runtime, cuBLAS, cuSPARSE, cuSOLVER, cuRAND, CUB, and the
other development libraries required by a native CUDA build. A runtime-only
package such as `cuda-version` is not sufficient for compiling COLMAP.

Verify that a stock COLMAP package is not installed:

```bash
conda list | grep '^colmap ' || true
```

The command should normally return no result.

Verify that the environment contains the development toolkit:

```bash
echo "$CONDA_PREFIX"
which python
which cmake
which ninja
which nvcc
python --version
cmake --version
nvcc --version
conda list | grep -E '^(cuda|libcublas|libcurand|libcusolver|libcusparse)'
```

`which nvcc` must point inside the active environment, for example:

```text
.../envs/colmap_4.1.1_priors_cuda/bin/nvcc
```

## 3. Verify CUDA 12.9 and RTX 5080 code generation

The RTX 5080 is compute capability 12.0. CUDA 12.9 must therefore list
`sm_120` as a supported binary target:

```bash
nvcc --list-gpu-code | tr ' ' '\n' | grep -w sm_120
nvcc --list-gpu-arch | tr ' ' '\n' | grep -w compute_120
```

Both commands must print a matching target. If `sm_120` is absent, the active
`nvcc` is too old or the wrong compiler is first on `PATH`.

Compile and execute a native `sm_120` test before building COLMAP:

```bash
cat > /tmp/cuda_sm120_check.cu <<'EOF'
#include <cuda_runtime.h>

#include <iostream>

__global__ void Increment(int* value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *value += 1;
  }
}

int main() {
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    std::cerr << "Could not query CUDA device 0\n";
    return 1;
  }

  std::cout << "GPU: " << properties.name << "\n";
  std::cout << "Compute capability: " << properties.major << "."
            << properties.minor << "\n";

  if (properties.major != 12 || properties.minor != 0) {
    std::cerr << "Expected an RTX 5080-class sm_120 device\n";
    return 2;
  }

  int input = 41;
  int output = 0;
  int* device_value = nullptr;
  if (cudaMalloc(&device_value, sizeof(int)) != cudaSuccess) {
    return 3;
  }
  if (cudaMemcpy(device_value,
                 &input,
                 sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(device_value);
    return 4;
  }

  Increment<<<1, 1>>>(device_value);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    cudaFree(device_value);
    return 5;
  }
  if (cudaMemcpy(&output,
                 device_value,
                 sizeof(int),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(device_value);
    return 6;
  }
  cudaFree(device_value);

  if (output != 42) {
    return 7;
  }

  std::cout << "Native sm_120 CUDA execution passed\n";
  return 0;
}
EOF

nvcc \
    -std=c++14 \
    -O2 \
    -arch=sm_120 \
    /tmp/cuda_sm120_check.cu \
    -o /tmp/cuda_sm120_check

/tmp/cuda_sm120_check
```

This test verifies four separate requirements:

- the Conda environment contains a CUDA compiler;
- the compiler recognizes `sm_120`;
- the NVIDIA driver can load CUDA 12.x code;
- a native kernel executes successfully on the RTX 5080.

Do not continue to the COLMAP build until this test passes.

## 4. Clone the custom branch

```bash
mkdir -p "$HOME/workspace"
cd "$HOME/workspace"

git clone https://github.com/jizongFox/colmap.git colmap-pose-priors
cd colmap-pose-priors

git fetch origin
git switch agent/bundle-adjuster-pose-priors
```

Verify the checkout:

```bash
git status -sb
git log -1 --oneline
grep 'COLMAP_VERSION' CMakeLists.txt
```

The version should be `4.1.1`.

### Updating an older checkout

The feature branch is based on COLMAP 4.1.1. To synchronize an old local
checkout, first preserve any local changes and then run:

```bash
cd "$HOME/workspace/colmap-pose-priors"
git fetch origin
git switch agent/bundle-adjuster-pose-priors
git reset --hard origin/agent/bundle-adjuster-pose-priors
```

`git reset --hard` deletes uncommitted work. Do not run it until anything
important has been committed or backed up.

## 5. Configure the CUDA/headless build

A headless CUDA build avoids Qt, OpenGL, and display-server dependencies while
retaining CUDA SIFT, GPU matching, and CUDA dense reconstruction.

Activate the environment and select the toolchain explicitly:

```bash
conda activate colmap_4.1.1_priors_cuda
cd "$HOME/workspace/colmap-pose-priors"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDA_HOME="$CONDA_PREFIX"
export CUDAToolkit_ROOT="$CONDA_PREFIX"
export CUDAARCHS=120
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

rm -rf build-conda-cuda
```

Configure COLMAP:

```bash
cmake -S . -B build-conda-cuda -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCUDAToolkit_ROOT="$CONDA_PREFIX" \
    -DCMAKE_BUILD_RPATH="$CONDA_PREFIX/lib;$CONDA_PREFIX/targets/x86_64-linux/lib" \
    -DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib;$CONDA_PREFIX/targets/x86_64-linux/lib" \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$CXX" \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DGUI_ENABLED=OFF \
    -DOPENGL_ENABLED=OFF \
    -DCUDA_ENABLED=ON \
    -DMVS_ENABLED=ON \
    -DONNX_ENABLED=OFF \
    -DTESTS_ENABLED=ON \
    -DFETCH_FAISS=ON \
    -DFETCH_POSELIB=ON
```

Important choices:

- `CUDA_ENABLED=ON` compiles COLMAP's CUDA feature paths.
- `MVS_ENABLED=ON` keeps CUDA PatchMatch stereo and dense reconstruction.
- `CMAKE_CUDA_ARCHITECTURES=120` generates native RTX 5080 cubin code and
  `compute_120` PTX. CMake generates both real and virtual code when no
  `-real` or `-virtual` suffix is supplied.
- `CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++` prevents `nvcc` from selecting an
  incompatible Conda host compiler.
- `CUDAToolkit_ROOT=$CONDA_PREFIX` prevents CMake from silently selecting an
  unrelated `/usr/local/cuda` installation.
- `CMAKE_CUDA_RUNTIME_LIBRARY=Shared` uses the Conda CUDA runtime dynamically.
- RPATH includes both common NVIDIA Conda library locations.
- `FETCH_FAISS=ON` avoids ABI mismatches with an unrelated FAISS installation.
- `GUI_ENABLED=OFF` and `OPENGL_ENABLED=OFF` do not disable CUDA.

Do not replace architecture `120` with `89`, `90`, or `100` for an RTX 5080.
`native` is also supported by recent CMake, but explicit `120` makes the
configuration reproducible and easy to audit.

## 6. Verify the generated CMake configuration

Inspect the cache before compiling:

```bash
grep -E \
  'CMAKE_CUDA_COMPILER:|CMAKE_CUDA_HOST_COMPILER:|CMAKE_CUDA_ARCHITECTURES:|CUDAToolkit_ROOT:|CUDA_ENABLED:|MVS_ENABLED:' \
  build-conda-cuda/CMakeCache.txt
```

The output must show values equivalent to:

```text
CMAKE_CUDA_COMPILER:FILEPATH=<conda-prefix>/bin/nvcc
CMAKE_CUDA_HOST_COMPILER:FILEPATH=/usr/bin/g++
CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=120
CUDAToolkit_ROOT:UNINITIALIZED=<conda-prefix>
CUDA_ENABLED:BOOL=ON
MVS_ENABLED:BOOL=ON
```

The exact CMake cache type may differ, but the values must be correct.

Also inspect the configure summary for a detected CUDA compiler and toolkit:

```bash
cmake -S . -B build-conda-cuda -GNinja -LAH \
    | grep -E 'CUDA|CUDAToolkit|MVS_ENABLED'
```

If the cache points to `/usr/local/cuda`, delete `build-conda-cuda` and repeat
the configuration after re-exporting `CUDACXX` and `CUDAToolkit_ROOT`.

## 7. Build COLMAP with CUDA

Build the full project:

```bash
cmake --build build-conda-cuda --parallel "$(nproc)"
```

CUDA compilation can use substantial RAM. For a laptop, a safer starting point
is:

```bash
cmake --build build-conda-cuda --parallel 8
```

To audit the actual CUDA compiler commands, rebuild one or more targets with
verbose output:

```bash
cmake --build build-conda-cuda --verbose 2>&1 \
    | tee /tmp/colmap_cuda_build.log

grep -E -- 'nvcc|compute_120|sm_120' /tmp/colmap_cuda_build.log | head -n 30
```

If the project was already fully built, Ninja may have nothing to compile. In
that case, remove one CUDA object or perform a clean rebuild before collecting
verbose commands.

## 8. Run tests and install

Run the existing bundle-adjustment regression test and the new database
pose-prior tests:

```bash
ctest \
    --test-dir build-conda-cuda \
    --output-on-failure \
    -R 'controllers/(database_pose_prior_bundle_adjustment|bundle_adjustment)_test'
```

Run every compiled test when practical:

```bash
ctest --test-dir build-conda-cuda --output-on-failure
```

Install into the CUDA Conda environment:

```bash
cmake --install build-conda-cuda
hash -r
```

Verify the installed executable:

```bash
which colmap
colmap --version
colmap -h
```

`which colmap` should point inside the active environment:

```text
.../envs/colmap_4.1.1_priors_cuda/bin/colmap
```

Check runtime libraries:

```bash
ldd "$CONDA_PREFIX/bin/colmap" | grep 'not found' || true
ldd "$CONDA_PREFIX/bin/colmap" \
    | grep -E 'cuda|cudart|cublas|cusparse|cusolver' || true
```

There must be no missing libraries.

## 9. Run a real COLMAP GPU smoke test

A successful CUDA compilation is necessary but not sufficient. This test forces
COLMAP's GPU feature-extraction and GPU matching paths on device 0.

Create two deterministic synthetic PGM images using only Python's standard
library:

```bash
rm -rf /tmp/colmap_cuda_smoke
mkdir -p /tmp/colmap_cuda_smoke/images

python - <<'PY'
from pathlib import Path

width = 1024
height = 768
root = Path('/tmp/colmap_cuda_smoke/images')

for image_index, shift in enumerate((0, 7)):
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            xs = (x + shift) % width
            checker = 160 if ((xs // 24) ^ (y // 24)) & 1 else 30
            texture = (13 * xs + 7 * y + 17 * image_index) % 80
            pixels[y * width + x] = min(255, checker + texture)

    path = root / f'image_{image_index:02d}.pgm'
    path.write_bytes(
        f'P5\n{width} {height}\n255\n'.encode('ascii') + bytes(pixels)
    )
PY
```

Force GPU SIFT extraction:

```bash
CUDA_VISIBLE_DEVICES=0 colmap feature_extractor \
    --database_path /tmp/colmap_cuda_smoke/database.db \
    --image_path /tmp/colmap_cuda_smoke/images \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.type SIFT \
    --FeatureExtraction.use_gpu 1 \
    --FeatureExtraction.gpu_index 0
```

Force GPU matching:

```bash
CUDA_VISIBLE_DEVICES=0 colmap exhaustive_matcher \
    --database_path /tmp/colmap_cuda_smoke/database.db \
    --FeatureMatching.use_gpu 1 \
    --FeatureMatching.gpu_index 0
```

Inspect the database:

```bash
sqlite3 /tmp/colmap_cuda_smoke/database.db \
    'SELECT COUNT(*) AS images FROM images;
     SELECT COUNT(*) AS keypoint_rows FROM keypoints;
     SELECT COUNT(*) AS match_rows FROM matches;'
```

The commands must complete without messages such as:

```text
CUDA driver version is insufficient for CUDA runtime version
no kernel image is available for execution on the device
SiftGPU not fully supported
```

This smoke test is the practical confirmation that the installed COLMAP binary
can execute its GPU paths on the RTX 5080.

## 10. Validate the pose-prior database

Inspect the schema before running bundle adjustment:

```bash
sqlite3 /path/to/database.db 'PRAGMA table_info(pose_priors);'
```

The custom reader requires the standard position-prior fields and a quaternion
rotation field.

Required standard columns:

```text
corr_data_id
corr_sensor_id
corr_sensor_type
position
position_covariance
coordinate_system
```

Supported quaternion column aliases:

```text
rotation
rotation_quaternion
rotation_prior
prior_qvec
qvec
```

Supported rotation-covariance aliases:

```text
rotation_covariance
rotation_prior_covariance
prior_qvec_covariance
qvec_covariance
```

Data conventions and validation:

- Position is the Cartesian camera center in world coordinates.
- `coordinate_system` must be exactly `CARTESIAN`; WGS84, undefined, and
  unrecognized values are rejected.
- Position covariance is a finite positive-definite 3x3 matrix.
- Quaternion BLOB order is `qw, qx, qy, qz`.
- Quaternion orientation is COLMAP world-to-camera rotation.
- Quaternion signs are arbitrary: `q` and `-q` are treated equivalently.
- Rotation covariance is a finite positive-definite 3x3 covariance of the
  right-invariant SO(3) residual `Log(R_current R_prior^T)`, measured in
  radians squared and expressed in the camera/sensor tangent frame.
- The complete `(corr_data_id, corr_sensor_id, corr_sensor_type)` identifier
  must equal the corresponding image data identifier in the input model.
- Duplicate rows for the same image are ignored rather than counted as
  independent absolute poses.
- At least three distinct registered camera poses must contribute prior
  residuals to the Ceres problem.
- Any SQLite error while stepping through the query aborts prior loading rather
  than accepting a truncated result set.

## 11. Run pose-prior bundle adjustment through the native CLI

The pose-prior feature is exposed directly by `colmap bundle_adjuster`. The
hidden environment variable and Python launcher are no longer used.

New native options:

```text
--BundleAdjustment.pose_prior_database_path
--BundleAdjustment.use_position_priors
--BundleAdjustment.use_rotation_priors
--BundleAdjustment.prior_position_fallback_stddev
--BundleAdjustment.prior_rotation_fallback_stddev_rad
--BundleAdjustment.prior_position_loss_scale
--BundleAdjustment.prior_rotation_loss_scale
```

Both prior switches default to `0`, so an ordinary invocation preserves stock
COLMAP behavior.

### Position and rotation priors

```bash
colmap bundle_adjuster \
    --input_path /path/to/project/sparse/input \
    --output_path /path/to/project/sparse/output \
    --BundleAdjustment.pose_prior_database_path /path/to/project/database.db \
    --BundleAdjustment.use_position_priors 1 \
    --BundleAdjustment.use_rotation_priors 1
```

### Position priors only

```bash
colmap bundle_adjuster \
    --input_path /path/to/project/sparse/input \
    --output_path /path/to/project/sparse/output_position \
    --BundleAdjustment.pose_prior_database_path /path/to/project/database.db \
    --BundleAdjustment.use_position_priors 1 \
    --BundleAdjustment.use_rotation_priors 0
```

Position priors establish the metric frame and remove the full similarity gauge.
At least three distinct registered images with matching Cartesian priors must
contribute residuals.

### Rotation priors only

```bash
colmap bundle_adjuster \
    --input_path /path/to/project/sparse/input \
    --output_path /path/to/project/sparse/output_rotation \
    --BundleAdjustment.pose_prior_database_path /path/to/project/database.db \
    --BundleAdjustment.use_position_priors 0 \
    --BundleAdjustment.use_rotation_priors 1
```

Rotation-only mode uses a translation-and-scale gauge that leaves global
orientation free. It requires at least two variable frames with a non-zero
baseline and at least one matching rotation prior.

### Stock bundle adjustment

```bash
colmap bundle_adjuster \
    --input_path /path/to/project/sparse/input \
    --output_path /path/to/project/sparse/output_stock
```

This is equivalent to explicitly setting both switches to `0`.

### Fixed calibrated intrinsics

```bash
colmap bundle_adjuster \
    --input_path /path/to/project/sparse/input \
    --output_path /path/to/project/sparse/output \
    --BundleAdjustment.pose_prior_database_path /path/to/project/database.db \
    --BundleAdjustment.use_position_priors 1 \
    --BundleAdjustment.use_rotation_priors 1 \
    --BundleAdjustment.refine_focal_length 0 \
    --BundleAdjustment.refine_principal_point 0 \
    --BundleAdjustment.refine_extra_params 0
```

A prior-configuration failure terminates the command with a non-zero exit code
and does not write an unchanged output model.

## 12. GPU bundle adjustment and cuDSS

The Conda `ceres-solver` package used above may be a CPU-only Ceres build. That
does not prevent COLMAP itself from compiling with CUDA, and it does not prevent
GPU feature extraction, matching, or dense reconstruction.

To request GPU sparse BA when Ceres was built with CUDA and cuDSS, pass:

```text
--BundleAdjustmentCeres.use_gpu 1
```

For example:

```bash
colmap bundle_adjuster \
    --input_path /path/to/project/sparse/input \
    --output_path /path/to/project/sparse/output_gpu_ba \
    --BundleAdjustment.pose_prior_database_path /path/to/project/database.db \
    --BundleAdjustment.use_position_priors 1 \
    --BundleAdjustment.use_rotation_priors 1 \
    --BundleAdjustmentCeres.use_gpu 1
```

If the log says that Ceres was compiled without cuDSS support and is falling
back to a CPU sparse solver, then the COLMAP CUDA build is still valid; only the
linear solver used by BA is running on the CPU. Enabling GPU sparse BA requires
installing NVIDIA cuDSS and rebuilding Ceres from source with both CUDA and
cuDSS enabled, followed by rebuilding COLMAP against that Ceres installation.

Do not interpret `CUDA_ENABLED=ON` as proof that Ceres itself contains cuDSS.
The GPU smoke test in section 9 verifies COLMAP's native CUDA paths. The BA log
verifies whether the optional Ceres/cuDSS path is available.

## 13. Inspect the optimized reconstruction

```bash
colmap model_analyzer \
    --path /path/to/project/sparse/output
```

Optionally convert it to text:

```bash
mkdir -p /path/to/project/sparse/output_txt

colmap model_converter \
    --input_path /path/to/project/sparse/output \
    --output_path /path/to/project/sparse/output_txt \
    --output_type TXT
```

For real validation, compare:

- reprojection error before and after optimization;
- camera-center error relative to position priors;
- quaternion angular error relative to rotation priors;
- registered-image count;
- point and observation counts;
- output scale and coordinate frame.

## 14. Updating and rebuilding

```bash
conda activate colmap_4.1.1_priors_cuda
cd "$HOME/workspace/colmap-pose-priors"

git pull --ff-only

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDAToolkit_ROOT="$CONDA_PREFIX"
export CUDAARCHS=120

cmake --build build-conda-cuda --parallel 8
ctest \
    --test-dir build-conda-cuda \
    --output-on-failure \
    -R 'controllers/(database_pose_prior_bundle_adjustment|bundle_adjustment)_test'
cmake --install build-conda-cuda
hash -r
```

After changing the CUDA toolkit, host compiler, architecture, or major CMake
options, delete `build-conda-cuda` and configure from scratch instead of reusing
the old cache.

## 15. Troubleshooting

### `nvcc` does not list `sm_120`

The wrong CUDA compiler is active or the toolkit is too old:

```bash
which -a nvcc
nvcc --version
nvcc --list-gpu-code | tr ' ' '\n' | grep -w sm_120
conda list | grep '^cuda '
```

Reactivate the CUDA environment and reinstall `cuda=12.9.*` from NVIDIA's
channel. Delete the CMake build directory afterward.

### `Unsupported gpu architecture 'compute_120'`

This is another indication that CMake selected an older `nvcc`. Check:

```bash
grep 'CMAKE_CUDA_COMPILER:' build-conda-cuda/CMakeCache.txt
```

It must point to `$CONDA_PREFIX/bin/nvcc`.

### `no kernel image is available for execution on the device`

The binary was built without compatible Blackwell code. Delete the build
folder and configure again with:

```text
-DCMAKE_CUDA_ARCHITECTURES=120
```

Do not reuse CUDA objects compiled only for older architectures.

### CUDA driver/runtime mismatch

Check:

```bash
nvidia-smi
nvcc --version
/tmp/cuda_sm120_check
```

The host NVIDIA driver must support the CUDA toolkit. For the full CUDA 12.9
Update 1 pairing, use Linux driver `575.57.08` or newer.

### CUDA host compiler is unsupported

Use Ubuntu's compiler explicitly:

```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
```

Configure from a clean build directory with:

```text
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
```

Do not add `--allow-unsupported-compiler` unless you have deliberately accepted
an unsupported toolchain combination.

### CMake finds `/usr/local/cuda` instead of Conda CUDA

Remove the old cache and export:

```bash
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDAToolkit_ROOT="$CONDA_PREFIX"
rm -rf build-conda-cuda
```

Then pass both `CMAKE_CUDA_COMPILER` and `CUDAToolkit_ROOT` during configuration.

### `GL/gl.h` is missing

Keep the build headless:

```text
-DGUI_ENABLED=OFF
-DOPENGL_ENABLED=OFF
```

Headless mode still supports CUDA feature extraction, matching, and MVS.

### FAISS undefined symbol

Do not mix the source build with an independently installed Conda FAISS
library. Remove conflicting FAISS packages, delete the build directory, and
configure again with:

```text
-DFETCH_FAISS=ON
```

### Wrong COLMAP executable

```bash
which -a colmap
conda list | grep '^colmap '
```

Remove the stock package if necessary, reinstall the source build, and refresh
the shell cache.

### Runtime library is missing

```bash
ldd "$CONDA_PREFIX/bin/colmap" | grep 'not found'
```

Reconfigure with the RPATH options shown in this guide. Also confirm that
`$CONDA_PREFIX/targets/x86_64-linux/lib` exists.

### BA falls back to CPU

A warning about missing cuDSS concerns the Ceres sparse linear solver, not the
COLMAP CUDA build. Verify GPU extraction and matching using section 9. Rebuild
Ceres with CUDA and cuDSS only when GPU sparse BA is specifically required.

## 16. Final verification checklist

```bash
conda activate colmap_4.1.1_priors_cuda

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
which nvcc
nvcc --version
nvcc --list-gpu-code | tr ' ' '\n' | grep -w sm_120
/tmp/cuda_sm120_check

which colmap
colmap --version

 grep -E \
  'CMAKE_CUDA_COMPILER:|CMAKE_CUDA_ARCHITECTURES:|CUDA_ENABLED:|MVS_ENABLED:' \
  "$HOME/workspace/colmap-pose-priors/build-conda-cuda/CMakeCache.txt"

ctest \
    --test-dir "$HOME/workspace/colmap-pose-priors/build-conda-cuda" \
    --output-on-failure \
    -R 'controllers/(database_pose_prior_bundle_adjustment|bundle_adjustment)_test'

sqlite3 /path/to/database.db 'PRAGMA table_info(pose_priors);'

colmap bundle_adjuster -h
```

Also complete the real COLMAP GPU smoke test in section 9. The build should not
be considered GPU-validated on the RTX 5080 until both the standalone
`sm_120` CUDA test and COLMAP GPU feature extraction complete successfully.

For the first real reconstruction, preserve the input model and write the
optimized reconstruction to a separate output directory.

## References

- NVIDIA CUDA GPUs and compute capability:
  https://developer.nvidia.com/cuda-gpus
- NVIDIA CUDA 12.9 Linux installation guide:
  https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-linux/
- NVIDIA CUDA 12.9 compiler target documentation:
  https://docs.nvidia.com/cuda/archive/12.9.0/cuda-compiler-driver-nvcc/
- NVIDIA CUDA 12.9 release notes and driver requirements:
  https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/
- CMake CUDA architectures property:
  https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
- COLMAP installation guide:
  https://colmap.github.io/install.html
- COLMAP command-line guide:
  https://colmap.github.io/cli.html
- NVIDIA cuDSS documentation:
  https://docs.nvidia.com/cuda/cudss/
