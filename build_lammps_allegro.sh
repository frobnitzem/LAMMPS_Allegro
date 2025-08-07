#!/bin/bash

LAMMPS_ALLEGRO_ROOT=${HOME}/code/lammps_allegro

# Set the environment
module load PrgEnv-gnu
module load cray-mpich
module load craype-accel-amd-gfx90a
module load rocm/6.4.1
module load cmake

LAMMPS_ROOT=$LAMMPS_ALLEGRO_ROOT/lammps
PAIR_ALLEGRO_ROOT=$LAMMPS_ALLEGRO_ROOT/pair_nequip_allegro
cd $LAMMPS_ALLEGRO_ROOT

# Download the torch C++ library
LIBTORCH_URL=https://download.pytorch.org/libtorch/rocm6.4/libtorch-shared-with-deps-2.8.0%2Brocm6.4.zip
if [ ! -d "libtorch" ]; then
  wget ${LIBTORCH_URL}
  unzip libtorch-*
  rm -r libtorch-*
fi

# Get LAMMPS and pair_allegro
if [ ! -d "${LAMMPS_ROOT}" ]; then
  git clone --depth=1 https://github.com/lammps/lammps $LAMMPS_ROOT
fi
if [ ! -d "${PAIR_ALLEGRO_ROOT}" ]; then
  git clone --depth=1 https://github.com/mir-group/pair_nequip_allegro $PAIR_ALLEGRO_ROOT
  # Patch LAMMPS
  cd ${PAIR_ALLEGRO_ROOT}; ./patch_lammps.sh ${LAMMPS_ROOT}
fi

# Build LAMMPS
mkdir ${LAMMPS_ROOT}/build; cd ${LAMMPS_ROOT}/build
MPICXX=$(which CC)
CXX=${ROCM_PATH}/bin/hipcc
TORCH_PATH="${LAMMPS_ALLEGRO_ROOT}/libtorch"
cmake -DBUILD_MPI=on \
      -DPKG_KOKKOS=ON \
      -DKokkos_ENABLE_HIP=on \
      -DMPI_CXX_COMPILER=${MPICXX} \
      -DCMAKE_CXX_COMPILER=${CXX} \
      -DKokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=ON \
      -DCMAKE_HIPFLAGS="--offload-arch=${GPU_ARCH}" \
      -DCMAKE_CXX_FLAGS="-fdenormal-fp-math=ieee -fgpu-flush-denormals-to-zero -munsafe-fp-atomics -I$MPICH_DIR/include" \
      -DMKL_INCLUDE_DIR="/tmp" \
      -DUSE_MKLDNN=OFF \
      -DNEQUIP_AOT_COMPILE=ON \
      -DPKG_MOLECULE=ON \
      -DCMAKE_PREFIX_PATH="${TORCH_PATH}" \
      ../cmake
make -j 16 install


