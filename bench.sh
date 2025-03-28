# single GPU
mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos neigh full binsize 2.8 -var x 1 -var y 1 -var z 1 -in lj.in -log 'log.lj.1-1'

# 8-GPU
mpirun -np 1 lmp -k on g 8 -sf kk -pk kokkos neigh full binsize 2.8 -var x 2 -var y 2 -var z 2 -in lj.in -log 'log.lj.8-8'

LD_LIBRARY_PATH=/usr/lib/libtorch/lib lmp -sf kk -k on g 4 -pk kokkos newton on neigh full -in lj.in

# create si.lj
LD_LIBRARY_PATH=/usr/lib/libtorch/lib lmp -sf kk -k on g 4 -pk kokkos newton on neigh full -in si.in

