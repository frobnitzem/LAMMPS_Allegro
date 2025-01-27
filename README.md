
Build instructions:
```
podman build -f Dockerfile.rocm -t rocm:6.1.3
podman build -f Dockerfile.mpi  -t rocm_mpich:6.1.3
```

Then
```
podman build -f Dockerfile.lammps -t lammps_rocm_mpich
```

or (on Frontier)
```
podman build -v /opt/cray:/opt/cray:ro,Z -f Dockerfile.lammps.frontier -t lammps_frontier
```

Refs:

* Velocity-images build recipe templates:
 - https://github.com/olcf/velocity-images/blob/main/rockylinux/templates/default.vtmp
 - https://github.com/olcf/velocity-images/blob/main/rocm/templates/rockylinux.vtmp
 - https://github.com/olcf/velocity-images/blob/main/mpich/templates/default.vtmp

* OLCF Containers https://developer.ornl.gov/2022-10-13-containers/

* https://github.com/amd/InfinityHub-CI/tree/main/lammps

* https://github.com/mir-group/pair\_allegro
