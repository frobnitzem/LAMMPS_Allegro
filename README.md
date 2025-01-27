# Container Build Repo

This build process creates a container for running accelerated
LAMMPS with the pair\_allegro style from the [mir-group](https://github.com/mir-group/pair_allegro).


# Build instructions

    podman build -f Dockerfile.rocm -t rocm:6.1.3
    podman build -f Dockerfile.mpi  -t rocm_mpich:6.1.3

Then, to test LAMMPS without the pair\_allegro addition,

    podman build -f Dockerfile.lammps -t lammps_rocm_mpich

or (on Frontier)

    podman build -v /opt/cray:/opt/cray:ro,Z -f Dockerfile.lammps.frontier -t lammps_frontier

---

Next, install pytorch for hip and setup the LAMMPS source

    podman build -f Dockerfile.lammps_allegro -t rmta

And finally do the LAMMPS build corresponding to local:

    podman build -f Dockerfile.allegro -t allegro

or Frontier:

    podman build -f Dockerfile.allegro.frontier -t allegro_frontier


# References

* Velocity-images build recipe templates:
 - [rockylinux](https://github.com/olcf/velocity-images/blob/main/rockylinux/templates/default.vtmp)
 - [rocm on rocky](https://github.com/olcf/velocity-images/blob/main/rocm/templates/rockylinux.vtmp)
 - [mpich](https://github.com/olcf/velocity-images/blob/main/mpich/templates/default.vtmp)

* [OLCF Containers Blog](https://developer.ornl.gov/2022-10-13-containers/)

* [LAMMPS@Infinity Hub](https://github.com/amd/InfinityHub-CI/tree/main/lammps)

* [pair\_allegro](https://github.com/mir-group/pair\_allegro)
