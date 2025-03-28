# Container Build Repo

This build process creates a container for running accelerated
LAMMPS with the pair\_allegro style from the [mir-group](https://github.com/mir-group/pair_allegro).


# Build instructions

    podman build -f Dockerfile.rocm -t rocm:6.2.4
    podman build -f Dockerfile.mpi  -t rocm_mpich:6.2.4

Then, to test LAMMPS without the pair\_allegro addition,

    podman build -f Dockerfile.lammps -t lammps_rocm_mpich

or (on Frontier) -- Note this does not work because podman build priviledges inside container are insufficient [sic]

    podman build -v /opt/cray:/opt/cray:ro,Z -f Dockerfile.lammps.frontier -t lammps_frontier

---

Next, install libtorch and miopen for hip

    podman build -f Dockerfile.libtorch -t libtorch:6.2.4

And finally do the LAMMPS-Allegro build:

    podman build -f Dockerfile.lammps_allegro -t rmta

<!--
or Frontier:

    podman build -f Dockerfile.allegro.frontier -t allegro_frontier
-->

# Running Images on Frontier

## Convert image to apptainer

    podman save -o oci-lammps.tar --format oci-archive lammps_rocm_mpich 
    scp oci-lammps.tar dtn:/lustre/orion/mat281/proj-shared/25_01_27_AllegroLammpsContainer/
    # on Frontier
    apptainer build lammps.sif oci-archive://oci-lammps.tar

## Example Frontier job-script

    #!/bin/bash
    #SBATCH -A mat281
    #SBATCH -N 2
    #SBATCH -t 30

    USER=`whoami`
    export APPTAINER_CACHEDIR=/lustre/orion/mat281/scratch/$USER/cache
    mkdir -p $APPTAINER_CACHEDIR

    module reset
    module load PrgEnv-gnu
    module load olcf-container-tools
    module load apptainer-enable-mpi apptainer-enable-gpu
    module load rocm/6.2.4

    srun -N 2 -n 16 --gpus-per-task=1 --gpu-bind=closest --unbuffered \
        singularity exec lammps.sif \
            lmp -k on g 1 -sf kk -pk kokkos gpu/aware on -in lj.in

Additional example LAMMPS invocations are present in `bench.sh`.

# References

* Velocity-images build recipe templates:
 - [rockylinux](https://github.com/olcf/velocity-images/blob/main/rockylinux/templates/default.vtmp)
 - [rocm on rocky](https://github.com/olcf/velocity-images/blob/main/rocm/templates/rockylinux.vtmp)
 - [mpich](https://github.com/olcf/velocity-images/blob/main/mpich/templates/default.vtmp)

* [OLCF Containers Blog](https://developer.ornl.gov/2022-10-13-containers/)

* [LAMMPS@Infinity Hub](https://github.com/amd/InfinityHub-CI/tree/main/lammps)

* [pair\_allegro](https://github.com/mir-group/pair\_allegro)
