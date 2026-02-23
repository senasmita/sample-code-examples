#!/bin/bash
#SBATCH --job-name=AL_SNAP
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=16                 # <-- ORCA MPI ranks
#SBATCH --cpus-per-task=1           # <-- pure MPI (no hybrid threads)
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# ---------------------------
# EDIT THESE PATHS
# ---------------------------
INP_DIR=/home/users/assen/proj1/active_site
OUT_DIR=/home/users/assen/proj1/active_site

ORCA_EXE=/home/users/assen/orca/orca

LAMMPS_PYDIR=/home/users/assen/mylammps/python
LAMMPS_LIBDIR=/home/users/assen/mylammps/src
# ---------------------------

# MPI ranks = Slurm tasks
ORCA_NPROCS="${SLURM_NTASKS:-16}"

# Scratch
SCRATCH_BASE="${SLURM_TMPDIR:-/dev/shm}"
SCRATCH="${SCRATCH_BASE}/${USER}/AL_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH}"

echo "SCRATCH=${SCRATCH}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-NOT_SET}"
echo "ORCA_NPROCS=${ORCA_NPROCS}"

module purge
module load python/3.11.6-gcc-8.5.0-x3v4mtg

# ---------------------------
# OpenMPI runtime for ORCA's internal MPI startup
# (mpirun + libmpi.so.40)
# ---------------------------
export OMPI_ROOT=/home/support/rl8/spack/0.21.2/spack/opt/spack/linux-rocky8-ivybridge/gcc-8.5.0/openmpi-4.1.6-j2yerfm6jbaucfqamgxvtruxcr53r5bv
export PATH="${OMPI_ROOT}/bin:${PATH}"

# Combine OpenMPI libs + LAMMPS libs in one LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${OMPI_ROOT}/lib:${OMPI_ROOT}/lib64:${LAMMPS_LIBDIR}:${LD_LIBRARY_PATH:-}"

echo "mpirun in job: $(which mpirun)"
mpirun --version | head -n 2 || true

echo "libmpi check:"
ls -l "${OMPI_ROOT}/lib/libmpi.so.40" 2>/dev/null || true
ls -l "${OMPI_ROOT}/lib64/libmpi.so.40" 2>/dev/null || true
ldd /home/users/assen/orca/orca_startup_mpi | grep -E "libmpi|not found" || true

# ---------------------------
# Keep LAMMPS serial + numpy/blas single-threaded
# ORCA is MPI-only here, so keep OMP threads at 1 to avoid oversubscription.
# ---------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ORCA scratch on local disk
export TMPDIR="${SCRATCH}"
export ORCA_TMPDIR="${SCRATCH}"
export ORCA_SCRDIR="${SCRATCH}"

# Make LAMMPS python visible (serial build is fine)
export PYTHONPATH="${LAMMPS_PYDIR}:${PYTHONPATH:-}"

# Unbuffered python logging
export PYTHONUNBUFFERED=1

# Sanity checks
test -x "${ORCA_EXE}" || { echo "ERROR: ORCA_EXE not executable: ${ORCA_EXE}"; exit 1; }
python3 -u -c "import lammps; print('OK: lammps imported from', lammps.__file__)"

# ---------------------------
# Stage work directory to scratch
# ---------------------------
rsync -a "${INP_DIR}/" "${SCRATCH}/work/"
cd "${SCRATCH}/work"

# Write outputs live to OUT_DIR
mkdir -p "${OUT_DIR}/runs" "${OUT_DIR}/train_pool"
rm -rf runs train_pool
ln -s "${OUT_DIR}/runs" runs
ln -s "${OUT_DIR}/train_pool" train_pool

echo "Linked outputs:"
echo "  runs -> $(readlink -f runs)"
echo "  train_pool -> $(readlink -f train_pool)"

# Pass ORCA settings to python
export AL_ORCA_EXE="${ORCA_EXE}"
export AL_ORCA_NPROCS="${ORCA_NPROCS}"

echo "AL_ORCA_EXE=${AL_ORCA_EXE}"
echo "AL_ORCA_NPROCS=${AL_ORCA_NPROCS}"
echo "Final PATH=${PATH}"
echo "Final LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# Run driver (single process)
echo "Running driver..."
python3 -u al_snap_loop_interrupt_nomin_ID.py 2>&1 | tee "${OUT_DIR}/driver_${SLURM_JOB_ID}.log"
py_rc=${PIPESTATUS[0]}

echo "Python exit code: ${py_rc}"
echo "Done: $(date)"
exit "${py_rc}"

