#!/bin/bash
# =============================================================================
#  run_mps_selftest.sh — validate the MPS-LGT solver core (DMRG vs dense ED
#  on small lattices) before building the 3×4 benchmark/quench drivers.
#
#  Usage:  sbatch run_mps_selftest.sh
# =============================================================================

#SBATCH --job-name=mps_selftest
#SBATCH --output=logs/mps_selftest_%j.out
#SBATCH --error=logs/mps_selftest_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "MPS-LGT solver self-test on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" mps_lgt.jl
echo "Exit code: $?  — $(date)"
