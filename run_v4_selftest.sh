#!/bin/bash
# =============================================================================
#  run_v4_selftest.sh — validate the v4 boundary-MPS environment under SLURM
#  (dedicated node memory; the reduced environment is multi-GB).
#
#  Usage:  sbatch run_v4_selftest.sh
# =============================================================================

#SBATCH --job-name=v4_selftest
#SBATCH --output=logs/v4_selftest_%j.out
#SBATCH --error=logs/v4_selftest_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export GKSwstype=nul

echo "v4 self-test on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" v4_selftest.jl
echo "Exit code: $?  — $(date)"
