#!/bin/bash
# =============================================================================
#  run_gs_benchmark_mps.sh — MPS-DMRG vs ED ground-state benchmark, 9 (m,g)
#  points on the 3×4 U(1) LGT, as a SLURM array (one point per task).
#
#  Usage:
#    sbatch run_gs_benchmark_mps.sh
#    # then aggregate:
#    julia --project=. gs_benchmark_mps_collect.jl
# =============================================================================

#SBATCH --job-name=gs_mps
#SBATCH --output=logs/gs_mps_%A_%a.out
#SBATCH --error=logs/gs_mps_%A_%a.err
#SBATCH --array=1-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs results

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "gs_benchmark_mps task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" gs_benchmark_mps.jl
echo "Exit code: $?  — $(date)"
