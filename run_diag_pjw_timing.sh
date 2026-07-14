#!/bin/bash
# =============================================================================
#  run_diag_pjw_timing.sh — quick diagnostic: plaquette+JW MPO bond and the cost
#  of one DMRG sweep at D = 40/60/100, to locate the runtime bottleneck before
#  committing a long array job.
#
#  Usage:  sbatch run_diag_pjw_timing.sh
#          grep -E "MPO|sweep|dim" logs/diag_pjw_timing_*.out
# =============================================================================

#SBATCH --job-name=diag_pjw
#SBATCH --output=logs/diag_pjw_timing_%j.out
#SBATCH --error=logs/diag_pjw_timing_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

echo "plaquette+JW timing diagnostic on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" diag_pjw_timing.jl
echo "Exit code: $?  — $(date)"
