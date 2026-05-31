#!/bin/bash
#SBATCH --job-name=v4_update_test
#SBATCH --output=logs/v4_update_test_%j.out
#SBATCH --error=logs/v4_update_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs
export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export GKSwstype=nul

echo "v4 update test on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" v4_update_test.jl
echo "Exit code: $?  — $(date)"
