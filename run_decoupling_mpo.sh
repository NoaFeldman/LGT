#!/bin/bash
# =============================================================================
#  run_decoupling_mpo.sh — Stage 3 self-test of the gauge-matter decoupling
#  exponent MPO (FSA compiler vs analytic target + truncation report).
#
#  Usage:  sbatch run_decoupling_mpo.sh
# =============================================================================

#SBATCH --job-name=decoup_mpo
#SBATCH --output=logs/decoup_mpo_%j.out
#SBATCH --error=logs/decoup_mpo_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "decoupling-MPO self-test on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" gauge_matter_decoupling_mpo.jl
echo "Exit code: $?  — $(date)"
