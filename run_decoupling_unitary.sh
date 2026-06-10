#!/bin/bash
# =============================================================================
#  run_decoupling_unitary.sh — Stage 4: build 𝒰 = exp(−iÔ) via Chebyshev MPO
#  exponentiation and verify unitarity (toy dense cross-check + ladder).
#
#  Usage:  sbatch run_decoupling_unitary.sh
# =============================================================================

#SBATCH --job-name=decoup_U
#SBATCH --output=logs/decoup_U_%j.out
#SBATCH --error=logs/decoup_U_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "decoupling-unitary self-test on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" gauge_matter_unitary.jl
echo "Exit code: $?  — $(date)"
