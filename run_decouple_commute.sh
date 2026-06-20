#!/bin/bash
# =============================================================================
#  run_decouple_commute.sh — does the Green's-function decoupling commute with
#  exact contraction of the MPS?  Quench on a 2×2 grid (only size where exact
#  contraction is feasible), then total + matter/gauge reduced-state fidelities
#  between [contract→decouple] and [decouple→contract] vs t.
#
#  Usage:  sbatch run_decouple_commute.sh
# =============================================================================

#SBATCH --job-name=dec_commute
#SBATCH --output=logs/dec_commute_%j.out
#SBATCH --error=logs/dec_commute_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs results

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export GKSwstype=nul

echo "decouple-commute test on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" decouple_commute_test.jl
echo "Exit code: $?  — $(date)"
