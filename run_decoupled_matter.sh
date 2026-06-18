#!/bin/bash
# =============================================================================
#  run_decoupled_matter.sh — matter-only decoupled DMRG (gauge field replaced
#  by the Green's-function Coulomb MPO) vs the full gauged ED, over the 9 (m,g)
#  points.  Quantifies how well the Coulomb-only decoupling reproduces the
#  matter observables.  Sequential (9 points in one job; ED dominates the cost).
#
#  Usage:  sbatch run_decoupled_matter.sh
# =============================================================================

#SBATCH --job-name=dec_matter
#SBATCH --output=logs/dec_matter_%j.out
#SBATCH --error=logs/dec_matter_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs results

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "decoupled-matter vs ED on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" decoupled_matter_benchmark.jl
echo "Exit code: $?  — $(date)"
