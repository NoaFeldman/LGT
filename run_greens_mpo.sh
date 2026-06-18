#!/bin/bash
# =============================================================================
#  run_greens_mpo.sh — self-test the Green's-function decoupling MPO
#  (MPS twin of the efficient PEPO): exact-G MPO vs dense Coulomb operator,
#  SoE-approximated bond ≪ N, and the decoupling-approximation error.
#
#  Usage:  sbatch run_greens_mpo.sh
# =============================================================================

#SBATCH --job-name=greens_mpo
#SBATCH --output=logs/greens_mpo_%j.out
#SBATCH --error=logs/greens_mpo_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "greens-MPO decoupling self-test on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" efficient_greens_mpo.jl
echo "Exit code: $?  — $(date)"
