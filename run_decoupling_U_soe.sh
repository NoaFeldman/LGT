#!/bin/bash
# =============================================================================
#  run_decoupling_U_soe.sh — SoE-approximated Bender–Zohar decoupling unitary
#  𝒰 = Π_j exp(−iÔ_j)·exp(−iÔ_bdry) on the column-major snake MPS.  Self-test:
#  each factor unitary; 𝒰_SoE → exact-M decoupler as K grows; +boundary helps;
#  per-string bonds O(1) ≪ exact.  Plus a 3×4 build to report efficiency.
#
#  Usage:  sbatch run_decoupling_U_soe.sh
# =============================================================================

#SBATCH --job-name=dec_U_soe
#SBATCH --output=logs/dec_U_soe_%j.out
#SBATCH --error=logs/dec_U_soe_%j.err
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

echo "SoE decoupling-U test on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" decoupling_U_soe.jl
echo "Exit code: $?  — $(date)"
