#!/bin/bash
# =============================================================================
#  run_matter_gauge_entanglement_pjw.sh — matter vs dual-plaquette gauge
#  half-system entanglement of the Bender–Zohar-decoupled 3×4 U(1) LGT ground
#  state WITH a magnetic plaquette term and Jordan–Wigner strings, over the
#  9 (m,g) points of the gs_benchmark grid, as a SLURM array (one point per task).
#
#  Optional one-off correctness gate (dense MPO self-test on 2×2):
#      PJW_SELFTEST=1 julia --project=. matter_gauge_entanglement_pjw.jl
#
#  Usage:
#    sbatch run_matter_gauge_entanglement_pjw.sh
#    # then aggregate + plot:
#    sbatch run_matter_gauge_entanglement_pjw_collect.sh
# =============================================================================

#SBATCH --job-name=mg_ent_pjw
#SBATCH --output=logs/mg_ent_pjw_%A_%a.out
#SBATCH --error=logs/mg_ent_pjw_%A_%a.err
#SBATCH --array=1-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs results

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "matter_gauge_entanglement_pjw task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" matter_gauge_entanglement_pjw.jl
echo "Exit code: $?  — $(date)"
