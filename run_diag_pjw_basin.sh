#!/bin/bash
# =============================================================================
#  run_diag_pjw_basin.sh — decide whether the cheap vacuum DMRG start is trapped
#  on 3×4 (vacuum vs flux-seeded basins), so we know if production needs the
#  expensive multi-start at all.  Per-sweep energies are printed, so even a
#  timeout leaves useful partial output.
#
#  Usage:  sbatch run_diag_pjw_basin.sh
#          grep -E "start|E_pen|verdict|TRAPPED|NOT trapped|RSS" logs/diag_pjw_basin_*.out
# =============================================================================

#SBATCH --job-name=diag_pjw_basin
#SBATCH --output=logs/diag_pjw_basin_%j.out
#SBATCH --error=logs/diag_pjw_basin_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

echo "plaquette+JW basin diagnostic on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" diag_pjw_basin.jl
echo "Exit code: $?  — $(date)"
