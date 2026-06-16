#!/bin/bash
# =============================================================================
#  run_mps_quench.sh — MPS vs ED real-time quenches A/B/C on the 3×4 U(1) LGT,
#  as a SLURM array (task 1/2/3 → quench A/B/C).  Each task runs both the MPS
#  (global-Krylov) and the aligned ED evolution and writes their CSVs.
#
#  Usage:
#    sbatch run_mps_quench.sh
#    # then plot:
#    julia --project=. plot_ed_vs_mps.jl
# =============================================================================

#SBATCH --job-name=mps_quench
#SBATCH --output=logs/mps_quench_%A_%a.out
#SBATCH --error=logs/mps_quench_%A_%a.err
#SBATCH --array=1-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs results

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "mps_quench task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" mps_quench.jl
echo "Exit code: $?  — $(date)"
