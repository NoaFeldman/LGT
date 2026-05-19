#!/bin/bash
# =============================================================================
#  SLURM job for finite-PEPS ground-state run with magnetic plaquette.
#
#  Usage:   sbatch submit_finite_peps_gs.slurm.sh <D> [n_steps] [noise]
#  Example: sbatch submit_finite_peps_gs.slurm.sh 8 2000 0.05
#
#  Memory & time scaling (per Trotter step ~ 9 plaquette updates):
#    D=6   ~10 GB,  ~15 hours    (overkill for cluster, runs locally)
#    D=8   ~50 GB,  ~6 days      ← recommended cluster target
#    D=10  ~200 GB, ~30 days     (needs bigmem; reduce n_steps to ~500)
#    D=12  ~600 GB, infeasible   (use full-update instead)
#
#  Pick --mem / --time / --cpus-per-task to match the D you submit.
# =============================================================================

#SBATCH --job-name=fpeps_gs
#SBATCH --output=logs/fpeps_gs_%j.out
#SBATCH --error=logs/fpeps_gs_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0                     # request all memory on the node
#SBATCH --time=144:00:00            # 6 days — adjust for larger D

# ─── Parse args ──────────────────────────────────────────────────────────────
D=${1:?"Usage: sbatch submit_finite_peps_gs.slurm.sh <D> [n_steps] [noise]"}
N_STEPS=${2:-2000}
NOISE=${3:-0.05}

# ─── Cluster paths (mirror cluster_example/submit.sh — adjust if needed) ─────
REPO_DIR="/scratch/noafeld/LGT"
JULIA_DEPOT="/scratch/noafeld/.julia"
JULIA_BIN="/net/opt/julia/julia-1.11.4/bin"

cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }

mkdir -p logs results_gs/D${D}

# ─── Environment ─────────────────────────────────────────────────────────────
export PATH="${JULIA_BIN}:${PATH}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT}"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# ─── Print job info ──────────────────────────────────────────────────────────
echo "================================================"
echo "  Finite-PEPS GS — U(1) LGT with magnetic plaquette"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "  D:        ${D}"
echo "  n_steps:  ${N_STEPS}"
echo "  noise:    ${NOISE}"
echo "  Node:     $(hostname)"
echo "  CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "  RAM:      $(free -h | awk '/^Mem:/{print $2}')"
echo "  Threads:  ${JULIA_NUM_THREADS}"
echo "  Started:  $(date)"
echo "================================================"

# ─── Run ─────────────────────────────────────────────────────────────────────
julia --project="${REPO_DIR}" \
      "${REPO_DIR}/benchmark_finite_peps_gs.jl" \
      ${D} ${N_STEPS} ${NOISE}

echo "================================================"
echo "  Finished: $(date)"
echo "  Output in: ${REPO_DIR}/results_gs/D${D}/"
echo "================================================"
