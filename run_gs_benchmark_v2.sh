#!/bin/bash
# =============================================================================
#  run_gs_benchmark_v2.sh
#
#  SLURM array driver for gs_benchmark_v2.jl (full-update PEPS).
#  9 parameter points: m ∈ {0.25, 0.50} × g ∈ {0.25, 0.50, 1.00, 2.00, 4.00}
#
#  Usage:
#      sbatch run_gs_benchmark_v2.sh
# =============================================================================

#SBATCH --job-name=gs_bench_v2
#SBATCH --output=logs/gs_bench_v2_%A_%a.out
#SBATCH --error=logs/gs_bench_v2_%A_%a.err
#SBATCH --array=1-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --partition=main

# ─── Paths ───────────────────────────────────────────────────────────────────
REPO_DIR="/scratch/noafeld/LGT"
JULIA_DEPOT="/scratch/noafeld/.julia"

cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs results/gs_bench_v2

# ─── Environment ─────────────────────────────────────────────────────────────
export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT}"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export GKSwstype=nul

# ─── Print job info ──────────────────────────────────────────────────────────
echo "========================================================================"
echo "  GS benchmark v2 (full update) — task ${SLURM_ARRAY_TASK_ID}"
echo "  Job    : ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "  Node   : $(hostname)"
echo "  CPUs   : ${SLURM_CPUS_PER_TASK}"
echo "  Julia  : $(julia --version 2>&1)"
echo "  Started: $(date)"
echo "========================================================================"

# Only the first task instantiates the project to avoid races.
if [[ "${SLURM_ARRAY_TASK_ID}" == "1" ]]; then
    echo "Instantiating project environment..."
    julia --project="${REPO_DIR}" -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
fi

julia --project="${REPO_DIR}" \
      --threads=${JULIA_NUM_THREADS} \
      "${REPO_DIR}/gs_benchmark_v2.jl"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "  Finished : $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "========================================================================"

exit ${EXIT_CODE}
