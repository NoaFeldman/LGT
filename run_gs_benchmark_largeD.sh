#!/bin/bash
# =============================================================================
#  run_gs_benchmark_largeD.sh
#
#  SLURM array driver for gs_benchmark_largeD.jl.
#  Runs PEPS ground-state ITE with larger bond dimensions (D_bond=8, D_max=24,
#  n_ite=600) compared to the original gs_benchmark.jl (D_bond=4, D_max=12,
#  n_ite=300).
#
#  The larger bond dimension and more ITE steps require significantly more
#  compute time:  ~8h per task (vs ~2h for the original).
#
#  Grid: 5 masses × 5 couplings = 25 tasks.
#
#  Usage:
#      sbatch run_gs_benchmark_largeD.sh
#      sbatch run_gs_benchmark_largeD.sh --export=D_BOND=12,D_MAX=32,N_ITE=800
# =============================================================================

#SBATCH --job-name=gs_bench_LD
#SBATCH --output=logs/gs_bench_LD_%A_%a.out
#SBATCH --error=logs/gs_bench_LD_%A_%a.err
#SBATCH --array=1-25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --partition=main

# ─── Paths ───────────────────────────────────────────────────────────────────
REPO_DIR="/scratch/noafeld/LGT"
JULIA_DEPOT="/scratch/noafeld/.julia"

cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs results/gs_bench_largeD

# ─── Environment ─────────────────────────────────────────────────────────────
export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT}"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export GKSwstype=nul

# ─── Tunable parameters (override via --export or environment) ───────────────
D_BOND=${D_BOND:-8}
D_MAX=${D_MAX:-24}
N_ITE=${N_ITE:-600}
TAU=${TAU:-0.02}
OUT_DIR="${OUT_DIR:-results/gs_bench_largeD}"

# ─── Print job info ──────────────────────────────────────────────────────────
echo "========================================================================"
echo "  GS benchmark (large D) — task ${SLURM_ARRAY_TASK_ID}"
echo "  Job    : ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "  Node   : $(hostname)"
echo "  CPUs   : ${SLURM_CPUS_PER_TASK}"
echo "  Julia  : $(julia --version 2>&1)"
echo "  D_bond : ${D_BOND}"
echo "  D_max  : ${D_MAX}"
echo "  N_ITE  : ${N_ITE}"
echo "  TAU    : ${TAU}"
echo "  Started: $(date)"
echo "========================================================================"

# Only the first task instantiates the project to avoid races.
if [[ "${SLURM_ARRAY_TASK_ID}" == "1" ]]; then
    echo "Instantiating project environment..."
    julia --project="${REPO_DIR}" -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
fi

julia --project="${REPO_DIR}" \
      --threads=${JULIA_NUM_THREADS} \
      "${REPO_DIR}/gs_benchmark_largeD.jl" "${SLURM_ARRAY_TASK_ID}" \
      --D-bond ${D_BOND} --D-max ${D_MAX} \
      --n-ite ${N_ITE} --tau ${TAU} \
      --out-dir "${REPO_DIR}/${OUT_DIR}"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "  Finished : $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "========================================================================"
echo ""
echo "  After all tasks complete, collect results with:"
echo "    julia --project=. gs_benchmark_collect.jl ${OUT_DIR}"

exit ${EXIT_CODE}
