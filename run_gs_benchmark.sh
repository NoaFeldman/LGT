#!/bin/bash
# =============================================================================
#  run_gs_benchmark.sh
#
#  SLURM array driver for gs_benchmark.jl.
#  Each task computes PEPS-ITE and ED ground states at one (m, g) point and
#  writes a single-row CSV.  Aggregate with gs_benchmark_collect.jl.
#
#  IMPORTANT: --array=1-N must match length(PARAM_GRID) in gs_benchmark.jl.
#  Default grid is 5 masses × 5 couplings = 25 tasks.
# =============================================================================

#SBATCH --job-name=gs_bench
#SBATCH --output=logs/gs_bench_%A_%a.out
#SBATCH --error=logs/gs_bench_%A_%a.err
#SBATCH --array=1-25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
JULIA_DEPOT="/scratch/noafeld/.julia"

cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs results/gs_bench

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT}"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export GKSwstype=nul

echo "========================================================================"
echo "  GS benchmark — task ${SLURM_ARRAY_TASK_ID}"
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
      "${REPO_DIR}/gs_benchmark.jl" "${SLURM_ARRAY_TASK_ID}" \
      --out-dir "${REPO_DIR}/results/gs_bench"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "  Finished : $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "========================================================================"

# When the last task ends, optionally trigger the collection script.
# (Not done automatically: dependencies are simpler to run manually:
#     julia --project=. gs_benchmark_collect.jl results/gs_bench
# )

exit ${EXIT_CODE}
