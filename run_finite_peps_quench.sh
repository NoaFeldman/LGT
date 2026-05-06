#!/bin/bash
# =============================================================================
#  run_finite_peps_quench.sh
#
#  SLURM submission script for finite_peps_quench.jl
#  3×4 finite PEPS lattice (2×3 plaquettes), three quench protocols.
# =============================================================================

#SBATCH --job-name=fpeps_quench
#SBATCH --output=logs/fpeps_quench_%j.out
#SBATCH --error=logs/fpeps_quench_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --partition=main

# ── Paths (mirror cluster_example conventions) ────────────────────────────────
REPO_DIR="/scratch/noafeld/LGT"
JULIA_DEPOT="/scratch/noafeld/.julia"

cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT}"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export GKSwstype=nul

# ── Job info ──────────────────────────────────────────────────────────────────
echo "========================================================================"
echo "  Finite PEPS Quench Benchmark"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Node    : $(hostname)"
echo "  CPUs    : ${SLURM_CPUS_PER_TASK}"
echo "  Julia   : $(julia --version 2>&1)"
echo "  Started : $(date)"
echo "========================================================================"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Instantiating project environment..."
julia --project="${REPO_DIR}" -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "Running finite_peps_quench.jl..."
julia --project="${REPO_DIR}" \
      --threads=${JULIA_NUM_THREADS} \
      "${REPO_DIR}/finite_peps_quench.jl"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "  Finished : $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "========================================================================"

echo ""
echo "Output files:"
ls -lh "${REPO_DIR}"/finite_peps_quench_* 2>/dev/null || echo "  (none found)"

exit ${EXIT_CODE}
