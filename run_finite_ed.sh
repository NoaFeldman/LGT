#!/bin/bash
# =============================================================================
#  run_finite_ed.sh
#
#  SLURM array submission for finite_ed.jl (exact diagonalization).
#  Three tasks run in parallel, one per quench protocol:
#    task 1 → Quench A (string breaking)
#    task 2 → Quench B (mass quench)
#    task 3 → Quench C (coupling quench)
#
#  Typical wall times at ED_NX=3, ED_NY=4, ED_DG=1:
#    Quench A  (~few k  states):  < 10 min
#    Quench B  (~10–50k states):  15–30 min
#    Quench C  (~10–50k states):  15–30 min
# =============================================================================

#SBATCH --job-name=fpeps_ed
#SBATCH --output=logs/fpeps_ed_%A_%a.out
#SBATCH --error=logs/fpeps_ed_%A_%a.err
#SBATCH --array=1-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --partition=main

# ── Paths ─────────────────────────────────────────────────────────────────────
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
QUENCH_NAMES=("" "A (String Breaking)" "B (Mass Quench)" "C (Coupling Quench)")
echo "========================================================================"
echo "  Finite PEPS ED Comparison — Quench ${QUENCH_NAMES[$SLURM_ARRAY_TASK_ID]}"
echo "  Job    : ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "  Node   : $(hostname)"
echo "  CPUs   : ${SLURM_CPUS_PER_TASK}"
echo "  Julia  : $(julia --version 2>&1)"
echo "  Started: $(date)"
echo "========================================================================"
echo ""

# ── Instantiate once (fast if already cached) ─────────────────────────────────
echo "Checking project environment..."
julia --project="${REPO_DIR}" -e 'import Pkg; Pkg.instantiate()'

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Running finite_ed.jl for quench ${SLURM_ARRAY_TASK_ID}..."
julia --project="${REPO_DIR}" \
      --threads=${JULIA_NUM_THREADS} \
      "${REPO_DIR}/finite_ed.jl" "${SLURM_ARRAY_TASK_ID}"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "  Finished: $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "========================================================================"
exit ${EXIT_CODE}
