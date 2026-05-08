#!/bin/bash
# =============================================================================
#  run_finite_peps_quench.sh
#
#  SLURM array submission for finite_peps_quench.jl
#  3×4 finite PEPS lattice (2×3 plaquettes), three quench protocols.
#  Each array task runs one quench independently:
#    task 1 → Quench A (string breaking,  ~RT only,   ~1–2 h)
#    task 2 → Quench B (mass quench,      300 ITE + RT, ~4–8 h)
#    task 3 → Quench C (coupling quench,  300 ITE + RT, ~4–8 h)
# =============================================================================

#SBATCH --job-name=fpeps_quench
#SBATCH --output=logs/fpeps_quench_%A_%a.out
#SBATCH --error=logs/fpeps_quench_%A_%a.err
#SBATCH --array=1-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=08:00:00
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
QUENCH_NAMES=("" "A (String Breaking)" "B (Mass Quench)" "C (Coupling Quench)")
echo "========================================================================"
echo "  Finite PEPS Quench — Quench ${QUENCH_NAMES[$SLURM_ARRAY_TASK_ID]}"
echo "  Job    : ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "  Node   : $(hostname)"
echo "  CPUs   : ${SLURM_CPUS_PER_TASK}"
echo "  Julia  : $(julia --version 2>&1)"
echo "  Started: $(date)"
echo "========================================================================"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Instantiating project environment..."
julia --project="${REPO_DIR}" -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "Running finite_peps_quench.jl for quench ${SLURM_ARRAY_TASK_ID}..."
julia --project="${REPO_DIR}" \
      --threads=${JULIA_NUM_THREADS} \
      "${REPO_DIR}/finite_peps_quench.jl" "${SLURM_ARRAY_TASK_ID}"

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
