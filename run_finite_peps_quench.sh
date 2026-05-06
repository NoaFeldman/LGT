#!/bin/bash
# =============================================================================
#  run_finite_peps_quench.sh
#
#  SLURM submission script for finite_peps_quench.jl
#
#  Runs three quench protocols (string breaking, mass quench, coupling quench)
#  on a 3×4 finite PEPS lattice (2×3 plaquettes).
#
#  Estimated wall time:
#    Quench A  (no ITE):          ~15 min   at D_max=6
#    Quench B  (300 ITE + RT):    ~60 min   at D_max=6
#    Quench C  (300 ITE + RT):    ~60 min   at D_max=6
#    Total:                      ~2.5 hours
#  Increase --time if using larger D_max or more steps.
#
#  Adjust the cluster-specific lines marked  ← EDIT  before submitting.
# =============================================================================

# ── Job metadata ──────────────────────────────────────────────────────────────
#SBATCH --job-name=fpeps_quench
#SBATCH --output=fpeps_quench_%j.out
#SBATCH --error=fpeps_quench_%j.err

# ── Resources ─────────────────────────────────────────────────────────────────
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # Julia BLAS uses all cores; 4–16 is typical
#SBATCH --mem=32G                 # 32 GB is comfortable; reduce to 16G if needed
#SBATCH --time=04:00:00           # hh:mm:ss; raise for larger D_max

# ── Partition / QOS  ← EDIT for your cluster ─────────────────────────────────
#SBATCH --partition=general       # replace with your cluster's partition name
##SBATCH --qos=normal             # uncomment / edit if your cluster uses QOS
##SBATCH --account=myproject      # uncomment / edit if required

# ── Email notifications (optional) ← EDIT ────────────────────────────────────
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your@email.address

# =============================================================================
#  Environment setup
# =============================================================================

echo "========================================================================"
echo "  Finite PEPS Quench Benchmark"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "  CPUs    : $SLURM_CPUS_PER_TASK"
echo "  Started : $(date)"
echo "========================================================================"
echo ""

# Change to the directory from which the job was submitted
cd "$SLURM_SUBMIT_DIR"

# ── Load Julia module  ← EDIT module name for your cluster ───────────────────
# Common variants:
#   module load julia/1.10.4
#   module load julia            (latest available)
#   module load languages/julia  (some HPC centres)
# If Julia is not available as a module, set the path directly:
#   export PATH="/path/to/julia-1.10.4/bin:$PATH"

module load julia                 # ← EDIT: replace with your cluster's module

# ── Julia / plotting settings ─────────────────────────────────────────────────
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK   # BLAS multi-threading
export GKSwstype=nul                             # GR backend: no display needed
export JULIA_DEPOT_PATH="$HOME/.julia"           # user package depot

# ── First-run package installation ───────────────────────────────────────────
# On the very first submission, packages may need to be resolved / precompiled.
# Uncomment the line below, run once, then comment it out again for production.
#
# julia --project="$SLURM_SUBMIT_DIR" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Running finite_peps_quench.jl ..."
echo ""

julia \
    --project="$SLURM_SUBMIT_DIR" \
    --threads=$SLURM_CPUS_PER_TASK \
    "$SLURM_SUBMIT_DIR/finite_peps_quench.jl"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "  Finished : $(date)"
echo "  Exit code: $EXIT_CODE"
echo "========================================================================"

# ── List output files ─────────────────────────────────────────────────────────
echo ""
echo "Output files:"
ls -lh "$SLURM_SUBMIT_DIR"/finite_peps_quench_* 2>/dev/null || echo "  (none found)"

exit $EXIT_CODE
