#!/bin/bash
# =============================================================================
#  run_matter_gauge_entanglement_pjw_collect.sh — aggregate the 9 per-task CSVs
#  from run_matter_gauge_entanglement_pjw.sh into results/mg_ent_pjw_summary.csv
#  and the figure results/mg_entanglement_pjw.png (plaquette + JW variant).
#
#  Submit with a dependency on the sweep array so it runs only after all 9 tasks:
#      SWEEP=$(sbatch --parsable run_matter_gauge_entanglement_pjw.sh)
#      sbatch --dependency=afterok:${SWEEP} run_matter_gauge_entanglement_pjw_collect.sh
#  (or just sbatch it on its own once the sweep has finished).
# =============================================================================

#SBATCH --job-name=mg_ent_pjw_collect
#SBATCH --output=logs/mg_ent_pjw_collect_%j.out
#SBATCH --error=logs/mg_ent_pjw_collect_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs results

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

echo "matter_gauge_entanglement_pjw collect on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" matter_gauge_entanglement_pjw_collect.jl
echo "Exit code: $?  — $(date)"
