#!/bin/bash
# =============================================================================
#  run_mps_collect_plot.sh — aggregate the MPS-vs-ED results and make figures.
#  Runs gs_benchmark_mps_collect.jl (→ gs_mps_summary.csv) then plot_ed_vs_mps.jl
#  (→ gs + quench PNGs).  Plots whatever CSVs are present (missing ones skipped).
#
#  Usage (after gs_benchmark / quench arrays finish):
#    sbatch run_mps_collect_plot.sh
# =============================================================================

#SBATCH --job-name=mps_collect_plot
#SBATCH --output=logs/mps_collect_plot_%j.out
#SBATCH --error=logs/mps_collect_plot_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs results

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export GKSwstype=nul          # headless GR for Plots

echo "collect + plot on $(hostname) — $(date)"
echo "── aggregating ground-state CSVs ──"
stdbuf -oL -eL julia --project="${REPO_DIR}" gs_benchmark_mps_collect.jl
echo "── plotting MPS vs ED ──"
stdbuf -oL -eL julia --project="${REPO_DIR}" plot_ed_vs_mps.jl
echo "Exit code: $?  — $(date)"
