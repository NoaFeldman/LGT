#!/bin/bash
# =============================================================================
#  run_matter_gauge_entanglement_pjw_selftest.sh — correctness gate for the
#  plaquette+JW Hamiltonian: dense 2×2 MPO self-test (DMRG penalized ground state
#  vs exact diagonalisation of the same MPO).  Run this BEFORE the 9-job sweep;
#  the log should print "PASS".
#
#  Usage:  sbatch run_matter_gauge_entanglement_pjw_selftest.sh
#          # inspect the result:
#          #   grep -E "PASS|WARN|Δ" logs/mg_ent_pjw_selftest_*.out
# =============================================================================

#SBATCH --job-name=mg_ent_pjw_selftest
#SBATCH --output=logs/mg_ent_pjw_selftest_%j.out
#SBATCH --error=logs/mg_ent_pjw_selftest_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PJW_SELFTEST=1

echo "plaquette+JW MPO self-test on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" matter_gauge_entanglement_pjw.jl
echo "Exit code: $?  — $(date)"
