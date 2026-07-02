#!/bin/bash
# =============================================================================
#  run_plaquette_entanglement_ed.sh — cross-check the dual-plaquette half-system
#  entanglement (plaquette_entanglement.jl) against full ED with EXACT decoupling.
#
#  Runs the MPS pipeline (which=:exact) and an independent ED reference (sparse Ô,
#  exact exp(−iÔ)|ψ⟩ on the dense state vector, dense plaquette partial trace) on
#  the 2×3 and 3×2 lattices (smallest with ≥2 plaquettes → nontrivial half-cut),
#  and asserts they agree on the source-free weight, purity, ρ_plaq and entropy S.
#  Exits 0 on PASS, 1 on FAIL.
#
#  Single job (correctness test, not a parameter sweep) — no array needed.
#  Usage:  sbatch run_plaquette_entanglement_ed.sh
# =============================================================================

#SBATCH --job-name=plaq_ent_ed
#SBATCH --output=logs/plaq_ent_ed_%j.out
#SBATCH --error=logs/plaq_ent_ed_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=main

REPO_DIR="/scratch/noafeld/LGT"
cd "${REPO_DIR}" || { echo "Cannot cd to ${REPO_DIR}"; exit 1; }
mkdir -p logs

export PATH="/net/opt/julia/julia-1.11.4/bin:${PATH}"
export JULIA_DEPOT_PATH="/scratch/noafeld/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "plaquette-entanglement ED cross-check on $(hostname) — $(date)"
stdbuf -oL -eL julia --project="${REPO_DIR}" test_plaquette_entanglement_ed.jl
CODE=$?
echo "Exit code: ${CODE}  — $(date)"
exit ${CODE}
