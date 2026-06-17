#= ═══════════════════════════════════════════════════════════════════════════════
   gs_benchmark_mps.jl

   Ground-state benchmark of the snake-MPS DMRG solver (mps_lgt.jl) against exact
   diagonalisation (finite_ed.jl) on the 3×4 U(1) LGT, over the SAME 9 (m,g)
   points as gs_benchmark_v3.jl (the PEPS benchmark).  Both solvers work in the
   staggered gauge sector (g_odd=−1, g_even=0); the MPS uses a Gauss-law penalty
   to select that sector, the ED enumerates it explicitly.

   SLURM array: task_id → (m,g) point.  Each task writes one CSV row; aggregate
   with gs_benchmark_mps_collect.jl.

   Usage:
       SLURM_ARRAY_TASK_ID=1 julia --project=. gs_benchmark_mps.jl
       sbatch run_gs_benchmark_mps.sh
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "mps_lgt.jl"))     # solver core (+ u1_lgt_hamiltonian)
include(joinpath(@__DIR__, "finite_ed.jl"))   # ED reference (gauge-sector)

using CSV, DataFrames, Printf, Statistics

# ── Same grid as gs_benchmark_v3.jl ──────────────────────────────────────────
const MPS_GRID = [
    (m=0.25, g=0.25), (m=0.25, g=0.50), (m=0.25, g=1.00),
    (m=0.25, g=2.00), (m=0.25, g=4.00),
    (m=0.50, g=0.25), (m=0.50, g=0.50), (m=0.50, g=2.00),
    (m=0.50, g=4.00),
]

const B_NX    = 3
const B_NY    = 4
const B_DG    = 1
const B_THOP  = 1.0
const B_DMPS  = 40       # DMRG bond dimension (gauge sector is small)
const B_NSW   = 8        # max DMRG sweeps (early-stops on convergence)
const B_LAM   = 5.0      # Gauss-law penalty (modest: pins sector w/o inflating spectrum)

function run_one_mps(task_id::Int)
    idx  = ((task_id - 1) % length(MPS_GRID)) + 1
    p    = MPS_GRID[idx]
    m, g = p.m, p.g
    nx, ny, dg = B_NX, B_NY, B_DG
    gch  = staggered_charges(nx, ny)

    @printf("=== Task %d: m=%.2f g=%.2f  (3×4 staggered, dg=1) ===\n", task_id, m, g)
    flush(stdout)

    # ── ED reference (gauge-invariant sector) ────────────────────────────────
    println("  [ED] building basis + Hamiltonian ...")
    states, key = build_basis(nx, ny, dg, Matrix{Int}(gch))
    Hed = build_hamiltonian(states, key, nx, ny, dg; g=g, t=B_THOP, m=m)
    t0 = time()
    Eed, ψed = find_ground_state(Hed)
    obs_ed = measure_ED(ψed, states, nx, ny, dg, 0.0)
    @printf("  [ED] E0 = %.8f  (%d states, %.1fs)\n", Eed, length(states), time()-t0)

    # ── MPS DMRG ground state (same sector via penalty) ──────────────────────
    println("  [MPS] DMRG ...")
    dims = node_dims(nx, ny, dg)
    Hpen = build_penalized_H(nx, ny, dg; g=g, t=B_THOP, m=m, gauss_g=gch, Λ=B_LAM)
    t1 = time()
    _, ψ = dmrg_ground_state(Hpen, dims; D=B_DMPS, nsweeps=B_NSW, verbose=true,
                             ψ0=staggered_mps(nx, ny, dg))   # start in the target sector
    Hbare = _assemble_mpo(dims, lgt_terms(nx, ny, dg; g=g, t=B_THOP, m=m))
    Emps  = real(mpo_expect(Hbare, ψ)) / real(mps_overlap(ψ, ψ))
    obs_mps = measure_mps(ψ, nx, ny, dg)
    viol = gauss_violation(ψ, nx, ny, dg, gch)
    bond = maximum(size(t, 2) for t in ψ)
    @printf("  [MPS] E = %.8f  bond=%d  Gauss-viol=%.2e  (%.1fs)\n",
            Emps, bond, viol, time()-t1)
    @printf("  ΔE/|E| = %.2e\n", abs(Emps - Eed) / abs(Eed))

    df = DataFrame(
        task=[task_id], m=[m], g=[g],
        E_ed=[Eed], E_mps=[Emps], dE_rel=[abs(Emps - Eed) / max(abs(Eed), 1e-12)],
        nf_mean_ed=[obs_ed.nf_mean], nf_mean_mps=[obs_mps.nf_mean],
        nf_even_ed=[obs_ed.nf_even], nf_even_mps=[obs_mps.nf_even],
        nf_odd_ed=[obs_ed.nf_odd],   nf_odd_mps=[obs_mps.nf_odd],
        E2_mean_ed=[obs_ed.E2_mean], E2_mean_mps=[obs_mps.E2_mean],
        gauss_viol=[viol], bond=[bond],
    )
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    out = joinpath(results_dir, "gs_mps_task$(task_id).csv")
    CSV.write(out, df)
    println("  Saved: $out")
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    task_id = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))
    run_one_mps(task_id)
end
