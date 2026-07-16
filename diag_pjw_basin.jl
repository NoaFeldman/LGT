#= ═══════════════════════════════════════════════════════════════════════════════
   diag_pjw_basin.jl

   Decisive test for the plaquette+JW ground-state strategy on 3×4: is the cheap
   VACUUM (staggered, electric-vacuum) DMRG start TRAPPED, or does it already give
   the ground state?  On 2×2 it trapped; on 3×4 we never checked.

   Runs, for one (m,g) point, with per-sweep energies printed (so partial progress
   survives a timeout):
     • the vacuum start at D=60 to convergence  → E_vac
     • a few FLUX-seeded starts at D=48         → E_flux[]

   Verdict:
     • min(E_flux) ≈ E_vac  → vacuum is NOT trapped ⇒ production can be vacuum-only
       (cheap: ~4 sweeps, low memory) and the restarts can be dropped.
     • min(E_flux) <  E_vac  → vacuum IS trapped ⇒ we need cheap flux screening.

   Usage:
       SLURM_ARRAY_TASK_ID=1 julia --project=. diag_pjw_basin.jl
       sbatch run_diag_pjw_basin.sh
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "matter_gauge_entanglement_pjw.jl"))

using Printf

function run_start(name, Hpen, dims, nx, ny, dg, gch, ψ0; D, nsweeps)
    @printf("\n  ── %s start (D=%d) ──\n", name, D); flush(stdout)
    t0 = time()
    E, ψ = dmrg_ground_state(Hpen, dims; D=D, nsweeps=nsweeps, verbose=true, ψ0=ψ0,
                             tol=PJW_KTOL, maxiter=PJW_KMAXIT, krylovdim=PJW_KDIM, etol=PJW_ETOL)
    viol = gauss_violation_cs(ψ, nx, ny, dg, gch)
    @printf("  %s: E_pen=%.8f  Gauss-viol=%.2e  bond=%d  %.0fs  peak RSS=%.1f GB\n",
            name, E, viol, maximum(size(t, 2) for t in ψ), time()-t0, Sys.maxrss()/2^30)
    flush(stdout)
    return E, viol
end

function basin(task_id::Int)
    idx = ((task_id - 1) % length(MG_GRID)) + 1
    p   = MG_GRID[idx]
    m, g = p.m, p.g
    nx, ny, dg = B_NX, B_NY, B_DG
    gch  = staggered_charges(nx, ny)
    dims = col_node_dims(nx, ny, dg)

    @printf("=== basin diag task %d: m=%.2f g=%.2f  (%d×%d, plaquette+JW) ===\n", task_id, m, g, nx, ny)
    Hpen = build_penalized_H_plaqjw_cs(nx, ny, dg; g=g, t=B_THOP, m=m, gauss_g=gch, Λ=B_LAM)
    @printf("  penalized-H MPO max bond = %d\n", maximum(size(W, 2) for W in Hpen)); flush(stdout)

    E_vac, _ = run_start("vacuum", Hpen, dims, nx, ny, dg, gch,
                         staggered_mps_cs(nx, ny, dg); D=60, nsweeps=8)

    E_flux = Float64[]
    for k in 1:3
        Ef, vf = run_start("flux$k", Hpen, dims, nx, ny, dg, gch,
                           flux_seeded_mps(nx, ny, dg; seed=10k + 1); D=48, nsweeps=8)
        vf < 1e-3 && push!(E_flux, Ef)
    end

    println("\n  ── verdict ──")
    @printf("  E_vac = %.8f   min E_flux(in-sector) = %s\n",
            E_vac, isempty(E_flux) ? "none in-sector" : @sprintf("%.8f", minimum(E_flux)))
    if !isempty(E_flux) && minimum(E_flux) < E_vac - 1e-4
        @printf("  VACUUM IS TRAPPED by %.6f — flux screening needed.\n", E_vac - minimum(E_flux))
    else
        println("  vacuum is NOT trapped — production can be vacuum-only (cheap).")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    basin(parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1")))
end
