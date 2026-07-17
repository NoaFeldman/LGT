#= ═══════════════════════════════════════════════════════════════════════════════
   diag_pjw_basin.jl

   Ground-truth cost + convergence of the plaquette+JW DMRG on 3×4.  The earlier
   timing diagnostic under-measured: it ran ONE sweep from the bond-1 product
   state (cheap, bonds still growing), whereas sweeps 2+ run at full bond where
   the d=18 two-site eigensolve forms ~900 MB intermediates hundreds of times per
   update → GC-bound and slow.

   This runs single sweeps ITERATIVELY (feeding ψ back), printing wall time,
   energy, bond and peak RSS after EACH sweep (flushed, so a timeout still leaves
   the numbers), at two bond dimensions — to answer:
     • how long is a real full-bond sweep, and how many do we need to converge?
     • is a small bond (D=30) enough (low-entanglement vacuum), i.e. can we cut the
       per-sweep cost by ~(30/60)² ≈ 4× and the memory likewise?

   Usage:
       SLURM_ARRAY_TASK_ID=1 julia --project=. diag_pjw_basin.jl
       sbatch run_diag_pjw_basin.sh
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "matter_gauge_entanglement_pjw.jl"))

using Printf

"""Run up to `nsw` single sweeps from `ψ0`, timing each; early-stop on |ΔE|<etol."""
function sweep_by_sweep(tag, Hpen, dims, nx, ny, dg, gch, ψ0; D, nsw=6, etol=1e-6)
    @printf("\n  ── %s : D=%d ─────────────────────────────────────────────\n", tag, D)
    flush(stdout)
    ψ = ψ0; Eprev = Inf
    for s in 1:nsw
        t0 = time()
        E, ψ = dmrg_ground_state(Hpen, dims; D=D, nsweeps=1, verbose=false, ψ0=ψ)  # DEFAULT eigensolver
        bond = maximum(size(t, 2) for t in ψ)
        @printf("    sweep %d: E_pen=%.8f  ΔE=%.2e  bond=%d  %6.0fs  peak RSS=%.1f GB\n",
                s, E, abs(E - Eprev), bond, time()-t0, Sys.maxrss()/2^30)
        flush(stdout)
        abs(E - Eprev) < etol && (println("    → converged"); return E, ψ)
        Eprev = E
    end
    return Eprev, ψ
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

    # small bond first (cheap — tells us the per-sweep cost and if D=30 suffices)
    E30, _ = sweep_by_sweep("vacuum", Hpen, dims, nx, ny, dg, gch,
                            staggered_mps_cs(nx, ny, dg); D=30, nsw=6)
    # then full bond, to see if the energy drops further (is D=30 enough?)
    E60, _ = sweep_by_sweep("vacuum", Hpen, dims, nx, ny, dg, gch,
                            staggered_mps_cs(nx, ny, dg); D=60, nsw=6)

    println("\n  ── summary ──")
    @printf("  vacuum E(D=30) = %.8f   vacuum E(D=60) = %.8f   ΔE(D) = %.2e\n",
            E30, E60, abs(E60 - E30))
    println("  (use the per-sweep times above to size the production walltime / bond.)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    basin(parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1")))
end
