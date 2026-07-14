#= ═══════════════════════════════════════════════════════════════════════════════
   diag_pjw_timing.jl

   Where does the plaquette+JW ground-state run spend its time?  Builds the
   Hamiltonian MPO for one (m,g) point, reports its bond dimension, and times a
   SINGLE DMRG sweep at a couple of bond dimensions so the full-run cost can be
   extrapolated before committing a long array job.

   Usage:
       SLURM_ARRAY_TASK_ID=1 julia --project=. diag_pjw_timing.jl
       sbatch run_diag_pjw_timing.sh
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "matter_gauge_entanglement_pjw.jl"))

using Printf

function diag(task_id::Int)
    idx = ((task_id - 1) % length(MG_GRID)) + 1
    p   = MG_GRID[idx]
    m, g = p.m, p.g
    nx, ny, dg = B_NX, B_NY, B_DG
    gch = staggered_charges(nx, ny)
    dims = col_node_dims(nx, ny, dg)

    @printf("=== diag task %d: m=%.2f g=%.2f  (%d×%d, plaquette+JW) ===\n", task_id, m, g, nx, ny)
    @printf("  node dims (column snake): %s   Hilbert dim = %.3e\n", string(dims), Float64(prod(dims)))

    # ── MPO build + bond ──────────────────────────────────────────────────────
    t0 = time()
    terms = lgt_terms_plaqjw_cs(nx, ny, dg; g=g, t=B_THOP, m=m)
    Hbare = _assemble_mpo(dims, terms)
    @printf("  bare  H: %d terms → MPO max bond = %d   (built in %.1fs)\n",
            length(terms), maximum(size(W, 2) for W in Hbare), time()-t0)
    t0 = time()
    Hpen = build_penalized_H_plaqjw_cs(nx, ny, dg; g=g, t=B_THOP, m=m, gauss_g=gch, Λ=B_LAM)
    @printf("  penal H: MPO max bond = %d   (built in %.1fs)\n",
            maximum(size(W, 2) for W in Hpen), time()-t0)
    flush(stdout)

    # ── time ONE sweep at increasing bond D (single staggered start) ──────────
    for D in (40, 60, 100)
        ψ0 = staggered_mps_cs(nx, ny, dg)
        t0 = time()
        E, _ = dmrg_ground_state(Hpen, dims; D=D, nsweeps=1, verbose=false, ψ0=ψ0)
        dt = time() - t0
        @printf("  D=%-3d : 1 sweep = %6.1fs   E_pen(1 sweep) = %.6f   → 30 sweeps ≈ %.1f min\n",
                D, dt, E, 30*dt/60)
        flush(stdout)
    end
    println("\n  Full run per point ≈ (1 + PJW_NRAND) starts × (above) + decoupling (~1h).")
end

if abspath(PROGRAM_FILE) == @__FILE__
    diag(parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1")))
end
