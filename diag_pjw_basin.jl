#= ═══════════════════════════════════════════════════════════════════════════════
   diag_pjw_basin.jl

   Validate the production ground-state strategy for the (highly entangled)
   plaquette+JW model on 3×4: BOND-RAMPING DMRG from the electric-vacuum start
   with the stronger Gauss penalty Λ=PJW_LAM.  Prints per sweep the energy, ΔE,
   Gauss violation, bond, wall time and peak RSS (flushed — a timeout keeps the
   trace), then reports the converged energy, whether it is in-sector, and the
   pre-decoupling half-system entanglement.

   Answers, from one (m,g) point:
     • does ramping actually CONVERGE (ΔE→0 at D=60), and at what wall-clock cost?
     • is the converged state IN the staggered sector (Gauss-viol ≈ 0)? — i.e. is
       Λ=PJW_LAM strong enough now that the plaquette term is present?
     • how entangled is it (S_pre) — sanity on why the bond is large.

   Usage:
       SLURM_ARRAY_TASK_ID=1 julia --project=. diag_pjw_basin.jl
       sbatch run_diag_pjw_basin.sh
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "matter_gauge_entanglement_pjw.jl"))

using Printf

function basin(task_id::Int)
    idx = ((task_id - 1) % length(MG_GRID)) + 1
    p   = MG_GRID[idx]
    m, g = p.m, p.g
    nx, ny, dg = B_NX, B_NY, B_DG
    gch  = staggered_charges(nx, ny)
    dims = col_node_dims(nx, ny, dg)

    @printf("=== basin diag task %d: m=%.2f g=%.2f  (%d×%d, plaquette+JW, Λ=%.0f) ===\n",
            task_id, m, g, nx, ny, PJW_LAM)
    @printf("  schedule (bond,sweeps): %s\n", string(PJW_SCHEDULE))
    Hpen = build_penalized_H_plaqjw_cs(nx, ny, dg; g=g, t=B_THOP, m=m, gauss_g=gch, Λ=PJW_LAM)
    @printf("  penalized-H MPO max bond = %d\n", maximum(size(W, 2) for W in Hpen)); flush(stdout)

    t0 = time()
    E, ψ, viol = ramped_ground_state(Hpen, dims, nx, ny, dg, gch;
                                     schedule=PJW_SCHEDULE, etol=PJW_ETOL, verbose=true)
    Hbare = _assemble_mpo(dims, lgt_terms_plaqjw_cs(nx, ny, dg; g=g, t=B_THOP, m=m))
    Ebare = real(mpo_expect(Hbare, ψ)) / real(mps_overlap(ψ, ψ))
    S_pre = half_chain_entropy(ψ)

    println("\n  ── summary ──")
    @printf("  E_pen=%.8f  E_bare=%.8f  Gauss-viol=%.2e  bond=%d  S_pre=%.4f nats  (%.0fs total)\n",
            E, Ebare, viol, maximum(size(t, 2) for t in ψ), S_pre, time()-t0)
    println(viol < 1e-3 ? "  IN-SECTOR ✓  — Λ holds the staggered sector." :
                          "  OUT OF SECTOR ✗ — raise PJW_LAM (plaquette term is beating the penalty).")
end

if abspath(PROGRAM_FILE) == @__FILE__
    basin(parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1")))
end
