#= ═══════════════════════════════════════════════════════════════════════════════
   gs_benchmark_v4.jl

   Ground-state benchmark using the GAUGE-INVARIANT BOUNDARY-MPS FULL UPDATE (v4).
   Same charge-resolved Vidal storage as v3, but the bond truncation is weighted
   by the real 2D environment (boundary MPS) via ALS instead of the diagonal
   Vidal weights — the fix for v3's collapse to the classical vacuum.

   Reports a true variational ⟨H⟩ (from the environment) for direct comparison
   with the ED ground-state energy.

   Usage:  julia --project=. gs_benchmark_v4.jl     (SLURM_ARRAY_TASK_ID=1)
           sbatch run_gs_benchmark_v4.sh
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "finite_peps_quench.jl"))
include(joinpath(@__DIR__, "finite_peps_full_update.jl"))
include(joinpath(@__DIR__, "finite_peps_gauge_invariant.jl"))
include(joinpath(@__DIR__, "finite_peps_boundary_mps.jl"))
include(joinpath(@__DIR__, "finite_peps_fullupdate_v4.jl"))
include(joinpath(@__DIR__, "finite_ed.jl"))

using CSV, DataFrames, Printf, Statistics

const V4_TASK_ID = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))

const V4_PARAM_GRID = [
    (m=0.25, g=0.25), (m=0.25, g=0.50), (m=0.25, g=1.00),
    (m=0.25, g=2.00), (m=0.25, g=4.00),
    (m=0.50, g=0.25), (m=0.50, g=0.50), (m=0.50, g=2.00),
    (m=0.50, g=4.00),
]

const V4_NX    = 3
const V4_NY    = 4
const V4_DG    = 1
const V4_D_MAX = 4         # env cost ~ (dp·D)⁴; D=4 keeps it ~270 MB/bond
const V4_N_ITE = 200       # full update is far heavier per step than v3
const V4_T_HOP = 1.0
const V4_NOISE = 0.1
const V4_CHI   = 48        # boundary-MPS bond dimension
const V4_NALS  = 30        # ALS sweeps per bond

function _staggered_charges_v4(nx, ny)
    g = zeros(Int, nx, ny)
    for iy in 1:ny, ix in 1:nx
        isodd(ix+iy) && (g[ix,iy] = -1)
    end
    return g
end

function run_one_v4(task_id::Int)
    idx = ((task_id - 1) % length(V4_PARAM_GRID)) + 1
    p = V4_PARAM_GRID[idx]
    m, g = p.m, p.g
    nx, ny, dg = V4_NX, V4_NY, V4_DG
    gch = _staggered_charges_v4(nx, ny)
    @printf("=== Task %d: m=%.2f g=%.2f D_max=%d n_ite=%d χ=%d n_als=%d (boundary-MPS full update v4) ===\n",
            task_id, m, g, V4_D_MAX, V4_N_ITE, V4_CHI, V4_NALS); flush(stdout)

    # ── v4 ground state ───────────────────────────────────────────────────
    @printf("  Running boundary-MPS full-update ITE...\n"); flush(stdout)
    t0 = time()
    gp = ite_ground_state_v4(nx, ny, dg, V4_D_MAX;
                             g=g, t_hop=V4_T_HOP, m=m, g_charges=gch,
                             n_ite=V4_N_ITE, noise=V4_NOISE,
                             χ=V4_CHI, n_als=V4_NALS, verbose=true)
    obs = measure_all_finite(gp.peps, nx, ny, dg, 0.0)
    gviol = forbidden_norm(gp)
    E_peps = compute_energy_v4(gp; g=g, t_hop=V4_T_HOP, m=m, χ=V4_CHI)
    t_peps = time() - t0
    @printf("[PEPS]  E=%.6f  ⟨n_f⟩=%.6f  nf_e=%.6f  nf_o=%.6f  ⟨E²⟩=%.6f  viol=%.1e  (%.1fs)\n",
            E_peps, obs.nf_mean, obs.nf_even, obs.nf_odd, obs.E2_mean, gviol, t_peps); flush(stdout)

    # ── ED ────────────────────────────────────────────────────────────────
    @printf("  Running ED...\n"); flush(stdout)
    t1 = time()
    states, key = build_basis(nx, ny, dg, gch)
    H = build_hamiltonian(states, key, nx, ny, dg; g=g, t=V4_T_HOP, m=m)
    E0, ψ0 = find_ground_state(H)
    obs_ed = measure_ED(ψ0, states, nx, ny, dg, 0.0)
    t_ed = time() - t1
    @printf("[ED ]  E0=%.6f  nf_e=%.6f  ⟨E²⟩=%.6f  (%.1fs)\n",
            E0, obs_ed.nf_even, obs_ed.E2_mean, t_ed); flush(stdout)

    @printf("\n  ── Comparison ──\n")
    @printf("           PEPS        ED          Δ\n")
    @printf("  E0       %.5f   %.5f   %.2e\n", E_peps, E0, abs(E_peps - E0))
    @printf("  nf_even  %.5f   %.5f   %.2e\n", obs.nf_even, obs_ed.nf_even, abs(obs.nf_even-obs_ed.nf_even))
    @printf("  E2_mean  %.5f   %.5f   %.2e\n", obs.E2_mean, obs_ed.E2_mean, abs(obs.E2_mean-obs_ed.E2_mean))
    flush(stdout)

    D_final = maximum(length(gp.qh[ix,iy]) for iy in 1:ny for ix in 1:nx-1; init=1)

    df = DataFrame(
        task_id=task_id, m=m, g=g, t_hop=V4_T_HOP, nx=nx, ny=ny, dg=dg,
        D_max=V4_D_MAX, n_ite=V4_N_ITE, noise=V4_NOISE, chi=V4_CHI, n_als=V4_NALS,
        D_final=D_final, gauge_viol=gviol,
        peps_energy=E_peps,
        peps_nf_mean=obs.nf_mean, peps_nf_even=obs.nf_even, peps_nf_odd=obs.nf_odd,
        peps_E2_mean=obs.E2_mean, peps_S_h=obs.S_h_mean, peps_S_v=obs.S_v_mean,
        peps_S_mean=obs.S_mean, peps_time_s=t_peps,
        ed_E0=E0, ed_nf_mean=obs_ed.nf_mean, ed_nf_even=obs_ed.nf_even,
        ed_nf_odd=obs_ed.nf_odd, ed_E2_mean=obs_ed.E2_mean,
        ed_basis_dim=length(states), ed_time_s=t_ed,
        d_energy=E_peps - E0,
        d_nf_mean=obs.nf_mean - obs_ed.nf_mean,
        d_nf_even=obs.nf_even - obs_ed.nf_even,
        d_nf_odd=obs.nf_odd - obs_ed.nf_odd,
        d_E2_mean=obs.E2_mean - obs_ed.E2_mean,
        d_cdw=(obs.nf_even-obs.nf_odd) - (obs_ed.nf_even-obs_ed.nf_odd),
    )

    outdir = joinpath(@__DIR__, "results", "gs_bench_v4")
    mkpath(outdir)
    outfile = joinpath(outdir, @sprintf("gs_benchmark_%03d.csv", task_id))
    CSV.write(outfile, df)
    @printf("  Saved: %s\n", outfile); flush(stdout)
end

run_one_v4(V4_TASK_ID)
