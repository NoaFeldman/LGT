#= ═══════════════════════════════════════════════════════════════════════════════
   gs_benchmark_v2.jl

   Ground-state benchmark using the full-update PEPS ITE (v2).
   Compares against exact diagonalization on a 3×4 lattice.

   Usage:
       julia --project=. gs_benchmark_v2.jl
       # or via SLURM: sbatch run_gs_benchmark_v2.sh
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "finite_peps_quench.jl"))        # loads finite_peps_ground_state.jl
include(joinpath(@__DIR__, "finite_peps_full_update.jl"))
include(joinpath(@__DIR__, "finite_ed.jl"))

using CSV, DataFrames, Printf, Statistics

# ── Parse SLURM array index ──────────────────────────────────────────────────
const V2_TASK_ID = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))

# ── Parameter grid ───────────────────────────────────────────────────────────
const V2_PARAM_GRID = [
    (m=0.25, g=0.25), (m=0.25, g=0.50), (m=0.25, g=1.00),
    (m=0.25, g=2.00), (m=0.25, g=4.00),
    (m=0.50, g=0.25), (m=0.50, g=0.50), (m=0.50, g=2.00),
    (m=0.50, g=4.00),
]

const V2_NX      = 3
const V2_NY      = 4
const V2_DG      = 1
const V2_D_MAX   = 12
const V2_N_ITE   = 600
const V2_T_HOP   = 1.0
const V2_NOISE   = 0.1     # larger init noise breaks the classical staggered attractor at strong g

function _make_staggered_charges_v2(nx::Int, ny::Int)
    g_charges = zeros(Int, nx, ny)
    for iy in 1:ny, ix in 1:nx
        isodd(ix + iy) && (g_charges[ix, iy] = -1)
    end
    return g_charges
end

function run_one_v2(task_id::Int)
    idx = ((task_id - 1) % length(V2_PARAM_GRID)) + 1
    p   = V2_PARAM_GRID[idx]
    m, g = p.m, p.g
    nx, ny, dg = V2_NX, V2_NY, V2_DG
    @printf("=== Task %d: m=%.2f g=%.2f D_max=%d n_ite=%d noise=%.3f ===\n",
            task_id, m, g, V2_D_MAX, V2_N_ITE, V2_NOISE)

    # ── Full-update PEPS ground state ─────────────────────────────────────
    @printf("  Running full-update PEPS ITE...\n")
    t_peps_0 = time()
    peps = ite_ground_state_v2(nx, ny, dg, V2_D_MAX;
                                g=g, t_hop=V2_T_HOP, m=m,
                                n_ite=V2_N_ITE, noise=V2_NOISE,
                                use_env=true, verbose=true)
    obs_peps = measure_all_finite(peps, nx, ny, dg, 0.0)
    t_peps = time() - t_peps_0
    @printf("[PEPS]  ⟨n_f⟩=%.6f  ⟨n_f⟩_e=%.6f  ⟨n_f⟩_o=%.6f  ⟨E²⟩=%.6f   (%.1fs)\n",
            obs_peps.nf_mean, obs_peps.nf_even, obs_peps.nf_odd, obs_peps.E2_mean, t_peps)

    # ── ED ground state ──────────────────────────────────────────────────
    @printf("  Running ED...\n")
    t_ed_0 = time()
    g_charges = _make_staggered_charges_v2(nx, ny)
    states, key_dict = build_basis(nx, ny, dg, g_charges)
    H = build_hamiltonian(states, key_dict, nx, ny, dg;
                          g=g, t=V2_T_HOP, m=m)
    println("[ED ] basis=$(length(states))  nnz(H)=$(nnz(H))")
    E0, ψ0 = find_ground_state(H)
    obs_ed = measure_ED(ψ0, states, nx, ny, dg, 0.0)
    t_ed = time() - t_ed_0
    @printf("[ED ]  ⟨n_f⟩=%.6f  ⟨n_f⟩_e=%.6f  ⟨n_f⟩_o=%.6f  ⟨E²⟩=%.6f   E0=%.6f   (%.1fs)\n",
            obs_ed.nf_mean, obs_ed.nf_even, obs_ed.nf_odd, obs_ed.E2_mean, E0, t_ed)

    # ── Print comparison ──────────────────────────────────────────────────
    @printf("\n  ── Comparison ──\n")
    @printf("           PEPS      ED       Δ\n")
    @printf("  nf_mean  %.5f   %.5f   %.2e\n",
            obs_peps.nf_mean, obs_ed.nf_mean, abs(obs_peps.nf_mean-obs_ed.nf_mean))
    @printf("  nf_even  %.5f   %.5f   %.2e\n",
            obs_peps.nf_even, obs_ed.nf_even, abs(obs_peps.nf_even-obs_ed.nf_even))
    @printf("  E2_mean  %.5f   %.5f   %.2e\n",
            obs_peps.E2_mean, obs_ed.E2_mean, abs(obs_peps.E2_mean-obs_ed.E2_mean))

    D_final = maximum(length(peps.λh[ix, iy])
                      for iy in 1:ny for ix in 1:nx-1; init=1)

    # ── Save ──────────────────────────────────────────────────────────────
    df = DataFrame(
        task_id      = task_id,
        m            = m,
        g            = g,
        t_hop        = V2_T_HOP,
        nx           = nx,
        ny           = ny,
        dg           = dg,
        D_max        = V2_D_MAX,
        n_ite        = V2_N_ITE,
        noise        = V2_NOISE,
        D_final      = D_final,
        # PEPS
        peps_nf_mean = obs_peps.nf_mean,
        peps_nf_even = obs_peps.nf_even,
        peps_nf_odd  = obs_peps.nf_odd,
        peps_E2_mean = obs_peps.E2_mean,
        peps_S_h     = obs_peps.S_h_mean,
        peps_S_v     = obs_peps.S_v_mean,
        peps_S_mean  = obs_peps.S_mean,
        peps_time_s  = t_peps,
        # ED
        ed_E0        = E0,
        ed_nf_mean   = obs_ed.nf_mean,
        ed_nf_even   = obs_ed.nf_even,
        ed_nf_odd    = obs_ed.nf_odd,
        ed_E2_mean   = obs_ed.E2_mean,
        ed_basis_dim = length(states),
        ed_time_s    = t_ed,
        # Differences
        d_nf_mean    = obs_peps.nf_mean - obs_ed.nf_mean,
        d_nf_even    = obs_peps.nf_even - obs_ed.nf_even,
        d_nf_odd     = obs_peps.nf_odd  - obs_ed.nf_odd,
        d_E2_mean    = obs_peps.E2_mean - obs_ed.E2_mean,
        d_cdw        = (obs_peps.nf_even - obs_peps.nf_odd) -
                       (obs_ed.nf_even   - obs_ed.nf_odd),
    )

    outdir = joinpath(@__DIR__, "results", "gs_bench_v2")
    mkpath(outdir)
    outfile = joinpath(outdir, @sprintf("gs_benchmark_%03d.csv", task_id))
    CSV.write(outfile, df)
    @printf("  Saved: %s\n", outfile)
end

run_one_v2(V2_TASK_ID)
