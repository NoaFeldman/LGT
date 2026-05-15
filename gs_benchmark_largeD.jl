# =============================================================================
#  gs_benchmark_largeD.jl
#
#  Ground-state benchmark: PEPS (simple-update ITE) vs ED on the 3×4 U(1) LGT.
#  Like gs_benchmark.jl but with tunable bond dimension and more ITE steps.
#
#  Usage:
#      julia --project=. gs_benchmark_largeD.jl <task_id> [--D-bond D] [--D-max Dmax] \
#            [--n-ite N] [--tau TAU] [--out-dir DIR]
#
#  Defaults:  D_bond=8, D_max=24, n_ite=600, tau=0.02
#
#  task_id ∈ 1 .. length(PARAM_GRID).
# =============================================================================

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "finite_peps_quench.jl"))
include(joinpath(@__DIR__, "finite_ed.jl"))

using CSV, DataFrames, Printf

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Parameter grid (same as gs_benchmark.jl)                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

const MASS_LIST = [0.25, 0.5, 1.0, 2.0, 4.0]
const G_LIST    = [0.25, 0.5, 1.0, 2.0, 4.0]
const PARAM_GRID = [(m=m, g=g) for m in MASS_LIST, g in G_LIST] |> vec

const BENCH_NX     = 3
const BENCH_NY     = 4
const BENCH_DG     = 1
const BENCH_T_HOP  = 1.0
const BENCH_MU_ITE = 0.0

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLI parsing                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function _parse_args_largeD(argv::Vector{String})
    isempty(argv) && error("Usage: julia gs_benchmark_largeD.jl <task_id> [options]")
    task_id = parse(Int, argv[1])

    # Defaults: larger than the original gs_benchmark.jl
    D_bond  = 8
    D_max   = 24
    n_ite   = 600
    tau     = 0.02
    out_dir = "results/gs_bench_largeD"

    i = 2
    while i <= length(argv)
        if argv[i] == "--D-bond" && i < length(argv)
            D_bond = parse(Int, argv[i+1]); i += 2
        elseif argv[i] == "--D-max" && i < length(argv)
            D_max = parse(Int, argv[i+1]); i += 2
        elseif argv[i] == "--n-ite" && i < length(argv)
            n_ite = parse(Int, argv[i+1]); i += 2
        elseif argv[i] == "--tau" && i < length(argv)
            tau = parse(Float64, argv[i+1]); i += 2
        elseif argv[i] == "--out-dir" && i < length(argv)
            out_dir = argv[i+1]; i += 2
        else
            i += 1
        end
    end
    return task_id, D_bond, D_max, n_ite, tau, out_dir
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Helpers                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function _make_staggered_charges(nx::Int, ny::Int)
    g_charges = zeros(Int, nx, ny)
    for iy in 1:ny, ix in 1:nx
        isodd(ix + iy) && (g_charges[ix, iy] = -1)
    end
    return g_charges
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function run_one_point_largeD(task_id::Int, D_bond::Int, D_max::Int,
                               n_ite::Int, tau::Float64, out_dir::String)
    1 <= task_id <= length(PARAM_GRID) ||
        error("task_id=$task_id outside 1..$(length(PARAM_GRID))")

    p = PARAM_GRID[task_id]
    m = p.m
    g = p.g
    nx, ny, dg = BENCH_NX, BENCH_NY, BENCH_DG

    println("=" ^ 72)
    println("  GS benchmark (large D) task $task_id / $(length(PARAM_GRID))")
    println("  m = $m   g = $g   t_hop = $BENCH_T_HOP")
    println("  lattice $(nx)×$(ny)  dg=$dg  D_bond=$D_bond  D_max=$D_max")
    println("  ITE: τ=$tau  n=$n_ite  μ_ite=$BENCH_MU_ITE")
    println("=" ^ 72)

    # ── PEPS ──────────────────────────────────────────────────────────────────
    println("\n[PEPS] imaginary-time evolution …")
    t_peps_0 = time()
    peps = ite_ground_state(nx, ny, dg, D_bond, D_max;
                             g=g, t_hop=BENCH_T_HOP, m=m,
                             τ_ite=tau, n_ite=n_ite,
                             μ=BENCH_MU_ITE)
    obs_peps = measure_all_finite(peps, nx, ny, dg, 0.0)
    t_peps   = time() - t_peps_0
    @printf("[PEPS]  ⟨n_f⟩=%.6f  ⟨n_f⟩_e=%.6f  ⟨n_f⟩_o=%.6f  ⟨E²⟩=%.6f   (%.1fs)\n",
            obs_peps.nf_mean, obs_peps.nf_even, obs_peps.nf_odd, obs_peps.E2_mean, t_peps)

    # ── ED ────────────────────────────────────────────────────────────────────
    println("\n[ED ] building basis + Hamiltonian …")
    t_ed_0 = time()
    g_charges = _make_staggered_charges(nx, ny)
    states, key_dict = build_basis(nx, ny, dg, g_charges)
    H = build_hamiltonian(states, key_dict, nx, ny, dg;
                          g=g, t=BENCH_T_HOP, m=m)
    println("[ED ] basis=$(length(states))  nnz(H)=$(nnz(H))")
    E0, ψ0 = find_ground_state(H)
    obs_ed = measure_ED(ψ0, states, nx, ny, dg, 0.0)
    t_ed = time() - t_ed_0
    @printf("[ED ]  ⟨n_f⟩=%.6f  ⟨n_f⟩_e=%.6f  ⟨n_f⟩_o=%.6f  ⟨E²⟩=%.6f   E0=%.6f   (%.1fs)\n",
            obs_ed.nf_mean, obs_ed.nf_even, obs_ed.nf_odd, obs_ed.E2_mean, E0, t_ed)

    # ── Pack one-row CSV ─────────────────────────────────────────────────────
    df = DataFrame(
        task_id      = task_id,
        m            = m,
        g            = g,
        t_hop        = BENCH_T_HOP,
        nx           = nx,
        ny           = ny,
        dg           = dg,
        D_bond       = D_bond,
        D_max        = D_max,
        tau_ite      = tau,
        n_ite        = n_ite,
        mu_ite       = BENCH_MU_ITE,
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
        # Differences (PEPS − ED)
        d_nf_mean    = obs_peps.nf_mean - obs_ed.nf_mean,
        d_nf_even    = obs_peps.nf_even - obs_ed.nf_even,
        d_nf_odd     = obs_peps.nf_odd  - obs_ed.nf_odd,
        d_E2_mean    = obs_peps.E2_mean - obs_ed.E2_mean,
        d_cdw        = (obs_peps.nf_even - obs_peps.nf_odd) -
                       (obs_ed.nf_even   - obs_ed.nf_odd),
    )

    mkpath(out_dir)
    fname = joinpath(out_dir, @sprintf("gs_benchmark_%03d.csv", task_id))
    CSV.write(fname, df)
    println("\nSaved: $fname")

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    task_id, D_bond, D_max, n_ite, tau, out_dir = _parse_args_largeD(ARGS)
    run_one_point_largeD(task_id, D_bond, D_max, n_ite, tau, out_dir)
end
