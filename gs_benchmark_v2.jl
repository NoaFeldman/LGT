#= ═══════════════════════════════════════════════════════════════════════════════
   gs_benchmark_v2.jl

   Ground-state benchmark using the full-update PEPS ITE (v2).
   Compares against exact diagonalization on a 3×4 lattice.
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "finite_peps_ground_state.jl"))
include(joinpath(@__DIR__, "finite_peps_full_update.jl"))
include(joinpath(@__DIR__, "finite_ed.jl"))

using CSV, DataFrames, Printf, Statistics

# ── Parse SLURM array index ──────────────────────────────────────────────────
const TASK_ID = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))

# ── Parameter grid ───────────────────────────────────────────────────────────
const PARAM_GRID = [
    (m=0.25, g=0.25), (m=0.25, g=0.50), (m=0.25, g=1.00),
    (m=0.25, g=2.00), (m=0.25, g=4.00),
    (m=0.50, g=0.25), (m=0.50, g=0.50), (m=0.50, g=2.00),
    (m=0.50, g=4.00),
]

const NX      = 3
const NY      = 4
const DG      = 1
const D_MAX   = 12
const N_ITE   = 600
const T_HOP   = 1.0
const NOISE   = 0.01

function run_one(task_id::Int)
    idx = ((task_id - 1) % length(PARAM_GRID)) + 1
    p   = PARAM_GRID[idx]
    m, g = p.m, p.g
    @printf("=== Task %d: m=%.2f g=%.2f D_max=%d n_ite=%d noise=%.3f ===\n",
            task_id, m, g, D_MAX, N_ITE, NOISE)

    # ── Full-update PEPS ground state ─────────────────────────────────────
    @printf("  Running full-update PEPS ITE...\n")
    peps = ite_ground_state_v2(NX, NY, DG, D_MAX;
                                g=g, t_hop=T_HOP, m=m,
                                n_ite=N_ITE, noise=NOISE,
                                use_env=true, verbose=true)

    # ── ED ground state ──────────────────────────────────────────────────
    @printf("  Running ED...\n")
    ed = ed_ground_state(NX, NY, DG; g=g, t_hop=T_HOP, m=m)

    # ── Measure PEPS observables ──────────────────────────────────────────
    nf_peps = zeros(NX, NY)
    E2_peps = zeros(NX, NY)
    for iy in 1:NY, ix in 1:NX
        _, d_gR, d_gU = site_dims(ix, iy, NX, NY, DG)
        nf_peps[ix,iy] = expect_site(peps, ix, iy,
                                      embed_f_site(op_nf(), d_gR, d_gU))
        O_E2 = zeros(ComplexF64, d_gR*d_gU*2, d_gR*d_gU*2)
        if ix < NX
            O_E2 .+= embed_R_site(op_E2(DG), d_gU)
        end
        if iy < NY
            O_E2 .+= embed_U_site(op_E2(DG), d_gR)
        end
        E2_peps[ix,iy] = expect_site(peps, ix, iy, O_E2)
    end

    even_mask = [iseven(ix+iy) for ix in 1:NX, iy in 1:NY]
    odd_mask  = .!even_mask

    peps_nf_even = mean(nf_peps[even_mask])
    peps_nf_odd  = mean(nf_peps[odd_mask])
    peps_nf_mean = mean(nf_peps)
    peps_E2_mean = mean(E2_peps)

    # ── ED observables ────────────────────────────────────────────────────
    ed_nf_even = mean(ed.nf_grid[even_mask])
    ed_nf_odd  = mean(ed.nf_grid[odd_mask])
    ed_nf_mean = mean(ed.nf_grid)
    ed_E2_mean = mean(ed.E2_grid)

    # ── Print comparison ──────────────────────────────────────────────────
    @printf("\n  ── Comparison ──\n")
    @printf("           PEPS      ED       Δ\n")
    @printf("  nf_mean  %.5f   %.5f   %.2e\n",
            peps_nf_mean, ed_nf_mean, abs(peps_nf_mean-ed_nf_mean))
    @printf("  nf_even  %.5f   %.5f   %.2e\n",
            peps_nf_even, ed_nf_even, abs(peps_nf_even-ed_nf_even))
    @printf("  E2_mean  %.5f   %.5f   %.2e\n",
            peps_E2_mean, ed_E2_mean, abs(peps_E2_mean-ed_E2_mean))

    D_final = maximum(length(peps.λh[ix, iy])
                      for iy in 1:NY for ix in 1:NX-1; init=1)

    # ── Save ──────────────────────────────────────────────────────────────
    df = DataFrame(
        m=m, g=g, D_max=D_MAX, n_ite=N_ITE, noise=NOISE, D_final=D_final,
        peps_nf_mean=peps_nf_mean, ed_nf_mean=ed_nf_mean,
        peps_nf_even=peps_nf_even, ed_nf_even=ed_nf_even,
        peps_nf_odd=peps_nf_odd,   ed_nf_odd=ed_nf_odd,
        peps_E2_mean=peps_E2_mean, ed_E2_mean=ed_E2_mean,
        d_nf_even=peps_nf_even-ed_nf_even,
        d_E2_mean=peps_E2_mean-ed_E2_mean,
    )

    outdir = joinpath(@__DIR__, "results", "gs_bench_v2")
    mkpath(outdir)
    outfile = joinpath(outdir, @sprintf("gs_benchmark_%03d.csv", task_id))
    CSV.write(outfile, df)
    @printf("  Saved: %s\n", outfile)
end

run_one(TASK_ID)
