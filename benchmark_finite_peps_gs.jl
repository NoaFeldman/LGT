# =============================================================================
#  Finite-PEPS ground-state benchmark (with magnetic plaquette)  —  cluster.
#
#  Usage:  julia benchmark_finite_peps_gs.jl <D> [n_steps] [noise] [nx] [ny]
#  e.g.:   julia benchmark_finite_peps_gs.jl 8 2000 0.05 4 4
#
#  Defaults:  n_steps=2000, noise=0.05, nx=ny=4.
#  Output goes to  results_gs/D<D>/  in the working directory.
# =============================================================================

if length(ARGS) < 1
    error("Usage: julia benchmark_finite_peps_gs.jl <D> [n_steps] [noise] [nx] [ny]")
end
const D_INPUT  = parse(Int, ARGS[1])
const N_STEPS  = length(ARGS) >= 2 ? parse(Int,     ARGS[2]) : 2000
const NOISE    = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.05
const NX_INPUT = length(ARGS) >= 4 ? parse(Int,     ARGS[4]) : 4
const NY_INPUT = length(ARGS) >= 5 ? parse(Int,     ARGS[5]) : 4

D_INPUT >= 2 || error("D must be >= 2, got $D_INPUT")

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
for pkg in ("Plots", "CSV", "DataFrames")
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg)
end

# Headless plotting on cluster
ENV["GKSwstype"] = "100"

include(joinpath(@__DIR__, "finite_peps_ground_state.jl"))

using CSV, DataFrames

const OUTDIR = joinpath(@__DIR__, "results_gs", "D$(D_INPUT)")
mkpath(OUTDIR)
println("Output directory: $OUTDIR")

# Physics parameters (tweak here if you want to scan)
const G_COUPLING = 1.0
const T_HOPPING  = 1.0
const M_MASS_VAL = 2.0
const TAU_STEP   = 0.01
const MEAS_EVERY = max(1, N_STEPS ÷ 100)   # ~100 measurement points

println("================================================================")
println("  Finite PEPS ground state — U(1) LGT with magnetic plaquette")
@printf("    D       = %d\n",      D_INPUT)
@printf("    lattice = %d × %d\n", NX_INPUT, NY_INPUT)
@printf("    steps   = %d  (measure every %d)\n", N_STEPS, MEAS_EVERY)
@printf("    τ       = %.4f\n",    TAU_STEP)
@printf("    noise   = %.4f\n",    NOISE)
@printf("    g, t, m = %.3f, %.3f, %.3f\n", G_COUPLING, T_HOPPING, M_MASS_VAL)
println("================================================================")

t0 = time()
peps_final, times, nf_hist, E_hist, Eplaq_hist =
    run_finite_peps_groundstate(
        nx            = NX_INPUT,
        ny            = NY_INPUT,
        dg            = 1,
        D_bond        = D_INPUT,
        D_max         = D_INPUT,
        g             = G_COUPLING,
        t_hop         = T_HOPPING,
        m_mass        = M_MASS_VAL,
        τ             = TAU_STEP,
        n_steps       = N_STEPS,
        measure_every = MEAS_EVERY,
        noise         = NOISE,
        plot_dir      = OUTDIR,
    )
elapsed = time() - t0
@printf("\n  Wall time: %.1f s  (%.2f h)\n", elapsed, elapsed / 3600)

# Save time series as CSV
df = DataFrame(
    step      = round.(Int, times ./ TAU_STEP),
    time      = times,
    nf_mean   = nf_hist,
    E_total   = E_hist,
    E_plaq    = Eplaq_hist,
)
CSV.write(joinpath(OUTDIR, "trajectory_D$(D_INPUT).csv"), df)
println("  Trajectory CSV: ", joinpath(OUTDIR, "trajectory_D$(D_INPUT).csv"))
println("================================================================")
