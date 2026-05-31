#= v4_update_test.jl — validate the Stage-2b environment-weighted full update.

   Runs a few v4 ITE steps and checks gauge invariance, half-filling, and that
   the variational energy is non-increasing.  Run under SLURM (needs memory).

   Usage:  sbatch run_v4_update_test.sh     (or julia --project=. v4_update_test.jl)
=#
ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "finite_peps_quench.jl"))
include(joinpath(@__DIR__, "finite_peps_full_update.jl"))
include(joinpath(@__DIR__, "finite_peps_gauge_invariant.jl"))
include(joinpath(@__DIR__, "finite_peps_boundary_mps.jl"))
include(joinpath(@__DIR__, "finite_peps_fullupdate_v4.jl"))

using Printf

# Intermediate coupling g=1 is where v3 collapsed to the classical vacuum —
# the decisive regime for whether the full update helps.
ok = v4_update_selftest(; nxv=3, nyv=4, dg=1, g=1.0, t_hop=1.0, m=0.25,
                         D_max=4, nsteps=8, χ=48, n_als=30, noise=0.1)

println()
println(ok ? "V4 UPDATE SELF-TEST PASSED" : "V4 UPDATE SELF-TEST FAILED — inspect above")
exit(ok ? 0 : 1)
