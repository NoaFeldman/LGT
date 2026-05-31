#= v4_selftest.jl — validate the Stage-1 boundary-MPS environment in isolation.

   Usage:
       julia --project=. v4_selftest.jl
=#
ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "finite_peps_quench.jl"))
include(joinpath(@__DIR__, "finite_peps_full_update.jl"))
include(joinpath(@__DIR__, "finite_peps_gauge_invariant.jl"))
include(joinpath(@__DIR__, "finite_peps_boundary_mps.jl"))

using Printf

ok = bmps_selftest(; nxv=3, nyv=4, dg=1, g=1.0, t_hop=1.0, m=0.25,
                    χ=64, nsteps=20, noise=0.1)

println()
println(ok ? "ALL STAGE-1 CHECKS PASSED" : "STAGE-1 CHECKS FAILED — inspect output above")
exit(ok ? 0 : 1)
