#= ═══════════════════════════════════════════════════════════════════════════════
   decouple_commute_test.jl

   Does the Green's-function gauge–matter decoupling COMMUTE with exact
   contraction of the MPS?  After a quench, for the time-evolved state ψ(t):

     case 1 :  contract ψ(t) → exact vector,  then decouple   φ₁ = 𝒰_dense · vec(ψ)
     case 2 :  decouple ψ(t) as an MPO,  then contract        φ₂ = vec(𝒰_MPO · ψ)

   Without bond truncation these are identical, so the fidelities below measure
   how much COMPRESSING the decoupled state degrades it as the quench entangles
   the system.  We report the total fidelity |⟨φ₁|φ₂⟩|² and the per-sector
   (matter / gauge) reduced-state fidelities, all vs t.

   ── Decoupling operator ──────────────────────────────────────────────────────
   𝒰 = exp(−iÔ),  Ô = Σ_{ℓ,n} M_{ℓ,n} φ_ℓ Q_n,  M = ∇G  (gradient of the 2D
   lattice Green's function G = (−∇²)⁻¹ — the shift field that slaves the
   electric field to the charges).  φ_ℓ = ½(U+U†) on link ℓ, Q_n = n_f − bg.

   ── Feasibility ──────────────────────────────────────────────────────────────
   "Exact contraction" requires the FULL Hilbert space (no gauge-sector
   restriction, since 𝒰 shifts E between sectors), so this runs on a 2×2 grid
   (dim 1296).  3×4 is ~18¹² and cannot be densely contracted.

   Requires: mps_lgt.jl, lgt_greens_function.jl
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "mps_lgt.jl"))            # MPS/MPO/DMRG/Krylov/quench + CMPO
include(joinpath(@__DIR__, "lgt_greens_function.jl"))# generate_greens_function (2D G)

using LinearAlgebra
using Printf
using CSV, DataFrames, Plots

default(fontfamily="Computer Modern", linewidth=2, framestyle=:box, grid=true,
        legend=:best, size=(900, 600), dpi=200)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Chebyshev MPO exponential (copied/adapted from gauge_matter_unitary.jl)  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

using QuadGK
besselJ(k::Int, a::Real) = quadgk(τ -> cos(k * τ - a * sin(τ)), 0, π; rtol=1e-12)[1] / π

mpo_identity_c(dims) = CMPO([reshape(ComplexF64.(Matrix(I, d, d)), 1, 1, d, d) for d in dims])
mpo_scale_c(α, A::CMPO) = CMPO([p == 1 ? α .* A[1] : A[p] for p in eachindex(A)])

function mpo_mult_c(A::CMPO, B::CMPO)
    n = length(A); C = CMPO(undef, n)
    for p in 1:n
        DlA, DrA, d, _ = size(A[p]); DlB, DrB, _, _ = size(B[p])
        Cp = zeros(ComplexF64, DlA * DlB, DrA * DrB, d, d)
        @inbounds for aL in 1:DlA, aR in 1:DrA, bL in 1:DlB, bR in 1:DrB
            Cp[(aL-1)*DlB+bL, (aR-1)*DrB+bR, :, :] = A[p][aL, aR, :, :] * B[p][bL, bR, :, :]
        end
        C[p] = Cp
    end
    return C
end

"""exp(−i O) as an MPO via Jacobi–Anger; `a` ≥ spectral radius of O."""
function expmi_mpo(O::CMPO, a::Float64; ε::Float64=1e-9, Dmax::Int=48, Kmax::Int=400)
    dims = [size(Op, 3) for Op in O]
    Hs = mpo_scale_c(1 / a, O); mpo_compress!(Hs; ε=ε, Dmax=Dmax)
    Tprev = mpo_identity_c(dims); Tcur = Hs
    U = mpo_axpby(besselJ(0, a), Tprev, 2 * (-im) * besselJ(1, a), Tcur)
    mpo_compress!(U; ε=ε, Dmax=Dmax)
    for k in 2:Kmax
        HT = mpo_mult_c(Hs, Tcur); mpo_compress!(HT; ε=ε, Dmax=Dmax)
        Tnext = mpo_axpby(2.0, HT, -1.0, Tprev); mpo_compress!(Tnext; ε=ε, Dmax=Dmax)
        ck = 2 * (-im)^k * besselJ(k, a)
        U = mpo_axpby(1.0, U, ck, Tnext); mpo_compress!(U; ε=ε, Dmax=Dmax)
        Tprev, Tcur = Tcur, Tnext
        (k > a && abs(besselJ(k, a)) < ε) && break
    end
    return U
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MPS → dense vector (site-1 OUTER, matching mpo_to_dense)                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function mps_to_dense(ψ::CMPS)
    P = permutedims(ψ[1][1, :, :], (2, 1))          # (k1, Dr)
    for p in 2:length(ψ)
        Dl, Dr, d = size(ψ[p]); ka = size(P, 1)
        Pnew = zeros(ComplexF64, ka * d, Dr)
        @inbounds for r in 1:Dr, l in 1:Dl
            Pnew[:, r] .+= kron(P[:, l], ψ[p][l, r, :])
        end
        P = Pnew
    end
    return P[:, 1]
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Decoupling exponent Ô = Σ M φ Q  on the grid                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

φ_op() = 0.5 * (op_U_gauge(1) + op_U_gauge(1)')      # ½(U+U†), Hermitian phase (dg=1)
bg_stag(ix, iy) = isodd(ix + iy) ? 1.0 : 0.0

"""Decoupling exponent Ô as a CMPO, plus a spectral-radius bound for exp."""
function build_decoupling_O(nx, ny, dg)
    @assert dg == 1
    _, pos = snake_nodes(nx, ny)
    _, Gt = generate_greens_function(nx, ny)         # G[ix,iy,jx,jy]
    dims = node_dims(nx, ny, dg)
    φ = ComplexF64.(φ_op())
    terms = HTerm[]
    Mabs = 0.0

    # Q_n operator (charge) on each node
    Qnode(jx, jy) = ComplexF64.(embed_f_site(op_nf() - bg_stag(jx, jy) * _Id(LGT_d_f),
                                             site_dims(jx, jy, nx, ny, dg)[2:3]...))
    # φ_ℓ on the node carrying link ℓ (right or up gauge DoF)
    φR(ix, iy) = ComplexF64.(embed_R_site(φ, site_dims(ix, iy, nx, ny, dg)[3]))   # needs d_gU
    φU(ix, iy) = ComplexF64.(embed_U_site(φ, site_dims(ix, iy, nx, ny, dg)[2]))   # needs d_gR

    function add_link!(linknode, φlink, nbx, nby, ix, iy)
        for jy in 1:ny, jx in 1:nx
            M = Gt[ix, iy, jx, jy] - Gt[nbx, nby, jx, jy]   # ∇G across the link
            abs(M) < 1e-14 && continue
            Mabs += abs(M)
            a, b = pos[linknode], pos[(jx, jy)]
            Q = Qnode(jx, jy)
            if a == b
                push!(terms, HTerm(M, Dict(a => φlink * Q)))
            else
                push!(terms, HTerm(M, Dict(a => copy(φlink), b => Q)))
            end
        end
    end

    for iy in 1:ny, ix in 1:nx-1                     # right links
        add_link!((ix, iy), φR(ix, iy), ix+1, iy, ix, iy)
    end
    for iy in 1:ny-1, ix in 1:nx                     # up links
        add_link!((ix, iy), φU(ix, iy), ix, iy+1, ix, iy)
    end

    O = _assemble_mpo(dims, terms)
    a_bound = Mabs * opnorm(Matrix(φ)) * opnorm(op_nf() - 0.5 * _Id(LGT_d_f))
    return O, a_bound
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Matter / gauge bipartition of the dense vector + reduced fidelity        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Reorder the full dense vector into (matter ⊗ gauge) and reshape to a matrix
Ψ[m,g], with matter = all fermion indices, gauge = all electric-field indices."""
function matter_gauge_matrix(v::Vector{ComplexF64}, nx, ny, dg)
    chain, _ = snake_nodes(nx, ny)
    dimsf = Int[]; dimsg = Int[]                    # per-node (fermion, gauge=d_gR*d_gU)
    for (ix, iy) in chain
        _, dgR, dgU = site_dims(ix, iy, nx, ny, dg)
        push!(dimsf, LGT_d_f); push!(dimsg, dgR * dgU)
    end
    Dm = prod(dimsf); Dg = prod(dimsg)
    Ψ = zeros(ComplexF64, Dm, Dg)
    # global index (site-1 outer): decode per node into (nf, eR*eU); recombine.
    nodes = length(chain)
    for g0 in 0:length(v)-1
        rem = g0; mi = 0; gi = 0
        # node-1 is most significant (outer)
        for k in 1:nodes
            blk = prod(LGT_d_f * dimsg[j] for j in k+1:nodes; init=1)  # size of remaining
            loc = rem ÷ blk; rem = rem % blk
            nf = loc ÷ dimsg[k]; eg = loc % dimsg[k]
            mi = mi * dimsf[k] + nf
            gi = gi * dimsg[k] + eg
        end
        Ψ[mi+1, gi+1] = v[g0+1]
    end
    return Ψ
end

"""Uhlmann fidelity F(ρ,σ) = (Tr √(√ρ σ √ρ))² for Hermitian PSD ρ,σ."""
function state_fidelity(ρ::Matrix{ComplexF64}, σ::Matrix{ComplexF64})
    Fρ = eigen(Hermitian(ρ)); s = sqrt.(max.(real.(Fρ.values), 0.0))
    sq = Fρ.vectors * Diagonal(s) * Fρ.vectors'
    M = Hermitian(sq * σ * sq)
    return (sum(sqrt.(max.(real.(eigen(M).values), 0.0))))^2
end

"""Total fidelity + matter/gauge reduced-state fidelities between φ₁ and φ₂."""
function sector_fidelities(φ1, φ2, nx, ny, dg)
    n1 = φ1 ./ norm(φ1); n2 = φ2 ./ norm(φ2)
    F_tot = abs2(dot(n1, n2))
    Ψ1 = matter_gauge_matrix(n1, nx, ny, dg); Ψ2 = matter_gauge_matrix(n2, nx, ny, dg)
    ρm1 = Ψ1 * Ψ1'; ρm2 = Ψ2 * Ψ2'                  # trace out gauge
    ρg1 = transpose(Ψ1) * conj(Ψ1); ρg2 = transpose(Ψ2) * conj(Ψ2)   # trace out matter
    return F_tot, state_fidelity(ρm1, ρm2), state_fidelity(Matrix(ρg1), Matrix(ρg2))
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Driver: quench, then commutativity fidelities vs t                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function run_commute_test(; nx=2, ny=2, dg=1, g=1.0, t_hop=1.0, m=0.5,
                          dt=0.1, n_steps=20, Dmax_evo=24, Dmax_dec=8)
    println("─── decoupling-vs-contraction commutativity test ($(nx)×$(ny)) ───")
    dims = node_dims(nx, ny, dg)
    @printf("  full Hilbert dim = %d\n", prod(dims))

    # decoupling unitary 𝒰 = exp(−iÔ), built once
    O, a = build_decoupling_O(nx, ny, dg)
    @printf("  Ô MPO bond = %d   spectral bound a = %.3f\n", maximum(size(W,2) for W in O), a)
    U = expmi_mpo(O, a; Dmax=32)
    Udense = mpo_to_dense(U)
    @printf("  𝒰 MPO bond = %d   (dense %d×%d)\n", maximum(size(W,2) for W in U), size(Udense)...)

    # quench setup
    H = build_H_mpo(nx, ny, dg; g=g, t=t_hop, m=m)
    ψ = staggered_mps(nx, ny, dg); ψ[1] ./= mps_norm(ψ)

    ts = Float64[]; Ftot = Float64[]; Fmat = Float64[]; Fgau = Float64[]
    println("    t      F_total    F_matter   F_gauge   (Dmax_dec=$Dmax_dec)")
    for step in 0:n_steps
        if step > 0
            ψ = krylov_evolve_step(H, ψ, dt; m=8, Dmax=Dmax_evo)
        end
        v  = mps_to_dense(ψ)
        φ1 = Udense * v                              # contract → decouple
        ψ2 = mpo_apply_mps_zipup(U, ψ; Dmax=Dmax_dec)
        φ2 = mps_to_dense(ψ2)                        # decouple (truncated) → contract
        ft, fm, fg = sector_fidelities(φ1, φ2, nx, ny, dg)
        push!(ts, step*dt); push!(Ftot, ft); push!(Fmat, fm); push!(Fgau, fg)
        @printf("  %5.2f   %.6f   %.6f   %.6f\n", step*dt, ft, fm, fg)
        flush(stdout)
    end

    df = DataFrame(t=ts, F_total=Ftot, F_matter=Fmat, F_gauge=Fgau)
    results_dir = joinpath(@__DIR__, "results"); mkpath(results_dir)
    CSV.write(joinpath(results_dir, "decouple_commute.csv"), df)

    p = plot(xlabel="t", ylabel="fidelity",
             title="Decouple∘contract vs contract∘decouple  ($(nx)×$(ny), Dmax_dec=$Dmax_dec)")
    plot!(p, ts, Ftot, label="total", marker=:circle)
    plot!(p, ts, Fmat, label="matter", marker=:square)
    plot!(p, ts, Fgau, label="gauge",  marker=:diamond)
    savefig(p, joinpath(results_dir, "decouple_commute.png"))
    println("  Saved: results/decouple_commute.csv + .png")
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_commute_test()
end
