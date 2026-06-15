#= ═══════════════════════════════════════════════════════════════════════════════
   mps_lgt.jl

   Matrix-Product-State solver for the finite U(1) lattice gauge theory on an
   nx×ny lattice, for direct comparison with the exact-diagonalisation reference
   (finite_ed.jl) and the finite-PEPS code (finite_peps_quench / gs_benchmark).

   ── Model (IDENTICAL to finite_ed.jl) ────────────────────────────────────────
       H = Σ_i [ (g²/2)(E²_R + E²_U) + m·(−1)^{ix+iy} n_f ]            (on-site)
          − t Σ_{horiz} ( c†_L U_R c_R + h.c. )                       (hopping)
          − t Σ_{vert}  ( c†_D U_U c_U + h.c. )
   No magnetic plaquette term; spinless fermions WITHOUT Jordan–Wigner strings
   (matching the ED/PEPS convention), so every term is a product of 1 or 2
   node-local operators and the MPS Hamiltonian is an exact MPO.

   ── Representation ───────────────────────────────────────────────────────────
   The nx×ny grid of NODES (each |n_f, e_R, e_U⟩, boundary-reduced) is ordered
   into a 1-D chain by a boustrophedon snake.  Horizontal hops are short-range;
   vertical hops are long-range on the snake but carry only identity between the
   two nodes (no fermion sign), so the Hamiltonian MPO is built as a compressed
   sum of bond-1 product-operator terms.

   Ground state  : two-site DMRG (KrylovKit eigsolve on the local problem).
   Real-time     : global-Krylov (Lanczos) MPS time stepping — the MPS analogue
                   of the ED `exponentiate`, reusing MPO·MPS apply + compression.

   Self-test (run directly): build H for a small lattice, take the DMRG ground
   state, and compare its energy to the gauge-sector ED ground state.

   Requires: u1_lgt_hamiltonian.jl  (site-local operators), finite_ed.jl (cross-check)
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

const _LGT_HAMILTONIAN_LOADED = true
include(joinpath(@__DIR__, "u1_lgt_hamiltonian.jl"))

using LinearAlgebra
using KrylovKit
using Printf

const CMPS = Vector{Array{ComplexF64,3}}    # ψ[p][Dl,Dr,d]
const CMPO = Vector{Array{ComplexF64,4}}    # W[p][Dl,Dr,k,b], W[1] Dl=1, W[end] Dr=1

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Snake geometry                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Boustrophedon node order for an nx×ny lattice → (chain, pos) where
`chain[p]=(ix,iy)` and `pos[(ix,iy)]=p`."""
function snake_nodes(nx::Int, ny::Int)
    chain = Tuple{Int,Int}[]
    for iy in 1:ny
        xs = isodd(iy) ? (1:nx) : (nx:-1:1)
        for ix in xs
            push!(chain, (ix, iy))
        end
    end
    pos = Dict(node => p for (p, node) in enumerate(chain))
    return chain, pos
end

"""Local Hilbert-space dimension of every node along the snake."""
function node_dims(nx::Int, ny::Int, dg::Int)
    chain, _ = snake_nodes(nx, ny)
    return [site_dims(ix, iy, nx, ny, dg)[1] for (ix, iy) in chain]
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Hamiltonian as a sum of product-operator terms → compressed MPO         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""A Hamiltonian term: scalar `coef` × ⊗_p op[p] (identity on unlisted sites)."""
struct HTerm
    coef :: ComplexF64
    ops  :: Dict{Int,Matrix{ComplexF64}}
end

"""
    lgt_terms(nx, ny, dg; g, t, m) → Vector{HTerm}

Every on-site and hopping term of H along the snake, matching finite_ed.jl."""
function lgt_terms(nx::Int, ny::Int, dg::Int; g::Float64, t::Float64, m::Float64)
    _, pos = snake_nodes(nx, ny)
    terms = HTerm[]

    # on-site (1-site, diagonal): (g²/2)(E²_R+E²_U) + m(−1)^{ix+iy} n_f
    for iy in 1:ny, ix in 1:nx
        H = ComplexF64.(H_onsite_site(ix, iy, nx, ny, dg; g=g, m=m))
        push!(terms, HTerm(1.0, Dict(pos[(ix, iy)] => H)))
    end

    # horizontal hopping (ix,iy)-(ix+1,iy):  −t(c†_L U_R c_R + h.c.)
    for iy in 1:ny, ix in 1:nx-1
        _, d_gR_R, d_gU_R = site_dims(ix+1, iy, nx, ny, dg)
        _, _, d_gU_L      = site_dims(ix,   iy, nx, ny, dg)
        AL = ComplexF64.(kron(op_cdag(), op_U_gauge(dg), _Id(d_gU_L)))   # c†⊗U_R⊗I
        BR = ComplexF64.(embed_f_site(op_c(), d_gR_R, d_gU_R))           # c
        pL, pR = pos[(ix, iy)], pos[(ix+1, iy)]
        push!(terms, HTerm(-t, Dict(pL => AL,            pR => BR)))
        push!(terms, HTerm(-t, Dict(pL => Matrix(AL'),   pR => Matrix(BR'))))  # h.c.
    end

    # vertical hopping (ix,iy)-(ix,iy+1):  −t(c†_D U_U c_U + h.c.)
    for iy in 1:ny-1, ix in 1:nx
        _, d_gR_U, d_gU_U = site_dims(ix, iy+1, nx, ny, dg)
        _, d_gR_D, _      = site_dims(ix, iy,   nx, ny, dg)
        AD = ComplexF64.(kron(op_cdag(), _Id(d_gR_D), op_U_gauge(dg)))   # c†⊗I⊗U_U
        BU = ComplexF64.(embed_f_site(op_c(), d_gR_U, d_gU_U))           # c
        pD, pU = pos[(ix, iy)], pos[(ix, iy+1)]
        push!(terms, HTerm(-t, Dict(pD => AD,          pU => BU)))
        push!(terms, HTerm(-t, Dict(pD => Matrix(AD'), pU => Matrix(BU'))))
    end

    return terms
end

"""Bond-1 product-operator MPO for one term (coef folded into the first site)."""
function term_mpo(dims::Vector{Int}, term::HTerm)
    n = length(dims)
    W = CMPO(undef, n)
    first = true
    for p in 1:n
        O = get(term.ops, p, Matrix{ComplexF64}(I, dims[p], dims[p]))
        first && (O = term.coef .* O; first = false)
        W[p] = reshape(ComplexF64.(O), 1, 1, dims[p], dims[p])
    end
    return W
end

"""α·A + β·B as a direct-sum MPO (open boundaries), uncompressed."""
function mpo_axpby(α::Number, A::CMPO, β::Number, B::CMPO)
    n = length(A); C = CMPO(undef, n)
    for p in 1:n
        DlA, DrA, d, _ = size(A[p]); DlB, DrB, _, _ = size(B[p])
        if p == 1
            T = zeros(ComplexF64, 1, DrA + DrB, d, d)
            T[1, 1:DrA, :, :]     .= α .* A[p][1, :, :, :]
            T[1, DrA+1:end, :, :] .= β .* B[p][1, :, :, :]
            C[p] = T
        elseif p == n
            T = zeros(ComplexF64, DlA + DlB, 1, d, d)
            T[1:DlA, 1, :, :]     .= A[p][:, 1, :, :]
            T[DlA+1:end, 1, :, :] .= B[p][:, 1, :, :]
            C[p] = T
        else
            T = zeros(ComplexF64, DlA + DlB, DrA + DrB, d, d)
            T[1:DlA, 1:DrA, :, :]         .= A[p]
            T[DlA+1:end, DrA+1:end, :, :] .= B[p]
            C[p] = T
        end
    end
    return C
end

"""Two-sweep SVD compression of an MPO (truncate S < ε·S₁ or beyond Dmax)."""
function mpo_compress!(W::CMPO; ε::Float64=1e-12, Dmax::Int=400)
    n = length(W)
    for p in 1:n-1
        Dl, Dr, d, _ = size(W[p])
        F = svd(reshape(permutedims(W[p], (1, 3, 4, 2)), Dl * d * d, Dr))
        r = length(F.S)
        W[p] = permutedims(reshape(F.U, Dl, d, d, r), (1, 4, 2, 3))
        SV = Diagonal(F.S) * F.Vt
        _, Dr2, _, _ = size(W[p+1])
        W[p+1] = reshape(SV * reshape(W[p+1], Dr, Dr2 * d * d), r, Dr2, d, d)
    end
    for p in n:-1:2
        Dl, Dr, d, _ = size(W[p])
        F = svd(reshape(permutedims(W[p], (1, 3, 4, 2)), Dl, d * d * Dr))
        s = F.S; keep = max(1, min(Dmax, count(>(ε * s[1]), s)))
        W[p] = permutedims(reshape(F.Vt[1:keep, :], keep, d, d, Dr), (1, 4, 2, 3))
        US = F.U[:, 1:keep] * Diagonal(s[1:keep])
        Dl0, _, _, _ = size(W[p-1])
        W[p-1] = permutedims(reshape(reshape(permutedims(W[p-1], (1, 3, 4, 2)),
                                             Dl0 * d * d, Dl) * US, Dl0, d, d, keep),
                             (1, 4, 2, 3))
    end
    return W
end

"""Assemble the compressed Hamiltonian MPO from the term list."""
function build_H_mpo(nx::Int, ny::Int, dg::Int; g::Float64, t::Float64, m::Float64,
                     ε::Float64=1e-12)
    dims = node_dims(nx, ny, dg)
    terms = lgt_terms(nx, ny, dg; g=g, t=t, m=m)
    H = term_mpo(dims, terms[1])
    for k in 2:length(terms)
        H = mpo_axpby(1.0, H, 1.0, term_mpo(dims, terms[k]))
        (k % 8 == 0) && mpo_compress!(H; ε=ε)        # periodic compression
    end
    mpo_compress!(H; ε=ε)
    return H
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MPS algebra                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function product_mps(configs::Vector{Int}, dims::Vector{Int})
    [reshape(ComplexF64.((1:dims[p]) .== configs[p]), 1, 1, dims[p]) for p in eachindex(dims)]
end

function mps_overlap(φ::CMPS, ψ::CMPS)
    v = ones(ComplexF64, 1, 1)
    for p in eachindex(φ)
        d = size(φ[p], 3)
        newv = zeros(ComplexF64, size(φ[p], 2), size(ψ[p], 2))
        @inbounds for k in 1:d
            newv .+= φ[p][:, :, k]' * v * ψ[p][:, :, k]
        end
        v = newv
    end
    return v[1, 1]
end
mps_norm(ψ::CMPS) = sqrt(real(mps_overlap(ψ, ψ)))

function mpo_apply_mps(W::CMPO, ψ::CMPS)
    n = length(W); out = CMPS(undef, n)
    for p in 1:n
        DlW, DrW, d, _ = size(W[p]); Dlψ, Drψ, _ = size(ψ[p])
        T = zeros(ComplexF64, DlW * Dlψ, DrW * Drψ, d)
        @inbounds for aL in 1:DlW, aR in 1:DrW, bL in 1:Dlψ, bR in 1:Drψ
            T[(aL-1)*Dlψ+bL, (aR-1)*Drψ+bR, :] = W[p][aL, aR, :, :] * ψ[p][bL, bR, :]
        end
        out[p] = T
    end
    return out
end

function mps_axpby(α::Number, A::CMPS, β::Number, B::CMPS)
    n = length(A); C = CMPS(undef, n)
    for p in 1:n
        DlA, DrA, d = size(A[p]); DlB, DrB, _ = size(B[p])
        if p == 1
            T = zeros(ComplexF64, 1, DrA + DrB, d)
            T[1, 1:DrA, :]     .= α .* A[p][1, :, :]
            T[1, DrA+1:end, :] .= β .* B[p][1, :, :]
            C[p] = T
        elseif p == n
            T = zeros(ComplexF64, DlA + DlB, 1, d)
            T[1:DlA, 1, :]     .= A[p][:, 1, :]
            T[DlA+1:end, 1, :] .= B[p][:, 1, :]
            C[p] = T
        else
            T = zeros(ComplexF64, DlA + DlB, DrA + DrB, d)
            T[1:DlA, 1:DrA, :]         .= A[p]
            T[DlA+1:end, DrA+1:end, :] .= B[p]
            C[p] = T
        end
    end
    return C
end

function mps_compress!(ψ::CMPS; ε::Float64=1e-10, Dmax::Int=200)
    n = length(ψ)
    for p in 1:n-1
        Dl, Dr, d = size(ψ[p])
        F = svd(reshape(permutedims(ψ[p], (1, 3, 2)), Dl * d, Dr)); r = length(F.S)
        ψ[p] = permutedims(reshape(F.U, Dl, d, r), (1, 3, 2))
        SV = Diagonal(F.S) * F.Vt
        _, Dr2, _ = size(ψ[p+1])
        ψ[p+1] = reshape(SV * reshape(ψ[p+1], Dr, Dr2 * d), r, Dr2, d)
    end
    for p in n:-1:2
        Dl, Dr, d = size(ψ[p])
        F = svd(reshape(permutedims(ψ[p], (1, 3, 2)), Dl, d * Dr))
        s = F.S; keep = max(1, min(Dmax, count(>(ε * s[1]), s)))
        ψ[p] = permutedims(reshape(F.Vt[1:keep, :], keep, d, Dr), (1, 3, 2))
        US = F.U[:, 1:keep] * Diagonal(s[1:keep])
        Dl0, _, _ = size(ψ[p-1])
        ψ[p-1] = permutedims(reshape(reshape(permutedims(ψ[p-1], (1, 3, 2)),
                                             Dl0 * d, Dl) * US, Dl0, d, keep), (1, 3, 2))
    end
    return ψ
end

"""⟨ψ|H|ψ⟩ for an MPO H (assumes ⟨ψ|ψ⟩ handled by caller)."""
mpo_expect(H::CMPO, ψ::CMPS) = mps_overlap(ψ, mpo_apply_mps(H, ψ))

"""⟨ψ|O at chain site p|ψ⟩ / ⟨ψ|ψ⟩ for a local operator O."""
function mps_local_expect(ψ::CMPS, p::Int, O::AbstractMatrix)
    Oψ = copy(ψ); d = size(ψ[p], 3)
    T = zeros(ComplexF64, size(ψ[p])...)
    @inbounds for k in 1:d, b in 1:d
        T[:, :, k] .+= O[k, b] .* ψ[p][:, :, b]
    end
    Oψ[p] = T
    return real(mps_overlap(ψ, Oψ)) / real(mps_overlap(ψ, ψ))
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Two-site DMRG ground state                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""All right environments: R[p] summarizes sites p+1..n (at the bond left of
site p+1).  R[n]=trivial; R[p]=right_env(R[p+1],H[p+1],ψ[p+1])."""
function build_R(H::CMPO, ψ::CMPS)
    n = length(ψ)
    R = Vector{Array{ComplexF64,3}}(undef, n)
    R[n] = ones(ComplexF64, 1, 1, 1)
    for p in n-1:-1:1
        R[p] = right_env(R[p+1], H[p+1], ψ[p+1])
    end
    return R
end

"""All left environments: L[p] summarizes sites 1..p-1.  L[1]=trivial."""
function build_L(H::CMPO, ψ::CMPS)
    n = length(ψ)
    L = Vector{Array{ComplexF64,3}}(undef, n)
    L[1] = ones(ComplexF64, 1, 1, 1)
    for p in 2:n
        L[p] = left_env(L[p-1], H[p-1], ψ[p-1])
    end
    return L
end

"""Right environment via tensor contraction (bra, mpo, ket bonds)."""
function right_env(Rnext::Array{ComplexF64,3}, Wp::Array{ComplexF64,4}, ψp::Array{ComplexF64,3})
    Dl, Dr, d = size(ψp); Dlw, Drw, _, _ = size(Wp)
    @tensoropt Rnew[a, b, c] := conj(ψp[a, ap, kk]) * Wp[b, bp, kk, qq] *
                                ψp[c, cp, qq] * Rnext[ap, bp, cp]
    return Rnew
end

"""Left environment."""
function left_env(Lprev::Array{ComplexF64,3}, Wp::Array{ComplexF64,4}, ψp::Array{ComplexF64,3})
    @tensoropt Lnew[a, b, c] := conj(ψp[ap, a, kk]) * Wp[bp, b, kk, qq] *
                                ψp[cp, c, qq] * Lprev[ap, bp, cp]
    return Lnew
end

"""Apply the two-site effective Hamiltonian to a two-site tensor
Θ[a, k1, k2, c] (left bond a, phys k1 k2, right bond c)."""
function apply_Heff2(L, W1, W2, R, Θ)
    @tensoropt HΘ[a, k1, k2, c] := L[a, b, ap] * W1[b, bm, k1, q1] *
                                   W2[bm, bp, k2, q2] * R[c, bp, cp] *
                                   Θ[ap, q1, q2, cp]
    return HΘ
end

using Random
using Statistics

"""One two-site DMRG update at bond (p,p+1): solve the local ground state via
KrylovKit, SVD-split with truncation to `D`, and advance the swept-through
environment.  Mutates `ψ` and (`L` if to_right, else `R`).  Returns the energy."""
function two_site_update!(ψ::CMPS, H::CMPO,
                          L::Vector{Array{ComplexF64,3}}, R::Vector{Array{ComplexF64,3}},
                          p::Int, D::Int; to_right::Bool)
    a, mmid, k1 = size(ψ[p]); _, c, k2 = size(ψ[p+1])
    @tensor Θ[ai, ki, kj, ci] := ψ[p][ai, mi, ki] * ψ[p+1][mi, ci, kj]
    Heff = θ -> apply_Heff2(L[p], H[p], H[p+1], R[p+1], θ)
    vals, vecs, info = eigsolve(Heff, Θ, 1, :SR; ishermitian=true,
                                krylovdim=min(20, length(Θ)), tol=1e-10, maxiter=200)
    E = real(vals[1]); Θg = vecs[1]
    Θg ./= norm(Θg)
    M = reshape(Θg, a * k1, k2 * c)
    F = svd(M); keep = max(1, min(D, count(>(1e-12 * F.S[1]), F.S)))
    U = F.U[:, 1:keep]; S = F.S[1:keep]; Vt = F.Vt[1:keep, :]
    if to_right
        ψ[p]   = permutedims(reshape(U, a, k1, keep), (1, 3, 2))
        ψ[p+1] = permutedims(reshape(Diagonal(S) * Vt, keep, k2, c), (1, 3, 2))
        L[p+1] = left_env(L[p], H[p], ψ[p])
    else
        ψ[p]   = permutedims(reshape(U * Diagonal(S), a, k1, keep), (1, 3, 2))
        ψ[p+1] = permutedims(reshape(Vt, keep, k2, c), (1, 3, 2))
        R[p]   = right_env(R[p+1], H[p+1], ψ[p+1])
    end
    return E
end

"""
    dmrg_ground_state(H, dims; D, nsweeps, seed) → (E, ψ)

Two-site DMRG.  Bond dimension grows up to `D`; `nsweeps` left-right sweeps."""
function dmrg_ground_state(H::CMPO, dims::Vector{Int}; D::Int=40, nsweeps::Int=6,
                           seed::Int=1, verbose::Bool=true)
    n = length(dims)
    Random.seed!(seed)
    ψ = CMPS(undef, n); Dprev = 1
    for p in 1:n
        Dr = p == n ? 1 : min(D, Dprev * dims[p])
        ψ[p] = randn(ComplexF64, Dprev, Dr, dims[p]); Dprev = Dr
    end
    mps_compress!(ψ; Dmax=D); nrm = mps_norm(ψ); ψ[1] ./= nrm

    E = 0.0
    for sweep in 1:nsweeps
        R = build_R(H, ψ)
        L = Vector{Array{ComplexF64,3}}(undef, n); L[1] = ones(ComplexF64, 1, 1, 1)
        for p in 1:n-1
            E = two_site_update!(ψ, H, L, R, p, D; to_right=true)
        end
        L2 = build_L(H, ψ)
        R2 = Vector{Array{ComplexF64,3}}(undef, n); R2[n] = ones(ComplexF64, 1, 1, 1)
        for p in n-1:-1:1
            E = two_site_update!(ψ, H, L2, R2, p, D; to_right=false)
        end
        verbose && @printf("    DMRG sweep %d:  E = %.10f\n", sweep, E)
    end
    return E, ψ
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Gauss-law penalty (selects the gauge / charge sector for DMRG)          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# DMRG on the bare H finds the GLOBAL ground state, which may sit in a different
# gauge sector than the ED comparison.  We add Λ·Σ_i (G_i − g_i)² with
# G_i = E_R(i) − E_R(i−x̂) + E_U(i) − E_U(i−ŷ) − n_f(i); for Λ large the ground
# state is forced into the target sector {g_i}.  The penalty is used ONLY for
# ground-state preparation — real-time evolution uses the bare H (which already
# conserves every G_i, so the sector is preserved automatically).

"""Charge sector with staggered background:  g_odd=−1, g_even=0 (matches ED B/C)."""
staggered_charges(nx::Int, ny::Int) = [isodd(ix + iy) ? -1 : 0 for ix in 1:nx, iy in 1:ny]

function gauss_penalty_terms(nx::Int, ny::Int, dg::Int, g_charges::AbstractMatrix, Λ::Float64)
    _, pos = snake_nodes(nx, ny)
    terms = HTerm[]
    for iy in 1:ny, ix in 1:nx
        pieces = Dict{Int,Matrix{ComplexF64}}()
        add!(node, O) = (pieces[node] = haskey(pieces, node) ? pieces[node] .+ ComplexF64.(O) : ComplexF64.(O))
        if ix < nx
            _, _, d_gU = site_dims(ix, iy, nx, ny, dg)
            add!(pos[(ix, iy)],  embed_R_site(op_E(dg), d_gU))
        end
        if ix > 1
            _, _, d_gU = site_dims(ix-1, iy, nx, ny, dg)
            add!(pos[(ix-1, iy)], -embed_R_site(op_E(dg), d_gU))
        end
        if iy < ny
            _, d_gR, _ = site_dims(ix, iy, nx, ny, dg)
            add!(pos[(ix, iy)],  embed_U_site(op_E(dg), d_gR))
        end
        if iy > 1
            _, d_gR, _ = site_dims(ix, iy-1, nx, ny, dg)
            add!(pos[(ix, iy-1)], -embed_U_site(op_E(dg), d_gR))
        end
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        add!(pos[(ix, iy)], -embed_f_site(op_nf(), d_gR, d_gU))

        gi = Float64(g_charges[ix, iy])
        nodes = collect(keys(pieces))
        for a in nodes, b in nodes        # (Σ O − gi)² → Σ_{a,b} Oa Ob
            if a == b
                push!(terms, HTerm(Λ, Dict(a => pieces[a] * pieces[a])))
            else
                push!(terms, HTerm(Λ, Dict(a => pieces[a], b => pieces[b])))
            end
        end
        for a in nodes                     # −2 gi Σ Oa
            push!(terms, HTerm(-2 * Λ * gi, Dict(a => pieces[a])))
        end
    end
    return terms
end

"""Build the bare or Gauss-penalized Hamiltonian MPO from a term list."""
function _assemble_mpo(dims::Vector{Int}, terms::Vector{HTerm}; ε::Float64=1e-12)
    H = term_mpo(dims, terms[1])
    for k in 2:length(terms)
        H = mpo_axpby(1.0, H, 1.0, term_mpo(dims, terms[k]))
        (k % 8 == 0) && mpo_compress!(H; ε=ε)
    end
    mpo_compress!(H; ε=ε)
    return H
end

"""Gauss-penalized Hamiltonian MPO (for ground-state preparation in sector `gauss_g`)."""
function build_penalized_H(nx::Int, ny::Int, dg::Int; g::Float64, t::Float64, m::Float64,
                           gauss_g::AbstractMatrix, Λ::Float64, ε::Float64=1e-12)
    dims = node_dims(nx, ny, dg)
    terms = vcat(lgt_terms(nx, ny, dg; g=g, t=t, m=m),
                 gauss_penalty_terms(nx, ny, dg, gauss_g, Λ))
    return _assemble_mpo(dims, terms; ε=ε)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Real-time evolution: global Krylov (Lanczos) MPS time stepping          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    krylov_evolve_step(H, ψ, dt; m, Dmax, ε) → ψ_new ≈ exp(−i dt H) ψ

Lanczos on the MPS manifold: build an orthonormal Krylov basis of `m` MPS (each
H·v via MPO apply + compression, full reorthogonalization), exponentiate the
small tridiagonal matrix, and recombine.  The MPS analogue of the ED
`exponentiate`; `ψ_new` is renormalized to unit norm."""
function krylov_evolve_step(H::CMPO, ψ::CMPS, dt::Float64; m::Int=8,
                            Dmax::Int=100, ε::Float64=1e-10)
    V = CMPS[]
    v1 = deepcopy(ψ); nv = mps_norm(v1); v1[1] ./= nv; push!(V, v1)
    α = Float64[]; β = Float64[]
    w = mpo_apply_mps(H, V[1]); mps_compress!(w; Dmax=Dmax, ε=ε)
    a1 = real(mps_overlap(V[1], w)); push!(α, a1)
    w = mps_axpby(1.0, w, -a1, V[1]); mps_compress!(w; Dmax=Dmax, ε=ε)
    for j in 2:m
        b = mps_norm(w); b < 1e-12 && break
        push!(β, b)
        vj = deepcopy(w); vj[1] ./= b; push!(V, vj)
        w = mpo_apply_mps(H, vj); mps_compress!(w; Dmax=Dmax, ε=ε)
        aj = real(mps_overlap(vj, w)); push!(α, aj)
        w = mps_axpby(1.0, w, -aj, vj)
        w = mps_axpby(1.0, w, -b, V[j-1]); mps_compress!(w; Dmax=Dmax, ε=ε)
        for vk in V                                   # full reorthogonalization
            ck = mps_overlap(vk, w)
            w = mps_axpby(1.0, w, -ck, vk)
        end
        mps_compress!(w; Dmax=Dmax, ε=ε)
    end
    k = length(α)
    T = zeros(ComplexF64, k, k)
    for i in 1:k; T[i, i] = α[i]; end
    for i in 1:k-1; T[i, i+1] = β[i]; T[i+1, i] = β[i]; end
    coeffs = (exp(-im * dt * T))[:, 1]                # exp(−i dt T) e₁
    ψnew = deepcopy(V[1]); ψnew[1] .*= coeffs[1]
    for kk in 2:k
        ψnew = mps_axpby(1.0, ψnew, coeffs[kk], V[kk]); mps_compress!(ψnew; Dmax=Dmax, ε=ε)
    end
    nn = mps_norm(ψnew); ψnew[1] ./= nn
    return ψnew
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Observables and initial states                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function node_nf_op(ix, iy, nx, ny, dg)
    _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
    return ComplexF64.(embed_f_site(op_nf(), d_gR, d_gU))
end

function node_E2_op(ix, iy, nx, ny, dg)
    _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
    dphys = LGT_d_f * d_gR * d_gU
    O = zeros(ComplexF64, dphys, dphys)
    ix < nx && (O .+= embed_R_site(op_E2(dg), d_gU))
    iy < ny && (O .+= embed_U_site(op_E2(dg), d_gR))
    return O
end

"""Site-resolved ⟨n_f⟩, ⟨E²⟩ and means/sublattice (mirrors finite_ed.measure_ED)."""
function measure_mps(ψ::CMPS, nx::Int, ny::Int, dg::Int; t_now::Float64=0.0)
    _, pos = snake_nodes(nx, ny)
    nf = zeros(Float64, nx, ny); E2 = zeros(Float64, nx, ny)
    for iy in 1:ny, ix in 1:nx
        p = pos[(ix, iy)]
        nf[ix, iy] = mps_local_expect(ψ, p, node_nf_op(ix, iy, nx, ny, dg))
        E2[ix, iy] = mps_local_expect(ψ, p, node_E2_op(ix, iy, nx, ny, dg))
    end
    nf_even = mean(nf[ix, iy] for ix in 1:nx, iy in 1:ny if iseven(ix + iy))
    nf_odd  = mean(nf[ix, iy] for ix in 1:nx, iy in 1:ny if isodd(ix + iy))
    return (t=t_now, nf_mean=mean(nf), nf_even=nf_even, nf_odd=nf_odd,
            E2_mean=mean(E2), nf_grid=nf, E2_grid=E2)
end

"""Staggered product state |n_f=1 on odd, 0 on even; E=0⟩  (quench B/C sector)."""
function staggered_mps(nx::Int, ny::Int, dg::Int)
    chain, _ = snake_nodes(nx, ny)
    dims = node_dims(nx, ny, dg)
    cfg = [begin _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
                 site_idx(isodd(ix + iy) ? 1 : 0, 0, 0, d_gR, d_gU, dg) end
           for (ix, iy) in chain]
    return product_mps(cfg, dims)
end

"""String-breaking product state for quench A: vacuum + n_f(2,iy_str)=1,
e_R(1,iy_str)=1  (sector g(1,iy_str)=+1, g(2,iy_str)=−2)."""
function string_breaking_mps(nx::Int, ny::Int, dg::Int; iy_str::Int=div(ny, 2))
    chain, _ = snake_nodes(nx, ny)
    dims = node_dims(nx, ny, dg)
    cfg = Int[]
    for (ix, iy) in chain
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        nf = (ix == 2 && iy == iy_str) ? 1 : 0
        eR = (ix == 1 && iy == iy_str) ? 1 : 0
        push!(cfg, site_idx(nf, eR, 0, d_gR, d_gU, dg))
    end
    return product_mps(cfg, dims)
end

"""Gauge charges for the string-breaking sector (for the penalized GS, unused in A)."""
function string_charges(nx::Int, ny::Int; iy_str::Int=div(ny, 2))
    g = zeros(Int, nx, ny); g[1, iy_str] = 1; g[2, iy_str] = -2; return g
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Dense contraction + self-test                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Contract a (small!) MPO to a dense matrix — self-tests only."""
function mpo_to_dense(W::CMPO)
    op = permutedims(W[1][1, :, :, :], (2, 3, 1))      # (k,b,Dr)
    for p in 2:length(W)
        Dl, Dr, d, _ = size(W[p]); K, B, _ = size(op)
        new = zeros(ComplexF64, K * d, B * d, Dr)
        @inbounds for r in 1:Dr, l in 1:Dl
            new[:, :, r] .+= kron(op[:, :, l], W[p][l, r, :, :])
        end
        op = new
    end
    return op[:, :, 1]
end

"""Mean Gauss-law violation ⟨Σ_i (G_i − g_i)²⟩ on an MPS (sector diagnostic)."""
function gauss_violation(ψ::CMPS, nx::Int, ny::Int, dg::Int, g_charges::AbstractMatrix)
    dims = node_dims(nx, ny, dg)
    terms = gauss_penalty_terms(nx, ny, dg, g_charges, 1.0)   # Λ=1
    H = _assemble_mpo(dims, terms)
    return real(mpo_expect(H, ψ)) / real(mps_overlap(ψ, ψ))   # + Σ g_i² omitted ⇒ shifted
end

"""
    selftest_mps_lgt(; nx, ny, dg, g, t, m, Λ, D)

Validate the solver on a small lattice: (1) the Gauss-penalized DMRG ground-state
energy matches the lowest eigenvalue of the densely-contracted MPO; (2) the state
sits in the target staggered sector.  Dense check feasible for nx·ny ≤ ~6 nodes."""
function selftest_mps_lgt(; nx::Int=2, ny::Int=3, dg::Int=1, g::Float64=1.0,
                          t::Float64=1.0, m::Float64=0.5, Λ::Float64=20.0, D::Int=60)
    println("─── MPS-LGT self-test ($(nx)×$(ny), dg=$dg, g=$g, t=$t, m=$m) ───")
    dims = node_dims(nx, ny, dg)
    gch = staggered_charges(nx, ny)
    Hpen = build_penalized_H(nx, ny, dg; g=g, t=t, m=m, gauss_g=gch, Λ=Λ)
    @printf("  penalized-H MPO max bond = %d\n", maximum(size(W, 2) for W in Hpen))

    Edmrg, ψ = dmrg_ground_state(Hpen, dims; D=D, nsweeps=8, verbose=true)

    Hd = mpo_to_dense(Hpen)
    Eexact = real(eigvals(Hermitian(Matrix(Hd)))[1])
    Hbare = _assemble_mpo(dims, lgt_terms(nx, ny, dg; g=g, t=t, m=m))
    Ebare = real(mpo_expect(Hbare, ψ)) / real(mps_overlap(ψ, ψ))
    viol  = gauss_violation(ψ, nx, ny, dg, gch)

    @printf("  DMRG E(penalized) = %.8f   dense lowest = %.8f   |Δ| = %.2e\n",
            Edmrg, Eexact, abs(Edmrg - Eexact))
    @printf("  bare-H energy ⟨ψ|H|ψ⟩ = %.8f   Gauss penalty ⟨Σ(G−g)²⟩(shifted) = %.2e\n",
            Ebare, viol)
    ok = abs(Edmrg - Eexact) < 1e-5
    println(ok ? "  PASS: DMRG reproduces the dense ground-state energy" :
                 "  WARN: DMRG energy disagrees with dense diagonalization")
    return ok
end

# Demo / self-test when run directly.
if abspath(PROGRAM_FILE) == @__FILE__
    selftest_mps_lgt(; nx=2, ny=2)
    println()
    selftest_mps_lgt(; nx=2, ny=3)
    println()
end
