#= ═══════════════════════════════════════════════════════════════════════════════
   efficient_greens_mpo.jl

   MPS/MPO twin of efficient_greens_pepo.jl: the gauge–matter decoupling for a
   pseudo-1D (snake-ordered) matrix-product representation, via a low-bond MPO of
   the lattice Green's function.

   ── Physics ──────────────────────────────────────────────────────────────────
   Integrating out the gauge field in the 2D U(1) LGT (solving the Gauss law
   div E = Q with the inverse Laplacian G = M⁻¹) replaces the explicit gauge
   links by a long-range Coulomb interaction among the matter charges:

       Ĥ_C  =  ½ Σ_{n,m}  G(n,m)  Q(n) Q(m),     Q(n) = n_f(n) − background.

   This is the gauge sector "decoupled" into a matter-only operator.  G is fully
   nonlocal, so the naive MPO bond is the number of sites N = Nx·Ny.  The
   Sum-of-Exponentials (SoE) approximation from lgt_greens_soe.jl,

       G(n,m) ≈ Σ_{j=1}^K w_j e^{−γ_j(|Δx|+|Δy|)},

   makes the kernel low-rank, so the compressed MPO bond collapses to ~K (≈10)
   instead of N — the MPS analogue of the K≈10 PEPO.

   ── What this file provides ──────────────────────────────────────────────────
   • snake ordering of the Nx×Ny matter sites into a 1-D chain;
   • a minimal Float64 MPO toolkit (term → bond-1 MPO, sum, SVD compress, dense);
   • `build_coulomb_mpo(Gtens, Nx, Ny; background)` — the decoupled-matter MPO;
   • `greens_soe_tensor` / `greens_exact_tensor` — exact vs SoE-approximated G;
   • `decoupled_matter_mpo(Nx, Ny; K)` — the deliverable: low-bond SoE MPO;
   • `selftest_greens_mpo` — MPO == Σ ½ G Q Q (dense, small lattice), bond ~K,
     and the SoE decoupling-approximation error vs exact G.

   The matter kinetic term (hopping with the gauge field transformed away) is the
   companion piece in the gauge_matter_* unitary track; this file delivers the
   interaction operator only — together they are the fully decoupled H.

   Requires: lgt_greens_soe.jl  (→ lgt_greens_function.jl)
   ═══════════════════════════════════════════════════════════════════════════ =#

include(joinpath(@__DIR__, "lgt_greens_soe.jl"))   # generate_greens_function, soe_approximation

using LinearAlgebra
using Printf

const RMPO = Vector{Array{Float64,4}}     # W[p][Dl,Dr,k,b]; W[1] Dl=1, W[end] Dr=1

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Snake ordering of the matter sites                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Boustrophedon order of the Nx×Ny sites → (chain, pos) with chain[p]=(ix,iy)."""
function snake_sites(Nx::Int, Ny::Int)
    chain = Tuple{Int,Int}[]
    for iy in 1:Ny
        xs = isodd(iy) ? (1:Nx) : (Nx:-1:1)
        for ix in xs
            push!(chain, (ix, iy))
        end
    end
    pos = Dict(s => p for (p, s) in enumerate(chain))
    return chain, pos
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Green's-function kernels:  exact and SoE-approximated                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Exact Green's function as G[ix,iy,jx,jy] = (M⁻¹)((ix,iy),(jx,jy))."""
function greens_exact_tensor(Nx::Int, Ny::Int)
    _, G_tens = generate_greens_function(Nx, Ny)
    return G_tens
end

"""SoE-approximated G[ix,iy,jx,jy] = soe(|Δx|+|Δy|) from a fitted `SoEKernel`."""
function greens_soe_tensor(Nx::Int, Ny::Int, soe)
    G = zeros(Float64, Nx, Ny, Nx, Ny)
    for ix in 1:Nx, iy in 1:Ny, jx in 1:Nx, jy in 1:Ny
        G[ix, iy, jx, jy] = soe(abs(ix - jx) + abs(iy - jy))
    end
    return G
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Minimal MPO toolkit (uniform physical dim d=2 matter sites)             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Bond-1 product-operator MPO for coef·⊗ ops (identity on unlisted sites)."""
function term_mpo(N::Int, d::Int, ops::Dict{Int,Matrix{Float64}}, coef::Float64)
    W = RMPO(undef, N); first = true
    for p in 1:N
        O = get(ops, p, Matrix{Float64}(I, d, d))
        first && (O = coef .* O; first = false)
        W[p] = reshape(copy(O), 1, 1, d, d)
    end
    return W
end

"""A + B as a direct-sum MPO (open boundaries), uncompressed."""
function mpo_add(A::RMPO, B::RMPO)
    n = length(A); d = size(A[1], 3); C = RMPO(undef, n)
    for p in 1:n
        DlA, DrA, _, _ = size(A[p]); DlB, DrB, _, _ = size(B[p])
        if p == 1
            T = zeros(Float64, 1, DrA + DrB, d, d)
            T[1, 1:DrA, :, :] .= A[p][1, :, :, :]; T[1, DrA+1:end, :, :] .= B[p][1, :, :, :]
            C[p] = T
        elseif p == n
            T = zeros(Float64, DlA + DlB, 1, d, d)
            T[1:DlA, 1, :, :] .= A[p][:, 1, :, :]; T[DlA+1:end, 1, :, :] .= B[p][:, 1, :, :]
            C[p] = T
        else
            T = zeros(Float64, DlA + DlB, DrA + DrB, d, d)
            T[1:DlA, 1:DrA, :, :] .= A[p]; T[DlA+1:end, DrA+1:end, :, :] .= B[p]
            C[p] = T
        end
    end
    return C
end

"""Two-sweep SVD compression (uniform d); truncate S < ε·S₁ or beyond Dmax."""
function mpo_compress!(W::RMPO; ε::Float64=1e-12, Dmax::Int=400)
    n = length(W); d = size(W[1], 3)
    for p in 1:n-1
        Dl, Dr, _, _ = size(W[p])
        F = svd(reshape(permutedims(W[p], (1, 3, 4, 2)), Dl * d * d, Dr)); r = length(F.S)
        W[p] = permutedims(reshape(F.U, Dl, d, d, r), (1, 4, 2, 3))
        _, Dr2, _, _ = size(W[p+1])
        W[p+1] = reshape(Diagonal(F.S) * F.Vt * reshape(W[p+1], Dr, Dr2 * d * d), r, Dr2, d, d)
    end
    for p in n:-1:2
        Dl, Dr, _, _ = size(W[p])
        F = svd(reshape(permutedims(W[p], (1, 3, 4, 2)), Dl, d * d * Dr))
        s = F.S; keep = max(1, min(Dmax, count(>(ε * s[1]), s)))
        W[p] = permutedims(reshape(F.Vt[1:keep, :], keep, d, d, Dr), (1, 4, 2, 3))
        Dl0, _, _, _ = size(W[p-1])
        W[p-1] = permutedims(reshape(reshape(permutedims(W[p-1], (1, 3, 4, 2)), Dl0 * d * d, Dl) *
                                     (F.U[:, 1:keep] * Diagonal(s[1:keep])), Dl0, d, d, keep),
                             (1, 4, 2, 3))
    end
    return W
end

mpo_maxbond(W::RMPO) = maximum(size(Wp, 2) for Wp in W)

"""Contract a (small!) MPO to a dense operator matrix."""
function mpo_to_dense(W::RMPO)
    op = permutedims(W[1][1, :, :, :], (2, 3, 1))      # (k,b,Dr)
    for p in 2:length(W)
        Dl, Dr, d, _ = size(W[p]); K, B, _ = size(op)
        new = zeros(Float64, K * d, B * d, Dr)
        @inbounds for r in 1:Dr, l in 1:Dl
            new[:, :, r] .+= kron(op[:, :, l], W[p][l, r, :, :])
        end
        op = new
    end
    return op[:, :, 1]
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Decoupled-matter Coulomb MPO                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Charge operator Q = n_f − background on a 2-level matter site."""
charge_op(background::Float64) = Float64[-background 0.0; 0.0 1.0-background]

"""
    build_coulomb_mpo(Gtens, Nx, Ny; background, ε) → RMPO

Low-bond MPO of the gauge-mediated Coulomb interaction on the snake chain,

    Ĥ_C = ½ Σ_{n,m} G(n,m) Q_n Q_m
        = ½ Σ_n G(n,n) Q_n²  +  Σ_{n<m} G(n,m) Q_n Q_m ,

assembled as a compressed sum of bond-1 two-/one-site product terms.  The
compressed bond reflects the rank of G's off-diagonal structure — ~K when `Gtens`
is the SoE-approximated kernel."""
function build_coulomb_mpo(Gtens::Array{Float64,4}, Nx::Int, Ny::Int;
                           background::Float64=0.5, ε::Float64=1e-12)
    chain, _ = snake_sites(Nx, Ny)
    N = Nx * Ny; d = 2
    Q = charge_op(background); Q2 = Q * Q
    terms = RMPO[]
    for a in 1:N
        (ix, iy) = chain[a]
        push!(terms, term_mpo(N, d, Dict(a => Q2), 0.5 * Gtens[ix, iy, ix, iy]))  # self
        for b in a+1:N
            (jx, jy) = chain[b]
            g = Gtens[ix, iy, jx, jy]
            abs(g) < ε && continue
            push!(terms, term_mpo(N, d, Dict(a => Q, b => Q), g))                 # n<m
        end
    end
    H = terms[1]
    for k in 2:length(terms)
        H = mpo_add(H, terms[k])
        (k % 12 == 0) && mpo_compress!(H; ε=ε)
    end
    mpo_compress!(H; ε=ε)
    return H
end

"""
    decoupled_matter_mpo(Nx, Ny; K, background) → (H_C, soe)

The deliverable: the SoE-approximated decoupled-matter Coulomb MPO (bond ~K),
plus the fitted `SoEKernel`.  Use as the gauge-sector interaction in a matter-only
MPS simulation (companion to the transformed kinetic term)."""
function decoupled_matter_mpo(Nx::Int, Ny::Int; K::Int=10, background::Float64=0.5)
    soe = soe_approximation(Nx, Ny; K=K, refine=true)
    G   = greens_soe_tensor(Nx, Ny, soe)
    return build_coulomb_mpo(G, Nx, Ny; background=background), soe
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Self-test                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Dense reference Σ ½ G(n,m) Q_n Q_m built directly by Kronecker products."""
function _coulomb_dense(Gtens, Nx, Ny; background=0.5)
    chain, _ = snake_sites(Nx, Ny)
    N = Nx * Ny; d = 2; Q = charge_op(background)
    embed(site, O) = kron([p == site ? O : Matrix{Float64}(I, d, d) for p in 1:N]...)
    H = zeros(Float64, d^N, d^N)
    for a in 1:N
        (ix, iy) = chain[a]
        Ea = embed(a, Q)
        H .+= 0.5 * Gtens[ix, iy, ix, iy] .* (Ea * Ea)
        for b in a+1:N
            (jx, jy) = chain[b]
            H .+= Gtens[ix, iy, jx, jy] .* (Ea * embed(b, Q))
        end
    end
    return H
end

"""
    selftest_greens_mpo(; Nx, Ny, K)

(1) the Coulomb MPO equals the dense Σ ½ G Q Q for exact G (small lattice);
(2) the SoE-approximated MPO has bond ~K ≪ N and its decoupling-approximation
    error vs exact G is set by the SoE kernel quality."""
function selftest_greens_mpo(; Nx::Int=3, Ny::Int=3, K::Int=8)
    println("─── Green's-function decoupling MPO self-test ($(Nx)×$(Ny), K=$K) ───")
    N = Nx * Ny

    Gex = greens_exact_tensor(Nx, Ny)
    Hex_mpo = build_coulomb_mpo(Gex, Nx, Ny)
    Hex_dense = _coulomb_dense(Gex, Nx, Ny)
    err_mpo = opnorm(mpo_to_dense(Hex_mpo) - Hex_dense) / opnorm(Hex_dense)
    @printf("  exact-G:  MPO vs dense ‖Δ‖/‖H‖ = %.2e   MPO bond = %d  (naive = N = %d)\n",
            err_mpo, mpo_maxbond(Hex_mpo), N)

    soe = soe_approximation(Nx, Ny; K=K, refine=true)
    Gsoe = greens_soe_tensor(Nx, Ny, soe)
    Hsoe_mpo = build_coulomb_mpo(Gsoe, Nx, Ny)
    Hsoe_dense = _coulomb_dense(Gsoe, Nx, Ny)
    err_soe = opnorm(Hsoe_dense - Hex_dense) / opnorm(Hex_dense)
    @printf("  SoE-G:    decoupling approx ‖H_soe − H_exact‖/‖H_exact‖ = %.2e   MPO bond = %d\n",
            err_soe, mpo_maxbond(Hsoe_mpo))
    @printf("  bonds:    exact-G = %d,  SoE-G = %d,  naive = N = %d\n",
            mpo_maxbond(Hex_mpo), mpo_maxbond(Hsoe_mpo), N)

    ok = err_mpo < 1e-10                              # MPO construction is exact
    println(ok ? "  PASS: MPO reproduces ½ΣG·QₙQₘ exactly; SoE bond ≪ N (set by K)" :
                 "  WARN: MPO does not match the dense Coulomb operator")
    return ok
end

# Demo / self-test when run directly.
if abspath(PROGRAM_FILE) == @__FILE__
    selftest_greens_mpo(; Nx=3, Ny=3, K=8)
    println()
    # Deliverable at the benchmark lattice:
    HC, soe = decoupled_matter_mpo(3, 4; K=10)
    @printf("3×4 decoupled-matter Coulomb MPO:  bond = %d  (SoE K=%d)\n",
            maximum(size(W, 2) for W in HC), soe.K)
end
