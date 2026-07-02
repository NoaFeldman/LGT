#= ═══════════════════════════════════════════════════════════════════════════════
   test_plaquette_entanglement_ed.jl

   Cross-check of `plaquette_entanglement.jl` against full exact diagonalisation
   (ED) with EXACT decoupling.

   ── What is being validated ──────────────────────────────────────────────────
   The MPS pipeline (plaquette_entanglement, which=:exact) computes the dual-
   plaquette half-system entanglement via: MPO exponent Ô (exact ∇G) → Krylov
   state decoupling exp(−iÔ)|ψ⟩ → per-config projection → Gram/eigenvector →
   Schmidt entropy.  The ED reference recomputes the SAME quantity along a fully
   independent path:
     • Ô as a matrix-free operator on the full Hilbert space (no MPO, no
       compression), assembled term-by-term from the same exact shift field ∇G
       and applied by contracting one node index at a time;
     • exp(−iÔ)|ψ⟩ by exact Krylov exponentiation of the DENSE state vector
       (no bond truncation) — i.e. ED, not MPS;
     • the plaquette reduced density matrix by dense partial trace over matter.

   The two must agree on: the source-free weight, the leading purity, the full
   plaquette density matrix ρ (phase-independent), and the half-system entropy S.

   Lattices: 2×3 and 3×2 — the smallest with (nx−1)(ny−1) ≥ 2 plaquettes, so the
   half-cut is nontrivial (Hilbert dim ≈ 1.4×10⁵: a dense 𝒰 or dense Ô is
   infeasible, but a matrix-free exp(−iÔ)|ψ⟩ and a dense state vector are cheap).

   Requires: plaquette_entanglement.jl (→ decoupling_U_soe.jl → mps_lgt.jl …)
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "plaquette_entanglement.jl"))

using LinearAlgebra
using KrylovKit
using Printf
using Random

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Shared initial state: identical MPS and dense vector (column-snake order) ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Random product state on the column snake, returned BOTH as an MPS and as a
dense state vector `⊗_p v_p` (node 1 most significant), built from the same
per-node vectors so the two routes see an identical state."""
function shared_random_state(nx::Int, ny::Int, dg::Int; seed::Int=1)
    Random.seed!(seed)
    dims = col_node_dims(nx, ny, dg)
    vs = [ (v = randn(ComplexF64, d); v ./ norm(v)) for d in dims ]
    ψ_mps = CMPS([reshape(vs[p], 1, 1, dims[p]) for p in eachindex(dims)])
    ψ_vec = vs[1]
    for p in 2:length(vs)
        ψ_vec = kron(ψ_vec, vs[p])
    end
    return ψ_mps, ψ_vec, dims
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ED route: matrix-free exponent Ô, exact exp(−iÔ)|ψ⟩, dense plaquette       ║
# ║  partial trace.  Assembled INDEPENDENTLY of the MPO path (genuine check).   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Apply a node-local operator `O` (d_p × d_p) at chain position `p` to the dense
state vector `ψ = ⊗_q v_q` (node 1 most significant).  Contracts only the node-p
index — no full-space matrix is ever formed."""
function apply_node_op(ψ::Vector{ComplexF64}, p::Int, O::Matrix{ComplexF64},
                       dims::Vector{Int})
    F    = prod(dims[p+1:end])                          # nodes after p (fastest)
    dp   = dims[p]
    slow = prod(dims[1:p-1])                            # nodes before p (slowest)
    T    = reshape(ψ, F, dp, slow)
    out  = similar(ψ)
    Tout = reshape(out, F, dp, slow)
    @inbounds for a in 1:slow, k in 1:dp
        acc = @view Tout[:, k, a]; fill!(acc, 0)
        for q in 1:dp
            Okq = O[k, q]
            Okq == 0 && continue
            acc .+= Okq .* @view T[:, q, a]
        end
    end
    return out
end

"""One decoupling term: `coef · O_a(a)` (single-site if `b==0`) or
`coef · O_a(a) O_b(b)` (two nodes)."""
struct OTerm
    coef :: ComplexF64
    a :: Int; Oa :: Matrix{ComplexF64}
    b :: Int; Ob :: Matrix{ComplexF64}
end

"""Term list for Ô = Σ_{ℓ,n} (∇G)_{ℓ,n} φ_ℓ Q_n on the column snake (mirrors
`build_O`, but as an independent matrix-free operator)."""
function build_O_terms(nx::Int, ny::Int, dg::Int, pos::Dict, Gt; tol::Float64=1e-12)
    φ = φ_op()
    Qloc(jx, jy) = ComplexF64.(embed_f_site(op_nf() - bg_stag(jx, jy) * _Id(LGT_d_f),
                                            site_dims(jx, jy, nx, ny, dg)[2:3]...))
    terms = OTerm[]
    function addlink!(ix, iy, dir)
        _, dgR, dgU = site_dims(ix, iy, nx, ny, dg)
        φloc = dir == :R ? ComplexF64.(embed_R_site(φ, dgU)) :
                           ComplexF64.(embed_U_site(φ, dgR))
        a = pos[(ix, iy)]
        for jy in 1:ny, jx in 1:nx
            M = shift_exact(Gt, ix, iy, dir, jx, jy)
            abs(M) < tol && continue
            b = pos[(jx, jy)]; Q = Qloc(jx, jy)
            push!(terms, a == b ? OTerm(M, a, φloc * Q, 0, φloc) :
                                  OTerm(M, a, copy(φloc), b, Q))
        end
    end
    for iy in 1:ny, ix in 1:nx-1; addlink!(ix, iy, :R); end
    for iy in 1:ny-1, ix in 1:nx; addlink!(ix, iy, :U); end
    return terms
end

"""Matrix-free action Ô·ψ from the term list."""
function apply_O(terms::Vector{OTerm}, ψ::Vector{ComplexF64}, dims::Vector{Int})
    out = zeros(ComplexF64, length(ψ))
    for t in terms
        w = t.b == 0 ? apply_node_op(ψ, t.a, t.Oa, dims) :
                       apply_node_op(apply_node_op(ψ, t.b, t.Ob, dims), t.a, t.Oa, dims)
        out .+= t.coef .* w
    end
    return out
end

"""exp(−iÔ)|ψ⟩ by exact Krylov exponentiation of the dense vector, sub-stepped
`n ≈ ⌈a⌉` times (mirrors `decouple_state`, but with no bond truncation)."""
function decouple_vector(terms::Vector{OTerm}, a::Float64, ψ::Vector{ComplexF64},
                         dims::Vector{Int}; tol::Float64=1e-12, krylovdim::Int=40)
    n = max(1, ceil(Int, a))
    φ = ψ ./ norm(ψ)
    for _ in 1:n
        φ, info = exponentiate(v -> -im .* apply_O(terms, v, dims), 1.0 / n, φ;
                               ishermitian=false, tol=tol, krylovdim=krylovdim)
        info.converged == 0 && @warn "exponentiate substep did not converge"
    end
    return φ
end

"""Dense plaquette reduced density matrix of a decoupled STATE VECTOR: project
onto every source-free link config E(h) (matter left free), then Gram-overlap."""
function ed_plaquette_density_matrix(ψvec::Vector{ComplexF64}, nx::Int, ny::Int, dg::Int,
                                     chain::Vector{Tuple{Int,Int}}, dims::Vector{Int})
    plaqs = plaquette_list(nx, ny)
    Nplaq = length(plaqs); n = length(chain)
    ncfg = 1 << Nplaq; nmat = 1 << n
    W = Vector{Vector{ComplexF64}}(undef, ncfg)        # matter vectors per height config
    for c in 0:ncfg-1
        h = config_to_height(c, plaqs, nx, ny)
        ER, EU = height_to_E(h, nx, ny)
        idx0 = Vector{Int}(undef, n); idx1 = Vector{Int}(undef, n)
        for (p, (ix, iy)) in enumerate(chain)
            _, dR, dU = site_dims(ix, iy, nx, ny, dg)
            eR = ix < nx ? ER[ix, iy] : 0
            eU = iy < ny ? EU[ix, iy] : 0
            idx0[p] = site_idx(0, eR, eU, dR, dU, dg)
            idx1[p] = site_idx(1, eR, eU, dR, dU, dg)
        end
        w = zeros(ComplexF64, nmat)
        for mc in 0:nmat-1
            fi = 0
            for p in 1:n
                lp = ((mc >> (p - 1)) & 1) == 1 ? idx1[p] : idx0[p]
                fi = fi * dims[p] + (lp - 1)
            end
            w[mc+1] = ψvec[fi+1]
        end
        W[c+1] = w
    end
    ρ = zeros(ComplexF64, ncfg, ncfg)
    for i in 1:ncfg, j in 1:ncfg
        ρ[i, j] = dot(W[i], W[j])                       # ⟨W_i|W_j⟩
    end
    weight = real(tr(ρ))
    return ρ / weight, plaqs, weight
end

"""Full ED reference: exact decoupling of the dense state, dual-plaquette
half-system entanglement.  Same output shape as `plaquette_entanglement`."""
function ed_plaquette_entanglement(ψvec::Vector{ComplexF64}, nx::Int, ny::Int;
                                   dg::Int=1, A::Union{Nothing,Vector{Int}}=nothing)
    chain, pos = column_snake(nx, ny)
    dims = col_node_dims(nx, ny, dg)
    _, Gt = generate_greens_function(nx, ny)
    terms = build_O_terms(nx, ny, dg, pos, Gt)
    Mabs = 0.0                                          # Σ|∇G| for the spectral bound
    for iy in 1:ny, ix in 1:nx-1, jy in 1:ny, jx in 1:nx
        Mabs += abs(shift_exact(Gt, ix, iy, :R, jx, jy))
    end
    for iy in 1:ny-1, ix in 1:nx, jy in 1:ny, jx in 1:nx
        Mabs += abs(shift_exact(Gt, ix, iy, :U, jx, jy))
    end
    a = _spectral_bound(Mabs)
    φ = decouple_vector(terms, a, ψvec, dims)
    ρ, plaqs, weight = ed_plaquette_density_matrix(φ, nx, ny, dg, chain, dims)
    Nplaq = length(plaqs)
    F = eigen(Hermitian(ρ)); purity = real(F.values[end]); v = F.vectors[:, end]
    Aset = A === nothing ? default_bipartition(nx, ny, plaqs) : A
    S = schmidt_entropy(schmidt_matrix(v, Nplaq, Aset))
    return (S=S, purity=purity, srcfree_weight=weight, ρ=ρ, plaqs=plaqs, A=Aset)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  The test                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function run_test(; cases=[(2, 3), (3, 2)], dg::Int=1, seed::Int=1,
                  Dmax::Int=256, tol_S::Float64=1e-3, tol_ρ::Float64=1e-3)
    println("═════════════════════════════════════════════════════════════════")
    println("  plaquette-entanglement  MPS pipeline  vs  ED (exact decoupling)")
    println("═════════════════════════════════════════════════════════════════")
    allok = true
    for (nx, ny) in cases
        Nplaq = (nx - 1) * (ny - 1)
        @printf("\n── %d×%d  (Hilbert dim = %d, plaquettes = %d) ──\n",
                nx, ny, prod(col_node_dims(nx, ny, dg)), Nplaq)

        ψ_mps, ψ_vec, _ = shared_random_state(nx, ny, dg; seed=seed)

        # Same bipartition on both routes.
        A = default_bipartition(nx, ny, plaquette_list(nx, ny))

        mps = plaquette_entanglement(ψ_mps, nx, ny; dg=dg, which=:exact,
                                     Dmax=Dmax, A=A, verbose=false)
        ed  = ed_plaquette_entanglement(ψ_vec, nx, ny; dg=dg, A=A)

        dS = abs(mps.S - ed.S)
        dW = abs(mps.srcfree_weight - ed.srcfree_weight)
        dP = abs(mps.purity - ed.purity)
        dρ = norm(mps.ρ - ed.ρ)                          # phase-independent

        @printf("  source-free weight :  MPS %.8f   ED %.8f   |Δ| %.2e\n",
                mps.srcfree_weight, ed.srcfree_weight, dW)
        @printf("  leading purity     :  MPS %.8f   ED %.8f   |Δ| %.2e\n",
                mps.purity, ed.purity, dP)
        @printf("  ρ_plaq Frobenius Δ :  %.2e\n", dρ)
        @printf("  half-system  S     :  MPS %.8f   ED %.8f   |Δ| %.2e  (%.6f bits)\n",
                mps.S, ed.S, dS, ed.S / log(2))

        ok = dS < tol_S && dρ < tol_ρ && dW < tol_S && dP < tol_S
        println(ok ? "  [PASS]" : "  [FAIL]")
        allok &= ok
        flush(stdout)
    end
    println("\n", allok ? "═══ ALL CASES PASS ═══" : "═══ SOME CASE FAILED ═══")
    return allok
end

if abspath(PROGRAM_FILE) == @__FILE__
    ok = run_test()
    exit(ok ? 0 : 1)
end
