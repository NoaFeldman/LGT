#= ═══════════════════════════════════════════════════════════════════════════════
   gauge_matter_exp_fit.jl

   Stage 2 of the gauge–matter decoupling MPO: compress the long-range shift
   field M (Stage 1) into a sum of K exponentials in the horizontal separation,

       M(x,y ; x',y')  ≈  Σ_{k=1}^K  c_k  λ_k^{|x−x'|},

   so that each exponential becomes one bond of a low-bond-dimension MPO.

   ── Why a few exponentials suffice ───────────────────────────────────────────
   The 2-row ladder has two transverse (rung) modes of the vertical Laplacian
   [[1,-1],[-1,1]]:  the SYMMETRIC mode (transverse eigenvalue 0) and the
   ANTISYMMETRIC mode (transverse eigenvalue m² = 2).  Decomposing −∇² into
   transverse modes, the longitudinal kernel of each mode obeys a 1D screened
   Poisson equation (−Δ_x + m²):

     • symmetric  (m²=0): the gradient of the massless ramp is ~constant ⇒ λ ≈ 1.
     • antisymmetric (m²=2): evanescent waves with cosh θ = 1 + m²/2 = 2 ⇒
            λ_± = e^{∓θ} = 2 ∓ √3  ≈ 0.2679 and 3.732.
       On a FINITE open chain the two reflected waves λ^{|x−x'|} and λ^{−|x−x'|}
       combine into the boundary-induced hyperbolic-sine (sinh/cosh) profile.

   Hence K = 3 exponentials  {1, 2−√3, 2+√3}  already reproduce the kernel to
   high accuracy; `validate_exp_fit` checks the fitted λ_k against 2−√3.

   ── Odd kernel ───────────────────────────────────────────────────────────────
   M is the GRADIENT of G, so it is ANTISYMMETRIC about the source:
   M(d) ≈ −M(−d) with d = x − x'.  We therefore fit the one-sided rightward
   kernel (d ≥ 0) and extend it antisymmetrically; the rightward fit needs only
   the DECAYING root λ = 2−√3 (the growing 2+√3 partner is the reflected wave
   that, together with λ, forms the finite-size sinh near the far boundary and
   shows up as the residual / extra exponentials).

   ── Method ───────────────────────────────────────────────────────────────────
   For each fixed (link-rung y, source-rung y') and link direction i:
     1. average M over pairs of equal signed offset d = x−x' ≥ 0 → sequence f(d);
     2. least-squares Prony on f(d) → decay parameters λ_k (polynomial roots);
     3. refine amplitudes c_k by a global least-squares over the rightward pairs;
     4. report the maximum absolute error over ALL pairs (odd-extended model).

   Requires: gauge_matter_ladder.jl
   ═══════════════════════════════════════════════════════════════════════════ =#

include(joinpath(@__DIR__, "gauge_matter_ladder.jl"))

using LinearAlgebra
using Printf

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Polynomial roots + least-squares Prony                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Roots of a monic-or-not polynomial  coeffs[1]·z^K + … + coeffs[K+1]  via the
companion matrix eigenvalues."""
function poly_roots(coeffs::AbstractVector)
    c = coeffs ./ coeffs[1]
    K = length(c) - 1
    K == 0 && return ComplexF64[]
    C = zeros(ComplexF64, K, K)
    @inbounds for i in 1:K
        C[1, i] = -c[i+1]
    end
    @inbounds for i in 2:K
        C[i, i-1] = 1.0
    end
    return eigvals(C)
end

"""
    prony_fit(f, K) → (c, λ)

Least-squares Prony fit of the sequence `f` (samples f(0),f(1),…) by K complex
exponentials:  f(r) ≈ Σ_k c_k λ_k^r.  Uses overdetermined linear prediction for
the λ_k (robust to finite-size deviations) and a Vandermonde least-squares for
the amplitudes c_k.
"""
function prony_fit(f::AbstractVector, K::Int)
    n = length(f)
    @assert n ≥ 2K "prony_fit: need ≥ 2K=$(2K) samples, got $n"

    # Linear prediction  f[m] = −Σ_{j=1}^K a_j f[m−j]   (rows m = K+1..n)
    rows = n - K
    A = zeros(Float64, rows, K)
    b = zeros(Float64, rows)
    @inbounds for r in 1:rows
        m = K + r
        for j in 1:K
            A[r, j] = f[m-j]
        end
        b[r] = f[m]
    end
    a = A \ (-b)                                  # least-squares prediction coeffs
    λ = poly_roots([1.0; a])                      # z^K + a₁z^{K−1} + … + a_K

    # Amplitudes via Vandermonde least squares  f[m] ≈ Σ c_k λ_k^{m−1}
    V = ComplexF64[λ[k]^(m-1) for m in 1:n, k in 1:K]
    c = V \ ComplexF64.(f)
    return c, λ
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Shift-field exponential fit                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Collect (signed offset d = x − x', value) pairs for horizontal links of row
`y` (i=1) or vertical links of column (i=2) at rung `y`, sourced from row `y'`.

The shift field is the GRADIENT of G, hence ODD about the source: M(d) ≈ −M(−d).
We therefore fit the one-sided (rightward, d ≥ 0) kernel and extend it
antisymmetrically; averaging over |d| would cancel the two signs to ≈ 0."""
function _collect_pairs(geo::LadderGeometry, M::AbstractMatrix, y::Int, yp::Int, i::Int)
    xs = (i == 1) ? (1:geo.N) : (1:geo.N+1)
    pairs = Tuple{Int,Float64}[]
    for x in xs, xp in 1:geo.N+1
        ℓ = geo.link_id[(x, y, i)]
        s = geo.vertex_id[(xp, yp)]
        push!(pairs, (x - xp, M[ℓ, s]))
    end
    return pairs
end

"""Result of one (y,y',i) exponential fit."""
struct ExpFit
    y       :: Int
    yp      :: Int
    i       :: Int
    c       :: Vector{ComplexF64}     # amplitudes
    λ       :: Vector{ComplexF64}     # decay parameters
    max_err :: Float64                # max |Σ c_k λ_k^{|x−x'|} − M| over all pairs
    K       :: Int
end

"""
    fit_one(geo, M, y, yp, i; K) → ExpFit

Fit the one-sided (rightward, d = x−x' ≥ 0) kernel
    M[link(x,y,i), src(x',y')]  ≈  Σ_k c_k λ_k^{d}      (d ≥ 0)
by K exponentials, and extend antisymmetrically (M(−d) = −M(d)) for d < 0.
`max_err` is the worst absolute residual over ALL pairs under this odd model.
"""
function fit_one(geo::LadderGeometry, M::AbstractMatrix, y::Int, yp::Int, i::Int; K::Int)
    pairs = _collect_pairs(geo, M, y, yp, i)

    # rightward (d ≥ 0) averaged sequence f(d)
    dmax = maximum(d for (d, _) in pairs if d ≥ 0)
    fsum = zeros(Float64, dmax + 1); fcnt = zeros(Int, dmax + 1)
    for (d, v) in pairs
        0 ≤ d ≤ dmax && (fsum[d+1] += v; fcnt[d+1] += 1)
    end
    @assert all(fcnt .> 0) "fit_one: gap in rightward offsets — geometry issue"
    f = fsum ./ fcnt

    Keff = min(K, length(f) ÷ 2)                  # need ≥ 2K samples
    c, λ = prony_fit(f, Keff)

    # global least-squares refinement of c over the rightward pairs
    rp = [(d, v) for (d, v) in pairs if d ≥ 0]
    D  = ComplexF64[λ[k]^d for (d, _) in rp, k in 1:Keff]
    c  = D \ ComplexF64[v for (_, v) in rp]

    # worst residual over ALL pairs, with antisymmetric extension for d < 0
    S(d) = sum(c[k] * λ[k]^d for k in 1:Keff)
    max_err = 0.0
    for (d, v) in pairs
        pred = d ≥ 0 ? S(d) : -S(-d)
        max_err = max(max_err, abs(pred - v))
    end

    return ExpFit(y, yp, i, c, λ, max_err, Keff)
end

"""
    fit_shift_field_exponentials(geo, M; K, i) → Dict{(y,yp)=>ExpFit}

Fit every transverse rung pair (y, y') for link direction `i` (default
horizontal, i=1).  Returns a dictionary keyed by (y, y').
"""
function fit_shift_field_exponentials(geo::LadderGeometry, M::AbstractMatrix;
                                       K::Int=3, i::Int=1)
    out = Dict{Tuple{Int,Int},ExpFit}()
    for y in 1:2, yp in 1:2
        out[(y, yp)] = fit_one(geo, M, y, yp, i; K=K)
    end
    return out
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Validation                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    validate_exp_fit(N; K, i, tol) → Bool

Build the ladder, its shift field, and the K-exponential fits; print the
maximum absolute error per rung pair and the fitted decay constants, and check
that the antisymmetric-mode rate λ = 2−√3 (the boundary-induced sinh scale) is
recovered by at least one rung pair.
"""
function validate_exp_fit(N::Int; K::Int=3, i::Int=1, tol::Float64=1e-6)
    geo = ladder_geometry(N)
    M   = shift_field(geo)
    fits = fit_shift_field_exponentials(geo, M; K=K, i=i)

    λ_anti = 2 - sqrt(3)                          # antisymmetric-mode decay (cosh θ = 2)

    println("─── exponential-sum fit (N=$N, K=$K, link i=$i) ───")
    @printf("  expected antisymmetric-mode λ = 2−√3 = %.6f  (and 1/λ = %.4f)\n",
            λ_anti, 1/λ_anti)
    worst = 0.0
    found_anti = false
    for y in 1:2, yp in 1:2
        F = fits[(y, yp)]
        worst = max(worst, F.max_err)
        λr = round.(real.(F.λ); digits=4)
        λi = round.(imag.(F.λ); digits=4)
        @printf("  (y=%d←y'=%d): max_err=%.2e   λ = %s%s\n",
                y, yp, F.max_err, string(λr),
                all(abs.(imag.(F.λ)) .< 1e-8) ? "" : " + i" * string(λi))
        if any(abs.(F.λ .- λ_anti) .< 1e-3) || any(abs.(F.λ .- 1/λ_anti) .< 1e-2)
            found_anti = true
        end
    end
    @printf("  worst max_err over all rung pairs = %.2e\n", worst)
    ok_err  = worst < tol
    println(ok_err     ? "  PASS: fit accurate to tol=$tol" :
                         "  WARN: fit error exceeds tol (increase K or N)")
    println(found_anti ? "  PASS: recovered boundary sinh scale λ=2−√3" :
                         "  WARN: antisymmetric-mode scale not found")
    return ok_err && found_anti
end

# Demo / self-test when run directly.
if abspath(PROGRAM_FILE) == @__FILE__
    for N in (8, 12, 20)
        validate_exp_fit(N; K=3, i=1)
        println()
    end
end
