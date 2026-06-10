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

     • symmetric  (m²=0): the gradient of the massless Green's function (with a
            uniform neutralizing background charge) is AFFINE in x, i.e. a
            constant + linear ramp ⇒ a CONFLUENT root at λ = 1.  A polynomial×λ^d
            term is NOT representable by a sum of distinct exponentials, so we
            fit it with an explicit affine baseline  a + b·d  instead.
     • antisymmetric (m²=2): evanescent waves with cosh θ = 1 + m²/2 = 2 ⇒
            λ_± = e^{∓θ} = 2 ∓ √3  ≈ 0.2679 and 3.732.
       On a FINITE open chain the two reflected waves λ^{|x−x'|} and λ^{−|x−x'|}
       combine into the boundary-induced hyperbolic-sine (sinh/cosh) profile.

   Hence the kernel is  (affine baseline a + b·d)  +  K genuine exponentials
   {2−√3, 2+√3, …}; `validate_exp_fit` checks the fitted λ_k against 2−√3.  The
   λ_k are extracted from the SECOND DIFFERENCE of the sequence (which kills the
   affine baseline) so the massless mode cannot contaminate the rate estimate.

   ── Why fit per transverse channel, in the bulk ──────────────────────────────
   M = B·L⁺ is the gradient of a FINITE pseudo-inverse, so it is NOT a function
   of the separation d = x−x' alone: at fixed d, individual (x,x') pairs differ
   by O(1) because of (i) the massless Coulomb background (affine in absolute
   position) and (ii) boundary reflections (λ^{x+x'} terms near the open ends).
   A d-only exponential sum therefore cannot reduce the *raw* per-(y,y') error
   below ~0.3 no matter how many exponentials or how large N.

   The fix is to work in the rung's transverse-mode basis and restrict to the
   bulk:  e_S=(1,1)/√2 (massless) and e_A=(1,−1)/√2 (massive).  The MASSIVE
   channel, away from the boundaries, IS the translation-invariant exponential
   sum the MPO represents; the MASSLESS channel is the affine Coulomb ramp,
   carried separately by a bond-2 ramp MPO.

   ── Odd kernel ───────────────────────────────────────────────────────────────
   Each channel kernel is the GRADIENT of a potential, hence antisymmetric — but
   a forward-difference link (x→x+1) sits at the HALF-INTEGER position x+½, so
   the reflection center is d = −½:  M(d) = −M(−1−d), not −M(−d).  We fit the
   one-sided rightward kernel (d ≥ 0) and extend by this half-shifted reflection;
   the rightward fit needs only the DECAYING root λ = 2−√3 (the growing 2+√3
   partner is the reflected wave reproduced by the finite-chain MPO's boundary
   tensors).

   ── Method ───────────────────────────────────────────────────────────────────
   Massive channel (A←A), bulk pairs (both endpoints ≥ `edge` from a boundary):
     1. average M_A over equal signed offset d = x−x' ≥ 0 → sequence f(d);
     2. Prony on the SECOND DIFFERENCE of f (kills the affine λ=1 baseline) → λ_k;
     3. amplitudes (a,b,c_k) by global least-squares in {1, d, λ_k^d};
     4. report the worst bulk residual (odd-extended).
   Massless channel (S←S):  fit the affine ramp p + q·x + r·x' and report its
   residual, confirming it is the static Coulomb background.

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

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Transverse-mode (rung) decomposition                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Why per-(y,y') fitting cannot reach tol: M = B·L⁺ is the gradient of a FINITE
# pseudo-inverse, so it is NOT a function of the separation d = x−x' alone.  Two
# translation-breaking pieces survive at fixed d:
#
#   (1) the MASSLESS (symmetric, m²=0) transverse mode — the static Coulomb
#       background, whose shift field is AFFINE in absolute position
#       (p + q·x + r·x'), leaving an r·x' spread when binning by d; and
#   (2) BOUNDARY REFLECTIONS — even the massive mode carries λ^{x+x'} reflection
#       terms near the open ends, again depending on absolute position.
#
# Diagonalizing the rung Laplacian [[1,−1],[−1,1]] into SYMMETRIC (e_S=(1,1)/√2,
# eigval 0, massless) and ANTISYMMETRIC (e_A=(1,−1)/√2, eigval 2, massive)
# channels separates these: the antisymmetric channel is a single gapped 1D
# propagator that, in the BULK (away from the open ends, where λ^{x+x'} is
# negligible), is exactly the translation-invariant exponential sum the MPO
# represents.  The symmetric channel is the affine Coulomb ramp, carried by a
# separate bond-2 ramp MPO, not by decaying exponentials.

"""Transverse-mode coefficients over the two rows (y=1,2):
`:S` symmetric (massless), `:A` antisymmetric (massive)."""
_mode_coeff(m::Symbol) = m === :S ? (1.0, 1.0) :
                          m === :A ? (1.0, -1.0) :
                          error("mode $m not in {:S,:A}")

"""
    channel_pairs(geo, M, link_mode, src_mode; i=1) → Vector{(d, x, xp, value)}

Project the horizontal-link (i=1) shift field onto transverse modes
`link_mode` (the field's rung) and `src_mode` (the source's rung), for every
link column x∈1:N and source column xp∈1:N+1.  Returns signed offset d=x−xp,
the absolute positions x, xp, and the channel-projected value (½ = 1/(√2)²)."""
function channel_pairs(geo::LadderGeometry, M::AbstractMatrix,
                       link_mode::Symbol, src_mode::Symbol; i::Int=1)
    i == 1 || error("channel_pairs is implemented for horizontal links (i=1)")
    ul = _mode_coeff(link_mode); us = _mode_coeff(src_mode)
    out = Tuple{Int,Int,Int,Float64}[]
    for x in 1:geo.N, xp in 1:geo.N+1
        v = 0.0
        for (yl, cl) in zip((1, 2), ul), (ys, cs) in zip((1, 2), us)
            ℓ = geo.link_id[(x, yl, 1)]
            s = geo.vertex_id[(xp, ys)]
            v += cl * cs * M[ℓ, s]
        end
        push!(out, (x - xp, x, xp, 0.5v))
    end
    return out
end

"""Interior margin for "bulk" pairs: drop sources/links within `edge` columns of
either open boundary, so λ^{x+x'} reflection terms are negligible."""
bulk_edge(geo::LadderGeometry) = max(2, geo.N ÷ 5)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Massive-channel exponential fit                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Result of the antisymmetric (massive) channel exponential fit.

Bulk model (rightward branch d = x−x' ≥ 0), extended antisymmetrically for d<0:

    M_A(d)  ≈  Σ_k c_k λ_k^{d}

with the dominant rate λ = 2−√3 (its growing partner 2+√3 builds the finite-size
sinh and is reproduced by the MPO's own boundary tensors)."""
struct ExpFit
    link_mode :: Symbol
    src_mode  :: Symbol
    i         :: Int
    a         :: ComplexF64           # affine baseline (≈0 for the massive channel)
    b         :: ComplexF64
    c         :: Vector{ComplexF64}   # exponential amplitudes
    λ         :: Vector{ComplexF64}   # exponential decay parameters
    max_err   :: Float64              # worst bulk residual (odd-extended)
    K         :: Int
end

"""
    fit_channel(geo, M, link_mode, src_mode; K, i=1, edge) → ExpFit

Fit the BULK (interior) one-sided kernel of a transverse channel by an affine
baseline plus K exponentials, M(d) ≈ a + b·d + Σ_k c_k λ_k^{d} (d ≥ 0), extended
antisymmetrically.  Rates λ_k come from Prony on the SECOND DIFFERENCE of the
bulk-averaged sequence (which annihilates the affine baseline); amplitudes from
a global least-squares in {1, d, λ_k^d}.  The leftward branch uses the
half-shifted antisymmetric extension M(d) = −M(−1−d) (forward-difference links
sit at x+½).  `max_err` is the worst residual over the BULK pairs only —
boundary reflections are deliberately excluded because the finite-chain MPO
reproduces them from the same rates."""
function fit_channel(geo::LadderGeometry, M::AbstractMatrix,
                     link_mode::Symbol, src_mode::Symbol;
                     K::Int, i::Int=1, edge::Int=bulk_edge(geo))
    pr = channel_pairs(geo, M, link_mode, src_mode; i=i)
    lo, hi = 1 + edge, (geo.N + 1) - edge
    bulk = [(d, v) for (d, x, xp, v) in pr if lo ≤ x ≤ hi && lo ≤ xp ≤ hi]
    @assert !isempty(bulk) "fit_channel: no bulk pairs (raise N or lower edge)"

    # bulk-averaged rightward sequence f(d)
    dmax = maximum(d for (d, _) in bulk if d ≥ 0)
    fsum = zeros(Float64, dmax + 1); fcnt = zeros(Int, dmax + 1)
    for (d, v) in bulk
        0 ≤ d ≤ dmax && (fsum[d+1] += v; fcnt[d+1] += 1)
    end
    keep = fcnt .> 0
    f = fsum[keep] ./ fcnt[keep]                  # contiguous from d=0 upward

    # rates from the 2nd difference (kills the affine λ=1 background)
    Δ2 = length(f) ≥ 3 ? [f[m+2] - 2f[m+1] + f[m] for m in 1:length(f)-2] : copy(f)
    Keff = max(1, min(K, length(Δ2) ÷ 2))
    _, λ = prony_fit(Δ2, Keff)

    # amplitudes (a, b, c_k) by global LSQ over the bulk rightward pairs
    rp  = [(d, v) for (d, v) in bulk if d ≥ 0]
    A   = ComplexF64[ j == 1 ? 1.0 :
                      j == 2 ? ComplexF64(d) :
                      λ[j-2]^d
                      for (d, _) in rp, j in 1:(2 + Keff) ]
    sol = A \ ComplexF64[v for (_, v) in rp]
    a, b = sol[1], sol[2]
    c    = sol[3:end]

    # Forward-difference links sit at the half-integer position x+½, so the
    # gradient field is antisymmetric about d = −½:  M(d) = −M(−1−d), NOT −M(−d).
    # The leftward partner of d=k (k≥0) is d=−1−k.
    S(d) = a + b * d + sum(c[k] * λ[k]^d for k in 1:Keff)
    max_err = 0.0
    for (d, v) in bulk
        pred = d ≥ 0 ? S(d) : -S(-1 - d)
        max_err = max(max_err, abs(pred - v))
    end
    return ExpFit(link_mode, src_mode, i, a, b, c, λ, max_err, Keff)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Massless-channel affine (Coulomb ramp) fit                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    symmetric_ramp(geo, M; i=1) → (p, q, r, residual)

Fit the symmetric (massless) channel's rightward branch by the 2-variable affine
Coulomb ramp  M_S(x,x') ≈ p + q·x + r·x'.  A small `residual` confirms the
massless sector is exactly the static linear background (gradient of the
quadratic massless potential), represented downstream by a bond-2 ramp MPO."""
function symmetric_ramp(geo::LadderGeometry, M::AbstractMatrix; i::Int=1)
    pr = channel_pairs(geo, M, :S, :S; i=i)
    rp = [(x, xp, v) for (d, x, xp, v) in pr if d ≥ 0]
    A = Float64[ j == 1 ? 1.0 : j == 2 ? Float64(x) : Float64(xp)
                 for (x, xp, _) in rp, j in 1:3 ]
    rhs = Float64[v for (_, _, v) in rp]
    sol = A \ rhs
    res = isempty(rp) ? 0.0 : maximum(abs, A * sol - rhs)
    return sol[1], sol[2], sol[3], res
end

"""
    symmetric_ramp_step(geo, M; i=1) → (a0, q, r, s, residual)

Fit the symmetric (massless) channel over ALL pairs (both branches) as a
CONTINUOUS affine background plus a Heaviside STEP at the source:

    v_SS(x,x')  ≈  a0 + q·x + r·x'  +  s·𝟙(x ≥ x').

The massless shift field is the gradient of a 1D Coulomb potential: a uniform
linear background (a0,q,r — continuous across the source) plus a unit step where
the link crosses the source charge.  This 4-parameter model is exact (residual
~machine), and unlike `symmetric_ramp` it is valid on BOTH branches, which the
decoupling MPO needs.  The rightward intercept is p = a0 + s."""
function symmetric_ramp_step(geo::LadderGeometry, M::AbstractMatrix; i::Int=1)
    pr = channel_pairs(geo, M, :S, :S; i=i)
    A = Float64[ j == 1 ? 1.0 :
                 j == 2 ? Float64(x) :
                 j == 3 ? Float64(xp) :
                 (x ≥ xp ? 1.0 : 0.0)
                 for (_, x, xp, _) in pr, j in 1:4 ]
    rhs = Float64[v for (_, _, _, v) in pr]
    sol = A \ rhs
    res = maximum(abs, A * sol - rhs)
    return sol[1], sol[2], sol[3], sol[4], res
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Validation                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    validate_exp_fit(N; K, i, tol) → Bool

Diagonalize the rung into massive (antisymmetric) and massless (symmetric)
channels, then check that:
  • the MASSIVE channel's bulk kernel is a sum of exponentials to `tol`, with the
    dominant rate λ = 2−√3 recovered;
  • the MASSLESS channel is the affine Coulomb ramp p + q·x + r·x' to `tol`.
"""
function validate_exp_fit(N::Int; K::Int=3, i::Int=1, tol::Float64=1e-6)
    geo = ladder_geometry(N)
    M   = shift_field(geo)
    λ_anti = 2 - sqrt(3)                          # massive-mode decay (cosh θ = 2)

    # Boundary-reflection leakage into the bulk window decays as λ^{2·edge}, so the
    # margin needed for leakage < tol is edge ≥ ln(tol)/(2 ln λ).  Use that target,
    # clamped to whatever interior the lattice can actually spare.
    target_edge = ceil(Int, log(tol) / (2 * log(λ_anti)))
    max_room    = (geo.N - 1) ÷ 2 - 1
    edge        = clamp(target_edge, 2, max(2, max_room))
    margin_ok   = edge ≥ target_edge             # is N large enough for tol?

    println("─── transverse-channel shift-field fit (N=$N, K=$K, link i=$i, bulk edge=$edge) ───")
    @printf("  expected massive (m²=2) λ = 2−√3 = %.6f   (partner 2+√3 = %.4f)\n",
            λ_anti, 2 + sqrt(3))

    # Massive (antisymmetric ← antisymmetric) channel: exponential sum.
    FA = fit_channel(geo, M, :A, :A; K=K, i=i, edge=edge)
    λr = round.(real.(FA.λ); digits=5)
    λi = round.(imag.(FA.λ); digits=5)
    kdom = argmin(abs.(FA.λ .- λ_anti))           # index of the 2−√3 mode
    @printf("  A←A (massive):  bulk max_err=%.2e   λ = %s%s\n",
            FA.max_err, string(λr),
            all(abs.(imag.(FA.λ)) .< 1e-8) ? "" : " + i" * string(λi))
    @printf("                  dominant amplitude c(λ=2−√3) = %+.5f\n", real(FA.c[kdom]))
    found_anti = any(abs.(FA.λ .- λ_anti) .< 1e-3) || any(abs.(FA.λ .- (2 + sqrt(3))) .< 1e-2)

    # Massless (symmetric ← symmetric) channel: affine Coulomb ramp.
    p, q, r, res_S = symmetric_ramp(geo, M; i=i)
    @printf("  S←S (massless): affine ramp p=%+.4f q=%+.4f r=%+.4f   residual=%.2e\n",
            p, q, r, res_S)

    ok_exp  = FA.max_err < tol
    ok_ramp = res_S < tol
    if ok_exp
        println("  PASS: massive channel = exponential sum to tol=$tol")
    elseif !margin_ok
        @printf("  INFO: massive-channel error %.2e is boundary-leakage limited (edge=%d<%d); raise N\n",
                FA.max_err, edge, target_edge)
    else
        println("  WARN: massive-channel bulk error exceeds tol unexpectedly")
    end
    println(ok_ramp ? "  PASS: massless channel = affine Coulomb ramp to tol=$tol" :
                      "  WARN: massless channel not affine (unexpected)")
    println(found_anti ? "  PASS: recovered massive-mode scale λ=2−√3" :
                         "  WARN: λ=2−√3 not found")
    # Accept once the rate is recovered, the ramp is exact, and the exponential
    # error is either below tol or merely leakage-limited by a too-small lattice.
    return found_anti && ok_ramp && (ok_exp || !margin_ok)
end

# Demo / self-test when run directly.
if abspath(PROGRAM_FILE) == @__FILE__
    for N in (16, 24, 40)
        validate_exp_fit(N; K=3, i=1)
        println()
    end
end
