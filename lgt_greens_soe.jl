#= ═══════════════════════════════════════════════════════════════════════════════
   lgt_greens_soe.jl  —  Part 1: Mathematical Core & Kernel Approximation

   Sum-of-Exponentials (SoE) approximation of the 2D lattice Green's function
   for the U(1) LGT with Open Boundary Conditions.

   ── Motivation ───────────────────────────────────────────────────────────────
   The exact PEPO representation of G = M⁻¹ has bond dimension D = Nx·Ny.
   If we can write

       G(rₓ, r_y) ≈ Σ_{j=1}^{K} wⱼ e^{-γⱼ |rₓ|} · e^{-γⱼ |r_y|}

   then the PEPO bond dimension drops to K (≈ 10), a dramatic compression.

   The key identity shows both representations are equivalent:
       G(rₓ, r_y) ≈ Σⱼ wⱼ e^{-γⱼ(|rₓ|+|r_y|)}  ⟺  g(d) = Σⱼ wⱼ e^{-γⱼ d}

   where g(d) = ⟨G(n,m)⟩_{|n−m|₁=d} is the kernel averaged over Manhattan
   distance d.  Fitting the 1-D sequence g(d) therefore recovers the (wⱼ,γⱼ)
   needed for the 2-D factored PEPO.

   ── Note on translation invariance ──────────────────────────────────────────
   The OBC Green's function is NOT translation-invariant; g(d) mixes
   contributions from bulk pairs and boundary-adjacent pairs.  Using a large
   reference lattice (default: the lattice itself; can override via Nx_ref,
   Ny_ref) reduces boundary contamination and gives a better bulk approximation.

   ── Contents ─────────────────────────────────────────────────────────────────
   §1  verify_laplacian_nonsingular   — analytic + numeric PD check for M
   §2  spectral_data                  — analytic eigensystem of M (OBC)
   §3  greens_kernel_by_distance      — g(d) = ⟨G(n,m)⟩ at distance d
   §4  SoEKernel                      — struct + functor + display
   §5  fit_soe_matrix_pencil          — Matrix Pencil (Hua–Sarkar 1990) fit
   §6  fit_soe_refine                 — Gauss–Newton nonlinear LS refinement
   §7  soe_approximation              — high-level driver
   §8  soe_quality                    — approximation quality metrics
   §9  run_soe_tests                  — test suite
   §10 demo / entry point
   ═══════════════════════════════════════════════════════════════════════════ =#

# ── Load base Green's function definitions ────────────────────────────────────
if !@isdefined(_LGT_SOE_LOADED)
    include(joinpath(@__DIR__, "lgt_greens_function.jl"))
    using LinearAlgebra
    using Printf
    using Test
    const _LGT_SOE_LOADED = true
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §1  Non-singularity of the 2D OBC Laplacian                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    verify_laplacian_nonsingular(Nx, Ny) -> (M, λ_min, λ_max, κ)

Construct the 2D OBC Laplacian

    M = L_{Nx} ⊗ I_{Ny} + I_{Nx} ⊗ L_{Ny}

and verify it is positive definite (non-singular) both analytically and
numerically.

Analytic eigenvalue bounds (OBC Dirichlet):
    λ_min = (2 − 2cos(π/(Nx+1))) + (2 − 2cos(π/(Ny+1)))  > 0
    λ_max = (2 − 2cos(Nx π/(Nx+1))) + (2 − 2cos(Ny π/(Ny+1)))

Returns
───────
- M      :: Matrix{Float64}   full Kronecker-sum Laplacian
- λ_min  :: Float64           smallest computed eigenvalue
- λ_max  :: Float64           largest computed eigenvalue
- κ      :: Float64           condition number λ_max/λ_min

Throws an AssertionError if any eigenvalue ≤ 0.
"""
function verify_laplacian_nonsingular(Nx::Int, Ny::Int)
    _, _, M = generate_greens_function_direct(Nx, Ny)

    # Analytic bounds (cheap check before expensive eigvals)
    λ_min_an = (2.0 - 2.0*cos(π/(Nx+1))) + (2.0 - 2.0*cos(π/(Ny+1)))
    λ_max_an = (2.0 - 2.0*cos(Nx*π/(Nx+1))) + (2.0 - 2.0*cos(Ny*π/(Ny+1)))
    @assert λ_min_an > 0.0 "Analytic λ_min ≤ 0 (should never happen with OBC)"

    # Numeric verification via full diagonalisation (exact, symmetric guarantee)
    evs   = eigvals(Symmetric(M))
    λ_min = minimum(evs)
    λ_max = maximum(evs)
    κ     = λ_max / λ_min

    @assert λ_min > 0.0 """
        Laplacian is NOT positive definite on a $(Nx)×$(Ny) OBC lattice.
        Minimum numeric eigenvalue: $λ_min
        (This indicates a bug — OBC discrete Laplacian must be PD.)
    """

    return M, λ_min, λ_max, κ
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §2  Analytic spectral decomposition                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    spectral_data(Nx, Ny) -> (λx, λy, Φx, Φy, Λ)

Return the analytic eigensystem for the 2D OBC Laplacian without forming M.

1-D Dirichlet eigenvalues and eigenvectors:
    λ_{k,d}   = 2 − 2cos(k π / (N_d + 1)),   k = 1 … N_d
    Φ_d[n, k] = √(2/(N_d+1)) · sin(k n π / (N_d + 1))

2-D eigenvalues (Kronecker sum):
    Λ[kx, ky] = λx[kx] + λy[ky]

2-D eigenvectors are outer products:
    Ψ_{kx,ky}(nx, ny) = Φx[nx, kx] · Φy[ny, ky]

and the Green's function is reconstructed via:
    G(n, m) = Σ_{kx,ky} Ψ_{kx,ky}(n) · Λ[kx,ky]⁻¹ · Ψ_{kx,ky}(m)

Returns
───────
- λx :: Vector{Float64}  length Nx   x-direction 1-D eigenvalues
- λy :: Vector{Float64}  length Ny   y-direction 1-D eigenvalues
- Φx :: Matrix{Float64}  (Nx, Nx)    Φx[n, k] = eigenvector component
- Φy :: Matrix{Float64}  (Ny, Ny)    Φy[n, k] = eigenvector component
- Λ  :: Matrix{Float64}  (Nx, Ny)    2-D eigenvalues Λ[kx, ky]
"""
function spectral_data(Nx::Int, Ny::Int)
    λx = [2.0 - 2.0*cos(kx*π/(Nx+1)) for kx in 1:Nx]
    λy = [2.0 - 2.0*cos(ky*π/(Ny+1)) for ky in 1:Ny]

    Φx = [sqrt(2.0/(Nx+1)) * sin(kx * nx * π / (Nx+1))
          for nx in 1:Nx, kx in 1:Nx]
    Φy = [sqrt(2.0/(Ny+1)) * sin(ky * ny * π / (Ny+1))
          for ny in 1:Ny, ky in 1:Ny]

    Λ = [λx[kx] + λy[ky] for kx in 1:Nx, ky in 1:Ny]

    return λx, λy, Φx, Φy, Λ
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §3  Reference kernel  g(d) = ⟨G(n,m)⟩_{|n−m|₁ = d}                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    greens_kernel_by_distance(Nx, Ny) -> (d_vals, g_mean, g_spread)

Compute the Green's function averaged over all site pairs at each Manhattan
distance d = |nx − mx| + |ny − my|.

The target of the SoE fit is the 1-D sequence g_mean, because if

    G(rₓ, r_y) ≈ Σⱼ wⱼ e^{-γⱼ |rₓ|} e^{-γⱼ |r_y|}

then every pair (n,m) with |n−m|₁ = d contributes the same factor e^{-γⱼ d},
so fitting g_mean(d) = Σⱼ wⱼ e^{-γⱼ d} recovers exactly the (wⱼ, γⱼ)
needed for the PEPO construction.

The returned g_spread (standard deviation at each d) quantifies how much the
true G deviates from a purely distance-dependent function — a measure of the
translation-invariance breaking by the OBC.

Returns
───────
- d_vals  :: Vector{Int}     0, 1, …, (Nx−1)+(Ny−1)
- g_mean  :: Vector{Float64} mean G value at each Manhattan distance
- g_spread:: Vector{Float64} standard deviation of G values at each distance
"""
function greens_kernel_by_distance(Nx::Int, Ny::Int)
    _, G_tens = generate_greens_function(Nx, Ny)

    d_max  = (Nx - 1) + (Ny - 1)
    sums   = zeros(Float64, d_max + 1)
    sums2  = zeros(Float64, d_max + 1)   # for std-dev
    counts = zeros(Int,     d_max + 1)

    for iy in 1:Ny, ix in 1:Nx
        for jy in 1:Ny, jx in 1:Nx
            d = abs(ix - jx) + abs(iy - jy)
            v = G_tens[ix, iy, jx, jy]
            sums[d+1]   += v
            sums2[d+1]  += v*v
            counts[d+1] += 1
        end
    end

    d_vals   = collect(0:d_max)
    g_mean   = sums ./ counts
    # Var = E[x²] - (E[x])²; clamp to 0 for round-off
    g_spread = sqrt.(max.(sums2 ./ counts .- g_mean.^2, 0.0))

    return d_vals, g_mean, g_spread
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §4  SoEKernel — data structure, evaluation, display                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    SoEKernel

Holds a K-term Sum-of-Exponentials approximation of the Green's function
kernel:

    G(|r|₁) ≈ Σ_{j=1}^{K} wⱼ · exp(−γⱼ · |r|₁)

Fields
──────
- K :: Int                  number of terms
- w :: Vector{Float64}      weights  (may be of either sign)
- γ :: Vector{Float64}      decay exponents (all > 0), sorted ascending
"""
struct SoEKernel
    K :: Int
    w :: Vector{Float64}
    γ :: Vector{Float64}

    function SoEKernel(K::Int, w::AbstractVector, γ::AbstractVector)
        @assert length(w) == K && length(γ) == K
        @assert all(γ .> 0) "All exponents γⱼ must be positive (got $(γ))"
        new(K, Vector{Float64}(w), Vector{Float64}(γ))
    end
end

"""Evaluate the SoE approximation at Manhattan distance `d`."""
(soe::SoEKernel)(d::Real) = sum(soe.w[j] * exp(-soe.γ[j] * d) for j in 1:soe.K)

function Base.show(io::IO, soe::SoEKernel)
    @printf(io, "SoEKernel  K = %d\n", soe.K)
    @printf(io, "   j     weight wⱼ       exponent γⱼ    decay length 1/γⱼ\n")
    @printf(io, "  ──────────────────────────────────────────────────────────\n")
    for j in 1:soe.K
        @printf(io, "  %2d   %+14.8f   %12.8f   %14.6f\n",
                j, soe.w[j], soe.γ[j], 1.0 / soe.γ[j])
    end
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §5  Matrix Pencil (Hua–Sarkar 1990) SoE fitting                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    fit_soe_matrix_pencil(d_vals, g_vals, K; L_frac=0.45) -> SoEKernel

Fit a K-term sum-of-exponentials to the real sequence

    g_vals[i] ≈ f(d_vals[i]) = Σ_{j=1}^K  wⱼ · zⱼ^{d_vals[i]}

where zⱼ = exp(−γⱼ) ∈ (0,1), using the **Matrix Pencil method**
(Hua & Sarkar, *IEEE Trans. Acoust.* 38(5), 1990).

Prerequisites
─────────────
- `d_vals` must be consecutive integers starting at 0 (uniform spacing 1).
- `length(g_vals) ≥ 2K + 1`.

Algorithm
─────────
1. Build an (N−L) × (L+1) Hankel matrix  H[i,j] = g_vals[i+j−1].
2. SVD H = U Σ Vᴴ; retain the K dominant left singular vectors U_K.
3. Partition H into  H₀ = H[:,1:L]  and  H₁ = H[:,2:L+1].
4. Form the signal-subspace pencil  MP = pinv(U_K' H₀) · (U_K' H₁).
5. The eigenvalues zⱼ of MP are the poles e^{−γⱼ}.
6. Poles with large imaginary part or |zⱼ| ≥ 1 are discarded (non-physical).
7. Solve the Vandermonde least-squares system for weights wⱼ.

Parameters
──────────
- L_frac : pencil parameter as fraction of N (best results with 0.35–0.50).
           L is clamped to [K+1, N−K−1].

Returns the SoEKernel sorted by ascending γⱼ (slowest decay first).
"""
function fit_soe_matrix_pencil(
        d_vals::AbstractVector{<:Real},
        g_vals::AbstractVector{<:Real},
        K::Int;
        L_frac::Float64 = 0.45)

    N = length(g_vals)
    @assert N == length(d_vals) "d_vals and g_vals must have equal length"
    @assert N >= 2*K + 1        "Need at least 2K+1 = $(2K+1) samples; got $N"

    # Pencil parameter L: larger → better noise suppression; must satisfy
    # K+1 ≤ L ≤ N−K−1 for the (N−L)×(L+1) Hankel matrix to have at least
    # K rows and K+1 columns.
    L    = clamp(round(Int, L_frac * N), K + 1, N - K - 1)
    nrow = N - L                           # number of Hankel rows ≥ K

    # ── Step 1: Hankel matrix ─────────────────────────────────────────────────
    H = Matrix{Float64}(undef, nrow, L + 1)
    for i in 1:nrow, j in 1:(L+1)
        H[i, j] = g_vals[i + j - 1]
    end

    # ── Step 2: Truncated SVD ─────────────────────────────────────────────────
    F   = svd(H)
    U_K = F.U[:, 1:K]                      # (nrow) × K signal subspace

    # ── Step 3–4: Matrix Pencil eigenvalue problem ────────────────────────────
    H0 = H[:, 1:L]                         # (nrow) × L
    H1 = H[:, 2:L+1]                       # (nrow) × L
    # Project both pencil matrices onto the K-dimensional signal subspace.
    # pinv handles the case where U_K' H0 is rank-deficient.
    MP = pinv(U_K' * H0) * (U_K' * H1)    # K × K matrix pencil

    z_cx = eigvals(MP)                     # K complex poles

    # ── Step 5: Filter poles ─────────────────────────────────────────────────
    # Retain only real, strictly decaying poles:  0 < zⱼ < 1.
    # Tolerance on imaginary part is scaled to the pole magnitudes.
    tol_im = max(1e-8, 1e-5 * maximum(abs.(z_cx)))
    mask   = (abs.(imag.(z_cx)) .< tol_im) .&
             (real.(z_cx)       .> 1e-14)  .&
             (real.(z_cx)       .< 1.0 - 1e-14)
    z_real = sort(real.(z_cx[mask]), rev = true)   # descending (largest z first)

    K_found = length(z_real)
    if K_found == 0
        error("Matrix Pencil found no real, stable poles in (0,1).  " *
              "Try a different K or L_frac, or check that g_vals is positive and decaying.")
    end

    # If we found fewer than K good poles, pad by geometric halving from the
    # smallest pole already found (adds fast-decaying basis functions).
    while length(z_real) < K
        push!(z_real, last(z_real) * 0.5)
    end
    z_use = z_real[1:K]

    # ── Step 6: Vandermonde least-squares for weights ─────────────────────────
    # Build V[i,j] = zⱼ^{d_vals[i]} and solve  V w ≈ g_vals.
    Vmat = [z_use[j]^Float64(d_vals[i]) for i in 1:N, j in 1:K]
    w    = real.(Vmat \ collect(Float64, g_vals))

    γ_vals = -log.(z_use)                  # γⱼ = −ln(zⱼ) > 0

    # ── Sort by ascending γ (slowest decay first) ────────────────────────────
    perm = sortperm(γ_vals)
    return SoEKernel(K, w[perm], γ_vals[perm])
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §6  Gauss–Newton nonlinear least-squares refinement                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    fit_soe_refine(soe0, d_vals, g_vals;
                   max_iter=100, tol=1e-12, λ_lm=1e-6) -> SoEKernel

Refine an initial SoE fit using the **Gauss–Newton** algorithm with a simple
Levenberg–Marquardt damping and backtracking line-search.

The nonlinear residual is

    r_i(w, γ) = g(d_i) − Σ_j wⱼ e^{−γⱼ d_i}

The Jacobian (N × 2K) has columns

    ∂rᵢ/∂wⱼ  = −e^{−γⱼ dᵢ}
    ∂rᵢ/∂γⱼ  = +wⱼ dᵢ e^{−γⱼ dᵢ}

The update Δθ = (JᵀJ + λ I)⁻¹ Jᵀ r is clipped so that γⱼ ≥ γ_min after the
step (no non-physical negative exponents).

Parameters
──────────
- max_iter : maximum Gauss–Newton iterations  (default 100)
- tol      : convergence tolerance ‖r‖₂ < tol  (default 1e-12)
- λ_lm     : initial LM damping (scales with residual improvement)

Returns the refined SoEKernel.  Falls back to soe0 if no improvement is found.
"""
function fit_soe_refine(
        soe0::SoEKernel,
        d_vals::AbstractVector{<:Real},
        g_vals::AbstractVector{<:Real};
        max_iter::Int   = 100,
        tol::Float64    = 1e-12,
        λ_lm::Float64   = 1e-6)

    K  = soe0.K
    w  = copy(soe0.w)
    γ  = copy(soe0.γ)
    N  = length(d_vals)
    d  = Float64.(d_vals)
    g  = Float64.(g_vals)
    γ_floor = 1e-12          # hard floor to keep exponents positive

    # Pre-allocate Jacobian and residual
    J = zeros(Float64, N, 2K)
    r = zeros(Float64, N)

    function compute_Jr!(J, r, w, γ)
        for i in 1:N
            fi = 0.0
            for j in 1:K
                eij        = exp(-γ[j] * d[i])
                J[i, j]    = -eij
                J[i, K+j]  =  w[j] * d[i] * eij
                fi        +=  w[j] * eij
            end
            r[i] = g[i] - fi
        end
        return dot(r, r)     # ‖r‖²
    end

    res² = compute_Jr!(J, r, w, γ)
    best_res² = res²
    best_w, best_γ = copy(w), copy(γ)

    λ = λ_lm
    for _ in 1:max_iter
        sqrt(res²) < tol && break

        # Normal equations with LM damping:  (JᵀJ + λI) Δθ = Jᵀr
        JtJ = J' * J
        Jtr = J' * r
        for k in 1:2K
            JtJ[k, k] += λ
        end
        Δθ = JtJ \ Jtr

        # Clip γ update to prevent negative exponents
        w_new = w + Δθ[1:K]
        γ_new = max.(γ + Δθ[K+1:2K], γ_floor)

        new_res² = compute_Jr!(J, r, w_new, γ_new)

        if new_res² < res²
            w, γ  = w_new, γ_new
            res²  = new_res²
            λ    *= 0.5          # relax damping on success
            if res² < best_res²
                best_res² = res²
                best_w    = copy(w)
                best_γ    = copy(γ)
            end
        else
            # Step rejected: increase damping, do not update parameters
            λ *= 4.0
            # Recompute J,r for current (w, γ)
            compute_Jr!(J, r, w, γ)
        end
    end

    perm = sortperm(best_γ)
    return SoEKernel(K, best_w[perm], best_γ[perm])
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §7  High-level driver                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    soe_approximation(Nx, Ny; K=10, L_frac=0.45, refine=true,
                      Nx_ref=nothing, Ny_ref=nothing) -> SoEKernel

Compute a K-term SoE approximation of the 2D OBC lattice Green's function.

Steps
─────
1. Compute G on the reference lattice (Nx_ref × Ny_ref, default Nx × Ny).
2. Build the 1-D kernel g(d) = ⟨G(n,m)⟩ at each Manhattan distance d.
3. Fit K exponentials via the Matrix Pencil method.
4. (Optional) Refine with Gauss–Newton to reduce the nonlinear LS residual.

Parameters
──────────
- K        : number of SoE terms (default 10)
- L_frac   : Matrix Pencil pencil-parameter fraction (default 0.45)
- refine   : whether to apply Gauss–Newton refinement (default true)
- Nx_ref, Ny_ref : reference lattice for kernel extraction.  Using a larger
  lattice reduces OBC boundary contamination.  Defaults to Nx × Ny.

Returns a SoEKernel and prints a brief quality summary.
"""
function soe_approximation(
        Nx::Int, Ny::Int;
        K::Int              = 10,
        L_frac::Float64     = 0.45,
        refine::Bool        = true,
        Nx_ref::Union{Int,Nothing} = nothing,
        Ny_ref::Union{Int,Nothing} = nothing)

    Nxr = isnothing(Nx_ref) ? Nx : Nx_ref
    Nyr = isnothing(Ny_ref) ? Ny : Ny_ref

    d_vals, g_mean, _ = greens_kernel_by_distance(Nxr, Nyr)
    N = length(d_vals)

    # Ensure we have enough points for the fit
    K_use = K
    if N < 2*K_use + 1
        K_use = (N - 1) ÷ 2
        @warn "Only $N distance samples available; reducing K from $K to $K_use."
    end

    soe = fit_soe_matrix_pencil(d_vals, g_mean, K_use; L_frac)

    if refine
        soe = fit_soe_refine(soe, d_vals, g_mean)
    end

    return soe
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §8  Approximation quality metrics                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    soe_quality(soe, d_vals, g_true) -> (ε_max, ε_rms)

Compute maximum and RMS relative errors of the SoE approximation:

    ε_rel(d) = |soe(d) − g_true(d)| / max(|g_true(d)|, floor)

where `floor = 1e-15` prevents division by zero for near-zero g values.

Returns
───────
- ε_max :: Float64   maximum relative error over all d in d_vals
- ε_rms :: Float64   root-mean-square relative error
"""
function soe_quality(
        soe::SoEKernel,
        d_vals::AbstractVector,
        g_true::AbstractVector{Float64})

    N    = length(d_vals)
    errs = Vector{Float64}(undef, N)
    for (i, d) in enumerate(d_vals)
        errs[i] = abs(soe(d) - g_true[i]) / max(abs(g_true[i]), 1e-15)
    end
    return maximum(errs), sqrt(sum(errs.^2) / N)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §9  Test suite                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    run_soe_tests()

Test suite covering:
  T1. Laplacian positive-definiteness (several lattice sizes).
  T2. Analytic spectral data — orthonormality + G reconstruction.
  T3. Kernel extraction — positivity, monotonicity, values in expected range.
  T4. Matrix Pencil on a synthetic exact-SoE signal (K=4, noiseless).
  T5. Full pipeline fit quality (K=10) on a 15×15 lattice.
  T6. Gauss–Newton refinement reduces (or maintains) the residual.
  T7. All fitted exponents γⱼ are positive.
"""
function run_soe_tests()
    println("=" ^ 70)
    println("  SoE Green's Function Test Suite")
    println("=" ^ 70)

    # ── T1: Positive-definiteness of M ───────────────────────────────────────
    @testset "T1  Laplacian positive-definite" begin
        for (Nx, Ny) in [(3,3), (4,5), (6,6), (8,4)]
            M, λ_min, λ_max, κ = verify_laplacian_nonsingular(Nx, Ny)
            @test λ_min > 0.0
            @test size(M) == (Nx*Ny, Nx*Ny)
            @printf("  [%d×%d]  λ_min = %.4e   λ_max = %.4e   κ = %.2e\n",
                    Nx, Ny, λ_min, λ_max, κ)
        end
    end

    # ── T2: Spectral decomposition ────────────────────────────────────────────
    @testset "T2  Spectral decomposition" begin
        Nx, Ny = 6, 5
        λx, λy, Φx, Φy, Λ = spectral_data(Nx, Ny)

        # All 2-D eigenvalues positive
        @test minimum(Λ) > 0.0

        # Eigenvector orthonormality  Φᵀ Φ = I
        @test norm(Φx' * Φx - I(Nx)) < 1e-10
        @test norm(Φy' * Φy - I(Ny)) < 1e-10

        # Spectral reconstruction of G matches generate_greens_function
        G_spec = zeros(Nx, Ny, Nx, Ny)
        for ky in 1:Ny, kx in 1:Nx
            inv_Λ = 1.0 / Λ[kx, ky]
            for jy in 1:Ny, jx in 1:Nx
                φjx_φjy = Φx[jx,kx] * Φy[jy,ky]
                for iy in 1:Ny, ix in 1:Nx
                    G_spec[ix,iy,jx,jy] += Φx[ix,kx]*Φy[iy,ky] * inv_Λ * φjx_φjy
                end
            end
        end
        _, G_tens = generate_greens_function(Nx, Ny)
        rel_err = norm(G_spec - G_tens) / norm(G_tens)
        @test rel_err < 1e-10
        @printf("  Spectral vs direct  ‖ΔG‖/‖G‖ = %.2e\n", rel_err)
    end

    # ── T3: Kernel extraction ─────────────────────────────────────────────────
    @testset "T3  Kernel g(d) properties" begin
        Nx, Ny = 10, 10
        d_vals, g_mean, g_spread = greens_kernel_by_distance(Nx, Ny)

        # g(d) must be positive (G is positive definite → diagonal dominates average)
        @test all(g_mean .> 0.0)

        # g(d) should decrease with distance (Green's function decays)
        # Check the mean trend: g_mean[1] > g_mean[end]
        @test g_mean[1] > last(g_mean)

        # Spread should be non-negative
        @test all(g_spread .>= 0.0)

        @printf("  [10×10]  g(0) = %.5f   g(1) = %.5f   g(%d) = %.5f\n",
                g_mean[1], g_mean[2], last(d_vals), last(g_mean))
        @printf("           spread(0) = %.5f   spread(1) = %.5f\n",
                g_spread[1], g_spread[2])
    end

    # ── T4: Matrix Pencil on a synthetic exact-SoE signal ─────────────────────
    @testset "T4  Matrix Pencil — synthetic exact signal" begin
        # Ground-truth SoE: 4 decaying exponentials
        w_true = [2.0, -0.5, 0.8, 0.3]
        γ_true = [0.05, 0.20, 0.60, 1.50]
        K_syn  = 4
        d_syn  = collect(0:40)
        g_syn  = [sum(w_true[j]*exp(-γ_true[j]*d) for j in 1:K_syn) for d in d_syn]

        soe_syn = fit_soe_matrix_pencil(d_syn, g_syn, K_syn; L_frac = 0.45)

        # Approximation error should be near machine precision on noiseless data
        ε_max, ε_rms = soe_quality(soe_syn, d_syn, g_syn)
        @test ε_max < 1e-6
        @printf("  Synthetic K=4  max-rel-err = %.2e   RMS-rel-err = %.2e\n",
                ε_max, ε_rms)
    end

    # ── T5: Full pipeline fit quality (K=10) ─────────────────────────────────
    @testset "T5  SoE fit quality K=10 on 15×15" begin
        Nx, Ny = 15, 15
        d_vals, g_mean, _ = greens_kernel_by_distance(Nx, Ny)

        soe_mp  = fit_soe_matrix_pencil(d_vals, g_mean, 10)
        ε_max_mp,  ε_rms_mp  = soe_quality(soe_mp,  d_vals, g_mean)

        soe_ref = fit_soe_refine(soe_mp, d_vals, g_mean)
        ε_max_ref, ε_rms_ref = soe_quality(soe_ref, d_vals, g_mean)

        @printf("  Matrix Pencil only:   max-rel-err = %.2e   RMS = %.2e\n",
                ε_max_mp, ε_rms_mp)
        @printf("  After GN refinement:  max-rel-err = %.2e   RMS = %.2e\n",
                ε_max_ref, ε_rms_ref)

        # After refinement: max relative error < 5 %
        @test ε_max_ref < 0.05
        # Refinement should not worsen the initial fit
        @test ε_rms_ref <= ε_rms_mp + 1e-10
    end

    # ── T6: Refinement reduces (or maintains) the residual ────────────────────
    @testset "T6  Refinement monotone" begin
        d_vals, g_mean, _ = greens_kernel_by_distance(12, 12)
        soe0 = fit_soe_matrix_pencil(d_vals, g_mean, 8)
        soe1 = fit_soe_refine(soe0, d_vals, g_mean)

        res0 = sum((soe0(d) - g_mean[i])^2 for (i,d) in enumerate(d_vals))
        res1 = sum((soe1(d) - g_mean[i])^2 for (i,d) in enumerate(d_vals))
        @test res1 <= res0 + 1e-10 * res0
    end

    # ── T7: All exponents positive ────────────────────────────────────────────
    @testset "T7  Exponents positive" begin
        for (Nx, Ny) in [(8,8), (10,12), (15,15)]
            soe = soe_approximation(Nx, Ny; K = 8, refine = true)
            @test all(soe.γ .> 0.0)
        end
    end

    println("=" ^ 70)
    println("  All SoE tests passed.")
    println("=" ^ 70)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §10 Demo / entry point                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if abspath(PROGRAM_FILE) == @__FILE__
    run_soe_tests()

    println()
    println("─" ^ 70)
    println("  Demo: SoE approximation for 14×14 lattice, K=10")
    println("─" ^ 70)

    Nx, Ny = 14, 14
    soe = soe_approximation(Nx, Ny; K = 10, refine = true)
    println()
    println(soe)

    d_vals, g_mean, g_spread = greens_kernel_by_distance(Nx, Ny)
    ε_max, ε_rms = soe_quality(soe, d_vals, g_mean)

    @printf("\nApproximation quality on %d×%d lattice (K=%d):\n", Nx, Ny, soe.K)
    @printf("  max relative error : %.3e\n", ε_max)
    @printf("  RMS relative error : %.3e\n", ε_rms)

    println("\nPoint-by-point comparison (g_true vs g_SoE):")
    println("    d     g_true      g_SoE      spread    rel_err")
    println("  ────────────────────────────────────────────────────")
    for (i, d) in enumerate(d_vals)
        g_approx = soe(d)
        rel_err  = abs(g_approx - g_mean[i]) / max(abs(g_mean[i]), 1e-15)
        @printf("  %3d   %9.6f   %9.6f   %8.5f   %.2e\n",
                d, g_mean[i], g_approx, g_spread[i], rel_err)
    end
end
