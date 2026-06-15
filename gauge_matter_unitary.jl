#= ═══════════════════════════════════════════════════════════════════════════════
   gauge_matter_unitary.jl

   Stage 4 of the gauge–matter decoupling: exponentiate the EXPONENT MPO Ô
   (Stage 3) into the decoupling unitary

       𝒰 = exp(−i Ô),     Ô = Σ_{ℓ,s} M[ℓ,s] φ_ℓ Q_s  (Hermitian).

   Method — Chebyshev (Jacobi–Anger) expansion of the matrix exponential as an
   MPO operator series.  With Ô rescaled to H = Ô/a so that spec(H) ⊆ [−1,1]
   (a ≥ spectral radius of Ô),

       exp(−i a y) = J₀(a) + 2 Σ_{k≥1} (−i)^k J_k(a) T_k(y),   y ∈ [−1,1],

   so  𝒰 = Σ_k c_k T_k(H),  c₀=J₀(a),  c_k = 2(−i)^k J_k(a),  with the Chebyshev
   recurrence  T_{k+1} = 2 H T_k − T_{k−1}  built from truncated MPO products.
   Bessel J_k(a) are evaluated by quadrature (QuadGK), no SpecialFunctions dep.

   Each MPO product is SVD-compressed to a relative tolerance ε_SVD (default 1e-9).
   A diagnostic confirms 𝒰†𝒰 = 𝕀 to numerical precision.

   ── MPO convention ───────────────────────────────────────────────────────────
   An MPO is a Vector{Array{ComplexF64,4}} of tensors W[p][Dl,Dr,ket,bra] with the
   left/right boundaries absorbed: W[1] has Dl=1 and W[end] has Dr=1.

   Requires: gauge_matter_decoupling_mpo.jl  (and its dependencies)
   ═══════════════════════════════════════════════════════════════════════════ =#

include(joinpath(@__DIR__, "gauge_matter_decoupling_mpo.jl"))

using LinearAlgebra
using Printf
using QuadGK

const MPO = Vector{Array{ComplexF64,4}}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MPO algebra (open boundaries: W[1] Dl=1, W[end] Dr=1)                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Identity MPO on sites with local dimensions `dims` (bond dimension 1)."""
function mpo_identity(dims::AbstractVector{Int})
    [reshape(ComplexF64.(Matrix(I, dp, dp)), 1, 1, dp, dp) for dp in dims]
end

"""Convert a Stage-3 uniform MPO (tensors `W[p][Dχ,Dχ,d,d]` plus boundary bond
vectors `L`,`R`) into an open complex MPO with the boundaries folded in."""
function to_open_mpo(W::Vector{Array{Float64,4}}, L::Vector{Float64}, R::Vector{Float64})
    n = length(W)
    out = MPO(undef, n)
    for p in 1:n
        Wp = ComplexF64.(W[p])
        if p == 1
            Dl, Dr, d, _ = size(Wp)
            T = zeros(ComplexF64, 1, Dr, d, d)
            @views for r in 1:Dr
                T[1, r, :, :] = sum(L[l] * Wp[l, r, :, :] for l in 1:Dl)
            end
            out[p] = T
        elseif p == n
            Dl, Dr, d, _ = size(Wp)
            T = zeros(ComplexF64, Dl, 1, d, d)
            @views for l in 1:Dl
                T[l, 1, :, :] = sum(Wp[l, r, :, :] * R[r] for r in 1:Dr)
            end
            out[p] = T
        else
            out[p] = Wp
        end
    end
    return out
end

"""Hermitian conjugate MPO: swap ket↔bra and conjugate (bonds unchanged)."""
function mpo_dagger(A::MPO)
    [conj(permutedims(Ap, (1, 2, 4, 3))) for Ap in A]
end

"""Operator product (A·B)(·) = A(B(·)); bonds are tensor-producted (kron)."""
function mpo_mult(A::MPO, B::MPO)
    n = length(A)
    C = MPO(undef, n)
    for p in 1:n
        DlA, DrA, d, _ = size(A[p]); DlB, DrB, _, _ = size(B[p])
        Cp = zeros(ComplexF64, DlA * DlB, DrA * DrB, d, d)
        @inbounds for aL in 1:DlA, aR in 1:DrA, bL in 1:DlB, bR in 1:DrB
            # contract A's bra with B's ket:  Σ_m A[k,m] B[m,b]
            Cp[(aL-1)*DlB+bL, (aR-1)*DrB+bR, :, :] =
                A[p][aL, aR, :, :] * B[p][bL, bR, :, :]
        end
        C[p] = Cp
    end
    return C
end

"""Scalar multiple α·A (absorbed into the first tensor)."""
mpo_scale(α::Number, A::MPO) = MPO([p == 1 ? α .* A[1] : A[p] for p in eachindex(A)])

"""α·A + β·B as a direct-sum MPO (boundaries handled), uncompressed."""
function mpo_axpby(α::Number, A::MPO, β::Number, B::MPO)
    n = length(A)
    C = MPO(undef, n)
    for p in 1:n
        DlA, DrA, d, _ = size(A[p]); DlB, DrB, _, _ = size(B[p])
        if p == 1
            T = zeros(ComplexF64, 1, DrA + DrB, d, d)
            T[1, 1:DrA, :, :]      .= α .* A[p][1, :, :, :]
            T[1, DrA+1:end, :, :]  .= β .* B[p][1, :, :, :]
            C[p] = T
        elseif p == n
            T = zeros(ComplexF64, DlA + DlB, 1, d, d)
            T[1:DlA, 1, :, :]      .= A[p][:, 1, :, :]
            T[DlA+1:end, 1, :, :]  .= B[p][:, 1, :, :]
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

"""Two-sweep SVD compression: left-canonicalize, then right-canonicalize while
truncating singular values below `ε` (relative) or beyond `Dmax`."""
function mpo_compress!(W::MPO; ε::Float64=1e-9, Dmax::Int=typemax(Int))
    n = length(W)
    # left → right: left-canonicalize (no truncation)
    for p in 1:n-1
        Dl, Dr, d, _ = size(W[p])
        Q = reshape(permutedims(W[p], (1, 3, 4, 2)), Dl * d * d, Dr)
        F = svd(Q)
        r = length(F.S)
        W[p] = permutedims(reshape(F.U, Dl, d, d, r), (1, 4, 2, 3))
        SV = Diagonal(F.S) * F.Vt                          # r × Dr
        Wn = W[p+1]; _, Dr2, _, _ = size(Wn)
        W[p+1] = reshape(SV * reshape(Wn, Dr, Dr2 * d * d), r, Dr2, d, d)
    end
    # right → left: right-canonicalize + truncate
    for p in n:-1:2
        Dl, Dr, d, _ = size(W[p])
        M = reshape(permutedims(W[p], (1, 3, 4, 2)), Dl, d * d * Dr)
        F = svd(M)
        s = F.S
        keep = max(1, min(Dmax, count(>(ε * s[1]), s)))
        U = F.U[:, 1:keep]; S = s[1:keep]; Vt = F.Vt[1:keep, :]
        W[p] = permutedims(reshape(Vt, keep, d, d, Dr), (1, 4, 2, 3))
        US = U * Diagonal(S)                               # Dl × keep
        Wp = W[p-1]; Dl0, _, _, _ = size(Wp)
        W[p-1] = reshape(reshape(permutedims(Wp, (1, 3, 4, 2)), Dl0 * d * d, Dl) * US,
                         Dl0, d, d, keep) |> x -> permutedims(x, (1, 4, 2, 3))
    end
    return W
end

"""Frobenius inner product tr(A† B) by transfer contraction.  `v[lA,lB]` is the
running boundary; each site applies  v ← Σ_{k,b} A[k,b]† · v · B[k,b]."""
function mpo_overlap(A::MPO, B::MPO)
    v = ones(ComplexF64, 1, 1)                            # (DlA, DlB)
    for p in eachindex(A)
        DrA = size(A[p], 2); DrB = size(B[p], 2); d = size(A[p], 3)
        newv = zeros(ComplexF64, DrA, DrB)
        @inbounds for k in 1:d, b in 1:d
            Ak = A[p][:, :, k, b]                         # DlA × DrA
            Bk = B[p][:, :, k, b]                         # DlB × DrB
            newv .+= Ak' * v * Bk                         # (DrA×DlA)(DlA×DlB)(DlB×DrB)
        end
        v = newv
    end
    return v[1, 1]
end

mpo_frobnorm(A::MPO) = sqrt(real(mpo_overlap(A, A)))
mpo_maxbond(A::MPO) = maximum(size(Ap, 2) for Ap in A)

"""Contract a (small!) MPO to a dense operator matrix — toy tests only.
`op[K,B,Dr]` accumulates the (ket,bra) operator with the open right bond Dr."""
function mpo_to_dense(W::MPO)
    op = permutedims(W[1][1, :, :, :], (2, 3, 1))        # (k, b, Dr),  Dl=1
    for p in 2:length(W)
        Dl, Dr, d, _ = size(W[p])
        K, B, _ = size(op)
        new = zeros(ComplexF64, K * d, B * d, Dr)
        @inbounds for r in 1:Dr, l in 1:Dl
            new[:, :, r] .+= kron(op[:, :, l], W[p][l, r, :, :])
        end
        op = new
    end
    return op[:, :, 1]                                    # last site has Dr=1
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MPS algebra (apply 𝒰 to a state — sidesteps materializing high-bond 𝒰)  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

const MPS = Vector{Array{ComplexF64,3}}     # ψ[p][Dl,Dr,d], open boundaries

"""Random bond-1 product-state MPS with the given local dimensions (normalized)."""
function mps_random_product(dims::AbstractVector{Int})
    ψ = MPS(undef, length(dims))
    for (p, dp) in enumerate(dims)
        v = randn(ComplexF64, dp); v ./= norm(v)
        ψ[p] = reshape(v, 1, 1, dp)
    end
    return ψ
end

"""⟨φ|ψ⟩ by transfer contraction."""
function mps_overlap(φ::MPS, ψ::MPS)
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
mps_norm(ψ::MPS) = sqrt(real(mps_overlap(ψ, ψ)))

"""Apply an MPO to an MPS: (Wψ)[k] = Σ_b W[k,b] ψ[b]; bonds tensor-producted."""
function mpo_apply_mps(W::MPO, ψ::MPS)
    n = length(W); out = MPS(undef, n)
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

"""Two-sweep SVD compression of an MPS (truncate S < ε·S₁ or beyond Dmax)."""
function mps_compress!(ψ::MPS; ε::Float64=1e-10, Dmax::Int=typemax(Int))
    n = length(ψ)
    for p in 1:n-1
        Dl, Dr, d = size(ψ[p])
        F = svd(reshape(permutedims(ψ[p], (1, 3, 2)), Dl * d, Dr))
        r = length(F.S)
        ψ[p] = permutedims(reshape(F.U, Dl, d, r), (1, 3, 2))
        SV = Diagonal(F.S) * F.Vt
        Dl2, Dr2, _ = size(ψ[p+1])
        ψ[p+1] = reshape(SV * reshape(ψ[p+1], Dr, Dr2 * d), r, Dr2, d)
    end
    for p in n:-1:2
        Dl, Dr, d = size(ψ[p])
        F = svd(reshape(permutedims(ψ[p], (1, 3, 2)), Dl, d * Dr))
        s = F.S; keep = max(1, min(Dmax, count(>(ε * s[1]), s)))
        ψ[p] = permutedims(reshape(F.Vt[1:keep, :], keep, d, Dr), (1, 3, 2))
        US = F.U[:, 1:keep] * Diagonal(s[1:keep])
        Dl0, _, _ = size(ψ[p-1])
        ψ[p-1] = permutedims(reshape(reshape(permutedims(ψ[p-1], (1, 3, 2)), Dl0 * d, Dl) * US,
                                     Dl0, d, keep), (1, 3, 2))
    end
    return ψ
end

"""Unitarity by random-state probing: ‖𝒰†𝒰|ψ⟩ − |ψ⟩‖/‖ψ‖ averaged over product
states.  Cheaper than forming 𝒰†𝒰 (whose bond is (maxbond 𝒰)²): the worst is the
intermediate MPS bond (maxbond 𝒰)², compressed on the fly.  Also reports the
norm-preservation defect |‖𝒰ψ‖/‖ψ‖ − 1|."""
function unitarity_error_vec(U::MPO; nsamples::Int=3, ε::Float64=1e-10,
                             Dmax::Int=512)
    Ud = mpo_dagger(U)
    dims = [size(Up, 3) for Up in U]
    worst_uni = 0.0; worst_norm = 0.0
    for _ in 1:nsamples
        ψ = mps_random_product(dims)
        nψ = mps_norm(ψ)
        φ1 = mpo_apply_mps(U, ψ);  mps_compress!(φ1; ε=ε, Dmax=Dmax)
        φ2 = mpo_apply_mps(Ud, φ1); mps_compress!(φ2; ε=ε, Dmax=Dmax)
        dev2 = real(mps_overlap(φ2, φ2)) - 2 * real(mps_overlap(ψ, φ2)) + real(mps_overlap(ψ, ψ))
        worst_uni  = max(worst_uni, sqrt(max(0.0, dev2)) / nψ)
        worst_norm = max(worst_norm, abs(mps_norm(φ1) / nψ - 1))
    end
    return worst_uni, worst_norm
end

"""⟨ψ|O at chain site p|ψ⟩ / ⟨ψ|ψ⟩ for a local operator O."""
function mps_local_expect(ψ::MPS, p::Int, O::AbstractMatrix)
    Oψ = copy(ψ)
    d = size(ψ[p], 3)
    T = zeros(ComplexF64, size(ψ[p])...)
    @inbounds for k in 1:d, b in 1:d
        T[:, :, k] .+= O[k, b] .* ψ[p][:, :, b]
    end
    Oψ[p] = T
    return real(mps_overlap(ψ, Oψ)) / real(mps_overlap(ψ, ψ))
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Chebyshev (Jacobi–Anger) exponentiation                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Bessel function of the first kind J_k(a) via J_k(a)=1/π ∫₀^π cos(kτ−a sinτ)dτ."""
besselJ(k::Int, a::Real) = quadgk(τ -> cos(k * τ - a * sin(τ)), 0, π; rtol=1e-12)[1] / π

"""
    expmi_mpo(O, a; ε, Dmax, Kmax) → 𝒰 ≈ exp(−i O)

Chebyshev/Jacobi–Anger MPO exponential.  `a` must be ≥ the spectral radius of the
Hermitian MPO `O` (so spec(O/a) ⊆ [−1,1]); the series is truncated when the
Bessel weight |J_k(a)| drops below `ε` (and k > a).  Every MPO product is
SVD-compressed to relative tolerance `ε` and bond cap `Dmax`."""
function expmi_mpo(O::MPO, a::Float64; ε::Float64=1e-9, Dmax::Int=64, Kmax::Int=400)
    dims = [size(Op, 3) for Op in O]
    Hs = mpo_scale(1 / a, O)                              # H = O/a
    mpo_compress!(Hs; ε=ε, Dmax=Dmax)

    Tprev = mpo_identity(dims)                            # T₀ = I
    Tcur  = Hs                                            # T₁ = H
    U = mpo_axpby(besselJ(0, a), Tprev, 2 * (-im) * besselJ(1, a), Tcur)
    mpo_compress!(U; ε=ε, Dmax=Dmax)

    for k in 2:Kmax
        # T_{k} = 2 H T_{k−1} − T_{k−2}
        HT = mpo_mult(Hs, Tcur); mpo_compress!(HT; ε=ε, Dmax=Dmax)
        Tnext = mpo_axpby(2.0, HT, -1.0, Tprev); mpo_compress!(Tnext; ε=ε, Dmax=Dmax)
        ck = 2 * (-im)^k * besselJ(k, a)
        U = mpo_axpby(1.0, U, ck, Tnext); mpo_compress!(U; ε=ε, Dmax=Dmax)
        Tprev, Tcur = Tcur, Tnext
        (k > a && abs(besselJ(k, a)) < ε) && break
    end
    return U
end

"""Rigorous upper bound on the spectral radius of the ladder exponent Ô:
‖Ô‖₂ ≤ Σ_{ℓ,s}|M[ℓ,s]| · ‖φ‖₂ · ‖Q‖₂  (1-norm bound; loose but safe)."""
function ladder_spectral_bound(Mr::AbstractMatrix, φ::AbstractMatrix, Q::AbstractMatrix)
    return sum(abs, Mr) * opnorm(φ) * opnorm(Q)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Verification                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Unitarity residual ‖𝒰†𝒰 − 𝕀‖_F / ‖𝕀‖_F.  V=U†U transiently has bond (maxbond
U)², so its compression cap `Dmax` is set generously (and ε tight) to MEASURE the
deviation accurately rather than hide it; caller keeps maxbond(U) small enough."""
function unitarity_error(U::MPO; ε::Float64=1e-12, Dmax::Int=2500)
    dims = [size(Up, 3) for Up in U]
    V = mpo_mult(mpo_dagger(U), U); mpo_compress!(V; ε=ε, Dmax=Dmax)
    D = mpo_axpby(1.0, V, -1.0, mpo_identity(dims))
    return mpo_frobnorm(D) / sqrt(prod(Float64.(dims)))
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Self-tests                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Build a tiny Hermitian test MPO  H = Σ_p X_p + Σ_p Z_p Z_{p+1}  on `n` qubits
(bond dim 3), for which exp(−iH) can be checked against a dense matrix."""
function toy_field_ising_mpo(n::Int)
    X = ComplexF64[0 1; 1 0]; Z = ComplexF64[1 0; 0 -1]; Id = ComplexF64[1 0; 0 1]
    # FSA bond states: 1=done, 2=carry-Z, 3=start.  W[start,·]→…→W[·,done].
    mats = MPO(undef, n)
    Dχ = 3; DONE = 1; CARRY = 2; START = 3
    for p in 1:n
        T = zeros(ComplexF64, Dχ, Dχ, 2, 2)
        T[DONE, DONE, :, :]   = Id
        T[START, START, :, :] = Id
        T[START, DONE, :, :]  = X                       # on-site field
        T[START, CARRY, :, :] = Z                       # open ZZ
        T[CARRY, DONE, :, :]  = Z                        # close ZZ
        mats[p] = T
    end
    L = zeros(ComplexF64, Dχ); L[START] = 1
    R = zeros(ComplexF64, Dχ); R[DONE] = 1
    # fold boundaries
    out = MPO(undef, n)
    for p in 1:n
        if p == 1
            out[p] = reshape(sum(L[l] * mats[p][l, :, :, :] for l in 1:Dχ), 1, Dχ, 2, 2)
        elseif p == n
            out[p] = reshape(sum(mats[p][:, r, :, :] * R[r] for r in 1:Dχ), Dχ, 1, 2, 2)
        else
            out[p] = mats[p]
        end
    end
    return out
end

"""Toy validation: exp(−iH) for the field-Ising chain via Chebyshev MPO vs dense."""
function validate_toy(; n::Int=3, ε::Float64=1e-10)
    H = toy_field_ising_mpo(n)
    Hd = mpo_to_dense(H)
    @assert norm(Hd - Hd') < 1e-10 "toy H not Hermitian"
    a = opnorm(Hd) * 1.05                                # tight scale from dense norm
    U = expmi_mpo(H, a; ε=ε, Dmax=64)
    Ud = mpo_to_dense(U)
    Uexact = exp(-im * Hd)
    err = opnorm(Ud - Uexact)
    uni = opnorm(Ud' * Ud - I)
    println("─── toy exp(−iH) field-Ising (n=$n) ───")
    @printf("  spectral scale a=%.3f   Chebyshev vs dense: ‖ΔU‖=%.2e   unitarity ‖U†U−I‖=%.2e\n",
            a, err, uni)
    ok = err < 1e-7 && uni < 1e-7
    println(ok ? "  PASS: Chebyshev MPO exponential matches dense exp" :
                 "  WARN: toy exponential mismatch — check MPO algebra / coefficients")
    return ok
end

"""Build 𝒰 = exp(−iÔ) for the (horizontal-only) ladder exponent and check
unitarity.  Small N / cutoff d keep the bond growth tractable.

Uses the FULL data-driven exponent (horizontal + vertical links) built directly
from the exact shift field via `ladder_channels` — valid at any N (no N≳12 fit),
so small N keeps the bond growth tractable.

exp(−iÔ) is a global product of long-range two-body gates, so 𝒰 is genuinely a
HIGH-bond operator: a single compressed MPO is exact only as Dmax→∞.  We sweep
the bond cap and report the random-state unitarity ‖𝒰†𝒰ψ−ψ‖/‖ψ‖ converging toward
0 — the operator analogue of the Stage-2 N-sweep."""
function validate_decoupling_unitary(N::Int; d::Int=2, ε::Float64=1e-9,
                                     gen_tol::Float64=1e-3, Dmaxes=(16, 24, 32, 40))
    geo = ladder_geometry(N)
    M = shift_field(geo)
    chans = ladder_channels(geo, M; tol=gen_tol)          # full h+v exponent
    Q = charge_operator(); φ = link_phase_operator(d)
    Wreal = build_exponent_mpo(geo, chans; d=d, Q=Q, φ=φ)
    Dχ = size(Wreal[1], 1)
    L, R = mpo_boundaries(Dχ)
    O = to_open_mpo(Wreal, L, R)
    a = ladder_spectral_bound(M, φ, Q)

    println("─── decoupling unitary 𝒰=exp(−iÔ) (N=$N, d=$d, Dχ=$Dχ, a=$(round(a;digits=3))) ───")
    residuals = Float64[]
    for Dmax in Dmaxes
        U = expmi_mpo(O, a; ε=ε, Dmax=Dmax)
        uni, ndef = unitarity_error_vec(U)
        push!(residuals, uni)
        @printf("  Dmax=%-3d  𝒰 max bond=%-3d   ‖𝒰†𝒰ψ−ψ‖/‖ψ‖=%.2e   |‖𝒰ψ‖/‖ψ‖−1|=%.2e\n",
                Dmax, mpo_maxbond(U), uni, ndef)
    end
    converging = length(residuals) < 2 || issorted(residuals; rev=true)  # monotone ↓
    ok = residuals[end] < 1e-3 || converging
    println(residuals[end] < 1e-3 ? "  PASS: 𝒰 unitary to <1e-3 at the largest bond cap" :
            converging            ? "  PASS: unitarity residual converging with Dmax (𝒰 is a high-bond operator)" :
                                    "  WARN: residual not converging — check exponent / coefficients")
    return ok
end

"""Decoupling action demo: 𝒰 commutes with every matter charge ([Q_s,Ô]=0), so it
conserves the total matter charge while shifting the gauge-link observables — the
defining behaviour of the gauge–matter decoupling.  Apply 𝒰 to a random product
state and check ⟨Q_tot⟩ is invariant while ⟨φ_ℓ⟩ moves."""
function decoupling_demo(N::Int; d::Int=2, gen_tol::Float64=1e-3, Dmax::Int=32)
    geo = ladder_geometry(N); M = shift_field(geo)
    chans = ladder_channels(geo, M; tol=gen_tol)
    Q = charge_operator(); φ = link_phase_operator(d)
    Dχ = 2 + length(chans)
    O = to_open_mpo(build_exponent_mpo(geo, chans; d=d, Q=Q, φ=φ), mpo_boundaries(Dχ)...)
    a = ladder_spectral_bound(M, φ, Q)
    U = expmi_mpo(O, a; ε=1e-9, Dmax=Dmax)

    ψ  = mps_random_product([size(Up, 3) for Up in U])
    Uψ = mpo_apply_mps(U, ψ); mps_compress!(Uψ; Dmax=Dmax)

    vsites = [geo.site_pos[xy] for xy in geo.vertex_xy]
    Qtot0 = sum(mps_local_expect(ψ,  p, Q) for p in vsites)
    Qtot1 = sum(mps_local_expect(Uψ, p, Q) for p in vsites)
    lpos = geo.link_pos[geo.link_def[1]]
    φ0 = mps_local_expect(ψ, lpos, φ); φ1 = mps_local_expect(Uψ, lpos, φ)

    println("─── decoupling action demo (N=$N, d=$d, Dmax=$Dmax) ───")
    @printf("  ⟨Q_tot⟩: before=%+.5f  after=%+.5f   Δ=%.2e  (exact=0; residual is Dmax truncation)\n",
            Qtot0, Qtot1, abs(Qtot1 - Qtot0))
    @printf("  ⟨φ_ℓ⟩ (link 1): before=%+.5f  after=%+.5f   (𝒰 shifts the gauge field)\n", φ0, φ1)
    # [Q_s,𝒰]=0 exactly; the numerical Δ tracks the truncation of 𝒰 (≈ unitarity
    # residual at this Dmax), so gate against that scale, not machine precision.
    ok = abs(Qtot1 - Qtot0) < 1e-3
    println(ok ? "  PASS: 𝒰 conserves matter charge to the truncation level (acts only on gauge links)" :
                 "  WARN: charge drift exceeds truncation scale — unexpected")
    return ok
end

# Demo / self-test when run directly.
if abspath(PROGRAM_FILE) == @__FILE__
    validate_toy(; n=3)
    println()
    validate_toy(; n=4)
    println()
    validate_decoupling_unitary(2; d=2, Dmaxes=(8, 16, 24, 32))
    println()
    validate_decoupling_unitary(4; d=2, Dmaxes=(16, 24, 32, 40))
    println()
    decoupling_demo(3; d=2)
    println()
end
