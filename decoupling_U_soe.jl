#= ═══════════════════════════════════════════════════════════════════════════════
   decoupling_U_soe.jl

   Approximated Bender–Zohar gauge–matter decoupling unitary for the pseudo-1D
   (snake-ordered) MPS, built from the SUM-OF-EXPONENTIALS approximation of the
   lattice Green's function.

   ── Construction (Form A: product of screened-string unitaries) ──────────────
   The exact decoupler is 𝒰 = exp(−iÔ), Ô = Σ_{ℓ,n} M_{ℓ,n} φ_ℓ Q_n, with the
   shift field M = ∇G (gradient of G = (−∇²)⁻¹).  Approximating
       G(n,m) ≈ Σ_{j=1}^K w_j e^{−γ_j(|Δx|+|Δy|)}
   gives M ≈ Σ_j M^{(j)}, each a single-decay "screened string".  Since all
   terms of Ô commute ([φ_ℓ Q_n, φ_ℓ' Q_n'] = 0),

       𝒰 = Π_{j=1}^K exp(−i Ô_j) · exp(−i Ô_bdry)        (exact product, no Trotter)

   where Ô_j = Σ M^{(j)} φ Q is the j-th screened string (low-bond MPO) and
   Ô_bdry = Σ_{edge (ℓ,n)} (M_exact − M_SoE) φ Q folds the exact field back in
   near the open boundary (where the bulk SoE is least accurate).

   Decisions implemented:  (1) snake-MPS, column-major snake;  (2) analytic SoE
   gradient;  (3) Form A;  (4) staggered Q, boundary terms folded in.

   ── Why this matters ─────────────────────────────────────────────────────────
   Each exp(−iÔ_j) has bond ∝ O(1) (single decay) instead of ∝ N, so 𝒰 is an
   efficient MPO operation usable on large lattices — unlike the exact decoupler
   (dense M), which is only feasible on tiny systems.

   Requires: mps_lgt.jl (MPS/MPO/HTerm), lgt_greens_soe.jl (SoE fit + exact G)
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "mps_lgt.jl"))
include(joinpath(@__DIR__, "lgt_greens_soe.jl"))

using LinearAlgebra
using Printf

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Chebyshev MPO exponential (Jacobi–Anger), CMPO types                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""All Bessel functions J_0(a),…,J_N(a) by Miller's stable downward recurrence
(robust for any a, unlike an oscillatory quadrature)."""
function besselJ_all(N::Int, a::Real)
    a == 0 && return Float64[k == 0 ? 1.0 : 0.0 for k in 0:N]
    Mtop = N + 25 + ceil(Int, abs(a))
    b = zeros(Float64, Mtop + 2)               # b[k+1] = J_k
    b[Mtop+1] = 1e-30
    for k in Mtop:-1:1
        b[k] = (2k / a) * b[k+1] - b[k+2]      # J_{k-1} = (2k/a)J_k − J_{k+1}
    end
    nrm = b[1] + 2 * sum(b[2k+1] for k in 1:(Mtop ÷ 2))   # J_0 + 2Σ J_{2k} = 1
    b ./= nrm
    return b[1:N+1]
end
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

"""exp(−i O) as a CMPO by SCALING-AND-SQUARING with a Taylor inner series.
Rescale so ‖O/2ˢ‖ < ½ (Taylor converges fast, no cancellation), then square s
times (each squaring preserves unitarity).  Far more stable under MPO truncation
than the Chebyshev three-term recurrence, which compounds compression errors."""
function expmi_mpo(O::CMPO, a::Float64; ε::Float64=1e-9, Dmax::Int=32, Mtaylor::Int=16)
    dims = [size(Op, 3) for Op in O]
    (a < 1e-12 || !isfinite(a)) && return mpo_identity_c(dims)      # O ≈ 0 ⇒ 𝒰 = I
    s = max(0, ceil(Int, log2(2 * a)))                             # 2ˢ ≥ 2a ⇒ ‖O/2ˢ‖ ≤ ½
    Os = mpo_scale_c(1 / 2.0^s, O); mpo_compress!(Os; ε=ε, Dmax=Dmax)
    U = mpo_identity_c(dims); term = mpo_identity_c(dims)
    for m in 1:Mtaylor                                             # exp(−iOs) = Σ (−iOs)ᵐ/m!
        term = mpo_scale_c(-im / m, mpo_mult_c(Os, term)); mpo_compress!(term; ε=ε, Dmax=Dmax)
        U = mpo_axpby(1.0, U, 1.0, term); mpo_compress!(U; ε=ε, Dmax=Dmax)
    end
    for _ in 1:s                                                   # square s times
        U = mpo_mult_c(U, U); mpo_compress!(U; ε=ε, Dmax=Dmax)
    end
    return U
end

function mps_to_dense(ψ::CMPS)
    P = permutedims(ψ[1][1, :, :], (2, 1))
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
# ║  Column-major snake  ("one step right, then all the way up/down")         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function column_snake(nx::Int, ny::Int)
    chain = Tuple{Int,Int}[]
    for ix in 1:nx
        ys = isodd(ix) ? (1:ny) : (ny:-1:1)
        for iy in ys
            push!(chain, (ix, iy))
        end
    end
    pos = Dict(s => p for (p, s) in enumerate(chain))
    return chain, pos
end

col_node_dims(nx, ny, dg) = [site_dims(ix, iy, nx, ny, dg)[1] for (ix, iy) in column_snake(nx, ny)[1]]

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Local operators and shift-field kernels                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

φ_op() = ComplexF64.(0.5 * (op_U_gauge(1) + op_U_gauge(1)'))     # ½(U+U†), dg=1
bg_stag(ix, iy) = isodd(ix + iy) ? 1.0 : 0.0

"""Exact shift field M_{ℓ,n} = G(a,n) − G(b,n) for link ℓ=(a→b), from exact G."""
function shift_exact(Gt, ix, iy, dir, jx, jy)
    bx, by = dir == :R ? (ix + 1, iy) : (ix, iy + 1)
    return Gt[ix, iy, jx, jy] - Gt[bx, by, jx, jy]
end

"""j-th SoE screened-string contribution to M (analytic gradient of the kernel)."""
function shift_soe_j(soe, j, ix, iy, dir, jx, jy)
    w = soe.w[j]; γ = soe.γ[j]
    ex(d) = exp(-γ * abs(d))
    if dir == :R
        return w * ex(iy - jy) * (ex(ix - jx) - ex(ix + 1 - jx))
    else
        return w * ex(ix - jx) * (ex(iy - jy) - ex(iy + 1 - jy))
    end
end
shift_soe_total(soe, ix, iy, dir, jx, jy) =
    sum(shift_soe_j(soe, j, ix, iy, dir, jx, jy) for j in 1:soe.K)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Exponent MPO  Ô = Σ M φ Q  from a shift-field function                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    build_O(nx, ny, dg, pos, dims, Mfun; tol) → (Ô::CMPO, Σ|M|)

Assemble Ô = Σ_{ℓ,n} Mfun(ℓ,n) φ_ℓ Q_n on the given snake (`pos`,`dims`).
`Mfun(ix,iy,dir,jx,jy)` returns the shift-field value for link (ix,iy,dir) and
charge site (jx,jy).  Q_n = n_f − staggered background."""
function build_O(nx, ny, dg, pos, dims, Mfun; tol::Float64=1e-12)
    φ = φ_op(); terms = HTerm[]; Mabs = 0.0
    Qnode(jx, jy) = ComplexF64.(embed_f_site(op_nf() - bg_stag(jx, jy) * _Id(LGT_d_f),
                                             site_dims(jx, jy, nx, ny, dg)[2:3]...))
    function addlink!(ix, iy, dir)
        _, dgR, dgU = site_dims(ix, iy, nx, ny, dg)
        φlink = dir == :R ? ComplexF64.(embed_R_site(φ, dgU)) :
                            ComplexF64.(embed_U_site(φ, dgR))
        a = pos[(ix, iy)]
        for jy in 1:ny, jx in 1:nx
            M = Mfun(ix, iy, dir, jx, jy)
            abs(M) < tol && continue
            Mabs += abs(M); b = pos[(jx, jy)]; Q = Qnode(jx, jy)
            push!(terms, a == b ? HTerm(M, Dict(a => φlink * Q)) :
                                  HTerm(M, Dict(a => copy(φlink), b => Q)))
        end
    end
    for iy in 1:ny, ix in 1:nx-1; addlink!(ix, iy, :R); end
    for iy in 1:ny-1, ix in 1:nx; addlink!(ix, iy, :U); end
    isempty(terms) && return mpo_identity_c(dims), 0.0
    return _assemble_mpo(dims, terms), Mabs
end

_spectral_bound(Mabs) = Mabs * opnorm(Matrix(φ_op())) * opnorm(op_nf() - 0.5 * _Id(LGT_d_f))

"""Is link (ix,iy,dir) or site within `bw` of the open boundary?"""
near_boundary(nx, ny, x, y, bw) = x ≤ bw || x > nx - bw || y ≤ bw || y > ny - bw

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Build the SoE decoupling unitary (Form A: list of factors)               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    build_decoupling_U_soe(nx, ny; dg, K, bw, Dmax) → (factors, soe, info)

Returns the decoupling unitary as a LIST of CMPO factors
`[𝒰_1, …, 𝒰_K, 𝒰_bdry]` (apply sequentially).  Each `𝒰_j = exp(−iÔ_j)` is a
single screened-string unitary (low bond); `𝒰_bdry` folds the exact field back
in within `bw` sites of the boundary."""
function build_decoupling_U_soe(nx::Int, ny::Int; dg::Int=1, K::Int=10, bw::Int=1,
                                Dmax::Int=32, Nx_ref::Int=8, Ny_ref::Int=8,
                                a_max::Float64=50.0)
    @assert dg == 1
    _, pos = column_snake(nx, ny)
    dims = col_node_dims(nx, ny, dg)
    # fit the SoE on a LARGER reference lattice so K isn't capped by the target's
    # few distance bins; the (w_j,γ_j) define the bulk kernel for any lattice.
    soe_full = soe_approximation(nx, ny; K=K, refine=true,
                                 Nx_ref=max(nx, Nx_ref), Ny_ref=max(ny, Ny_ref))
    _, Gt = generate_greens_function(nx, ny)

    # Build each screened-string factor, but SKIP channels whose spectral bound
    # exceeds a_max — an ill-conditioned fit can give large cancelling amplitudes
    # w_j, so an individual channel is a huge operator (slow/inaccurate to
    # exponentiate) even when the SUM is benign.  The used set is reused for the
    # boundary correction so M_SoE stays consistent.
    factors = CMPO[]; bonds = Int[]; used = Int[]
    for j in 1:soe_full.K
        Oj, Mabs = build_O(nx, ny, dg, pos, dims,
                           (ix, iy, dir, jx, jy) -> shift_soe_j(soe_full, j, ix, iy, dir, jx, jy))
        aj = _spectral_bound(Mabs)
        if !isfinite(aj) || aj > a_max
            @printf("  skip channel j=%d (γ=%.3f, a=%.1f > %.1f)\n", j, soe_full.γ[j], aj, a_max)
            continue
        end
        Uj = expmi_mpo(Oj, aj; Dmax=Dmax)
        mx = maximum(maximum(abs, W) for W in Uj)         # stability: 𝒰 must be ~O(1)
        if !isfinite(mx) || mx > 10.0
            @printf("  skip channel j=%d (γ=%.3f: Chebyshev unstable, max|𝒰|=%.1e)\n",
                    j, soe_full.γ[j], mx)
            continue
        end
        push!(factors, Uj); push!(bonds, maximum(size(W, 2) for W in Uj)); push!(used, j)
    end
    n_string = length(used)
    @printf("  SoE: used %d/%d channels;  γ = %s\n",
            n_string, soe_full.K, string(round.(soe_full.γ[used]; digits=3)))

    shift_soe_used(ix, iy, dir, jx, jy) =
        isempty(used) ? 0.0 : sum(shift_soe_j(soe_full, j, ix, iy, dir, jx, jy) for j in used)

    # boundary correction: (M_exact − M_SoE) on edge links/charges (used channels)
    function Mbdry(ix, iy, dir, jx, jy)
        bx, by = dir == :R ? (ix + 1, iy) : (ix, iy + 1)
        (near_boundary(nx, ny, ix, iy, bw) || near_boundary(nx, ny, bx, by, bw) ||
         near_boundary(nx, ny, jx, jy, bw)) || return 0.0
        return shift_exact(Gt, ix, iy, dir, jx, jy) - shift_soe_used(ix, iy, dir, jx, jy)
    end
    Obd, Mabs_bd = build_O(nx, ny, dg, pos, dims, Mbdry)
    a_bd = _spectral_bound(Mabs_bd); bdry_bond = 0
    if isfinite(a_bd) && a_bd ≤ a_max
        Ubd = expmi_mpo(Obd, a_bd; Dmax=Dmax)
        mxb = maximum(maximum(abs, W) for W in Ubd)
        if isfinite(mxb) && mxb ≤ 10.0
            push!(factors, Ubd); bdry_bond = maximum(size(W, 2) for W in Ubd); push!(bonds, bdry_bond)
        else
            @printf("  boundary factor unstable (max|𝒰|=%.1e) — dropped\n", mxb)
        end
    else
        @printf("  boundary a=%.1f > a_max — dropped (SoE_used too coarse)\n", a_bd)
    end
    isempty(factors) && (factors = CMPO[mpo_identity_c(dims)])   # nothing stable ⇒ 𝒰 = I

    info = (K=n_string, bonds=bonds, bdry_bond=bdry_bond, n_string=n_string)
    return factors, soe_full, info
end

"""Apply 𝒰 = Π factors to an MPS (sequential, zip-up compressed).  Factors
commute, so order is immaterial up to truncation."""
function apply_U(factors::Vector{CMPO}, ψ::CMPS; Dmax::Int=64, ε::Float64=1e-10)
    for U in factors
        ψ = mpo_apply_mps_zipup(U, ψ; Dmax=Dmax, ε=ε)
    end
    return ψ
end

"""Multiply the factor list into a single CMPO (for dense checks / small N)."""
function fold_factors(factors::Vector{CMPO}; Dmax::Int=64, ε::Float64=1e-10)
    U = factors[1]
    for k in 2:length(factors)
        U = mpo_mult_c(factors[k], U); mpo_compress!(U; ε=ε, Dmax=Dmax)
    end
    return U
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Self-test: unitarity + SoE-vs-exact decoupler vs K + bonds               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    validate_U_soe(; nx, ny, Ks, bw)

On a small lattice (dense-feasible): each factor is exactly unitary; the
SoE decoupler approaches the exact-M decoupler as K grows; the boundary factor
reduces the error; and the per-string bonds are O(1) ≪ the exact bond."""
function validate_U_soe(; nx::Int=2, ny::Int=2, Ks=(1, 2, 3), bw::Int=1)
    println("─── SoE decoupling unitary 𝒰 ($(nx)×$(ny), column snake) ───")
    _, pos = column_snake(nx, ny); dims = col_node_dims(nx, ny, 1)
    @printf("  full Hilbert dim = %d\n", prod(dims))

    # exact-M decoupler (reference), built on the SAME snake
    _, Gt = generate_greens_function(nx, ny)
    Oex, Mabs_ex = build_O(nx, ny, 1, pos, dims,
                           (ix, iy, dir, jx, jy) -> shift_exact(Gt, ix, iy, dir, jx, jy))
    Uex = expmi_mpo(Oex, _spectral_bound(Mabs_ex); Dmax=64)
    Uex_d = mpo_to_dense(Uex)
    @printf("  exact-M decoupler bond = %d\n", maximum(size(W, 2) for W in Uex))

    for K in Ks
        factors, soe, info = build_decoupling_U_soe(nx, ny; K=K, bw=bw)
        strings = factors[1:info.n_string]                 # first n_string are screened strings
        has_bd  = info.bdry_bond > 0
        for (tag, fac) in (("no-bdry", strings), ("+bdry", has_bd ? factors : strings))
            isempty(fac) && (@printf("  K=%-2d %-8s: no stable channels\n", K, tag); continue)
            Ud = mpo_to_dense(fold_factors(fac; Dmax=128))
            uni = opnorm(Ud' * Ud - I); err = opnorm(Ud - Uex_d)
            sb = info.n_string > 0 ? maximum(info.bonds[1:info.n_string]) : 0
            @printf("  K=%-2d %-8s: strings=%d  ‖𝒰−𝒰_exact‖=%.2e  ‖𝒰†𝒰−I‖=%.2e  max string bond=%d\n",
                    K, tag, info.n_string, err, uni, sb)
        end
    end
    println("  (string bonds are O(1) ≪ exact bond; +bdry should lower the error)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_U_soe(; nx=2, ny=2, Ks=(2, 4, 6))
    println()
    # Larger lattice: build only (no dense), report efficiency
    factors, soe, info = build_decoupling_U_soe(3, 4; K=10, bw=1)
    @printf("3×4 SoE decoupler: %d factors (%d strings + bdry), string bonds=%s, bdry bond=%d\n",
            length(factors), info.n_string, string(info.bonds[1:info.n_string]), info.bdry_bond)
end
