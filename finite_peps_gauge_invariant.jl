#= ═══════════════════════════════════════════════════════════════════════════════
   finite_peps_gauge_invariant.jl

   Gauge-invariant (charge-resolved) finite PEPS for the U(1) LGT — "v3".

   ── Idea ──────────────────────────────────────────────────────────────────────
   Each PEPS virtual leg carries a definite electric-flux quantum number.  The
   horizontal bond (ix,iy)–(ix+1,iy) carries the flux e_R of node (ix,iy); the
   vertical bond (ix,iy)–(ix,iy+1) carries the flux e_U of node (ix,iy).  A node
   tensor T[p,l,r,u,d], with physical p = |n_f, e_R, e_U⟩, is non-zero only when

       (copy)   flux(r) = e_R          flux(u) = e_U
       (Gauss)  e_R + e_U − flux(l) − flux(d) = n_f + q_bg(ix,iy)

   where flux(l), flux(d) are the fluxes carried by the left/down bonds (= the
   right/up fluxes of the neighbouring nodes) and q_bg = g_charges(ix,iy) is the
   static background charge.  Boundary (missing) links carry flux 0.

   This is exactly the ED Gauss law
       E_R(i) − E_R(i−x̂) + E_U(i) − E_U(i−ŷ) − n_f(i) = g_charges(i)
   imposed as a *block-sparsity* pattern on every tensor.

   ── Why it helps ──────────────────────────────────────────────────────────────
   • Total fermion number is conserved exactly (boundary fluxes pinned to 0), so
     the ⟨n_f⟩ ≈ 0.5 drift seen in v2 vanishes by construction.
   • Every SVD truncation is done block-by-block within a flux sector, so the
     truncation can never leak weight across charge sectors — it only discards
     directions that genuinely carry small amplitude.
   • The bond carries (flux, multiplicity), so matter entanglement is captured by
     the per-sector multiplicity while the gauge flux is tracked exactly.

   ── Storage ───────────────────────────────────────────────────────────────────
   A `GaugePEPS` wraps an ordinary dense `FinitePEPS` (so all of the existing
   measurement code — expect_site, measure_all_finite, bond_entropy — works
   unchanged) plus two charge tables:
       qh[ix,iy] :: Vector{Int}   flux label of every state on horizontal bond
       qv[ix,iy] :: Vector{Int}   flux label of every state on vertical   bond
   Gauge-forbidden tensor entries are kept *exactly* zero.

   Requires: finite_peps_ground_state.jl (FinitePEPS, site_dims, embeddings, …)
             finite_peps_quench.jl        (measure_all_finite, bond_entropy, …)
   ═══════════════════════════════════════════════════════════════════════════ =#

using LinearAlgebra
using Printf

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Physical-index decode (inverse of site_idx)                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    decode_phys(p, d_gR, d_gU, dg) → (nf, eR, eU)

Inverse of `site_idx`: map a 1-based physical index `p` of a node with local
gauge dims `(d_gR, d_gU)` back to its quantum numbers `(n_f, e_R, e_U)`.
Missing links (d_g* == 1) decode to flux 0.
"""
@inline function decode_phys(p::Int, d_gR::Int, d_gU::Int, dg::Int)
    p0  = p - 1
    nf  = p0 ÷ (d_gR * d_gU)
    rem = p0 % (d_gR * d_gU)
    iR1 = rem ÷ d_gU          # 0-based right-gauge index
    iU0 = rem % d_gU          # 0-based up-gauge index
    eR  = (d_gR == 1) ? 0 : (iR1 - dg)
    eU  = (d_gU == 1) ? 0 : (iU0 - dg)
    return nf, eR, eU
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  GaugePEPS data structure                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
GaugePEPS wraps a dense `FinitePEPS` with per-bond flux labels.

    peps       :: FinitePEPS         dense masked tensors + Vidal weights
    qh[ix,iy]  :: Vector{Int}        flux of each state on horizontal bond, len = |λh|
    qv[ix,iy]  :: Vector{Int}        flux of each state on vertical   bond, len = |λv|
    g_charges  :: Matrix{Int}        static background charge per site
    dg         :: Int
"""
mutable struct GaugePEPS
    peps      :: FinitePEPS
    qh        :: Matrix{Vector{Int}}    # [nx-1, ny]
    qv        :: Matrix{Vector{Int}}    # [nx, ny-1]
    g_charges :: Matrix{Int}            # [nx, ny]
    dg        :: Int
end

nx(g::GaugePEPS) = g.peps.nx
ny(g::GaugePEPS) = g.peps.ny

"""Flux carried by the left bond of node (ix,iy) state index `l` (0 at boundary)."""
@inline _flux_l(g::GaugePEPS, ix, iy, l) = (ix > 1)          ? g.qh[ix-1, iy][l] : 0
@inline _flux_r(g::GaugePEPS, ix, iy, r) = (ix < nx(g))      ? g.qh[ix,   iy][r] : 0
@inline _flux_u(g::GaugePEPS, ix, iy, u) = (iy < ny(g))      ? g.qv[ix,   iy][u] : 0
@inline _flux_d(g::GaugePEPS, ix, iy, d) = (iy > 1)          ? g.qv[ix, iy-1][d] : 0

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Gauss-law mask                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    gauss_mask(gpeps, ix, iy) → BitArray{5}

Boolean tensor over (p, l, r, u, d): `true` exactly where the entry is allowed
by the copy + Gauss constraints for node (ix, iy).
"""
function gauss_mask(g::GaugePEPS, ix::Int, iy::Int)
    nxv, nyv = nx(g), ny(g)
    dg = g.dg
    d_phys, d_gR, d_gU = site_dims(ix, iy, nxv, nyv, dg)
    D_l, D_r, D_u, D_d = vdims(g.peps, ix, iy)
    qbg = g.g_charges[ix, iy]

    mask = falses(d_phys, D_l, D_r, D_u, D_d)
    for d in 1:D_d
        f_d = _flux_d(g, ix, iy, d)
        for u in 1:D_u
            f_u = _flux_u(g, ix, iy, u)
            for r in 1:D_r
                f_r = _flux_r(g, ix, iy, r)
                for l in 1:D_l
                    f_l = _flux_l(g, ix, iy, l)
                    for p in 1:d_phys
                        nf, eR, eU = decode_phys(p, d_gR, d_gU, dg)
                        ok = (f_r == eR) && (f_u == eU) &&
                             (eR + eU - f_l - f_d == nf + qbg)
                        ok && (mask[p, l, r, u, d] = true)
                    end
                end
            end
        end
    end
    return mask
end

"""Zero out every gauge-forbidden entry of tensor (ix,iy) in place."""
function apply_mask!(g::GaugePEPS, ix::Int, iy::Int)
    mask = gauss_mask(g, ix, iy)
    T = g.peps.tensors[ix, iy]
    @inbounds for i in eachindex(T)
        mask[i] || (T[i] = 0)
    end
    return nothing
end

"""
    forbidden_norm(gpeps) → Float64

Frobenius norm of all gauge-forbidden tensor entries, summed over the lattice.
A faithful gauge-invariant state has this exactly 0 (up to round-off).
"""
function forbidden_norm(g::GaugePEPS)
    tot = 0.0
    for iy in 1:ny(g), ix in 1:nx(g)
        mask = gauss_mask(g, ix, iy)
        T = g.peps.tensors[ix, iy]
        @inbounds for i in eachindex(T)
            mask[i] || (tot += abs2(T[i]))
        end
    end
    return sqrt(tot)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Initialisation — staggered product state on the gauge-invariant manifold ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    init_gauge_peps(nx, ny, dg, g_charges; noise=0.0)

Initialise in the staggered product state: every bond has dimension 1 carrying
flux 0, and each site holds the unique state allowed by Gauss's law at zero
flux, namely n_f = −g_charges(ix,iy), e_R = e_U = 0.

`noise` perturbs the (single) allowed amplitude; bond growth is gate-driven, so
noise mainly lifts accidental degeneracies and is optional.
"""
function init_gauge_peps(nx::Int, ny::Int, dg::Int, g_charges::Matrix{Int};
                          noise::Float64=0.0)
    tensors = Matrix{Array{ComplexF64,5}}(undef, nx, ny)
    λh = Matrix{Vector{Float64}}(undef, nx-1, ny)
    λv = Matrix{Vector{Float64}}(undef, nx, ny-1)
    qh = Matrix{Vector{Int}}(undef, nx-1, ny)
    qv = Matrix{Vector{Int}}(undef, nx, ny-1)

    for iy in 1:ny, ix in 1:nx-1
        λh[ix, iy] = [1.0]; qh[ix, iy] = [0]
    end
    for iy in 1:ny-1, ix in 1:nx
        λv[ix, iy] = [1.0]; qv[ix, iy] = [0]
    end

    for iy in 1:ny, ix in 1:nx
        d_phys, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        arr = zeros(ComplexF64, d_phys, 1, 1, 1, 1)
        nf_init = -g_charges[ix, iy]                 # Gauss at zero flux
        @assert 0 ≤ nf_init ≤ 1 "init: n_f=$nf_init out of range at ($ix,$iy)"
        idx = site_idx(nf_init, 0, 0, d_gR, d_gU, dg)
        arr[idx, 1, 1, 1, 1] = 1.0
        tensors[ix, iy] = arr
    end

    peps = FinitePEPS(tensors, λh, λv, nx, ny)
    g = GaugePEPS(peps, qh, qv, g_charges, dg)

    if noise > 0
        for iy in 1:ny, ix in 1:nx
            T = g.peps.tensors[ix, iy]
            T .+= noise .* randn(ComplexF64, size(T))
            apply_mask!(g, ix, iy)                  # keep noise gauge-invariant
            nrm = sqrt(sum(abs2, T))
            nrm > 1e-15 && (T ./= nrm)
        end
    end
    return g
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Charge-resolved (block-diagonal) SVD                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

const GI_REG = 1e-12     # per-sector singular-value floor

"""
    charge_svd(Θ, row_charge, col_charge, D_max; reg=GI_REG)
        → (U, S, Vt, bond_charge)

Block-diagonal SVD of `Θ` (nrow×ncol) where rows/cols carry the conserved bond
flux `row_charge` / `col_charge`.  Each flux sector is decomposed independently;
the largest `D_max` singular values *across all sectors* are kept.  Returns the
assembled `U (nrow×Dk)`, `S (Dk)`, `Vt (Dk×ncol)` and the flux label of each
retained bond state.
"""
function charge_svd(Θ::AbstractMatrix{ComplexF64},
                    row_charge::Vector{Int}, col_charge::Vector{Int},
                    D_max::Int; reg::Float64=GI_REG)
    nrow, ncol = size(Θ)
    sectors = sort(collect(intersect(Set(row_charge), Set(col_charge))))

    # Per-candidate (compact) storage: short singular vectors + their index sets.
    u_short = Vector{Vector{ComplexF64}}()   # length |Rs|
    v_short = Vector{Vector{ComplexF64}}()   # length |Cs|
    rows_of = Vector{Vector{Int}}()
    cols_of = Vector{Vector{Int}}()
    svals   = Float64[]
    charges = Int[]

    for s in sectors
        Rs = findall(==(s), row_charge)
        Cs = findall(==(s), col_charge)
        (isempty(Rs) || isempty(Cs)) && continue
        F = svd(Θ[Rs, Cs])
        # Globally we keep ≤ D_max, so never more than D_max from one sector.
        kmax = min(D_max, length(F.S))
        for k in 1:kmax
            F.S[k] ≤ reg && continue
            push!(u_short, F.U[:, k]); push!(v_short, F.Vt[k, :])
            push!(rows_of, Rs);        push!(cols_of, Cs)
            push!(svals, F.S[k]);      push!(charges, s)
        end
    end

    isempty(svals) && error("charge_svd: no singular values above reg in any sector")

    keep = sortperm(svals; rev=true)[1:min(D_max, length(svals))]
    Dk = length(keep)

    U  = zeros(ComplexF64, nrow, Dk)
    Vt = zeros(ComplexF64, Dk, ncol)
    S  = zeros(Float64, Dk)
    bond_charge = zeros(Int, Dk)
    for (j, idx) in enumerate(keep)
        @inbounds U[rows_of[idx], j]  .= u_short[idx]
        @inbounds Vt[j, cols_of[idx]] .= v_short[idx]
        S[j] = svals[idx]
        bond_charge[j] = charges[idx]
    end
    return U, S, Vt, bond_charge
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Charge-resolved simple-update bond updates                              ║
# ║                                                                          ║
# ║  Weight absorb/peel mirrors finite_peps_full_update.jl; the dense SVD is  ║
# ║  replaced by charge_svd and the new tensors are re-masked.               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    simple_update_bond_h!(gpeps, ix, iy, gate, D_max)

Charge-resolved horizontal simple-update of bond (ix,iy)–(ix+1,iy).
The new bond carries flux = e_R of node (ix,iy).
"""
function simple_update_bond_h!(g::GaugePEPS, ix::Int, iy::Int,
                                gate::AbstractMatrix, D_max::Int)
    peps = g.peps
    nxv, nyv, dg = nx(g), ny(g), g.dg
    apply_mask!(g, ix, iy); apply_mask!(g, ix+1, iy)

    T_L = peps.tensors[ix,   iy]
    T_R = peps.tensors[ix+1, iy]
    dp_L = size(T_L, 1); dp_R = size(T_R, 1)
    _, d_gR_L, d_gU_L = site_dims(ix,   iy, nxv, nyv, dg)
    _, d_gR_R, d_gU_R = site_dims(ix+1, iy, nxv, nyv, dg)

    λ_bond = peps.λh[ix, iy]; bond_dim = length(λ_bond)

    λ_l   = (ix > 1)         ? peps.λh[ix-1, iy]   : ones(1)
    λ_r   = (ix+1 < nxv)     ? peps.λh[ix+1, iy]   : ones(1)
    λ_u_L = (iy < nyv)       ? peps.λv[ix,   iy]   : ones(1)
    λ_d_L = (iy > 1)         ? peps.λv[ix, iy-1]   : ones(1)
    λ_u_R = (iy < nyv)       ? peps.λv[ix+1, iy]   : ones(1)
    λ_d_R = (iy > 1)         ? peps.λv[ix+1, iy-1] : ones(1)

    sq_bond = sqrt.(λ_bond)
    sq_l   = sqrt.(λ_l);    sq_r   = sqrt.(λ_r)
    sq_u_L = sqrt.(λ_u_L);  sq_d_L = sqrt.(λ_d_L)
    sq_u_R = sqrt.(λ_u_R);  sq_d_R = sqrt.(λ_d_R)

    # Absorb sqrt weights
    L_w = copy(T_L)
    for p in 1:dp_L, l in 1:size(L_w,2), r in 1:bond_dim, u in 1:size(L_w,4), d in 1:size(L_w,5)
        L_w[p,l,r,u,d] *= sq_l[l] * sq_bond[r] * sq_u_L[u] * sq_d_L[d]
    end
    R_w = copy(T_R)
    for p in 1:dp_R, l in 1:bond_dim, r in 1:size(R_w,3), u in 1:size(R_w,4), d in 1:size(R_w,5)
        R_w[p,l,r,u,d] *= sq_bond[l] * sq_r[r] * sq_u_R[u] * sq_d_R[d]
    end

    sz_L = size(L_w); sz_R = size(R_w)
    L_perm = permutedims(L_w, (1,2,4,5,3))
    L_mat  = reshape(L_perm, dp_L*sz_L[2]*sz_L[4]*sz_L[5], bond_dim)
    R_perm = permutedims(R_w, (2,1,3,4,5))
    R_mat  = reshape(R_perm, bond_dim, dp_R*sz_R[3]*sz_R[4]*sz_R[5])
    Θ_raw  = L_mat * R_mat

    Θ = reshape(Θ_raw, dp_L, sz_L[2], sz_L[4], sz_L[5], dp_R, sz_R[3], sz_R[4], sz_R[5])
    sz = size(Θ)

    # Apply gate (pR fast, pL slow — matches kron(H_L,H_R))
    Θ_pp = permutedims(Θ, (5,1,2,3,4,6,7,8))
    Θ_pp_mat = reshape(Θ_pp, dp_R*dp_L, sz[2]*sz[3]*sz[4]*sz[6]*sz[7]*sz[8])
    Θ_new_pp_mat = gate * Θ_pp_mat
    Θ_new_pp = reshape(Θ_new_pp_mat, dp_R, dp_L, sz[2], sz[3], sz[4], sz[6], sz[7], sz[8])
    Θ_new = permutedims(Θ_new_pp, (2,3,4,5,1,6,7,8))

    Θ_mat = reshape(Θ_new, dp_L*sz[2]*sz[3]*sz[4], dp_R*sz[6]*sz[7]*sz[8])
    nrow, ncol = size(Θ_mat)

    # ── Sector labels: bond flux s = e_R of the left node ──────────────────
    row_charge = Vector{Int}(undef, nrow)
    for r0 in 0:nrow-1
        pL = (r0 % dp_L) + 1
        _, eR, _ = decode_phys(pL, d_gR_L, d_gU_L, dg)
        row_charge[r0+1] = eR
    end
    col_charge = Vector{Int}(undef, ncol)
    for c0 in 0:ncol-1
        pR    = (c0 % dp_R) + 1
        rest  = c0 ÷ dp_R                        # (rR, uR, dR), rR fastest
        dR0   = (rest ÷ sz[6]) ÷ sz[7]           # drop rR then uR → dR (0-based)
        f_dR  = (iy > 1) ? g.qv[ix+1, iy-1][dR0+1] : 0
        nfR, eRR, eUR = decode_phys(pR, d_gR_R, d_gU_R, dg)
        col_charge[c0+1] = eRR + eUR - f_dR - nfR - g.g_charges[ix+1, iy]
    end

    # Defensive: kill any entry that bridges different sectors (gate is
    # block-diagonal in s, so these should already be ~0).
    @inbounds for c in 1:ncol, r in 1:nrow
        row_charge[r] == col_charge[c] || (Θ_mat[r, c] = 0)
    end

    U, S_tr, Vt, new_charge = charge_svd(Θ_mat, row_charge, col_charge, D_max)
    D_keep = length(S_tr)
    λ_new = S_tr ./ max(norm(S_tr), 1e-15)
    sqS = sqrt.(S_tr)

    L_raw = reshape(U * Diagonal(sqS), dp_L, sz[2], sz[3], sz[4], D_keep)
    L_new = permutedims(L_raw, (1,2,5,3,4))
    R_tmp = reshape(Diagonal(sqS) * Vt, D_keep, dp_R, sz[6], sz[7], sz[8])
    R_new = permutedims(R_tmp, (2,1,3,4,5))

    isq_l   = 1.0 ./ (sq_l   .+ FU_REG)
    isq_u_L = 1.0 ./ (sq_u_L .+ FU_REG); isq_d_L = 1.0 ./ (sq_d_L .+ FU_REG)
    isq_r   = 1.0 ./ (sq_r   .+ FU_REG)
    isq_u_R = 1.0 ./ (sq_u_R .+ FU_REG); isq_d_R = 1.0 ./ (sq_d_R .+ FU_REG)

    for p in 1:dp_L, l in 1:sz[2], r in 1:D_keep, u in 1:sz[3], d in 1:sz[4]
        L_new[p,l,r,u,d] *= isq_l[l] * isq_u_L[u] * isq_d_L[d]
    end
    for p in 1:dp_R, l in 1:D_keep, r in 1:sz[6], u in 1:sz[7], d in 1:sz[8]
        R_new[p,l,r,u,d] *= isq_r[r] * isq_u_R[u] * isq_d_R[d]
    end

    peps.tensors[ix,   iy] = L_new
    peps.tensors[ix+1, iy] = R_new
    peps.λh[ix, iy] = λ_new
    g.qh[ix, iy]    = new_charge

    apply_mask!(g, ix, iy); apply_mask!(g, ix+1, iy)
    return nothing
end

"""
    simple_update_bond_v!(gpeps, ix, iy, gate, D_max)

Charge-resolved vertical simple-update of bond (ix,iy)[down]–(ix,iy+1)[up].
The new bond carries flux = e_U of node (ix,iy).
"""
function simple_update_bond_v!(g::GaugePEPS, ix::Int, iy::Int,
                                gate::AbstractMatrix, D_max::Int)
    peps = g.peps
    nxv, nyv, dg = nx(g), ny(g), g.dg
    apply_mask!(g, ix, iy); apply_mask!(g, ix, iy+1)

    T_D = peps.tensors[ix, iy]
    T_U = peps.tensors[ix, iy+1]
    dp_D = size(T_D, 1); dp_U = size(T_U, 1)
    _, d_gR_D, d_gU_D = site_dims(ix, iy,   nxv, nyv, dg)
    _, d_gR_U, d_gU_U = site_dims(ix, iy+1, nxv, nyv, dg)

    λ_bond = peps.λv[ix, iy]; bond_dim = length(λ_bond)

    λ_l_D = (ix > 1)       ? peps.λh[ix-1, iy]   : ones(1)
    λ_r_D = (ix < nxv)     ? peps.λh[ix,   iy]   : ones(1)
    λ_d_D = (iy > 1)       ? peps.λv[ix, iy-1]   : ones(1)
    λ_l_U = (ix > 1)       ? peps.λh[ix-1, iy+1] : ones(1)
    λ_r_U = (ix < nxv)     ? peps.λh[ix,   iy+1] : ones(1)
    λ_u_U = (iy+1 < nyv)   ? peps.λv[ix, iy+1]   : ones(1)

    sq_bond = sqrt.(λ_bond)
    sq_l_D = sqrt.(λ_l_D); sq_r_D = sqrt.(λ_r_D); sq_d_D = sqrt.(λ_d_D)
    sq_l_U = sqrt.(λ_l_U); sq_r_U = sqrt.(λ_r_U); sq_u_U = sqrt.(λ_u_U)

    D_w = copy(T_D)
    for p in 1:dp_D, l in 1:size(D_w,2), r in 1:size(D_w,3), u in 1:bond_dim, d in 1:size(D_w,5)
        D_w[p,l,r,u,d] *= sq_l_D[l] * sq_r_D[r] * sq_bond[u] * sq_d_D[d]
    end
    U_w = copy(T_U)
    for p in 1:dp_U, l in 1:size(U_w,2), r in 1:size(U_w,3), u in 1:size(U_w,4), d in 1:bond_dim
        U_w[p,l,r,u,d] *= sq_l_U[l] * sq_r_U[r] * sq_u_U[u] * sq_bond[d]
    end

    sz_D = size(D_w); sz_U = size(U_w)
    D_perm = permutedims(D_w, (1,2,3,5,4))
    D_mat  = reshape(D_perm, dp_D*sz_D[2]*sz_D[3]*sz_D[5], bond_dim)
    U_perm = permutedims(U_w, (5,1,2,3,4))
    U_mat  = reshape(U_perm, bond_dim, dp_U*sz_U[2]*sz_U[3]*sz_U[4])
    Θ_raw  = D_mat * U_mat

    Θ = reshape(Θ_raw, dp_D, sz_D[2], sz_D[3], sz_D[5], dp_U, sz_U[2], sz_U[3], sz_U[4])
    szΘ = size(Θ)

    Θ_pp = permutedims(Θ, (5,1,2,3,4,6,7,8))
    Θ_pp_mat = reshape(Θ_pp, dp_U*dp_D, szΘ[2]*szΘ[3]*szΘ[4]*szΘ[6]*szΘ[7]*szΘ[8])
    Θ_new_pp_mat = gate * Θ_pp_mat
    Θ_new_pp = reshape(Θ_new_pp_mat, dp_U, dp_D, szΘ[2], szΘ[3], szΘ[4], szΘ[6], szΘ[7], szΘ[8])
    Θ_new = permutedims(Θ_new_pp, (2,3,4,5,1,6,7,8))

    Θ_mat = reshape(Θ_new, dp_D*szΘ[2]*szΘ[3]*szΘ[4], dp_U*szΘ[6]*szΘ[7]*szΘ[8])
    nrow, ncol = size(Θ_mat)

    # ── Sector labels: bond flux s = e_U of the down node ──────────────────
    row_charge = Vector{Int}(undef, nrow)
    for r0 in 0:nrow-1
        pD = (r0 % dp_D) + 1
        _, _, eU = decode_phys(pD, d_gR_D, d_gU_D, dg)
        row_charge[r0+1] = eU
    end
    col_charge = Vector{Int}(undef, ncol)
    for c0 in 0:ncol-1
        pU   = (c0 % dp_U) + 1
        rest = c0 ÷ dp_U
        lU0  = rest % szΘ[6]
        # (rest÷szΘ[6]) holds (rU0, uU0) — unused for the sector label
        f_lU = (ix > 1) ? g.qh[ix-1, iy+1][lU0+1] : 0
        nfU, eRU, eUU = decode_phys(pU, d_gR_U, d_gU_U, dg)
        col_charge[c0+1] = eRU + eUU - f_lU - nfU - g.g_charges[ix, iy+1]
    end

    @inbounds for c in 1:ncol, r in 1:nrow
        row_charge[r] == col_charge[c] || (Θ_mat[r, c] = 0)
    end

    U, S_tr, Vt, new_charge = charge_svd(Θ_mat, row_charge, col_charge, D_max)
    D_keep = length(S_tr)
    λ_new = S_tr ./ max(norm(S_tr), 1e-15)
    sqS = sqrt.(S_tr)

    D_raw = reshape(U * Diagonal(sqS), dp_D, szΘ[2], szΘ[3], szΘ[4], D_keep)
    D_new = permutedims(D_raw, (1,2,3,5,4))
    U_tmp = reshape(Diagonal(sqS) * Vt, D_keep, dp_U, szΘ[6], szΘ[7], szΘ[8])
    U_new = permutedims(U_tmp, (2,3,4,5,1))

    isq_l_D = 1.0 ./ (sq_l_D .+ FU_REG); isq_r_D = 1.0 ./ (sq_r_D .+ FU_REG)
    isq_d_D = 1.0 ./ (sq_d_D .+ FU_REG)
    isq_l_U = 1.0 ./ (sq_l_U .+ FU_REG); isq_r_U = 1.0 ./ (sq_r_U .+ FU_REG)
    isq_u_U = 1.0 ./ (sq_u_U .+ FU_REG)

    for p in 1:dp_D, l in 1:szΘ[2], r in 1:szΘ[3], u in 1:D_keep, d in 1:szΘ[4]
        D_new[p,l,r,u,d] *= isq_l_D[l] * isq_r_D[r] * isq_d_D[d]
    end
    for p in 1:dp_U, l in 1:szΘ[6], r in 1:szΘ[7], u in 1:szΘ[8], d in 1:D_keep
        U_new[p,l,r,u,d] *= isq_l_U[l] * isq_r_U[r] * isq_u_U[u]
    end

    peps.tensors[ix, iy]   = D_new
    peps.tensors[ix, iy+1] = U_new
    peps.λv[ix, iy] = λ_new
    g.qv[ix, iy]    = new_charge

    apply_mask!(g, ix, iy); apply_mask!(g, ix, iy+1)
    return nothing
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Trotter step + ITE driver                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Second-order Trotter ITE step (H half, V full, H half) — charge-resolved."""
function trotter_step_gauge!(g::GaugePEPS; gcoup::Float64, t_hop::Float64,
                              m::Float64, τ::Float64, D_max::Int)
    nxv, nyv, dg = nx(g), ny(g), g.dg
    for iy in 1:nyv, ix in 1:nxv-1
        Hh = H_merged_h_site(ix, iy, nxv, nyv, dg; g=gcoup, t=t_hop, m=m)
        simple_update_bond_h!(g, ix, iy, exp(-(τ/2) .* Hh), D_max)
    end
    for iy in 1:nyv-1, ix in 1:nxv
        Hv = H_merged_v_site(ix, iy, nxv, nyv, dg; g=gcoup, t=t_hop, m=m)
        simple_update_bond_v!(g, ix, iy, exp(-τ .* Hv), D_max)
    end
    for iy in 1:nyv, ix in nxv-1:-1:1
        Hh = H_merged_h_site(ix, iy, nxv, nyv, dg; g=gcoup, t=t_hop, m=m)
        simple_update_bond_h!(g, ix, iy, exp(-(τ/2) .* Hh), D_max)
    end
    return nothing
end

"""
    ite_ground_state_v3(nx, ny, dg, D_max; g, t_hop, m, g_charges,
                        n_ite, noise, verbose) → GaugePEPS

Gauge-invariant imaginary-time ground-state search with τ annealing.
Bond fluxes are tracked exactly; total fermion number is conserved.
"""
function ite_ground_state_v3(nx::Int, ny::Int, dg::Int, D_max::Int;
                              g::Float64, t_hop::Float64, m::Float64,
                              g_charges::Matrix{Int},
                              n_ite::Int=600, noise::Float64=0.1,
                              verbose::Bool=true)
    gp = init_gauge_peps(nx, ny, dg, g_charges; noise=noise)

    s1 = div(n_ite, 8); s2 = div(n_ite, 8); s3 = div(n_ite, 4)
    s4 = div(n_ite, 6); s5 = div(n_ite, 6)
    s6 = n_ite - (s1+s2+s3+s4+s5)
    τ_stages = [(0.1,s1), (0.05,s2), (0.02,s3), (0.01,s4), (0.005,s5), (0.001,s6)]

    step_total = 0
    for (τ, n_stage) in τ_stages
        for _ in 1:n_stage
            step_total += 1
            trotter_step_gauge!(gp; gcoup=g, t_hop=t_hop, m=m, τ=τ, D_max=D_max)
            if verbose && (step_total % 50 == 0 || step_total == 1)
                nf_val = mean_nf(gp.peps, nx, ny, dg)
                D_now  = maximum(length(gp.peps.λh[ix, iy])
                                 for iy in 1:ny for ix in 1:nx-1; init=1)
                viol = forbidden_norm(gp)
                @printf("    ITE step %4d / %d (τ=%.4f):  ⟨n_f⟩ = %.5f  D = %d  gauge_viol = %.1e\n",
                        step_total, n_ite, τ, nf_val, D_now, viol)
                flush(stdout)
            end
        end
    end
    return gp
end
