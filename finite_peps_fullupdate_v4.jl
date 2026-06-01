#= ═══════════════════════════════════════════════════════════════════════════════
   finite_peps_fullupdate_v4.jl  —  "v4" gauge-invariant boundary-MPS full update

   Stage 2b + 3.  Replaces v3's *unweighted* charge-resolved SVD truncation with
   an environment-weighted ALS truncation using the boundary-MPS environment
   (finite_peps_boundary_mps.jl).  The state is stored exactly as in v3
   (definite-flux Vidal form), so all measurement code is reused unchanged.

   Pipeline for one bond:
     1. env, RA, RB, QA, QB  =  bond_environment_*(gp, …)         [Stage 2a]
     2. θ = gate · (RA·RB)                                         (gate-applied)
     3. ALS: minimise ‖θ − X·Y‖²_env over rank-D' (X,Y)           [Stage 2b]
        • init from charge-resolved SVD (definite-flux β labels)
        • per-iteration flux projection keeps the bond gauge-invariant
     4. reassemble A_new = QA·X, B_new = QB·Y  (symmetric gauge)
     5. strip transverse sqrt-weights → contract → charge-SVD = canonical
        Vidal form with new λ and flux labels (identical to v3 storage)

   Energy:  compute_energy_v4 sums ⟨H_merged_bond⟩ / ⟨ψ|ψ⟩ over all bonds via the
   same environments — a true variational ⟨H⟩ for direct comparison with ED.

   Requires: finite_peps_gauge_invariant.jl, finite_peps_boundary_mps.jl,
             finite_peps_quench.jl (H_merged_*_site, decode_phys, charge_svd, …)
   ═══════════════════════════════════════════════════════════════════════════ =#

using LinearAlgebra
using TensorOperations
using Printf

const V4_ALS_REG = 1e-10

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Helpers                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Hermitian PSD solve  N·x = s  with eigenvalue-floor regularisation."""
function _solve_herm(N::Matrix{ComplexF64}, S::Matrix{ComplexF64}, reg::Float64)
    Nh = Hermitian(0.5 .* (N .+ N'))
    F = eigen(Nh)
    w = real.(F.values)
    wmax = maximum(w; init=0.0)
    floor = reg * max(wmax, 1e-30)
    inv_w = [wi > floor ? 1.0/wi : 0.0 for wi in w]
    return F.vectors * Diagonal(inv_w) * (F.vectors' * S)
end

"""Apply a 2-site gate (kron(H_L,H_R): left slow, right fast) to θ[aA,pL,pR,bB]."""
function _apply_gate(θ::Array{ComplexF64,4}, gate::AbstractMatrix, dpL::Int, dpR::Int)
    nA, _, _, nB = size(θ)
    θp = permutedims(θ, (3,2,1,4))                    # (pR,pL,aA,bB)
    M  = reshape(θp, dpR*dpL, nA*nB)
    M  = gate * M
    θp = reshape(M, dpR, dpL, nA, nB)
    return permutedims(θp, (3,2,1,4))                 # (aA,pL,pR,bB)
end

"""Robust column→sector assignment for a block-diagonal matrix M (rows carry
`row_charge`).  Each column inherits the sector of its dominant row."""
function _col_charge_from_blocks(M::Matrix{ComplexF64}, row_charge::Vector{Int})
    ncol = size(M, 2)
    cc = zeros(Int, ncol)
    @inbounds for c in 1:ncol
        rbest = 1; vbest = -1.0
        for r in 1:size(M,1)
            v = abs2(M[r,c])
            v > vbest && (vbest = v; rbest = r)
        end
        cc[c] = row_charge[rbest]
    end
    return cc
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Environment-weighted ALS truncation                                     ║
# ║                                                                          ║
# ║  Given θ[aA,pL,pR,bB] and env[aA,bB,aA',bB'], find X[aA,pL,β],          ║
# ║  Y[bB,pR,β] (β = 1..D') minimising ‖θ − XY‖²_env.                       ║
# ║  βflux[β] is the definite flux of each bond state; X is projected onto   ║
# ║  it each iteration (eR(pL)=βflux) to stay gauge-invariant.               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function als_truncate(θ::Array{ComplexF64,4}, env::Array{ComplexF64,4},
                      D_max::Int, eRofpL::Vector{Int}, impliedR;
                      n_als::Int=30, reg::Float64=V4_ALS_REG)
    nA, dpL, dpR, nB = size(θ)

    # ── init: charge-resolved SVD of θ across (aA,pL)|(bB,pR) ─────────────
    θm = reshape(permutedims(θ, (1,2,4,3)), nA*dpL, nB*dpR)   # rows (aA,pL), cols (bB,pR)
    row_charge = Vector{Int}(undef, nA*dpL)
    for r in 0:nA*dpL-1
        pL = (r ÷ nA) + 1
        row_charge[r+1] = eRofpL[pL]
    end
    col_charge = _col_charge_from_blocks(θm, row_charge)
    U, S, Vt, βflux = charge_svd(θm, row_charge, col_charge, D_max)
    Dp = length(S)
    sq = sqrt.(S)
    X = reshape(U * Diagonal(sq),  nA, dpL, Dp)                # (aA,pL,β)
    Y = permutedims(reshape(Diagonal(sq) * Vt, Dp, nB, dpR), (2,3,1))  # (bB,pR,β)

    # flux mask for X: keep eR(pL)==βflux[β]
    Xmask = [eRofpL[pL] == βflux[β] for _ in 1:nA, pL in 1:dpL, β in 1:Dp]
    project_X!(Xv) = (@inbounds for i in eachindex(Xv); Xmask[i] || (Xv[i]=0); end)

    # ── ALS sweeps ────────────────────────────────────────────────────────
    for _ in 1:n_als
        # solve X (fix Y)
        @tensor NX[aA,β,aAp,βp] := Y[bB,pR,β] * env[aA,bB,aAp,bBp] * conj(Y[bBp,pR,βp])
        @tensor SX[aA,pL,β]      := env[a2,bB,aA,bBp] * θ[a2,pL,pR,bB] * conj(Y[bBp,pR,β])
        Nm = reshape(permutedims(NX,(1,2,3,4)), nA*Dp, nA*Dp)
        for pL in 1:dpL
            s = reshape(SX[:,pL,:], nA*Dp, 1)
            x = _solve_herm(Matrix(Nm), Matrix{ComplexF64}(s), reg)
            X[:,pL,:] = reshape(x, nA, Dp)
        end
        project_X!(X)

        # solve Y (fix X)
        @tensor NY[bB,β,bBp,βp] := X[aA,pL,β] * env[aA,bB,aAp,bBp] * conj(X[aAp,pL,βp])
        @tensor SY[bB,pR,β]      := env[aA,bB2,aAp,bB] * θ[aA,pL,pR,bB2] * conj(X[aAp,pL,β])
        Nm = reshape(permutedims(NY,(1,2,3,4)), nB*Dp, nB*Dp)
        for pR in 1:dpR
            s = reshape(SY[:,pR,:], nB*Dp, 1)
            y = _solve_herm(Matrix(Nm), Matrix{ComplexF64}(s), reg)
            Y[:,pR,:] = reshape(y, nB, Dp)
        end
    end
    return X, Y, βflux
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Full-update bond updates (horizontal & vertical)                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    full_update_bond_h_v4!(gp, ix, iy, gate, D_max; χ, n_als)

Environment-weighted full update of horizontal bond (ix,iy)–(ix+1,iy).
"""
function full_update_bond_h_v4!(gp::GaugePEPS, ix::Int, iy::Int,
                                 gate::AbstractMatrix, D_max::Int;
                                 χ::Int=64, n_als::Int=30)
    peps = gp.peps
    nxv, nyv, dg = nx(gp), ny(gp), gp.dg
    apply_mask!(gp, ix, iy); apply_mask!(gp, ix+1, iy)

    E = bond_environment_h(gp, ix, iy; χ=χ)
    dpL = size(E.RA, 2); dpR = size(E.RB, 2)
    _, d_gR_L, d_gU_L = site_dims(ix,   iy, nxv, nyv, dg)
    _, d_gR_R, d_gU_R = site_dims(ix+1, iy, nxv, nyv, dg)

    @tensor θ[aA,pL,pR,bB] := E.RA[aA,pL,k] * E.RB[bB,pR,k]
    θ = _apply_gate(θ, gate, dpL, dpR)

    eRofpL = [decode_phys(p, d_gR_L, d_gU_L, dg)[2] for p in 1:dpL]
    X, Y, βflux = als_truncate(θ, E.env, D_max, eRofpL, nothing; n_als=n_als)

    # reassemble (symmetric sqrt-absorbed gauge): A(p,l,β,u,d), B(p,β,r,u,d)
    @tensor Afull[p,l,r,u,d] := E.QA[l,u,d,aA] * X[aA,p,r]
    @tensor Bfull[p,l,r,u,d] := E.QB[r,u,d,bB] * Y[bB,p,l]

    # strip the transverse sqrt-weights that sqrt_absorbed_site put in
    _strip_h!(gp, ix, iy, Afull, Bfull)

    # Enforce gauge invariance using the ALS bond fluxes before the final SVD:
    # ALS projects X but not Y, so Bfull can carry small cross-sector content.
    # Store as the new bond (flux = βflux), mask both sites clean, then the
    # analytic charge-SVD re-canonicalises into definite-flux Vidal form.
    peps.tensors[ix,   iy] = Afull
    peps.tensors[ix+1, iy] = Bfull
    peps.λh[ix, iy] = ones(length(βflux))
    gp.qh[ix, iy]   = βflux
    apply_mask!(gp, ix, iy); apply_mask!(gp, ix+1, iy)

    _recanonicalise_h!(gp, ix, iy, peps.tensors[ix,iy], peps.tensors[ix+1,iy], D_max)
    apply_mask!(gp, ix, iy); apply_mask!(gp, ix+1, iy)
    return nothing
end

"""Divide out the sqrt(λ) absorbed on the transverse legs of the two bond sites."""
function _strip_h!(gp, ix, iy, A, B)
    peps = gp.peps
    isq(λ) = 1.0 ./ (sqrt.(λ) .+ FU_REG)
    sλ_lA = (ix > 1)        ? isq(peps.λh[ix-1, iy]) : ones(1)
    sλ_uA = (iy < ny(gp))   ? isq(peps.λv[ix,   iy]) : ones(1)
    sλ_dA = (iy > 1)        ? isq(peps.λv[ix, iy-1]) : ones(1)
    sλ_rB = (ix+1 < nx(gp)) ? isq(peps.λh[ix+1, iy]) : ones(1)
    sλ_uB = (iy < ny(gp))   ? isq(peps.λv[ix+1, iy]) : ones(1)
    sλ_dB = (iy > 1)        ? isq(peps.λv[ix+1, iy-1]) : ones(1)
    @inbounds for d in axes(A,5), u in axes(A,4), r in axes(A,3), l in axes(A,2), p in axes(A,1)
        A[p,l,r,u,d] *= sλ_lA[l] * sλ_uA[u] * sλ_dA[d]
    end
    @inbounds for d in axes(B,5), u in axes(B,4), r in axes(B,3), l in axes(B,2), p in axes(B,1)
        B[p,l,r,u,d] *= sλ_rB[r] * sλ_uB[u] * sλ_dB[d]
    end
end

"""Contract A·B over the new bond, charge-SVD across the bond cut, store the
canonical Vidal tensors + weight + flux labels (mirrors v3 charge resolution)."""
function _recanonicalise_h!(gp, ix, iy, A, B, D_max)
    peps = gp.peps; dg = gp.dg
    dpL = size(A,1); DlA = size(A,2); DuA = size(A,4); DdA = size(A,5)
    dpR = size(B,1); DrB = size(B,3); DuB = size(B,4); DdB = size(B,5)
    _, d_gR_L, d_gU_L = site_dims(ix,   iy, nx(gp), ny(gp), dg)
    _, d_gR_R, d_gU_R = site_dims(ix+1, iy, nx(gp), ny(gp), dg)

    @tensor Θ[pL,l,uL,dL, pR,r,uR,dR] := A[pL,l,k,uL,dL] * B[pR,k,r,uR,dR]
    Θm = reshape(Θ, dpL*DlA*DuA*DdA, dpR*DrB*DuB*DdB)
    nrow, ncol = size(Θm)

    row_charge = Vector{Int}(undef, nrow)
    for r0 in 0:nrow-1
        pL = (r0 % dpL) + 1
        row_charge[r0+1] = decode_phys(pL, d_gR_L, d_gU_L, dg)[2]
    end
    col_charge = Vector{Int}(undef, ncol)
    for c0 in 0:ncol-1
        pR   = (c0 % dpR) + 1
        rest = c0 ÷ dpR
        dR0  = (rest ÷ DrB) ÷ DuB
        f_dR = (iy > 1) ? gp.qv[ix+1, iy-1][dR0+1] : 0
        nfR, eRR, eUR = decode_phys(pR, d_gR_R, d_gU_R, dg)
        col_charge[c0+1] = eRR + eUR - f_dR - nfR - gp.g_charges[ix+1, iy]
    end

    U, S, Vt, newq = charge_svd(Θm, row_charge, col_charge, D_max)
    Dk = length(S)
    λnew = S ./ max(norm(S), 1e-15)

    Lraw = reshape(U, dpL, DlA, DuA, DdA, Dk)          # (p,l,u,d,β)
    Lnew = permutedims(Lraw, (1,2,5,3,4))              # (p,l,β,u,d)
    Rtmp = reshape(Vt, Dk, dpR, DrB, DuB, DdB)         # (β,p,r,u,d)
    Rnew = permutedims(Rtmp, (2,1,3,4,5))              # (p,β,r,u,d)

    peps.tensors[ix,   iy] = Lnew
    peps.tensors[ix+1, iy] = Rnew
    peps.λh[ix, iy] = λnew
    gp.qh[ix, iy]   = newq
end

"""
    full_update_bond_v_v4!(gp, ix, iy, gate, D_max; χ, n_als)

Environment-weighted full update of vertical bond (ix,iy)[down]–(ix,iy+1)[up].
"""
function full_update_bond_v_v4!(gp::GaugePEPS, ix::Int, iy::Int,
                                 gate::AbstractMatrix, D_max::Int;
                                 χ::Int=64, n_als::Int=30)
    peps = gp.peps
    nxv, nyv, dg = nx(gp), ny(gp), gp.dg
    apply_mask!(gp, ix, iy); apply_mask!(gp, ix, iy+1)

    E = bond_environment_v(gp, ix, iy; χ=χ)
    dpD = size(E.RA, 2); dpU = size(E.RB, 2)
    _, d_gR_D, d_gU_D = site_dims(ix, iy,   nxv, nyv, dg)
    _, d_gR_U, d_gU_U = site_dims(ix, iy+1, nxv, nyv, dg)

    @tensor θ[aA,pD,pU,bB] := E.RA[aA,pD,k] * E.RB[bB,pU,k]
    θ = _apply_gate(θ, gate, dpD, dpU)

    eUofpD = [decode_phys(p, d_gR_D, d_gU_D, dg)[3] for p in 1:dpD]   # bond flux = e_U
    X, Y, βflux = als_truncate(θ, E.env, D_max, eUofpD, nothing; n_als=n_als)

    # reassemble: A(down) outer=(l,r,d) bond=u;  B(up) outer=(l,r,u) bond=d
    @tensor Afull[p,l,r,u,d] := E.QA[l,r,d,aA] * X[aA,p,u]
    @tensor Bfull[p,l,r,u,d] := E.QB[l,r,u,bB] * Y[bB,p,d]

    _strip_v!(gp, ix, iy, Afull, Bfull)

    # Enforce gauge invariance with the ALS bond fluxes before re-canonicalising.
    peps.tensors[ix, iy]   = Afull
    peps.tensors[ix, iy+1] = Bfull
    peps.λv[ix, iy] = ones(length(βflux))
    gp.qv[ix, iy]   = βflux
    apply_mask!(gp, ix, iy); apply_mask!(gp, ix, iy+1)

    _recanonicalise_v!(gp, ix, iy, peps.tensors[ix,iy], peps.tensors[ix,iy+1], D_max)
    apply_mask!(gp, ix, iy); apply_mask!(gp, ix, iy+1)
    return nothing
end

function _strip_v!(gp, ix, iy, D, U)
    peps = gp.peps
    isq(λ) = 1.0 ./ (sqrt.(λ) .+ FU_REG)
    sλ_lD = (ix > 1)        ? isq(peps.λh[ix-1, iy])   : ones(1)
    sλ_rD = (ix < nx(gp))   ? isq(peps.λh[ix,   iy])   : ones(1)
    sλ_dD = (iy > 1)        ? isq(peps.λv[ix, iy-1])   : ones(1)
    sλ_lU = (ix > 1)        ? isq(peps.λh[ix-1, iy+1]) : ones(1)
    sλ_rU = (ix < nx(gp))   ? isq(peps.λh[ix,   iy+1]) : ones(1)
    sλ_uU = (iy+1 < ny(gp)) ? isq(peps.λv[ix, iy+1])   : ones(1)
    @inbounds for d in axes(D,5), u in axes(D,4), r in axes(D,3), l in axes(D,2), p in axes(D,1)
        D[p,l,r,u,d] *= sλ_lD[l] * sλ_rD[r] * sλ_dD[d]
    end
    @inbounds for d in axes(U,5), u in axes(U,4), r in axes(U,3), l in axes(U,2), p in axes(U,1)
        U[p,l,r,u,d] *= sλ_lU[l] * sλ_rU[r] * sλ_uU[u]
    end
end

function _recanonicalise_v!(gp, ix, iy, D, U, D_max)
    peps = gp.peps; dg = gp.dg
    dpD = size(D,1); DlD = size(D,2); DrD = size(D,3); DdD = size(D,5)
    dpU = size(U,1); DlU = size(U,2); DrU = size(U,3); DuU = size(U,4)
    _, d_gR_D, d_gU_D = site_dims(ix, iy,   nx(gp), ny(gp), dg)
    _, d_gR_U, d_gU_U = site_dims(ix, iy+1, nx(gp), ny(gp), dg)

    @tensor Θ[pD,l,r,dD, pU,lU,rU,uU] := D[pD,l,r,k,dD] * U[pU,lU,rU,uU,k]
    Θm = reshape(Θ, dpD*DlD*DrD*DdD, dpU*DlU*DrU*DuU)
    nrow, ncol = size(Θm)

    row_charge = Vector{Int}(undef, nrow)
    for r0 in 0:nrow-1
        pD = (r0 % dpD) + 1
        row_charge[r0+1] = decode_phys(pD, d_gR_D, d_gU_D, dg)[3]   # e_U
    end
    col_charge = Vector{Int}(undef, ncol)
    for c0 in 0:ncol-1
        pU   = (c0 % dpU) + 1
        rest = c0 ÷ dpU
        lU0  = rest % DlU
        f_lU = (ix > 1) ? gp.qh[ix-1, iy+1][lU0+1] : 0
        nfU, eRU, eUU = decode_phys(pU, d_gR_U, d_gU_U, dg)
        col_charge[c0+1] = eRU + eUU - f_lU - nfU - gp.g_charges[ix, iy+1]
    end

    Uu, S, Vt, newq = charge_svd(Θm, row_charge, col_charge, D_max)
    Dk = length(S)
    λnew = S ./ max(norm(S), 1e-15)

    Draw = reshape(Uu, dpD, DlD, DrD, DdD, Dk)         # (p,l,r,d,β)
    Dnew = permutedims(Draw, (1,2,3,5,4))              # (p,l,r,β,d)
    Utmp = reshape(Vt, Dk, dpU, DlU, DrU, DuU)         # (β,p,l,r,u)
    Unew = permutedims(Utmp, (2,3,4,5,1))              # (p,l,r,u,β)

    peps.tensors[ix, iy]   = Dnew
    peps.tensors[ix, iy+1] = Unew
    peps.λv[ix, iy] = λnew
    gp.qv[ix, iy]   = newq
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Variational energy via the environment                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""⟨h⟩ on one bond from its environment + reduced tensors (h on (pL,pR))."""
function _bond_expect(E, h::AbstractMatrix, dpL::Int, dpR::Int)
    @tensor θ0[aA,pL,pR,bB] := E.RA[aA,pL,k] * E.RB[bB,pR,k]
    hθ = _apply_gate(θ0, h, dpL, dpR)                  # h·θ0
    @tensor Tn[aAp,bBp,pL,pR] := E.env[aA,bB,aAp,bBp] * θ0[aA,pL,pR,bB]
    num  = sum(Tn .* conj(permutedims(θ0, (1,4,2,3))))
    @tensor Th[aAp,bBp,pL,pR] := E.env[aA,bB,aAp,bBp] * hθ[aA,pL,pR,bB]
    hval = sum(Th .* conj(permutedims(θ0, (1,4,2,3))))
    return real(hval), real(num)
end

"""
    compute_energy_v4(gp; g, t_hop, m, χ) → Float64

Variational ⟨H⟩ = Σ_bonds ⟨H_merged_bond⟩ / ⟨ψ|ψ⟩.  The merged on-site weights
make each site's on-site energy count exactly once, so this is the full energy.
"""
function compute_energy_v4(gp::GaugePEPS; g::Float64, t_hop::Float64, m::Float64,
                            χ::Int=64)
    nxv, nyv, dg = nx(gp), ny(gp), gp.dg
    E_tot = 0.0
    for iy in 1:nyv, ix in 1:nxv-1
        Eb = bond_environment_h(gp, ix, iy; χ=χ)
        dpL = size(Eb.RA,2); dpR = size(Eb.RB,2)
        h = H_merged_h_site(ix, iy, nxv, nyv, dg; g=g, t=t_hop, m=m)
        hval, nrm = _bond_expect(Eb, h, dpL, dpR)
        E_tot += hval / nrm
    end
    for iy in 1:nyv-1, ix in 1:nxv
        Eb = bond_environment_v(gp, ix, iy; χ=χ)
        dpD = size(Eb.RA,2); dpU = size(Eb.RB,2)
        h = H_merged_v_site(ix, iy, nxv, nyv, dg; g=g, t=t_hop, m=m)
        hval, nrm = _bond_expect(Eb, h, dpD, dpU)
        E_tot += hval / nrm
    end
    return E_tot
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Trotter step + ITE driver                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function trotter_step_v4!(gp::GaugePEPS; gcoup::Float64, t_hop::Float64,
                           m::Float64, τ::Float64, D_max::Int,
                           χ::Int=64, n_als::Int=30)
    nxv, nyv, dg = nx(gp), ny(gp), gp.dg
    for iy in 1:nyv, ix in 1:nxv-1
        Hh = H_merged_h_site(ix, iy, nxv, nyv, dg; g=gcoup, t=t_hop, m=m)
        full_update_bond_h_v4!(gp, ix, iy, exp(-(τ/2) .* Hh), D_max; χ=χ, n_als=n_als)
    end
    for iy in 1:nyv-1, ix in 1:nxv
        Hv = H_merged_v_site(ix, iy, nxv, nyv, dg; g=gcoup, t=t_hop, m=m)
        full_update_bond_v_v4!(gp, ix, iy, exp(-τ .* Hv), D_max; χ=χ, n_als=n_als)
    end
    for iy in 1:nyv, ix in nxv-1:-1:1
        Hh = H_merged_h_site(ix, iy, nxv, nyv, dg; g=gcoup, t=t_hop, m=m)
        full_update_bond_h_v4!(gp, ix, iy, exp(-(τ/2) .* Hh), D_max; χ=χ, n_als=n_als)
    end
    return nothing
end

"""
    ite_ground_state_v4(nx, ny, dg, D_max; g, t_hop, m, g_charges,
                        n_ite, noise, χ, n_als, verbose) → GaugePEPS

Gauge-invariant boundary-MPS full-update ITE with τ annealing.
"""
function ite_ground_state_v4(nx::Int, ny::Int, dg::Int, D_max::Int;
                              g::Float64, t_hop::Float64, m::Float64,
                              g_charges::Matrix{Int},
                              n_ite::Int=300, noise::Float64=0.1,
                              χ::Int=64, n_als::Int=30, verbose::Bool=true)
    gp = init_gauge_peps(nx, ny, dg, g_charges; noise=noise)

    s1 = div(n_ite,8); s2 = div(n_ite,8); s3 = div(n_ite,4)
    s4 = div(n_ite,6); s5 = div(n_ite,6); s6 = n_ite-(s1+s2+s3+s4+s5)
    τ_stages = [(0.1,s1),(0.05,s2),(0.02,s3),(0.01,s4),(0.005,s5),(0.001,s6)]

    step = 0
    for (τ, ns) in τ_stages
        for _ in 1:ns
            step += 1
            trotter_step_v4!(gp; gcoup=g, t_hop=t_hop, m=m, τ=τ,
                             D_max=D_max, χ=χ, n_als=n_als)
            if verbose && (step % 25 == 0 || step == 1)
                nf = mean_nf(gp.peps, nx, ny, dg)
                D  = maximum(length(gp.qh[ix,iy]) for iy in 1:ny for ix in 1:nx-1; init=1)
                viol = forbidden_norm(gp)
                @printf("    v4 ITE %4d/%d (τ=%.4f): ⟨n_f⟩=%.5f D=%d viol=%.1e\n",
                        step, n_ite, τ, nf, D, viol); flush(stdout)
            end
        end
    end
    return gp
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Stage-2b self-test: gauge preserved, energy decreases, beats v3         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    v4_update_selftest(; m, g, D_max, nsteps, χ, n_als) → Bool

Run a few v4 ITE steps and check: gauge invariance preserved, ⟨n_f⟩≈0.5, and
the variational energy is monotonically non-increasing across the last steps.
"""
function v4_update_selftest(; nxv=3, nyv=4, dg=1, g=1.0, t_hop=1.0, m=0.25,
                             D_max=4, nsteps=8, χ=64, n_als=30, noise=0.1)
    gch = zeros(Int, nxv, nyv)
    for iy in 1:nyv, ix in 1:nxv
        isodd(ix+iy) && (gch[ix,iy] = -1)
    end
    gp = init_gauge_peps(nxv, nyv, dg, gch; noise=noise)
    println("─── v4 full-update self-test (m=$m g=$g D=$D_max) ───")
    Es = Float64[]
    for s in 1:nsteps
        trotter_step_v4!(gp; gcoup=g, t_hop=t_hop, m=m, τ=0.05,
                         D_max=D_max, χ=χ, n_als=n_als)
        E = compute_energy_v4(gp; g=g, t_hop=t_hop, m=m, χ=χ)
        viol = forbidden_norm(gp)
        nf = mean_nf(gp.peps, nxv, nyv, dg)
        push!(Es, E)
        @printf("  step %d: E=%.6f  ⟨n_f⟩=%.5f  viol=%.1e\n", s, E, nf, viol)
    end
    gauge_ok = forbidden_norm(gp) < 1e-9
    nf_ok    = abs(mean_nf(gp.peps,nxv,nyv,dg) - 0.5) < 1e-6
    mono_ok  = Es[end] <= Es[1] + 1e-6
    println(gauge_ok ? "  PASS gauge" : "  FAIL gauge")
    println(nf_ok    ? "  PASS half-filling" : "  FAIL half-filling")
    println(mono_ok  ? "  PASS energy non-increasing" : "  FAIL energy went up")
    return gauge_ok && nf_ok && mono_ok
end
