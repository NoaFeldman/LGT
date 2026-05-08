#= ═══════════════════════════════════════════════════════════════════════════════
   finite_peps_ground_state.jl

   Imaginary-time simple-update to find the ground state of a finite nx×ny PEPS
   for the U(1) LGT, using the same Hamiltonian conventions as lgt_evolution.jl.

   ── Lattice & DoF conventions ─────────────────────────────────────────────────
   Each node (ix, iy) with ix ∈ 1:nx, iy ∈ 1:ny carries:
     • fermion         (always present)
     • right-gauge DoF (only if ix < nx, i.e. not the rightmost column)
     • up-gauge DoF    (only if iy < ny, i.e. not the topmost row)

   Accordingly the physical Hilbert-space dimension of node (ix, iy) is:
     d_phys = d_f × d_gR × d_gU
   where d_gR = d_gU = 1 (trivial) when the corresponding link does not exist.

   ── Boundary conditions ───────────────────────────────────────────────────────
   Open boundary conditions.  Edge/corner nodes have trivial gauge DoFs for the
   missing links.  No Hamiltonian term is generated for a non-existent bond.

   ── Ground state search ───────────────────────────────────────────────────────
   Imaginary-time evolution  e^{-τ H}  applied via nearest-neighbour 2-site
   simple update (sweep over all existing bonds).  The result converges to the
   ground state as τ → ∞.

   ── Initial state ─────────────────────────────────────────────────────────────
   Staggered: even sites (ix+iy even) n_f=0, odd sites n_f=1, all gauge links 0.
   ═══════════════════════════════════════════════════════════════════════════ =#

using Pkg
Pkg.activate(".")
for pkg in ["TensorKit", "TensorOperations", "KrylovKit", "LinearAlgebra",
            "Plots", "Printf"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

using TensorKit
using TensorOperations
using LinearAlgebra
using Printf
using Plots

const _LGT_HAMILTONIAN_LOADED = true
include(joinpath(@__DIR__, "u1_lgt_hamiltonian.jl"))

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Parameters                                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

const NX      = 4        # lattice columns
const NY      = 4        # lattice rows
const DG      = 1        # gauge truncation
const D_BOND  = 2        # initial virtual bond dimension
const D_MAX   = 4        # max bond dimension
const SVD_CUT = 1e-12

const G_COUP  = 1.0
const T_HOP   = 1.0
const M_MASS  = 2.0

const TAU     = 0.01
const N_STEPS = 2000
const MEASURE_EVERY = 100

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Finite PEPS data structure                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
FinitePEPS stores the tensors and bond weights for an nx×ny lattice.

Tensor convention for node (ix, iy):
    tensors[ix, iy]  :  Array{ComplexF64, 5}  with axes
        (p, l, r, u, d)
    where:
        p  = physical index  (1:d_phys)
        l  = left  virtual   (1:D_l),  D_l = D_BOND if ix > 1 else 1
        r  = right virtual   (1:D_r),  D_r = D_BOND if ix < nx else 1
        u  = up    virtual   (1:D_u),  D_u = D_BOND if iy < ny else 1
        d  = down  virtual   (1:D_d),  D_d = D_BOND if iy > 1 else 1

Bond weights:
    λh[ix, iy]  : Vector{Float64}  on horizontal bond (ix,iy)—(ix+1,iy)
    λv[ix, iy]  : Vector{Float64}  on vertical   bond (ix,iy)—(ix,iy+1)
"""
mutable struct FinitePEPS
    tensors :: Matrix{Array{ComplexF64, 5}}   # [nx, ny]
    λh      :: Matrix{Vector{Float64}}        # [nx-1, ny]
    λv      :: Matrix{Vector{Float64}}        # [nx, ny-1]
    nx      :: Int
    ny      :: Int
end

"""Virtual bond dimension of node (ix, iy) along each direction."""
function vdims(peps::FinitePEPS, ix::Int, iy::Int)
    D_l = (ix > 1)        ? length(peps.λh[ix-1, iy]) : 1
    D_r = (ix < peps.nx)  ? length(peps.λh[ix,   iy]) : 1
    D_u = (iy < peps.ny)  ? length(peps.λv[ix,   iy]) : 1
    D_d = (iy > 1)        ? length(peps.λv[ix, iy-1]) : 1
    return D_l, D_r, D_u, D_d
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Initialisation                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
Initialise the finite PEPS in the staggered large-mass product state:
  Site (ix,iy) with ix+iy even → n_f = 0
  Site (ix,iy) with ix+iy odd  → n_f = 1
  All gauge links: e = 0
"""
function init_finite_peps(nx::Int, ny::Int, dg::Int, D::Int)
    tensors = Matrix{Array{ComplexF64, 5}}(undef, nx, ny)
    λh      = Matrix{Vector{Float64}}(undef, nx-1, ny)
    λv      = Matrix{Vector{Float64}}(undef, nx, ny-1)

    # Initialise bond weights (trivial = 1, size D)
    for iy in 1:ny, ix in 1:nx-1
        λh[ix, iy] = ones(Float64, D)
    end
    for iy in 1:ny-1, ix in 1:nx
        λv[ix, iy] = ones(Float64, D)
    end

    for iy in 1:ny, ix in 1:nx
        d_phys, _, _ = site_dims(ix, iy, nx, ny, dg)

        # Virtual dimensions (1 at boundaries)
        D_l = (ix > 1)   ? D : 1
        D_r = (ix < nx)  ? D : 1
        D_u = (iy < ny)  ? D : 1
        D_d = (iy > 1)   ? D : 1

        arr = zeros(ComplexF64, d_phys, D_l, D_r, D_u, D_d)

        # Product state
        nf_init = iseven(ix + iy) ? 0 : 1
        # Index for |nf_init, eR=0, eU=0⟩ in site's Hilbert space
        # d_gR, d_gU determine whether R/U gauge exist
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        idx = site_idx(nf_init, 0, 0, d_gR, d_gU, dg)
        arr[idx, 1, 1, 1, 1] = 1.0

        tensors[ix, iy] = arr
    end

    return FinitePEPS(tensors, λh, λv, nx, ny)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Simple-update bond update                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    update_bond_h!(peps, ix, iy, gate, D_trunc)

Apply a horizontal 2-site gate on the bond (ix,iy)—(ix+1,iy).
Updates `peps.tensors[ix,iy]`, `peps.tensors[ix+1,iy]`, and `peps.λh[ix,iy]`
in-place using the Vidal simple-update rule (bare tensors stored, weights in λ).
"""
function update_bond_h!(peps::FinitePEPS, ix::Int, iy::Int,
                         gate::AbstractMatrix, D_trunc::Int)
    T_L = peps.tensors[ix,   iy]
    T_R = peps.tensors[ix+1, iy]
    dp_L = size(T_L, 1)
    dp_R = size(T_R, 1)

    λ_bond = peps.λh[ix, iy]
    bond_dim = length(λ_bond)
    @assert size(T_L, 3) == bond_dim && size(T_R, 2) == bond_dim

    # Weights for transverse legs
    λ_l   = (ix > 1)         ? peps.λh[ix-1, iy]  : ones(1)
    λ_r   = (ix+1 < peps.nx) ? peps.λh[ix+1, iy]  : ones(1)
    λ_u_L = (iy < peps.ny)   ? peps.λv[ix,   iy]  : ones(1)
    λ_d_L = (iy > 1)         ? peps.λv[ix, iy-1]  : ones(1)
    λ_u_R = (iy < peps.ny)   ? peps.λv[ix+1, iy]  : ones(1)
    λ_d_R = (iy > 1)         ? peps.λv[ix+1, iy-1] : ones(1)

    sq_bond = sqrt.(λ_bond)
    sq_l   = sqrt.(λ_l)
    sq_r   = sqrt.(λ_r)
    sq_u_L = sqrt.(λ_u_L)
    sq_d_L = sqrt.(λ_d_L)
    sq_u_R = sqrt.(λ_u_R)
    sq_d_R = sqrt.(λ_d_R)

    # Absorb weights: L absorbs sq_bond on right, transverse weights on l, u_L, d_L
    L_w = copy(T_L)
    for p in 1:dp_L, l in 1:size(L_w,2), r in 1:bond_dim,
        u in 1:size(L_w,4), d in 1:size(L_w,5)
        L_w[p, l, r, u, d] *= sq_l[l] * sq_bond[r] * sq_u_L[u] * sq_d_L[d]
    end

    # R absorbs sq_bond on left, transverse weights on r_R, u_R, d_R
    R_w = copy(T_R)
    for p in 1:dp_R, l in 1:bond_dim, r in 1:size(R_w,3),
        u in 1:size(R_w,4), d in 1:size(R_w,5)
        R_w[p, l, r, u, d] *= sq_bond[l] * sq_r[r] * sq_u_R[u] * sq_d_R[d]
    end

    # Contract along bond axis: L(p1,l,χ,uL,dL) * R(p2,χ,r,uR,dR) → Θ(p1,l,uL,dL,p2,r,uR,dR)
    # Reshape to (p1*l*uL*dL, χ) and (χ, p2*r*uR*dR), then matrix multiply
    sz_L = size(L_w);  sz_R = size(R_w)
    L_perm = permutedims(L_w, (1, 2, 4, 5, 3))  # (p1, l, uL, dL, χ)
    L_mat  = reshape(L_perm, dp_L*sz_L[2]*sz_L[4]*sz_L[5], bond_dim)
    R_perm = permutedims(R_w, (2, 1, 3, 4, 5))  # (χ, p2, r, uR, dR)
    R_mat  = reshape(R_perm, bond_dim, dp_R*sz_R[3]*sz_R[4]*sz_R[5])
    Θ_mat_raw = L_mat * R_mat   # (p1*l*uL*dL, p2*r*uR*dR)

    # Reshape to (p1, l, uL, dL, p2, r, uR, dR) then apply gate
    Θ = reshape(Θ_mat_raw, dp_L, sz_L[2], sz_L[4], sz_L[5], dp_R, sz_R[3], sz_R[4], sz_R[5])
    sz = size(Θ)

    # Apply gate:  Θ_new[p1',l,uL,dL,p2',r,uR,dR] = Σ_{p1,p2} G[p1',p2',p1,p2]*Θ[p1,l,uL,dL,p2,r,uR,dR]
    # Reshape to (p1*p2, rest) and apply gate matrix (dp_L*dp_R × dp_L*dp_R)
    Θ_pp = permutedims(Θ, (1, 5, 2, 3, 4, 6, 7, 8))   # (p1, p2, l, uL, dL, r, uR, dR)
    Θ_pp_mat = reshape(Θ_pp, dp_L*dp_R, sz[2]*sz[3]*sz[4]*sz[6]*sz[7]*sz[8])
    Θ_new_pp_mat = gate * Θ_pp_mat
    Θ_new_pp = reshape(Θ_new_pp_mat, dp_L, dp_R, sz[2], sz[3], sz[4], sz[6], sz[7], sz[8])
    Θ_new = permutedims(Θ_new_pp, (1, 3, 4, 5, 2, 6, 7, 8))  # (p1,l,uL,dL,p2,r,uR,dR)

    # SVD split
    Θ_mat = reshape(Θ_new, dp_L*sz[2]*sz[3]*sz[4], dp_R*sz[6]*sz[7]*sz[8])
    F = svd(Θ_mat)
    D_keep = min(D_trunc, count(F.S .> SVD_CUT), length(F.S))
    D_keep = max(D_keep, 1)
    S_tr = F.S[1:D_keep]
    λ_new = S_tr ./ norm(S_tr)
    sqS = sqrt.(S_tr)

    L_raw = reshape(F.U[:, 1:D_keep] * Diagonal(sqS),
                    dp_L, sz[2], sz[3], sz[4], D_keep)
    # L_raw axes: (p, l, uL, dL, r_new);  need to store as (p, l, r, u, d)
    L_new = permutedims(L_raw, (1, 2, 5, 3, 4))  # (p, l, r_new, uL, dL)

    R_tmp = reshape(Diagonal(sqS) * F.Vt[1:D_keep, :],
                    D_keep, dp_R, sz[6], sz[7], sz[8])
    # R_tmp axes: (l_new, p, r, uR, dR);  permute to (p, l, r, u, d)
    R_new = permutedims(R_tmp, (2, 1, 3, 4, 5))   # (dp_R, D_keep, r, uR, dR)

    # Remove absorbed transverse weights
    isq_l  = 1.0 ./ (sq_l  .+ 1e-15)
    isq_u_L = 1.0 ./ (sq_u_L .+ 1e-15)
    isq_d_L = 1.0 ./ (sq_d_L .+ 1e-15)
    isq_r  = 1.0 ./ (sq_r  .+ 1e-15)
    isq_u_R = 1.0 ./ (sq_u_R .+ 1e-15)
    isq_d_R = 1.0 ./ (sq_d_R .+ 1e-15)

    # L_new: (p, l, r_new, uL, dL) → remove sq_l[l], sq_u_L[uL], sq_d_L[dL]
    for p in 1:dp_L, l in 1:sz[2], r in 1:D_keep, u in 1:sz[3], d in 1:sz[4]
        L_new[p, l, r, u, d] *= isq_l[l] * isq_u_L[u] * isq_d_L[d]
    end
    # R_new: (p, l_new, r, uR, dR) → remove sq_r[r], sq_u_R[uR], sq_d_R[dR]
    for p in 1:dp_R, l in 1:D_keep, r in 1:sz[6], u in 1:sz[7], d in 1:sz[8]
        R_new[p, l, r, u, d] *= isq_r[r] * isq_u_R[u] * isq_d_R[d]
    end

    peps.tensors[ix,   iy] = L_new
    peps.tensors[ix+1, iy] = R_new
    peps.λh[ix, iy] = λ_new
    return nothing
end

"""
    update_bond_v!(peps, ix, iy, gate, D_trunc)

Apply a vertical 2-site gate on the bond (ix,iy) [lower] — (ix,iy+1) [upper].
"""
function update_bond_v!(peps::FinitePEPS, ix::Int, iy::Int,
                         gate::AbstractMatrix, D_trunc::Int)
    T_D = peps.tensors[ix, iy]
    T_U = peps.tensors[ix, iy+1]
    dp_D = size(T_D, 1)
    dp_U = size(T_U, 1)

    λ_bond = peps.λv[ix, iy]
    bond_dim = length(λ_bond)
    @assert size(T_D, 4) == bond_dim && size(T_U, 5) == bond_dim

    # Transverse weights for lower (D) node: left(2), right(3)
    λ_l_D = (ix > 1)       ? peps.λh[ix-1, iy]   : ones(1)
    λ_r_D = (ix < peps.nx) ? peps.λh[ix,   iy]   : ones(1)
    λ_d_D = (iy > 1)       ? peps.λv[ix, iy-1]   : ones(1)
    # Transverse weights for upper (U) node
    λ_l_U = (ix > 1)       ? peps.λh[ix-1, iy+1] : ones(1)
    λ_r_U = (ix < peps.nx) ? peps.λh[ix,   iy+1] : ones(1)
    λ_u_U = (iy+1 < peps.ny) ? peps.λv[ix, iy+1] : ones(1)

    sq_bond = sqrt.(λ_bond)
    sq_l_D = sqrt.(λ_l_D);  sq_r_D = sqrt.(λ_r_D);  sq_d_D = sqrt.(λ_d_D)
    sq_l_U = sqrt.(λ_l_U);  sq_r_U = sqrt.(λ_r_U);  sq_u_U = sqrt.(λ_u_U)

    # Absorb: D absorbs sq_bond on up(4), transverse on l(2), r(3), d(5)
    D_w = copy(T_D)
    for p in 1:dp_D, l in 1:size(D_w,2), r in 1:size(D_w,3),
        u in 1:bond_dim, d in 1:size(D_w,5)
        D_w[p, l, r, u, d] *= sq_l_D[l] * sq_r_D[r] * sq_bond[u] * sq_d_D[d]
    end

    # U absorbs sq_bond on down(5), transverse on l(2), r(3), u(4)
    U_w = copy(T_U)
    for p in 1:dp_U, l in 1:size(U_w,2), r in 1:size(U_w,3),
        u in 1:size(U_w,4), d in 1:bond_dim
        U_w[p, l, r, u, d] *= sq_l_U[l] * sq_r_U[r] * sq_u_U[u] * sq_bond[d]
    end

    # Contract along bond: D(p1,lD,rD,χ,dD) * U(p2,lU,rU,uU,χ) → Θ(p1,lD,rD,dD,p2,lU,rU,uU)
    sz_D = size(D_w);  sz_U = size(U_w)
    D_perm = permutedims(D_w, (1, 2, 3, 5, 4))  # (p1, lD, rD, dD, χ)
    D_mat  = reshape(D_perm, dp_D*sz_D[2]*sz_D[3]*sz_D[5], bond_dim)
    U_perm = permutedims(U_w, (5, 1, 2, 3, 4))  # (χ, p2, lU, rU, uU)
    U_mat  = reshape(U_perm, bond_dim, dp_U*sz_U[2]*sz_U[3]*sz_U[4])
    Θ_raw  = D_mat * U_mat   # (p1*lD*rD*dD, p2*lU*rU*uU)
    Θ = reshape(Θ_raw, dp_D, sz_D[2], sz_D[3], sz_D[5], dp_U, sz_U[2], sz_U[3], sz_U[4])
    szΘ = size(Θ)

    # Apply gate via matrix multiply on physical indices
    Θ_pp = permutedims(Θ, (1, 5, 2, 3, 4, 6, 7, 8))   # (p1, p2, lD, rD, dD, lU, rU, uU)
    Θ_pp_mat = reshape(Θ_pp, dp_D*dp_U, szΘ[2]*szΘ[3]*szΘ[4]*szΘ[6]*szΘ[7]*szΘ[8])
    Θ_new_pp_mat = gate * Θ_pp_mat
    Θ_new_pp = reshape(Θ_new_pp_mat, dp_D, dp_U, szΘ[2], szΘ[3], szΘ[4], szΘ[6], szΘ[7], szΘ[8])
    Θ_new = permutedims(Θ_new_pp, (1, 3, 4, 5, 2, 6, 7, 8))  # (p1,lD,rD,dD,p2,lU,rU,uU)

    # SVD: (p1, lD, rD, dD) | (p2, lU, rU, uU)
    Θ_mat = reshape(Θ_new, dp_D*szΘ[2]*szΘ[3]*szΘ[4], dp_U*szΘ[6]*szΘ[7]*szΘ[8])
    F = svd(Θ_mat)
    D_keep = min(D_trunc, count(F.S .> SVD_CUT), length(F.S))
    D_keep = max(D_keep, 1)
    S_tr = F.S[1:D_keep]
    λ_new = S_tr ./ norm(S_tr)
    sqS = sqrt.(S_tr)

    # D_raw axes: (p, lD, rD, dD, u_new); need (p, l, r, u, d) → permute u_new to position 4
    D_raw = reshape(F.U[:, 1:D_keep] * Diagonal(sqS),
                    dp_D, szΘ[2], szΘ[3], szΘ[4], D_keep)
    D_new = permutedims(D_raw, (1, 2, 3, 5, 4))  # (p, lD, rD, u_new, dD)

    # U_tmp axes: (d_new, p, lU, rU, uU); permute to (p, lU, rU, uU, d_new)
    U_tmp = reshape(Diagonal(sqS) * F.Vt[1:D_keep, :],
                    D_keep, dp_U, szΘ[6], szΘ[7], szΘ[8])
    U_new = permutedims(U_tmp, (2, 3, 4, 5, 1))   # (dp_U, lU, rU, uU, d_new)

    # Remove absorbed transverse weights
    isq_l_D = 1.0 ./ (sq_l_D .+ 1e-15);  isq_r_D = 1.0 ./ (sq_r_D .+ 1e-15)
    isq_d_D = 1.0 ./ (sq_d_D .+ 1e-15)
    isq_l_U = 1.0 ./ (sq_l_U .+ 1e-15);  isq_r_U = 1.0 ./ (sq_r_U .+ 1e-15)
    isq_u_U = 1.0 ./ (sq_u_U .+ 1e-15)

    # D_new: (p, lD, rD, u_new, dD) → remove sq_l_D[lD], sq_r_D[rD], sq_d_D[dD]
    for p in 1:dp_D, l in 1:szΘ[2], r in 1:szΘ[3], u in 1:D_keep, d in 1:szΘ[4]
        D_new[p, l, r, u, d] *= isq_l_D[l] * isq_r_D[r] * isq_d_D[d]
    end
    # U_new: (p, lU, rU, uU, d_new) → remove sq_l_U[lU], sq_r_U[rU], sq_u_U[uU]
    for p in 1:dp_U, l in 1:szΘ[6], r in 1:szΘ[7], u in 1:szΘ[8], d in 1:D_keep
        U_new[p, l, r, u, d] *= isq_l_U[l] * isq_r_U[r] * isq_u_U[u]
    end

    peps.tensors[ix, iy]   = D_new
    peps.tensors[ix, iy+1] = U_new
    peps.λv[ix, iy] = λ_new
    return nothing
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Trotter sweep                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
One second-order Trotter step over all bonds of the finite lattice.
Order: [h-half-forward] → [v-full-forward] → [h-half-backward]
"""
function trotter_step!(peps::FinitePEPS, nx::Int, ny::Int, dg::Int;
                        g::Float64, t_hop::Float64, m::Float64,
                        τ::Float64, D_trunc::Int, μ::Float64=0.0)
    # ── Horizontal half-step (left to right sweep) ────────────────────────
    for iy in 1:ny, ix in 1:nx-1
        Hh = H_merged_h_site(ix, iy, nx, ny, dg; g=g, t=t_hop, m=m, μ=μ)
        gate = exp(-(τ / 2) .* Hh)
        update_bond_h!(peps, ix, iy, gate, D_trunc)
    end

    # ── Vertical full-step (bottom to top sweep) ──────────────────────────
    for iy in 1:ny-1, ix in 1:nx
        Hv = H_merged_v_site(ix, iy, nx, ny, dg; g=g, t=t_hop, m=m, μ=μ)
        gate = exp(-τ .* Hv)
        update_bond_v!(peps, ix, iy, gate, D_trunc)
    end

    # ── Horizontal half-step (right to left sweep, for symmetry) ─────────
    for iy in 1:ny, ix in nx-1:-1:1
        Hh = H_merged_h_site(ix, iy, nx, ny, dg; g=g, t=t_hop, m=m, μ=μ)
        gate = exp(-(τ / 2) .* Hh)
        update_bond_h!(peps, ix, iy, gate, D_trunc)
    end

    return nothing
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Observables                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
Single-site expectation value ⟨O⟩ at node (ix, iy) using simple-update weights.
"""
function expect_site(peps::FinitePEPS, ix::Int, iy::Int, O_mat::AbstractMatrix)
    T = peps.tensors[ix, iy]
    Tw = copy(T)
    dp = size(T, 1)

    λ_l = (ix > 1)        ? peps.λh[ix-1, iy] : ones(1)
    λ_r = (ix < peps.nx)  ? peps.λh[ix,   iy] : ones(1)
    λ_u = (iy < peps.ny)  ? peps.λv[ix,   iy] : ones(1)
    λ_d = (iy > 1)        ? peps.λv[ix, iy-1] : ones(1)

    for p in 1:dp, l in axes(Tw,2), r in axes(Tw,3),
        u in axes(Tw,4), d in axes(Tw,5)
        Tw[p, l, r, u, d] *= λ_l[l] * λ_r[r] * λ_u[u] * λ_d[d]
    end

    T_flat  = reshape(Tw, dp, :)
    num = real(tr(T_flat' * O_mat * T_flat))
    den = real(tr(T_flat' * T_flat))
    return den > 1e-15 ? num / den : 0.0
end

"""Mean fermion density averaged over all sites."""
function mean_nf(peps::FinitePEPS, nx::Int, ny::Int, dg::Int)
    total = 0.0
    for iy in 1:ny, ix in 1:nx
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        O = embed_f_site(op_nf(), d_gR, d_gU)
        total += expect_site(peps, ix, iy, O)
    end
    return total / (nx * ny)
end

"""
Energy expectation value (sum over all bonds via 2-site expectation values
using simple-update weights — approximate in non-MPS geometries).
"""
function total_energy(peps::FinitePEPS, nx::Int, ny::Int, dg::Int;
                       g::Float64, t_hop::Float64, m::Float64)
    E = 0.0

    # On-site contributions
    for iy in 1:ny, ix in 1:nx
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        Hos = H_onsite_site(ix, iy, nx, ny, dg; g=g, m=m)
        E += expect_site(peps, ix, iy, Hos)
    end

    # Hopping: horizontal bonds
    for iy in 1:ny, ix in 1:nx-1
        T_L = peps.tensors[ix,   iy]
        T_R = peps.tensors[ix+1, iy]
        dp_L = size(T_L, 1);  dp_R = size(T_R, 1)

        λ_bond = peps.λh[ix, iy]
        bond_dim = length(λ_bond)
        sq_bond = sqrt.(λ_bond)

        λ_l  = (ix > 1)        ? peps.λh[ix-1, iy] : ones(1)
        λ_u_L = (iy < ny)      ? peps.λv[ix,  iy]  : ones(1)
        λ_d_L = (iy > 1)       ? peps.λv[ix, iy-1] : ones(1)
        λ_r  = (ix+1 < nx)     ? peps.λh[ix+1, iy] : ones(1)
        λ_u_R = (iy < ny)      ? peps.λv[ix+1, iy]  : ones(1)
        λ_d_R = (iy > 1)       ? peps.λv[ix+1, iy-1] : ones(1)

        L_w = copy(T_L)
        for p in 1:dp_L, l in axes(L_w,2), r in 1:bond_dim, u in axes(L_w,4), d in axes(L_w,5)
            L_w[p,l,r,u,d] *= sqrt(λ_l[l]) * sq_bond[r] * sqrt(λ_u_L[u]) * sqrt(λ_d_L[d])
        end
        R_w = copy(T_R)
        for p in 1:dp_R, l in 1:bond_dim, r in axes(R_w,3), u in axes(R_w,4), d in axes(R_w,5)
            R_w[p,l,r,u,d] *= sq_bond[l] * sqrt(λ_r[r]) * sqrt(λ_u_R[u]) * sqrt(λ_d_R[d])
        end

        # Contract: rho_LR[p1,p2,p1',p2'] = Σ_{l,r,u,d,χ} L*R conj
        L_flat = reshape(L_w, dp_L, :)   # (dp_L, rest)
        R_flat = reshape(R_w, dp_R, :)
        # Bond index χ is at position 3 of T_L and position 2 of T_R (after p)
        # Reshape to (dp, bond, other)
        sz_L = size(L_w)
        sz_R = size(R_w)
        L_bond = reshape(permutedims(L_w, (1, 3, 2, 4, 5)),
                         dp_L, bond_dim, sz_L[2]*sz_L[4]*sz_L[5])
        R_bond = reshape(permutedims(R_w, (1, 2, 3, 4, 5)),
                         dp_R, bond_dim, sz_R[3]*sz_R[4]*sz_R[5])

        # rho[p1,p2,p1',p2'] = Σ_{χ,aux} L[p1,χ,aux]*conj(L[p1',χ,aux]) * ...
        # Use: contract Θ = Σ_χ L[p1,χ,a]*R[p2,χ,b], shape (dp_L, dp_R, a, b)
        Θ = zeros(ComplexF64, dp_L, dp_R, sz_L[2]*sz_L[4]*sz_L[5], sz_R[3]*sz_R[4]*sz_R[5])
        for p1 in 1:dp_L, p2 in 1:dp_R, χ in 1:bond_dim
            Θ[p1, p2, :, :] .+= L_bond[p1, χ, :] * R_bond[p2, χ, :]'
        end
        Θ_mat = reshape(Θ, dp_L*dp_R, :)

        H_hop = H_hop_h_site(ix, iy, nx, ny, dg; t=t_hop)
        H_hop_r = reshape(H_hop, dp_L, dp_R, dp_L, dp_R)

        # ⟨H⟩ = Σ_{p1,p2,p1',p2'} H[p1,p2,p1',p2'] ρ[p1,p2,p1',p2']
        # ρ[p1,p2,p1',p2'] = Σ_{aux} Θ[p1,p2,aux]*conj(Θ[p1',p2',aux])
        Θ_flat = reshape(Θ, dp_L*dp_R, size(Θ_mat, 2))
        ρ = Θ_flat * Θ_flat'  # (dp_L*dp_R) × (dp_L*dp_R)
        numerator   = real(tr(reshape(H_hop, dp_L*dp_R, dp_L*dp_R) * ρ))
        denominator = real(tr(ρ))
        E += denominator > 1e-15 ? numerator / denominator : 0.0
    end

    return E
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main ground state search                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function run_finite_peps_groundstate(;
        nx::Int      = NX,
        ny::Int      = NY,
        dg::Int      = DG,
        D_bond::Int  = D_BOND,
        D_max::Int   = D_MAX,
        g::Float64   = G_COUP,
        t_hop::Float64 = T_HOP,
        m_mass::Float64 = M_MASS,
        τ::Float64   = TAU,
        n_steps::Int = N_STEPS,
        measure_every::Int = MEASURE_EVERY)

    println("══════════════════════════════════════════════════════════════")
    println("  U(1) LGT — finite PEPS ground state (imaginary-time SU)")
    println("══════════════════════════════════════════════════════════════")
    @printf("  Lattice: %d × %d  (nx × ny)\n", nx, ny)
    @printf("  Gauge truncation dg = %d  (link dim = %d)\n", dg, gauge_dim(dg))
    @printf("  Bond D = %d → D_max = %d\n", D_bond, D_max)
    @printf("  g = %.3f,  t = %.3f,  m = %.3f\n", g, t_hop, m_mass)
    @printf("  τ = %.4e,  steps = %d\n\n", τ, n_steps)

    # Print site physical dimensions
    println("  Site physical dimensions (d_phys = d_f × d_gR × d_gU):")
    for iy in ny:-1:1
        print("    iy=$iy: ")
        for ix in 1:nx
            dp, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
            print("[$ix,$iy](d=$(dp)) ")
        end
        println()
    end
    println()

    peps = init_finite_peps(nx, ny, dg, D_bond)

    times   = Float64[]
    nf_hist = Float64[]

    push!(times, 0.0)
    push!(nf_hist, mean_nf(peps, nx, ny, dg))

    @printf("  %6s  |  ⟨n_f⟩_mean\n", "step")
    @printf("  ───────┼────────────\n")
    @printf("  %6d  |  %.6f\n", 0, nf_hist[end])

    for step in 1:n_steps
        trotter_step!(peps, nx, ny, dg; g=g, t_hop=t_hop, m=m_mass,
                      τ=τ, D_trunc=D_max)

        if step % measure_every == 0 || step == n_steps
            nf = mean_nf(peps, nx, ny, dg)
            push!(times, step * τ)
            push!(nf_hist, nf)
            D_now = maximum(length(peps.λh[ix, iy]) for iy in 1:ny for ix in 1:nx-1;
                            init=1)
            @printf("  %6d  |  %.6f   D=%d\n", step, nf, D_now)
        end
    end

    # Final site-resolved densities
    println("\n  Final fermion densities ⟨n_f⟩ (iy rows, ix cols):")
    for iy in ny:-1:1
        print("    iy=$iy: ")
        for ix in 1:nx
            _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
            O = embed_f_site(op_nf(), d_gR, d_gU)
            nf = expect_site(peps, ix, iy, O)
            @printf("%6.4f  ", nf)
        end
        println()
    end
    println()

    # Plot
    p = plot(times, nf_hist;
             xlabel = "imaginary time",
             ylabel = "⟨n_f⟩ (mean)",
             title  = "Finite PEPS ground state  ($(nx)×$(ny), m=$(m_mass))",
             lw = 2, marker = :circle, ms = 2, label = "⟨n_f⟩")
    hline!(p, [0.5]; label = "0.5", ls = :dash, color = :gray)
    savefig(p, joinpath(@__DIR__, "finite_peps_groundstate.png"))
    display(p)

    println("  Plot saved to finite_peps_groundstate.png")
    println("══════════════════════════════════════════════════════════════")

    return peps, times, nf_hist
end

# ── Run (only when executed directly, not when included) ─────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    peps_final, times, nf_hist = run_finite_peps_groundstate()
end
