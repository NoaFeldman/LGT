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
    init_finite_peps(nx, ny, dg, D; noise=0.0)

Initialise the finite PEPS in the staggered large-mass product state:
  Site (ix,iy) with ix+iy even → n_f = 0
  Site (ix,iy) with ix+iy odd  → n_f = 1
  All gauge links: e = 0

If `noise > 0`, add an independent complex-Gaussian perturbation of standard
deviation `noise` to every entry of every tensor.  This gives the bonds rank
> 1 from step 0, which the simple update needs in order for the plaquette /
magnetic dynamics to take hold (see notes in the README).
"""
function init_finite_peps(nx::Int, ny::Int, dg::Int, D::Int; noise::Float64 = 0.0)
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

        if noise > 0
            arr .+= noise .* (randn(ComplexF64, size(arr)))
        end

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
    # Gate is built as kron(Hos_L, Idn_R) etc., where Julia's kron puts the SECOND
    # factor on the FAST (column-major) flat axis.  We therefore put p2 (R site)
    # first so it becomes the fast flat index; otherwise the gate's L-operator
    # would act on the R-site tensor leg (and vice versa).
    Θ_pp = permutedims(Θ, (5, 1, 2, 3, 4, 6, 7, 8))   # (p2, p1, l, uL, dL, r, uR, dR)
    Θ_pp_mat = reshape(Θ_pp, dp_R*dp_L, sz[2]*sz[3]*sz[4]*sz[6]*sz[7]*sz[8])
    Θ_new_pp_mat = gate * Θ_pp_mat
    Θ_new_pp = reshape(Θ_new_pp_mat, dp_R, dp_L, sz[2], sz[3], sz[4], sz[6], sz[7], sz[8])
    Θ_new = permutedims(Θ_new_pp, (2, 3, 4, 5, 1, 6, 7, 8))  # (p1,l,uL,dL,p2,r,uR,dR)

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

    # Apply gate via matrix multiply on physical indices.
    # Gate is kron(Hos_D, Idn_U) etc., which puts the U site on the FAST flat
    # axis.  Match that by flattening (p2=U, p1=D) with p2 first.
    Θ_pp = permutedims(Θ, (5, 1, 2, 3, 4, 6, 7, 8))   # (p2, p1, lD, rD, dD, lU, rU, uU)
    Θ_pp_mat = reshape(Θ_pp, dp_U*dp_D, szΘ[2]*szΘ[3]*szΘ[4]*szΘ[6]*szΘ[7]*szΘ[8])
    Θ_new_pp_mat = gate * Θ_pp_mat
    Θ_new_pp = reshape(Θ_new_pp_mat, dp_U, dp_D, szΘ[2], szΘ[3], szΘ[4], szΘ[6], szΘ[7], szΘ[8])
    Θ_new = permutedims(Θ_new_pp, (2, 3, 4, 5, 1, 6, 7, 8))  # (p1,lD,rD,dD,p2,lU,rU,uU)

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
# ║  Plaquette update — 3-node L-shape via SVD-decomposed gate              ║
# ║                                                                          ║
# ║  Plaquette with lower-left corner at (ix,iy) involves nodes              ║
# ║    A = (ix,   iy  )   bottom & left links live on A                     ║
# ║    B = (ix+1, iy  )   right  link  lives on B's up-gauge                ║
# ║    C = (ix,   iy+1)   top    link  lives on C's right-gauge             ║
# ║                                                                          ║
# ║  We absorb the decomposed gate (GA, GB, GC) onto each node, enlarging   ║
# ║  the A–B bond by D_α (auxiliary index of GA·GB) and the A–C bond by    ║
# ║  D_β (auxiliary of GA·GC).  We then compress both bonds back to        ║
# ║  D_trunc using the standard 2-site simple-update SVD with identity     ║
# ║  gate (since the physics is already encoded in the absorbed tensors).  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    update_plaquette!(peps, ix, iy, GA, GB, GC, dg, D_trunc)

Apply the SVD-decomposed plaquette gate on nodes A=(ix,iy), B=(ix+1,iy),
C=(ix,iy+1).  See `decompose_plaquette_LShape` for the GA/GB/GC convention.
"""
function update_plaquette!(peps::FinitePEPS, ix::Int, iy::Int,
                            GA::AbstractArray, GB::AbstractArray, GC::AbstractArray,
                            dg::Int, D_trunc::Int)
    nx, ny = peps.nx, peps.ny
    @assert ix < nx && iy < ny "plaquette ($ix,$iy) needs ix<nx and iy<ny"
    gd = gauge_dim(dg)
    D_α = size(GB, 3)
    D_β = size(GC, 3)
    df  = LGT_d_f

    T_A = peps.tensors[ix,   iy]
    T_B = peps.tensors[ix+1, iy]
    T_C = peps.tensors[ix,   iy+1]

    d_pA, d_gR_A, d_gU_A = site_dims(ix,   iy,   nx, ny, dg)
    d_pB, d_gR_B, d_gU_B = site_dims(ix+1, iy,   nx, ny, dg)
    d_pC, d_gR_C, d_gU_C = site_dims(ix,   iy+1, nx, ny, dg)
    @assert d_gR_A == gd && d_gU_A == gd
    @assert d_gU_B == gd                         # right link of B must exist
    @assert d_gR_C == gd                         # top link of C must exist

    D_lA, D_rA, D_uA, D_dA = size(T_A, 2), size(T_A, 3), size(T_A, 4), size(T_A, 5)
    D_lB, D_rB, D_uB, D_dB = size(T_B, 2), size(T_B, 3), size(T_B, 4), size(T_B, 5)
    D_lC, D_rC, D_uC, D_dC = size(T_C, 2), size(T_C, 3), size(T_C, 4), size(T_C, 5)

    # ── 1. Apply GA on T_A — gate acts on (iR_A = bottom-link, iU_A = left-link) ──
    # Physical flat index of T_A obeys site_idx ordering: nf*gd² + (iR-1)*gd + iU.
    # In Julia column-major reshape, this matches reshape(., gd, gd, df) where
    # the first dim is iU (fastest), second iR, third nf — see derivation in
    # `site_idx`.  We then permute to (nf, iR, iU, …) for natural contraction.
    T_A_s   = reshape(T_A, gd, gd, df, D_lA, D_rA, D_uA, D_dA)              # (iU, iR, nf, l, r, u, d)
    T_A_3d  = permutedims(T_A_s, (3, 2, 1, 4, 5, 6, 7))                     # (nf, iR, iU, l, r, u, d)
    @tensor T_A_g[nf, iRp, iUp, α, β, lA, rA, uA, dA] :=
        GA[iR, iU, iRp, iUp, α, β] * T_A_3d[nf, iR, iU, lA, rA, uA, dA]
    # Combine (rA, α) → new r-axis  and  (uA, β) → new u-axis (rA/uA varying fastest).
    T_A_p  = permutedims(T_A_g, (1, 2, 3, 6, 7, 4, 8, 5, 9))                # (nf, iRp, iUp, lA, rA, α, uA, β, dA)
    T_A_p2 = permutedims(T_A_p, (3, 2, 1, 4, 5, 6, 7, 8, 9))                # (iUp, iRp, nf, …)
    T_A_new = reshape(T_A_p2, d_pA, D_lA, D_rA * D_α, D_uA * D_β, D_dA)

    # ── 2. Apply GB on T_B — gate acts on iU_B (B's up-gauge = right link) ──
    T_B_s   = reshape(T_B, d_gU_B, d_gR_B, df, D_lB, D_rB, D_uB, D_dB)
    T_B_3d  = permutedims(T_B_s, (3, 2, 1, 4, 5, 6, 7))                     # (nf, iR_B, iU_B, l, r, u, d)
    @tensor T_B_g[nf, iR, iUp, α, lB, rB, uB, dB] :=
        GB[iU, iUp, α] * T_B_3d[nf, iR, iU, lB, rB, uB, dB]
    # Combine (lB, α) → new l-axis (lB fastest).
    T_B_p  = permutedims(T_B_g, (1, 2, 3, 5, 4, 6, 7, 8))                   # (nf, iR, iUp, lB, α, rB, uB, dB)
    T_B_p2 = permutedims(T_B_p, (3, 2, 1, 4, 5, 6, 7, 8))                   # (iUp, iR, nf, lB, α, …)
    T_B_new = reshape(T_B_p2, d_pB, D_lB * D_α, D_rB, D_uB, D_dB)

    # ── 3. Apply GC on T_C — gate acts on iR_C (C's right-gauge = top link) ──
    T_C_s   = reshape(T_C, d_gU_C, d_gR_C, df, D_lC, D_rC, D_uC, D_dC)
    T_C_3d  = permutedims(T_C_s, (3, 2, 1, 4, 5, 6, 7))                     # (nf, iR_C, iU_C, l, r, u, d)
    @tensor T_C_g[nf, iRp, iU, β, lC, rC, uC, dC] :=
        GC[iR, iRp, β] * T_C_3d[nf, iR, iU, lC, rC, uC, dC]
    # Combine (dC, β) → new d-axis (dC fastest).
    T_C_p  = permutedims(T_C_g, (1, 2, 3, 5, 6, 7, 8, 4))                   # (nf, iRp, iU, lC, rC, uC, dC, β)
    T_C_p2 = permutedims(T_C_p, (3, 2, 1, 4, 5, 6, 7, 8))                   # (iU, iRp, nf, lC, rC, uC, dC, β)
    T_C_new = reshape(T_C_p2, d_pC, D_lC, D_rC, D_uC, D_dC * D_β)

    # ── 4. Stash enlarged tensors and extend bond weights to match new dims ──
    peps.tensors[ix,   iy]   = T_A_new
    peps.tensors[ix+1, iy]   = T_B_new
    peps.tensors[ix,   iy+1] = T_C_new

    λ_AB_old = peps.λh[ix, iy]
    λ_AC_old = peps.λv[ix, iy]
    peps.λh[ix, iy] = repeat(λ_AB_old, D_α)   # length D_rA*D_α, matches new r/l-axes
    peps.λv[ix, iy] = repeat(λ_AC_old, D_β)   # length D_uA*D_β, matches new u/d-axes

    # ── 5. Compress A-B bond back to ≤ D_trunc (identity gate; SVD truncates) ──
    Id_AB = Matrix{ComplexF64}(I, d_pA * d_pB, d_pA * d_pB)
    update_bond_h!(peps, ix, iy, Id_AB, D_trunc)

    # ── 6. Compress A-C bond back to ≤ D_trunc ──
    Id_AC = Matrix{ComplexF64}(I, d_pA * d_pC, d_pA * d_pC)
    update_bond_v!(peps, ix, iy, Id_AC, D_trunc)

    return nothing
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Trotter sweep                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
One second-order Trotter step over all bonds of the finite lattice.
Order:
  [h-half] → [v-half] → [plaq-full] → [v-half] → [h-half]

The magnetic-plaquette gate is sandwiched in the middle of a symmetric
second-order split so that, combined with the symmetric h/v sweeps, the
overall step is O(τ³) accurate (modulo simple-update truncation error).
"""
function trotter_step!(peps::FinitePEPS, nx::Int, ny::Int, dg::Int;
                        g::Float64, t_hop::Float64, m::Float64,
                        τ::Float64, D_trunc::Int, μ::Float64=0.0)
    # Pre-build the plaquette gate once per call (same gauge gate for every plaquette).
    G_plaq = exp(-τ .* H_plaquette_gauge(dg; g=g))
    plaq_dec = decompose_plaquette_LShape(G_plaq, dg; cutoff=SVD_CUT)

    # ── Horizontal half-step (left to right sweep) ────────────────────────
    for iy in 1:ny, ix in 1:nx-1
        Hh = H_merged_h_site(ix, iy, nx, ny, dg; g=g, t=t_hop, m=m, μ=μ)
        gate = exp(-(τ / 2) .* Hh)
        update_bond_h!(peps, ix, iy, gate, D_trunc)
    end

    # ── Vertical half-step (bottom to top sweep) ──────────────────────────
    for iy in 1:ny-1, ix in 1:nx
        Hv = H_merged_v_site(ix, iy, nx, ny, dg; g=g, t=t_hop, m=m, μ=μ)
        gate = exp(-(τ / 2) .* Hv)
        update_bond_v!(peps, ix, iy, gate, D_trunc)
    end

    # ── Plaquette full-step (magnetic term) ───────────────────────────────
    for iy in 1:ny-1, ix in 1:nx-1
        update_plaquette!(peps, ix, iy,
                          plaq_dec.GA, plaq_dec.GB, plaq_dec.GC,
                          dg, D_trunc)
    end

    # ── Vertical half-step (top to bottom sweep, for symmetry) ───────────
    for iy in ny-1:-1:1, ix in nx:-1:1
        Hv = H_merged_v_site(ix, iy, nx, ny, dg; g=g, t=t_hop, m=m, μ=μ)
        gate = exp(-(τ / 2) .* Hv)
        update_bond_v!(peps, ix, iy, gate, D_trunc)
    end

    # ── Horizontal half-step (right to left sweep, for symmetry) ─────────
    for iy in ny:-1:1, ix in nx-1:-1:1
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
    plaquette_energy(peps, ix, iy, dg; g)

Expectation value of the magnetic-plaquette Hamiltonian on the L-shape
(A=(ix,iy), B=(ix+1,iy), C=(ix,iy+1)) using the simple-update environment
(sqrt(λ) absorbed on each transverse bond, diagonal sum).

Smart contraction: only ket/bra indices of the *internal* A–B and A–C bonds
survive in the intermediate tensors, keeping memory at O(D⁴).
"""
function plaquette_energy(peps::FinitePEPS, ix::Int, iy::Int, dg::Int; g::Float64)
    nx, ny = peps.nx, peps.ny
    @assert ix < nx && iy < ny
    gd = gauge_dim(dg)

    T_A = peps.tensors[ix,   iy]
    T_B = peps.tensors[ix+1, iy]
    T_C = peps.tensors[ix,   iy+1]

    _, d_gR_A, d_gU_A = site_dims(ix,   iy,   nx, ny, dg)
    _, d_gR_B, d_gU_B = site_dims(ix+1, iy,   nx, ny, dg)
    _, d_gR_C, d_gU_C = site_dims(ix,   iy+1, nx, ny, dg)

    # sqrt bond weights — internal (A–B, A–C) and external
    sq_lA = (ix > 1)         ? sqrt.(peps.λh[ix-1, iy])   : ones(1)
    sq_dA = (iy > 1)         ? sqrt.(peps.λv[ix,   iy-1]) : ones(1)
    sq_AB = sqrt.(peps.λh[ix, iy])                          # internal
    sq_AC = sqrt.(peps.λv[ix, iy])                          # internal
    sq_rB = (ix+1 < nx)      ? sqrt.(peps.λh[ix+1, iy])   : ones(1)
    sq_uB = (iy   < ny)      ? sqrt.(peps.λv[ix+1, iy])   : ones(1)
    sq_dB = (iy > 1)         ? sqrt.(peps.λv[ix+1, iy-1]) : ones(1)
    sq_lC = (ix > 1)         ? sqrt.(peps.λh[ix-1, iy+1]) : ones(1)
    sq_rC = (ix   < nx)      ? sqrt.(peps.λh[ix,   iy+1]) : ones(1)
    sq_uC = (iy+1 < ny)      ? sqrt.(peps.λv[ix,   iy+1]) : ones(1)

    function _absorb(T, wl, wr, wu, wd)
        S = copy(T)
        for p in axes(S,1), l in axes(S,2), r in axes(S,3), u in axes(S,4), d in axes(S,5)
            S[p,l,r,u,d] *= wl[l] * wr[r] * wu[u] * wd[d]
        end
        return S
    end

    T_A_eff = _absorb(T_A, sq_lA, sq_AB, sq_AC, sq_dA)
    T_B_eff = _absorb(T_B, sq_AB, sq_rB, sq_uB, sq_dB)
    T_C_eff = _absorb(T_C, sq_lC, sq_rC, sq_uC, sq_AC)

    # Single-site Wilson-loop operators on each node's physical leg.
    #   W = U†_bottom ⊗ U_left ⊗ U†_right ⊗ U_top
    #     A: bottom = R-gauge → U†_R;   left = U-gauge → U_U
    #     B: right  = U-gauge → U†_U
    #     C: top    = R-gauge → U_R
    Ug = op_U_gauge(dg)
    Ud = op_Udag_gauge(dg)
    W_A = kron(_Id(LGT_d_f), Ud, Ug)                                          # d_f × gd × gd
    W_B = (d_gR_B == 1) ? kron(_Id(LGT_d_f), Ud) : kron(_Id(LGT_d_f), _Id(d_gR_B), Ud)
    W_C = (d_gU_C == 1) ? kron(_Id(LGT_d_f), Ug) : kron(_Id(LGT_d_f), Ug, _Id(d_gU_C))

    function _apply(T, O)
        sz = size(T)
        return reshape(O * reshape(T, sz[1], :), sz...)
    end

    T_A_W = _apply(T_A_eff, W_A)
    T_B_W = _apply(T_B_eff, W_B)
    T_C_W = _apply(T_C_eff, W_C)

    # Local M tensors — physical and *external* bonds summed (external = diagonal).
    function _M_A(Tk, Tb)
        @tensor MA[rk, rb, uk, ub] :=
            Tk[p, lA, rk, uk, dA] * conj(Tb[p, lA, rb, ub, dA])
        return MA
    end
    function _M_B(Tk, Tb)
        @tensor MB[lk, lb] :=
            Tk[p, lk, rB, uB, dB] * conj(Tb[p, lb, rB, uB, dB])
        return MB
    end
    function _M_C(Tk, Tb)
        @tensor MC[dk, db] :=
            Tk[p, lC, rC, uC, dk] * conj(Tb[p, lC, rC, uC, db])
        return MC
    end

    function _scalar(MA, MB, MC)
        @tensor MAB[uk, ub] := MA[rk, rb, uk, ub] * MB[rk, rb]
        @tensor val[]       := MAB[uk, ub] * MC[uk, ub]
        return val[]
    end

    num = _scalar(_M_A(T_A_W, T_A_eff), _M_B(T_B_W, T_B_eff), _M_C(T_C_W, T_C_eff))
    den = _scalar(_M_A(T_A_eff, T_A_eff), _M_B(T_B_eff, T_B_eff), _M_C(T_C_eff, T_C_eff))

    # ⟨H_plaq⟩ = -1/(2g²) (⟨W⟩ + ⟨W†⟩) = -1/g² · Re⟨W⟩
    return abs(den) > 1e-15 ? -(1 / g^2) * real(num) / real(den) : 0.0
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

    # Hopping: vertical bonds  (D = lower = (ix,iy),  U = upper = (ix,iy+1))
    for iy in 1:ny-1, ix in 1:nx
        T_D = peps.tensors[ix, iy]
        T_U = peps.tensors[ix, iy+1]
        dp_D = size(T_D, 1);  dp_U = size(T_U, 1)

        λ_bond = peps.λv[ix, iy]
        bond_dim = length(λ_bond)
        sq_bond = sqrt.(λ_bond)

        λ_l_D = (ix > 1)         ? peps.λh[ix-1, iy]   : ones(1)
        λ_r_D = (ix < nx)        ? peps.λh[ix,   iy]   : ones(1)
        λ_d_D = (iy > 1)         ? peps.λv[ix,   iy-1] : ones(1)
        λ_l_U = (ix > 1)         ? peps.λh[ix-1, iy+1] : ones(1)
        λ_r_U = (ix < nx)        ? peps.λh[ix,   iy+1] : ones(1)
        λ_u_U = (iy+1 < ny)      ? peps.λv[ix,   iy+1] : ones(1)

        D_w = copy(T_D)
        for p in 1:dp_D, l in axes(D_w,2), r in axes(D_w,3), u in 1:bond_dim, d in axes(D_w,5)
            D_w[p,l,r,u,d] *= sqrt(λ_l_D[l]) * sqrt(λ_r_D[r]) * sq_bond[u] * sqrt(λ_d_D[d])
        end
        U_w = copy(T_U)
        for p in 1:dp_U, l in axes(U_w,2), r in axes(U_w,3), u in axes(U_w,4), d in 1:bond_dim
            U_w[p,l,r,u,d] *= sqrt(λ_l_U[l]) * sqrt(λ_r_U[r]) * sqrt(λ_u_U[u]) * sq_bond[d]
        end

        sz_D = size(D_w);  sz_U = size(U_w)
        # Bond index χ is at position 4 of T_D and position 5 of T_U.
        D_bond_t = reshape(permutedims(D_w, (1, 4, 2, 3, 5)),
                           dp_D, bond_dim, sz_D[2]*sz_D[3]*sz_D[5])
        U_bond_t = reshape(permutedims(U_w, (1, 5, 2, 3, 4)),
                           dp_U, bond_dim, sz_U[2]*sz_U[3]*sz_U[4])

        Θ = zeros(ComplexF64, dp_D, dp_U, sz_D[2]*sz_D[3]*sz_D[5], sz_U[2]*sz_U[3]*sz_U[4])
        for p1 in 1:dp_D, p2 in 1:dp_U, χ in 1:bond_dim
            Θ[p1, p2, :, :] .+= D_bond_t[p1, χ, :] * U_bond_t[p2, χ, :]'
        end

        H_hop = H_hop_v_site(ix, iy, nx, ny, dg; t=t_hop)

        Θ_flat = reshape(Θ, dp_D*dp_U, :)
        ρ = Θ_flat * Θ_flat'
        numerator   = real(tr(reshape(H_hop, dp_D*dp_U, dp_D*dp_U) * ρ))
        denominator = real(tr(ρ))
        E += denominator > 1e-15 ? numerator / denominator : 0.0
    end

    # Plaquettes (magnetic term, L-shape over A=(ix,iy), B=(ix+1,iy), C=(ix,iy+1))
    for iy in 1:ny-1, ix in 1:nx-1
        E += plaquette_energy(peps, ix, iy, dg; g=g)
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
        measure_every::Int = MEASURE_EVERY,
        noise::Float64 = 0.0,
        plot_dir::String = @__DIR__)

    println("══════════════════════════════════════════════════════════════")
    println("  U(1) LGT — finite PEPS ground state (imaginary-time SU)")
    println("══════════════════════════════════════════════════════════════")
    @printf("  Lattice: %d × %d  (nx × ny)\n", nx, ny)
    @printf("  Gauge truncation dg = %d  (link dim = %d)\n", dg, gauge_dim(dg))
    @printf("  Bond D = %d → D_max = %d\n", D_bond, D_max)
    @printf("  g = %.3f,  t = %.3f,  m = %.3f\n", g, t_hop, m_mass)
    @printf("  τ = %.4e,  steps = %d,  noise = %.2e\n\n", τ, n_steps, noise)

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

    peps = init_finite_peps(nx, ny, dg, D_bond; noise=noise)

    times     = Float64[]
    nf_hist   = Float64[]
    E_hist    = Float64[]
    Eplaq_hist = Float64[]

    function _measure!(step::Int)
        nf  = mean_nf(peps, nx, ny, dg)
        E   = total_energy(peps, nx, ny, dg; g=g, t_hop=t_hop, m=m_mass)
        Ep  = sum(plaquette_energy(peps, ix, iy, dg; g=g)
                  for iy in 1:ny-1, ix in 1:nx-1; init=0.0)
        push!(times,      step * τ)
        push!(nf_hist,    nf)
        push!(E_hist,     E)
        push!(Eplaq_hist, Ep)
        D_now = maximum(length(peps.λh[ix, iy]) for iy in 1:ny for ix in 1:nx-1;
                        init=1)
        @printf("  %6d  |  %.6f  | %12.6f | %12.6f |  D=%d\n",
                step, nf, E, Ep, D_now)
    end

    @printf("  %6s  |  %-10s | %-12s | %-12s |\n",
            "step", "⟨n_f⟩_mean", "E_total", "E_plaq")
    @printf("  ───────┼────────────┼──────────────┼──────────────┼────────\n")
    _measure!(0)

    for step in 1:n_steps
        trotter_step!(peps, nx, ny, dg; g=g, t_hop=t_hop, m=m_mass,
                      τ=τ, D_trunc=D_max)

        if step % measure_every == 0 || step == n_steps
            _measure!(step)
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
             title  = "Finite PEPS ground state  ($(nx)×$(ny), m=$(m_mass), D=$D_max)",
             lw = 2, marker = :circle, ms = 2, label = "⟨n_f⟩")
    hline!(p, [0.5]; label = "0.5", ls = :dash, color = :gray)
    savefig(p, joinpath(plot_dir, "finite_peps_groundstate_nf.png"))

    p2 = plot(times, E_hist;     lw=2, marker=:circle, ms=2, label="E_total",
              xlabel="imaginary time", ylabel="energy",
              title="Energy trajectory  (D=$D_max)")
    plot!(p2, times, Eplaq_hist; lw=2, marker=:square, ms=2, label="E_plaq")
    savefig(p2, joinpath(plot_dir, "finite_peps_groundstate_energy.png"))

    println("  Plots saved to $(plot_dir).")
    println("══════════════════════════════════════════════════════════════")

    return peps, times, nf_hist, E_hist, Eplaq_hist
end

# ── Run (only when executed directly, not when included) ─────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    peps_final, times, nf_hist, E_hist, Eplaq_hist = run_finite_peps_groundstate()
end
