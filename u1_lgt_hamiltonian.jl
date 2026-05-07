#= ═══════════════════════════════════════════════════════════════════════════════
   u1_lgt_hamiltonian.jl

   U(1) Lattice Gauge Theory Hamiltonian for iPEPS / iPEPO Trotter evolution.
   Compatible with the simple-update Trotter code in tensorkit_tst.jl.

   ── Lattice ──────────────────────────────────────────────────────────────────
   Square lattice.  Each iPEPS node covers:
     • one lattice site          (spinless fermion,   dim = 2)
     • the rightward link        (U(1) gauge field,   dim = 2·d_gauge+1)
     • the upward link           (U(1) gauge field,   dim = 2·d_gauge+1)

   Total physical dimension per node:  d_node = d_f × d_g²
   Basis ordering inside a node:  |n_f, e_R, e_U⟩
     n_f  ∈ {0, 1}                    fermion occupation
     e_R  ∈ {-d_gauge, …, d_gauge}    right-link electric field
     e_U  ∈ {-d_gauge, …, d_gauge}    up-link electric field

   ── Hamiltonian ──────────────────────────────────────────────────────────────
     H  =  (g²/2) Σ_l E_l²
          - (1/2g²) Σ_plaq [ U_top U†_right U†_bottom U_left + h.c. ]
          - t Σ_{i,μ} [ ψ†_i U_{i,μ} ψ_{i+μ} + h.c. ]
          + m Σ_i (-1)^{ix+iy} ψ†_i ψ_i

   where E|e⟩ = e|e⟩,  U|e⟩ = |e+1⟩  (annihilated if e = d_gauge).

   ── Trotter decomposition ────────────────────────────────────────────────────
   Terms are grouped for a second-order Suzuki-Trotter step:
     1. On-site  (electric-field energy + staggered mass)  →  1-node gate
     2. Horizontal hopping  (-t ψ†_L U_{R,L} ψ_R + h.c.)  →  2-node gate
     3. Vertical hopping    (-t ψ†_D U_{U,D} ψ_U + h.c.)  →  2-node gate
     4. Plaquette           →  3-node / 4-gauge-link gate (see notes below)

   On-site energy is folded (¼ + ¼) into the horizontal and vertical 2-site
   gates so that the simple-update scheme from tensorkit_tst.jl can be used
   directly for terms 1–3.

   The plaquette term couples gauge DoFs on 3 distinct nodes and cannot be
   represented as a 2-site gate.  It is provided both as a raw exp(-τ H_plaq)
   and as an SVD decomposition into three tensors (one per node).
   ═══════════════════════════════════════════════════════════════════════════ =#

# When included from another script, skip Pkg setup (caller handles it).
if !@isdefined(_LGT_HAMILTONIAN_LOADED)

using Pkg
Pkg.activate(".")
for pkg in ["TensorKit", "TensorOperations", "LinearAlgebra"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

const _LGT_HAMILTONIAN_LOADED = true
end # guard

using TensorKit
using TensorOperations
using LinearAlgebra
using Printf

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Fermion flavour flag ─────────────────────────────────────────────────────
# TODO (SPINFUL): Set SPINFUL = true and update the following:
#   • d_f  → 4   (basis |0⟩, |↑⟩, |↓⟩, |↑↓⟩)
#   • op_c()       → return (c_↑, c_↓) tuple
#   • op_nf()      → n_↑ + n_↓
#   • op_parity_f()→ diag(1, -1, -1, 1)
#   • hopping terms: sum over spin components   -t Σ_σ ψ†_{i,σ} U ψ_{i+μ,σ}
const SPINFUL = false
const LGT_d_f = SPINFUL ? 4 : 2     # fermion Hilbert-space dimension

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Hilbert-space dimensions & basis indexing                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Dimension of one U(1) gauge link truncated at ±`dg`."""
gauge_dim(dg::Int) = 2dg + 1

"""Total Hilbert-space dimension of one iPEPS node."""
node_dim(dg::Int) = LGT_d_f * gauge_dim(dg)^2

"""
    node_idx(nf, er, eu, dg)

1-based flat index for node basis state |nf, er, eu⟩.
"""
function node_idx(nf::Int, er::Int, eu::Int, dg::Int)
    gd = gauge_dim(dg)
    return nf * gd^2 + (er + dg) * gd + (eu + dg) + 1
end

"""1-based index for gauge-link state |e⟩."""
gauge_idx(e::Int, dg::Int) = e + dg + 1

"""TensorKit physical space for one iPEPS node."""
node_physical_space(dg::Int) = ℂ^node_dim(dg)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Elementary operators                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

_Id(n::Int) = Matrix{ComplexF64}(I, n, n)

# ── Gauge-link operators  (gauge_dim × gauge_dim) ────────────────────────────

"""Electric field  E|e⟩ = e|e⟩."""
function op_E(dg::Int)
    gd = gauge_dim(dg)
    E = zeros(ComplexF64, gd, gd)
    for e in -dg:dg
        E[gauge_idx(e, dg), gauge_idx(e, dg)] = e
    end
    return E
end

"""E² operator on a single link."""
op_E2(dg::Int) = let E = op_E(dg); E * E end

"""Parallel transporter (raising):  U|e⟩ = |e+1⟩,  U|d_gauge⟩ = 0."""
function op_U_gauge(dg::Int)
    gd = gauge_dim(dg)
    U = zeros(ComplexF64, gd, gd)
    for e in -dg:dg-1
        U[gauge_idx(e + 1, dg), gauge_idx(e, dg)] = 1.0
    end
    return U
end

"""Lowering:  U†|e⟩ = |e-1⟩,  U†|−d_gauge⟩ = 0."""
op_Udag_gauge(dg::Int) = collect(op_U_gauge(dg)')

# ── Fermion operators  (LGT_d_f × LGT_d_f) ─────────────────────────────────

"""Annihilation operator  c|1⟩ = |0⟩."""
function op_c()
    # TODO (SPINFUL): return (c_up, c_dn) for the two spin components
    c = zeros(ComplexF64, LGT_d_f, LGT_d_f)
    c[1, 2] = 1.0
    return c
end

"""Creation operator  c†."""
op_cdag() = collect(op_c()')

"""Number operator  n_f = c†c."""
function op_nf()
    # TODO (SPINFUL): n_f = n_↑ + n_↓
    nf = zeros(ComplexF64, LGT_d_f, LGT_d_f)
    nf[2, 2] = 1.0
    return nf
end

"""Parity  P_f = (-1)^{n_f}."""
function op_parity_f()
    # TODO (SPINFUL): P = diag(1, -1, -1, 1)
    P = _Id(LGT_d_f)
    P[2, 2] = -1.0
    return P
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Node-level embedding                                                   ║
# ║                                                                          ║
# ║  Node basis:  fermion ⊗ right-gauge ⊗ up-gauge                          ║
# ║  kron(A, B, C)  ↔  A acts on fermion, B on right-gauge, C on up-gauge   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Embed a fermion operator (d_f × d_f) into the full node space."""
embed_f(O, dg) = kron(O, _Id(gauge_dim(dg)), _Id(gauge_dim(dg)))

"""Embed a right-gauge operator (dg × dg) into the full node space."""
embed_R(O, dg) = kron(_Id(LGT_d_f), O, _Id(gauge_dim(dg)))

"""Embed an up-gauge operator (dg × dg) into the full node space."""
embed_U(O, dg) = kron(_Id(LGT_d_f), _Id(gauge_dim(dg)), O)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Hamiltonian terms                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── 1. On-site (single node) ────────────────────────────────────────────────

"""
    H_onsite(dg; g, m, staggered_sign)

Single-node Hamiltonian:

    H = (g²/2)(E²_R + E²_U) + m·staggered_sign·n_f

`staggered_sign` = (-1)^{ix+iy} = ±1.

Returns a Hermitian `(dn × dn)` matrix.
"""
function H_onsite(dg::Int; g::Float64, m::Float64, staggered_sign::Int = 1)
    H  = (g^2 / 2) .* (embed_R(op_E2(dg), dg) .+ embed_U(op_E2(dg), dg))
    H .+= m * staggered_sign .* embed_f(op_nf(), dg)
    return 0.5 .* (H .+ H')   # ensure Hermitianity
end

# ── 2. Horizontal hopping (two nodes: L → R) ────────────────────────────────

"""
    H_hop_h(dg; t)

Two-node horizontal hopping:

    -t [ ψ†_L · U_{R-gauge,L} · ψ_R  +  h.c. ]

The gauge link `U` acts on the right-gauge DoF of node L.
Returns a Hermitian `(dn² × dn²)` matrix.

Jordan-Wigner note: for nearest-neighbour spinless fermions the JW
string is trivial (no intervening sites).  P_f is not needed here.
"""
function H_hop_h(dg::Int; t::Float64)
    gd = gauge_dim(dg)
    # Left node: c† ⊗ U_right ⊗ I_up
    cdag_U_L = kron(op_cdag(), op_U_gauge(dg), _Id(gd))
    # Right node: c ⊗ I_right ⊗ I_up
    c_R = embed_f(op_c(), dg)

    hop = kron(cdag_U_L, c_R)     # (dn² × dn²)
    H = -t .* (hop .+ hop')
    return 0.5 .* (H .+ H')
end

# ── 3. Vertical hopping (two nodes: D → U) ──────────────────────────────────

"""
    H_hop_v(dg; t)

Two-node vertical hopping:

    -t [ ψ†_D · U_{U-gauge,D} · ψ_U  +  h.c. ]

The gauge link `U` acts on the up-gauge DoF of node D (lower).
Returns a Hermitian `(dn² × dn²)` matrix.
"""
function H_hop_v(dg::Int; t::Float64)
    gd = gauge_dim(dg)
    # Lower node: c† ⊗ I_right ⊗ U_up
    cdag_U_D = kron(op_cdag(), _Id(gd), op_U_gauge(dg))
    # Upper node: c ⊗ I_right ⊗ I_up
    c_U = embed_f(op_c(), dg)

    hop = kron(cdag_U_D, c_U)
    H = -t .* (hop .+ hop')
    return 0.5 .* (H .+ H')
end

# ── 4. Plaquette ─────────────────────────────────────────────────────────────
#
#  The plaquette with lower-left corner at (i,j) involves 4 gauge links
#  distributed across 3 iPEPS nodes:
#
#     Node A = (i,  j  ):  bottom ≡ right-gauge,  left ≡ up-gauge
#     Node B = (i+1,j  ):  right  ≡ up-gauge
#     Node C = (i,  j+1):  top    ≡ right-gauge
#
#  Wilson loop (user convention):
#     W = U_top · U†_right · U†_bottom · U_left
#
#  As a tensor product on the 4 links in order (bottom, left, right, top):
#     W = U†_bottom ⊗ U_left ⊗ U†_right ⊗ U_top
#
#  H_plaq = -(1/2g²)(W + W†)
# ─────────────────────────────────────────────────────────────────────────────

"""
    H_plaquette_gauge(dg; g)

Pure-gauge plaquette Hamiltonian on the 4 links:

    -(1/2g²) [ U†_bottom ⊗ U_left ⊗ U†_right ⊗ U_top  +  h.c. ]

Returned matrix has size `(dg⁴ × dg⁴)` in the basis
`|e_bottom, e_left, e_right, e_top⟩`.
"""
function H_plaquette_gauge(dg::Int; g::Float64)
    Ug  = op_U_gauge(dg)
    Ud  = op_Udag_gauge(dg)
    W   = kron(Ud, Ug, Ud, Ug)             # U†_b ⊗ U_l ⊗ U†_r ⊗ U_t
    H   = -(1 / (2 * g^2)) .* (W .+ W')
    return 0.5 .* (H .+ H')
end

"""
    H_plaquette_full(dg; g)

Plaquette Hamiltonian embedded into the full node Hilbert spaces
of the 3 nodes:  node_A ⊗ node_B ⊗ node_C.

    node_A = (i,j):      U† on right-gauge,  U on up-gauge
    node_B = (i+1,j):    U† on up-gauge
    node_C = (i,j+1):    U on right-gauge

Returns `(dn³ × dn³)`.  Practical only for small `dg`.
"""
function H_plaquette_full(dg::Int; g::Float64)
    gd = gauge_dim(dg)
    Ug = op_U_gauge(dg)
    Ud = op_Udag_gauge(dg)
    # Node A:  I_f ⊗ U†_right ⊗ U_up
    OA = kron(_Id(LGT_d_f), Ud, Ug)
    # Node B:  I_f ⊗ I_right ⊗ U†_up
    OB = kron(_Id(LGT_d_f), _Id(gd), Ud)
    # Node C:  I_f ⊗ U_right ⊗ I_up
    OC = kron(_Id(LGT_d_f), Ug, _Id(gd))

    W = kron(OA, OB, OC)
    H = -(1 / (2 * g^2)) .* (W .+ W')
    return 0.5 .* (H .+ H')
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Merged 2-site Hamiltonians  (on-site energy folded in)                 ║
# ║                                                                          ║
# ║  Each site participates in one horizontal and one vertical 2-site gate.  ║
# ║  Distributing ½ of the on-site energy to each bond direction:           ║
# ║    H_h_merged = H_hop_h + ¼ (H_onsite ⊗ I  +  I ⊗ H_onsite)            ║
# ║    H_v_merged = H_hop_v + ¼ (H_onsite ⊗ I  +  I ⊗ H_onsite)            ║
# ║                                                                          ║
# ║  For the staggered-mass term (-1)^{ix+iy} a ≥2-site unit cell is       ║
# ║  needed.  For a 1-site unit cell set m=0 or use the AB-sublattice       ║
# ║  variants below.                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    H_merged_h(dg; g, t, m)

Horizontal 2-site Hamiltonian with on-site energy folded in (stagger = +1).
For a 1×1 unit cell with no staggering; set `m=0` if staggering is needed
and use `H_merged_h_AB` instead.
"""
function H_merged_h(dg::Int; g::Float64, t::Float64, m::Float64 = 0.0)
    dn  = node_dim(dg)
    Idn = _Id(dn)
    Hos = H_onsite(dg; g=g, m=m, staggered_sign=1)
    Hh  = H_hop_h(dg; t=t)
    return Hh .+ 0.25 .* (kron(Hos, Idn) .+ kron(Idn, Hos))
end

"""
    H_merged_v(dg; g, t, m)

Vertical 2-site Hamiltonian with on-site energy folded in (stagger = +1).
"""
function H_merged_v(dg::Int; g::Float64, t::Float64, m::Float64 = 0.0)
    dn  = node_dim(dg)
    Idn = _Id(dn)
    Hos = H_onsite(dg; g=g, m=m, staggered_sign=1)
    Hv  = H_hop_v(dg; t=t)
    return Hv .+ 0.25 .* (kron(Hos, Idn) .+ kron(Idn, Hos))
end

"""
    H_merged_h_AB(dg; g, t, m, sign_L, sign_R)

Horizontal 2-site Hamiltonian for an AB-sublattice unit cell.
`sign_L`, `sign_R` are (-1)^{ix+iy} for the left and right sites.
"""
function H_merged_h_AB(dg::Int; g::Float64, t::Float64, m::Float64,
                        sign_L::Int, sign_R::Int)
    dn  = node_dim(dg)
    Idn = _Id(dn)
    Hos_L = H_onsite(dg; g=g, m=m, staggered_sign=sign_L)
    Hos_R = H_onsite(dg; g=g, m=m, staggered_sign=sign_R)
    Hh    = H_hop_h(dg; t=t)
    return Hh .+ 0.25 .* (kron(Hos_L, Idn) .+ kron(Idn, Hos_R))
end

"""
    H_merged_v_AB(dg; g, t, m, sign_D, sign_U)

Vertical 2-site Hamiltonian for an AB-sublattice unit cell.
"""
function H_merged_v_AB(dg::Int; g::Float64, t::Float64, m::Float64,
                        sign_D::Int, sign_U::Int)
    dn  = node_dim(dg)
    Idn = _Id(dn)
    Hos_D = H_onsite(dg; g=g, m=m, staggered_sign=sign_D)
    Hos_U = H_onsite(dg; g=g, m=m, staggered_sign=sign_U)
    Hv    = H_hop_v(dg; t=t)
    return Hv .+ 0.25 .* (kron(Hos_D, Idn) .+ kron(Idn, Hos_U))
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Trotter gate construction                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Exponentiate: G = exp(-τ H)."""
lgt_gate(H::AbstractMatrix, τ::Real) = exp(-τ .* H)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Plaquette gate SVD decomposition                                       ║
# ║                                                                          ║
# ║  G_plaq ≈ Σ_{α,β}  G_A[b,l,b',l',α] · G_B[r,r',α,β] · G_C[t,t',β]   ║
# ║                                                                          ║
# ║  where (b,l) are gauge DoFs of node A, r of node B, t of node C.       ║
# ║  The plaquette touches 3 nodes that form an L-shape:                    ║
# ║                                                                          ║
# ║          C ── (top link) ──→                                            ║
# ║          |                                                              ║
# ║     (left link)                                                         ║
# ║          |                                                              ║
# ║          A ── (bottom link) ── B                                        ║
# ║                     (right link of B goes ↑)                            ║
# ║                                                                          ║
# ║  This decomposition allows sequential application as 2-body operations  ║
# ║  with enlarged bond dimension (auxiliary indices α, β).                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    decompose_plaquette_gate(G_gauge, dg; D_aux_max, cutoff)

SVD-decompose the gauge-only plaquette gate `(dg⁴ × dg⁴)` into three
tensors for application on nodes A, B, C.

Returns a named tuple `(GA, GB, GC, D_alpha, D_beta)`:
  `GA` : `(dg, dg, dg, dg, D_alpha)`   — indices `(b, l, b', l', α)`
  `GB` : `(dg, dg, D_alpha, D_beta)`   — indices `(r, r', α, β)`
  `GC` : `(dg, dg, D_beta)`            — indices `(t, t', β)`
"""
function decompose_plaquette_gate(G_gauge::AbstractMatrix, dg::Int;
                                   D_aux_max::Int = 20,
                                   cutoff::Float64 = 1e-12)
    gd = gauge_dim(dg)

    # G is (gd⁴ × gd⁴). Reshape to 8-index tensor:
    #   (b, l, r, t,  b', l', r', t')
    G8 = reshape(G_gauge, gd, gd, gd, gd, gd, gd, gd, gd)

    # Permute to  (b, l, b', l',   r, r',   t, t')
    #   old indices: 1  2  3  4   5  6  7  8
    #   new order:   1  2  5  6   3  7   4  8
    G_perm = permutedims(G8, (1, 2, 5, 6,  3, 7,  4, 8))

    # ── First SVD: {b, l, b', l'}  vs  {r, r', t, t'} ────────────────────
    dim_A  = gd^4
    dim_BC = gd^4
    M1 = reshape(G_perm, dim_A, dim_BC)
    F1 = svd(M1)

    D_alpha = min(D_aux_max, count(F1.S .> cutoff), length(F1.S))
    D_alpha = max(D_alpha, 1)
    sqS1 = sqrt.(F1.S[1:D_alpha])

    GA_mat = F1.U[:, 1:D_alpha]  * Diagonal(sqS1)     # (gd⁴, D_α)
    BC_mat = Diagonal(sqS1)      * F1.Vt[1:D_alpha, :] # (D_α, gd⁴)

    GA = reshape(GA_mat, gd, gd, gd, gd, D_alpha)      # (b, l, b', l', α)

    # ── Second SVD: {r, r', α}  vs  {t, t'} ──────────────────────────────
    # BC_mat → (D_α, r, r', t, t')
    BC5 = reshape(BC_mat, D_alpha, gd, gd, gd, gd)
    # Permute to (r, r', α, t, t')
    BC_perm = permutedims(BC5, (2, 3, 1, 4, 5))

    dim_B = gd * gd * D_alpha
    dim_C = gd * gd
    M2 = reshape(BC_perm, dim_B, dim_C)
    F2 = svd(M2)

    D_beta = min(D_aux_max, count(F2.S .> cutoff), length(F2.S))
    D_beta = max(D_beta, 1)
    sqS2 = sqrt.(F2.S[1:D_beta])

    GB_mat = F2.U[:, 1:D_beta]  * Diagonal(sqS2)       # (dim_B, D_β)
    GC_mat = Diagonal(sqS2)     * F2.Vt[1:D_beta, :]   # (D_β, dim_C)

    GB = reshape(GB_mat, gd, gd, D_alpha, D_beta)       # (r, r', α, β)
    GC_raw = reshape(GC_mat, D_beta, gd, gd)            # (β, t, t')
    GC = permutedims(GC_raw, (2, 3, 1))                 # (t, t', β)

    return (GA=GA, GB=GB, GC=GC, D_alpha=D_alpha, D_beta=D_beta)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Full gate-set builder                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    build_lgt_gates(dg; g, t, m, τ, D_aux_max) → NamedTuple

Build every Trotter gate required for one imaginary-time step of the U(1) LGT.

Fields of the returned tuple
─────────────────────────────
  `G_hop_h`     2-site horizontal gate (merged with ½ onsite), size `(dn²×dn²)`
  `G_hop_v`     2-site vertical gate   (merged with ½ onsite), size `(dn²×dn²)`
  `G_plaq_raw`  plaquette gate on 4 gauge links, size `(dg⁴×dg⁴)`
  `G_plaq`      SVD-decomposed plaquette `(GA, GB, GC, D_alpha, D_beta)`
  `H_hop_h_mat` horizontal Hamiltonian matrix (for custom Trotter splitting)
  `H_hop_v_mat` vertical Hamiltonian matrix
  `H_plaq_mat`  plaquette Hamiltonian matrix (gauge-only)

Usage with tensorkit_tst.jl simple-update
─────────────────────────────────────────
    gates = build_lgt_gates(dg; g=1.0, t=1.0, m=0.5, τ=0.01)
    # horizontal half-step + vertical full-step + horizontal half-step:
    simple_update_horizontal!(peps, gates.G_hop_h_half, D_max)
    simple_update_vertical!(peps,   gates.G_hop_v_full, D_max)
    simple_update_horizontal!(peps, gates.G_hop_h_half, D_max)
    # plaquette: see notes at top of file

Note: for a lattice with staggered mass (m ≠ 0), use a 2-site unit cell and
      the `_AB` merged Hamiltonian variants.
"""
function build_lgt_gates(dg::Int;
                          g::Float64    = 1.0,
                          t::Float64    = 1.0,
                          m::Float64    = 0.0,
                          τ::Float64    = 0.01,
                          D_aux_max::Int = 20)
    # ── merged 2-site Hamiltonians (on-site energy folded in) ─────────────
    Hh = H_merged_h(dg; g=g, t=t, m=m)
    Hv = H_merged_v(dg; g=g, t=t, m=m)

    # ── plaquette (gauge DoFs only) ───────────────────────────────────────
    Hp = H_plaquette_gauge(dg; g=g)

    # ── exponentiate ──────────────────────────────────────────────────────
    G_hop_h_half = lgt_gate(Hh, τ / 2)
    G_hop_h_full = lgt_gate(Hh, τ)
    G_hop_v_half = lgt_gate(Hv, τ / 2)
    G_hop_v_full = lgt_gate(Hv, τ)
    G_plaq_raw   = lgt_gate(Hp, τ)

    # ── decompose plaquette ───────────────────────────────────────────────
    G_plaq = decompose_plaquette_gate(G_plaq_raw, dg; D_aux_max=D_aux_max)

    return (
        G_hop_h_half = G_hop_h_half,
        G_hop_h_full = G_hop_h_full,
        G_hop_v_half = G_hop_v_half,
        G_hop_v_full = G_hop_v_full,
        G_plaq_raw   = G_plaq_raw,
        G_plaq       = G_plaq,
        H_hop_h_mat  = Hh,
        H_hop_v_mat  = Hv,
        H_plaq_mat   = Hp,
    )
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Observables (single-node expectation values)                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Fermion number operator embedded in the node space."""
obs_nf(dg::Int) = embed_f(op_nf(), dg)

"""Electric-field operator on the right link, embedded in node space."""
obs_E_right(dg::Int) = embed_R(op_E(dg), dg)

"""Electric-field operator on the up link, embedded in node space."""
obs_E_up(dg::Int) = embed_U(op_E(dg), dg)

"""E² summed over both links of a node."""
obs_E2_total(dg::Int) = embed_R(op_E2(dg), dg) .+ embed_U(op_E2(dg), dg)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Gauss-law projector  (for diagnostics / initialisation)                ║
# ║                                                                          ║
# ║  Gauss's law at site i:                                                  ║
# ║    E_{i,R} - E_{i-x̂,R} + E_{i,U} - E_{i-ŷ,U}  =  n_f,i  (− background)║
# ║                                                                          ║
# ║  For a 1-site unit cell (all nodes identical) with zero background:     ║
# ║    (E_R - E_R) + (E_U - E_U)  =  n_f   →   0 = n_f   (trivial)        ║
# ║  A nontrivial check requires ≥2-site unit cell.                        ║
# ║                                                                          ║
# ║  Below we provide the single-node Gauss operator for embedding into     ║
# ║  a larger unit cell.                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    gauss_operator_node(dg)

Returns `G_node = E_R + E_U - n_f` as a `(dn × dn)` matrix.
For a full Gauss-law check on a 2-site unit cell, combine with the
incoming links from neighbouring nodes:
    G_i = E_{i,R} + E_{i,U} - E_{i-x̂,R} - E_{i-ŷ,U} - n_{f,i}
"""
function gauss_operator_node(dg::Int)
    return embed_R(op_E(dg), dg) .+ embed_U(op_E(dg), dg) .- embed_f(op_nf(), dg)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Finite-lattice site-local Hilbert-space dimensions                     ║
# ║                                                                          ║
# ║  For a finite nx×ny lattice each node (ix, iy) may have trivial gauge   ║
# ║  DoFs for missing boundary links.                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    site_dims(ix, iy, nx, ny, dg)

Returns (d_phys, d_gR, d_gU) for node (ix, iy) on an nx×ny lattice.
  d_gR = gauge_dim(dg) if ix < nx, else 1
  d_gU = gauge_dim(dg) if iy < ny, else 1
"""
function site_dims(ix::Int, iy::Int, nx::Int, ny::Int, dg::Int)
    gd    = gauge_dim(dg)
    d_gR  = (ix < nx) ? gd : 1
    d_gU  = (iy < ny) ? gd : 1
    d_phys = LGT_d_f * d_gR * d_gU
    return d_phys, d_gR, d_gU
end

"""1-based flat index into states of node (ix,iy): |nf, eR, eU⟩."""
function site_idx(nf::Int, eR::Int, eU::Int, d_gR::Int, d_gU::Int, dg::Int)
    iR = (d_gR == 1) ? 1 : (eR + dg + 1)
    iU = (d_gU == 1) ? 1 : (eU + dg + 1)
    return nf * d_gR * d_gU + (iR - 1) * d_gU + iU
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Finite-lattice site-local operator embedding                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    embed_f_site(O, d_gR, d_gU)

Embed a fermion operator into the local Hilbert space d_f × d_gR × d_gU.
"""
embed_f_site(O, d_gR::Int, d_gU::Int) = kron(O, _Id(d_gR), _Id(d_gU))

"""
    embed_R_site(O, d_gU)

Embed a right-gauge operator (d_gR × d_gR) into d_f × d_gR × d_gU.
"""
embed_R_site(O, d_gU::Int) = kron(_Id(LGT_d_f), O, _Id(d_gU))

"""
    embed_U_site(O, d_gR)

Embed an up-gauge operator (d_gU × d_gU) into d_f × d_gR × d_gU.
"""
embed_U_site(O, d_gR::Int) = kron(_Id(LGT_d_f), _Id(d_gR), O)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Finite-lattice on-site Hamiltonian  (site-specific Hilbert space)      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    H_onsite_site(ix, iy, nx, ny, dg; g, m)

Single-node Hamiltonian for node (ix, iy).  Includes:
  • (g²/2) E²_R  if right link exists (ix < nx)
  • (g²/2) E²_U  if up link exists    (iy < ny)
  • m · (-1)^{ix+iy} · n_f
"""
function H_onsite_site(ix::Int, iy::Int, nx::Int, ny::Int, dg::Int;
                        g::Float64, m::Float64)
    d_phys, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
    H = zeros(ComplexF64, d_phys, d_phys)

    if ix < nx
        H .+= (g^2 / 2) .* embed_R_site(op_E2(dg), d_gU)
    end
    if iy < ny
        H .+= (g^2 / 2) .* embed_U_site(op_E2(dg), d_gR)
    end

    stag = (iseven(ix + iy)) ? 1 : -1
    H .+= m * stag .* embed_f_site(op_nf(), d_gR, d_gU)

    return 0.5 .* (H .+ H')
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Finite-lattice two-site Hamiltonians                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    H_hop_h_site(ix, iy, nx, ny, dg; t)

Horizontal 2-site hopping between node (ix, iy) [left] and (ix+1, iy) [right].
Requires ix < nx.
"""
function H_hop_h_site(ix::Int, iy::Int, nx::Int, ny::Int, dg::Int; t::Float64)
    @assert ix < nx "H_hop_h_site: no right bond at ix=$ix (nx=$nx)"
    _, d_gR_L, d_gU_L = site_dims(ix,   iy, nx, ny, dg)
    _, d_gR_R, d_gU_R = site_dims(ix+1, iy, nx, ny, dg)

    cdag_U_L = kron(op_cdag(), op_U_gauge(dg), _Id(d_gU_L))
    c_R = embed_f_site(op_c(), d_gR_R, d_gU_R)

    hop = kron(cdag_U_L, c_R)
    H = -t .* (hop .+ hop')
    return 0.5 .* (H .+ H')
end

"""
    H_hop_v_site(ix, iy, nx, ny, dg; t)

Vertical 2-site hopping between node (ix, iy) [lower] and (ix, iy+1) [upper].
Requires iy < ny.
"""
function H_hop_v_site(ix::Int, iy::Int, nx::Int, ny::Int, dg::Int; t::Float64)
    @assert iy < ny "H_hop_v_site: no up bond at iy=$iy (ny=$ny)"
    _, d_gR_D, d_gU_D = site_dims(ix, iy,   nx, ny, dg)
    _, d_gR_U, d_gU_U = site_dims(ix, iy+1, nx, ny, dg)

    cdag_U_D = kron(op_cdag(), _Id(d_gR_D), op_U_gauge(dg))
    c_U = embed_f_site(op_c(), d_gR_U, d_gU_U)

    hop = kron(cdag_U_D, c_U)
    H = -t .* (hop .+ hop')
    return 0.5 .* (H .+ H')
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Finite-lattice merged 2-site Hamiltonians  (boundary-corrected)        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    _onsite_weight(ix, iy, nx, ny) → Float64

Weight for the on-site energy of site (ix,iy) in each bond Hamiltonian.
Each site participates in `num_bonds` bonds; summing 1/num_bonds over all bonds
gives exactly 1, so the full on-site energy is applied exactly once per Trotter step.
This corrects the boundary under-counting that occurs when using the naive 1/4 weight:
corner sites (2 bonds) would otherwise receive only 50%, edge sites (3 bonds) 75%.
"""
function _onsite_weight(ix::Int, iy::Int, nx::Int, ny::Int)
    nb = (ix > 1 ? 1 : 0) + (ix < nx ? 1 : 0) + (iy > 1 ? 1 : 0) + (iy < ny ? 1 : 0)
    return 1.0 / nb
end

"""
    H_merged_h_site(ix, iy, nx, ny, dg; g, t, m)

Horizontal merged Hamiltonian for the bond (ix,iy)—(ix+1,iy):
    H_hop_h_site + αL·H_onsite_L ⊗ I + αR·I ⊗ H_onsite_R

αL = 1/num_bonds(ix,iy), αR = 1/num_bonds(ix+1,iy), ensuring each site's
on-site energy is counted exactly once summed over all bonds it belongs to.
"""
function H_merged_h_site(ix::Int, iy::Int, nx::Int, ny::Int, dg::Int;
                          g::Float64, t::Float64, m::Float64)
    d_phys_L, _, _ = site_dims(ix,   iy, nx, ny, dg)
    d_phys_R, _, _ = site_dims(ix+1, iy, nx, ny, dg)
    Idn_L = _Id(d_phys_L)
    Idn_R = _Id(d_phys_R)
    Hos_L = H_onsite_site(ix,   iy, nx, ny, dg; g=g, m=m)
    Hos_R = H_onsite_site(ix+1, iy, nx, ny, dg; g=g, m=m)
    Hh    = H_hop_h_site(ix, iy, nx, ny, dg; t=t)
    αL = _onsite_weight(ix,   iy, nx, ny)
    αR = _onsite_weight(ix+1, iy, nx, ny)
    return Hh .+ αL .* kron(Hos_L, Idn_R) .+ αR .* kron(Idn_L, Hos_R)
end

"""
    H_merged_v_site(ix, iy, nx, ny, dg; g, t, m)

Vertical merged Hamiltonian for the bond (ix,iy)—(ix,iy+1):
    H_hop_v_site + αD·H_onsite_D ⊗ I + αU·I ⊗ H_onsite_U

αD = 1/num_bonds(ix,iy), αU = 1/num_bonds(ix,iy+1).
"""
function H_merged_v_site(ix::Int, iy::Int, nx::Int, ny::Int, dg::Int;
                          g::Float64, t::Float64, m::Float64)
    d_phys_D, _, _ = site_dims(ix, iy,   nx, ny, dg)
    d_phys_U, _, _ = site_dims(ix, iy+1, nx, ny, dg)
    Idn_D = _Id(d_phys_D)
    Idn_U = _Id(d_phys_U)
    Hos_D = H_onsite_site(ix, iy,   nx, ny, dg; g=g, m=m)
    Hos_U = H_onsite_site(ix, iy+1, nx, ny, dg; g=g, m=m)
    Hv    = H_hop_v_site(ix, iy, nx, ny, dg; t=t)
    αD = _onsite_weight(ix, iy,   nx, ny)
    αU = _onsite_weight(ix, iy+1, nx, ny)
    return Hv .+ αD .* kron(Hos_D, Idn_U) .+ αU .* kron(Idn_D, Hos_U)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Demo / self-test                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function lgt_demo(; dg = 1, g = 1.0, t_hop = 1.0, m_mass = 0.0, τ = 0.01)
    println("═══════════════════════════════════════════════════════════")
    println("  U(1) LGT Hamiltonian — demo")
    println("═══════════════════════════════════════════════════════════")
    gd = gauge_dim(dg)
    dn = node_dim(dg)
    @printf("  Gauge truncation  d_gauge = %d  →  link dim = %d\n", dg, gd)
    @printf("  Fermion dim = %d  (%s)\n", LGT_d_f, SPINFUL ? "spinful" : "spinless")
    @printf("  Node dim    = %d  (fermion × right-gauge × up-gauge = %d×%d×%d)\n",
            dn, LGT_d_f, gd, gd)
    @printf("  Couplings:  g = %.3f,  t = %.3f,  m = %.3f\n", g, t_hop, m_mass)
    @printf("  Trotter step  τ = %.4f\n\n", τ)

    # ── Inspect spectra ───────────────────────────────────────────────────
    Hos = H_onsite(dg; g=g, m=m_mass, staggered_sign=1)
    ev_os = eigvals(Hermitian(Hos))
    println("  H_onsite eigenvalues (first 8):")
    println("    ", round.(ev_os[1:min(8, end)], digits=4))

    Hh = H_hop_h(dg; t=t_hop)
    ev_h = eigvals(Hermitian(Hh))
    println("  H_hop_horizontal eigenvalues (first 8):")
    println("    ", round.(ev_h[1:min(8, end)], digits=4))

    Hp = H_plaquette_gauge(dg; g=g)
    ev_p = eigvals(Hermitian(Hp))
    println("  H_plaquette (gauge-only) eigenvalues (first 8):")
    println("    ", round.(ev_p[1:min(8, end)], digits=4))

    # ── Build gates ───────────────────────────────────────────────────────
    gates = build_lgt_gates(dg; g=g, t=t_hop, m=m_mass, τ=τ)
    println()
    @printf("  Trotter gates built.  Plaquette SVD: D_α = %d, D_β = %d\n",
            gates.G_plaq.D_alpha, gates.G_plaq.D_beta)

    # ── Verify plaquette SVD reconstruction ───────────────────────────────
    GA, GB, GC = gates.G_plaq.GA, gates.G_plaq.GB, gates.G_plaq.GC
    # Reconstruct:  G_recon[b,l,r,t,b',l',r',t']
    #   = Σ_{α,β} GA[b,l,b',l',α] GB[r,r',α,β] GC[t,t',β]
    @tensor G_recon[b, l, r, t, bp, lp, rp, tp] :=
        GA[b, l, bp, lp, α] * GB[r, rp, α, β] * GC[t, tp, β]
    # Permute G_recon to match original ordering (b,l,r,t, b',l',r',t')
    G_recon_mat = reshape(G_recon, gd^4, gd^4)
    # The original gate has index order (b,l,r,t,b',l',r',t') before
    # the permutation.  We need to undo the permutation we did.
    # Original: (b,l,r,t,b',l',r',t').  Permuted to: (b,l,b',l',r,r',t,t').
    # G_recon has (b,l,r,t,b',l',r',t') from the tensor contraction.
    # But we permuted (1,2,5,6,3,7,4,8), so to get back we do inverse perm.
    G_orig = reshape(gates.G_plaq_raw, gd, gd, gd, gd, gd, gd, gd, gd)
    err = norm(reshape(G_recon, :) .- reshape(G_orig, :)) / norm(reshape(G_orig, :))
    @printf("  Plaquette SVD reconstruction error: %.2e\n", err)

    # ── Gate dimensions summary ───────────────────────────────────────────
    @printf("\n  Gate dimensions:\n")
    @printf("    2-site hopping gate:  %d × %d\n", size(gates.G_hop_h_half)...)
    @printf("    Plaquette (gauge):    %d × %d\n", size(gates.G_plaq_raw)...)
    @printf("    Node physical space:  ℂ^%d\n\n", dn)

    println("  ── Integration with tensorkit_tst.jl ──")
    println("  V_phys = node_physical_space($dg)        # = ℂ^$dn")
    println("  peps   = init_ipeps(V_phys, ℂ^D)")
    println("  simple_update_horizontal!(peps, gates.G_hop_h_half, D_max)")
    println("  simple_update_vertical!(peps,   gates.G_hop_v_full, D_max)")
    println("  simple_update_horizontal!(peps, gates.G_hop_h_half, D_max)")
    println("  # Plaquette: apply decomposed gate (see SVD fields above)")
    println()
    println("═══════════════════════════════════════════════════════════")
    return gates
end

# Run demo if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    lgt_demo()
end
