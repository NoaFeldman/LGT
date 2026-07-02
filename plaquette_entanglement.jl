#= ═══════════════════════════════════════════════════════════════════════════════
   plaquette_entanglement.jl

   Half-system entanglement of the GAUGE sector of a ladder-like U(1) LGT MPS,
   read out in the DUAL PLAQUETTE representation.

   ── Pipeline ─────────────────────────────────────────────────────────────────
     1. Decouple matter from gauge with the Bender–Zohar SoE decoupler,
        exactly following decoupler-usage.md:
            E = build_exponents(nx, ny; K, bw)
            φ = decouple_state(E.Ofull, E.a_full, ψ)          # 𝒰|ψ⟩
        In the decoupled frame the static charges are folded into a shift of the
        electric field, so the transformed gauge field E' is (ideally) SOURCE-FREE
        — the condition under which the dual plaquette (height) field exists.

     2. Dual plaquette mode.  A source-free integer electric field on the links is
        the discrete curl of a height field h_p living on the PLAQUETTES (dual
        sites).  There are (nx−1)(ny−1) plaquettes — half the links, no gauge
        redundancy.  Local Hilbert space per plaquette = {0, 1}: h_p = 1 iff, in
        the link (loop) picture, the plaquette lies INSIDE an excited unit loop,
        0 otherwise.  For dg=1 (E ∈ {−1,0,1}) the curl of h ∈ {0,1} always lands
        back in {−1,0,1}, so heights ↔ source-free link configs is a bijection:

            E_R(ix,iy) = h(ix, iy)   − h(ix, iy−1)     (right link: above − below)
            E_U(ix,iy) = h(ix−1, iy) − h(ix, iy)       (up link:    left  − right)

        with h ≡ 0 for any plaquette index outside 1:(nx−1) × 1:(ny−1) (exterior).

     3. Gauge state in the plaquette basis.  For every height config h we project
        the decoupled MPS onto its link config E(h) (fixing e_R, e_U on every
        node, leaving n_f free) → a matter-only MPS ψ_h.  The Gram matrix
            ρ[h′,h] = ⟨ψ_{h′} | ψ_h⟩
        is the gauge reduced density matrix (matter traced out) written in the
        plaquette basis.  tr ρ = weight captured in the source-free sector
        (→ 1 for a clean decoupling); the dominant eigenvector is the pure
        plaquette wavefunction |G⟩ = Σ_h v_h |h⟩ and its weight (purity) measures
        how well 𝒰 factorised matter ⊗ gauge.

     4. Half-system entanglement.  Reshape |G⟩ across a spatial bipartition of the
        plaquettes and take the von Neumann entropy of the Schmidt spectrum.

   NOTE on ordering: the decoupler works on the COLUMN-major snake
   (`column_snake`, see decoupler-usage.md).  The input MPS ψ must be in that same
   node order (i.e. its local dims == `col_node_dims(nx,ny,dg)`), as produced by
   `random_product_mps(build_exponents(...).dims)` or a quench built on the column
   snake.  A row-snake state from mps_lgt must be reordered first.

   Requires: decoupling_U_soe.jl (→ mps_lgt.jl, lgt_greens_soe.jl, u1 ops)
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "decoupling_U_soe.jl"))

using LinearAlgebra
using Printf

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Dual lattice: plaquette height field  ↔  link electric field             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Plaquettes of an nx×ny lattice, indexed by lower-left corner (px,py),
px ∈ 1:(nx−1), py ∈ 1:(ny−1).  Order: px fastest (a plaquette row at a time)."""
plaquette_list(nx::Int, ny::Int) = [(px, py) for py in 1:ny-1 for px in 1:nx-1]

"""Value of the height field at plaquette (px,py); 0 outside the plaquette grid
(the exterior region, which is fixed to height 0)."""
@inline hval(h::AbstractMatrix{Int}, px::Int, py::Int, nx::Int, ny::Int) =
    (1 ≤ px ≤ nx - 1 && 1 ≤ py ≤ ny - 1) ? h[px, py] : 0

"""
    height_to_E(h, nx, ny) → (ER, EU)

Discrete curl of a plaquette height field `h[px,py]` → link electric fields.
`ER[ix,iy]` (right link, ix<nx) and `EU[ix,iy]` (up link, iy<ny).  A single unit
loop (one plaquette at height 1) maps to ±1 on its four boundary links and 0
elsewhere."""
function height_to_E(h::AbstractMatrix{Int}, nx::Int, ny::Int)
    ER = zeros(Int, nx, ny)
    EU = zeros(Int, nx, ny)
    for iy in 1:ny, ix in 1:nx-1                       # right links
        ER[ix, iy] = hval(h, ix, iy, nx, ny) - hval(h, ix, iy - 1, nx, ny)
    end
    for iy in 1:ny-1, ix in 1:nx                       # up links
        EU[ix, iy] = hval(h, ix - 1, iy, nx, ny) - hval(h, ix, iy, nx, ny)
    end
    return ER, EU
end

"""Unpack a config integer `c` (bit i = height of plaquette i) into the height
grid `h[px,py]` over `plaqs`."""
function config_to_height(c::Int, plaqs, nx::Int, ny::Int)
    h = zeros(Int, nx - 1, ny - 1)
    for (i, (px, py)) in enumerate(plaqs)
        h[px, py] = (c >> (i - 1)) & 1
    end
    return h
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Project the decoupled MPS onto a source-free gauge (link) configuration   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    project_matter(φ, nx, ny, dg, chain, ER, EU) → matter CMPS

Fix the gauge indices (e_R, e_U) of every node of the decoupled MPS `φ` to the
link config (`ER`,`EU`), leaving the fermion index n_f free.  Returns a
matter-only MPS (local dim = LGT_d_f = 2), the residual amplitude of `φ` on that
gauge configuration."""
function project_matter(φ::CMPS, nx::Int, ny::Int, dg::Int,
                        chain::Vector{Tuple{Int,Int}},
                        ER::AbstractMatrix{Int}, EU::AbstractMatrix{Int})
    ψ = CMPS(undef, length(chain))
    for (p, (ix, iy)) in enumerate(chain)
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        eR = ix < nx ? ER[ix, iy] : 0                  # 0 is a no-op when d_gR==1
        eU = iy < ny ? EU[ix, iy] : 0
        i0 = site_idx(0, eR, eU, d_gR, d_gU, dg)        # |n_f=0, eR, eU⟩
        i1 = site_idx(1, eR, eU, d_gR, d_gU, dg)        # |n_f=1, eR, eU⟩
        ψ[p] = φ[p][:, :, [i0, i1]]
    end
    return ψ
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Gauge reduced density matrix in the plaquette basis                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    plaquette_density_matrix(φ, nx, ny; dg) → (ρ, plaqs, weight)

Gauge reduced density matrix of the decoupled MPS `φ`, written in the dual
plaquette basis: `ρ[h′,h] = ⟨ψ_{h′}|ψ_h⟩` where `ψ_h` is `φ` projected onto the
link config E(h) with matter left open (matter traced out by the overlap).
`ρ` is (2^Nplaq × 2^Nplaq); `weight = tr ρ` is the state's weight in the
source-free sector (→ 1 for a clean decoupling).  `ρ` is returned normalised
(trace 1)."""
function plaquette_density_matrix(φ::CMPS, nx::Int, ny::Int; dg::Int=1)
    plaqs = plaquette_list(nx, ny)
    Nplaq = length(plaqs)
    Nplaq ≤ 16 || error("plaquette_density_matrix: $Nplaq plaquettes → 2^$Nplaq " *
                        "configs is too large for the dense readout (small-lattice tool).")
    chain, _ = column_snake(nx, ny)
    @assert length(chain) == length(φ) "MPS length ≠ number of nodes — is φ on the column snake?"

    ncfg = 1 << Nplaq
    projected = Vector{CMPS}(undef, ncfg)              # ψ_h for every height config
    for c in 0:ncfg-1
        h = config_to_height(c, plaqs, nx, ny)
        ER, EU = height_to_E(h, nx, ny)
        projected[c+1] = project_matter(φ, nx, ny, dg, chain, ER, EU)
    end

    ρ = zeros(ComplexF64, ncfg, ncfg)
    for i in 1:ncfg
        ρ[i, i] = real(mps_overlap(projected[i], projected[i]))
        for j in i+1:ncfg
            z = mps_overlap(projected[i], projected[j])   # ⟨ψ_i|ψ_j⟩
            ρ[i, j] = z
            ρ[j, i] = conj(z)
        end
    end
    weight = real(tr(ρ))
    weight > 0 || error("no weight in the source-free sector — decoupling failed or wrong sector")
    return ρ / weight, plaqs, weight
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Spatial bipartition of the plaquettes + entanglement entropy             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Balanced spatial bipartition of the plaquettes: a vertical (x) cut when there
are ≥2 plaquette columns, else a horizontal (y) cut.  Returns the list of
plaquette indices (into `plaqs`) forming region A."""
function default_bipartition(nx::Int, ny::Int, plaqs)
    npx, npy = nx - 1, ny - 1
    if npx ≥ 2
        xcut = div(npx, 2)
        return [i for (i, (px, _)) in enumerate(plaqs) if px ≤ xcut]
    else
        ycut = div(npy, 2)
        return [i for (i, (_, py)) in enumerate(plaqs) if py ≤ ycut]
    end
end

"""
    schmidt_matrix(v, Nplaq, A) → M

Reshape a plaquette wavefunction `v` (length 2^Nplaq, index = config integer +1)
into the Schmidt matrix `M[a,b]` for the bipartition (A, B=rest), where `a`/`b`
enumerate the sub-configs of the plaquettes in A/B."""
function schmidt_matrix(v::AbstractVector, Nplaq::Int, A::Vector{Int})
    B = setdiff(1:Nplaq, A)
    NA, NB = length(A), length(B)
    M = zeros(eltype(v), 1 << NA, 1 << NB)
    for c in 0:(1<<Nplaq)-1
        a = 0
        for (k, i) in enumerate(A); a |= ((c >> (i - 1)) & 1) << (k - 1); end
        b = 0
        for (k, i) in enumerate(B); b |= ((c >> (i - 1)) & 1) << (k - 1); end
        M[a+1, b+1] = v[c+1]
    end
    return M
end

"""Von Neumann entropy (nats) of the Schmidt spectrum of matrix `M`."""
function schmidt_entropy(M::AbstractMatrix)
    s = svdvals(M)
    p = abs2.(s); p ./= sum(p)
    return -sum(x -> x > 0 ? x * log(x) : 0.0, p)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Top-level driver                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    plaquette_entanglement(ψ, nx, ny; dg, K, bw, Dmax, which, A, verbose) → NamedTuple

Take a ladder-like U(1) LGT MPS `ψ` (in COLUMN-snake node order — see the note at
the top of this file), decouple matter from gauge with the Bender–Zohar SoE
decoupler (decoupler-usage.md), express the gauge sector in the DUAL PLAQUETTE
mode, and return the HALF-SYSTEM entanglement of that plaquette wavefunction.

Keywords
  `K`, `bw`     SoE decoupler parameters (K=2 is enough; bw = boundary shell).
  `Dmax`        state bond kept during the Krylov decoupling.
  `which`       :full (Ofull, default), :bulk (Obulk), or :exact (Oexact).
  `A`           plaquette indices (into the plaquette list) forming half A;
                defaults to a balanced spatial cut.
  `verbose`     print a summary.

Returns `(S, purity, srcfree_weight, v, ρ, plaqs, A)`:
  `S`               half-system entanglement entropy of the plaquette state (nats).
  `purity`          largest eigenvalue of ρ_plaq = weight of the leading pure
                    gauge state (→ 1 for a clean matter⊗gauge factorisation).
  `srcfree_weight`  weight of ψ (decoupled) captured in the source-free sector.
  `v`               the plaquette wavefunction (dominant eigenvector of ρ_plaq).
  `ρ`               the plaquette reduced density matrix (trace-normalised).
  `plaqs`, `A`      plaquette list and the region-A index set used for the cut.
"""
function plaquette_entanglement(ψ::CMPS, nx::Int, ny::Int; dg::Int=1, K::Int=2,
                                bw::Int=1, Dmax::Int=64, which::Symbol=:full,
                                A::Union{Nothing,Vector{Int}}=nothing,
                                verbose::Bool=true)
    @assert dg == 1 "dual plaquette readout assumes dg=1 (E ∈ {−1,0,1})"
    @assert length(ψ) == length(col_node_dims(nx, ny, dg)) "ψ must be on the column snake"

    # ── 1. decouple matter from gauge (decoupler-usage.md) ────────────────────
    Ex = build_exponents(nx, ny; dg=dg, K=K, bw=bw)
    O, a = which === :full  ? (Ex.Ofull,  Ex.a_full)  :
           which === :bulk  ? (Ex.Obulk,  Ex.a_bulk)  :
           which === :exact ? (Ex.Oexact, Ex.a_exact) :
           error("which must be :full, :bulk or :exact")
    φ = decouple_state(O, a, ψ; Dmax=Dmax)

    # ── 2+3. gauge state in the plaquette basis ───────────────────────────────
    ρ, plaqs, srcfree_weight = plaquette_density_matrix(φ, nx, ny; dg=dg)
    Nplaq = length(plaqs)
    F = eigen(Hermitian(ρ))                            # ascending eigenvalues
    purity = real(F.values[end])
    v = F.vectors[:, end]                              # plaquette wavefunction

    # ── 4. half-system entanglement of the plaquette wavefunction ─────────────
    Aset = A === nothing ? default_bipartition(nx, ny, plaqs) : A
    M = schmidt_matrix(v, Nplaq, Aset)
    S = schmidt_entropy(M)

    if verbose
        println("─── dual-plaquette half-system entanglement ($(nx)×$(ny)) ───")
        @printf("  plaquettes: %d   bipartition |A|=%d |B|=%d\n",
                Nplaq, length(Aset), Nplaq - length(Aset))
        @printf("  decoupling: source-free weight = %.6f   plaquette purity = %.6f\n",
                srcfree_weight, purity)
        @printf("  half-system entanglement  S = %.6f nats  (= %.6f bits)\n",
                S, S / log(2))
    end
    return (S=S, purity=purity, srcfree_weight=srcfree_weight,
            v=v, ρ=ρ, plaqs=plaqs, A=Aset)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Demo (writes only — not auto-run per the no-implicit-execution policy)    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Demo on the 3×4 target: decouple a random product state and report the
dual-plaquette half-system entanglement.  Substitute a real quenched/ground-state
MPS (in COLUMN-snake order) for `ψ` to get a physical number."""
function demo_plaquette_entanglement(; nx::Int=3, ny::Int=4, K::Int=2, Dmax::Int=64)
    Random.seed!(1)
    ψ = random_product_mps(col_node_dims(nx, ny, 1))
    return plaquette_entanglement(ψ, nx, ny; K=K, Dmax=Dmax)
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo_plaquette_entanglement()
end
