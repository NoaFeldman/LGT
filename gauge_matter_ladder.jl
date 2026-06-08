#= ═══════════════════════════════════════════════════════════════════════════════
   gauge_matter_ladder.jl

   Stage 1 of the gauge–matter decoupling (Bender & Zohar, arXiv:2008.01349) on a
   2×N-plaquette ladder, in preparation for building the decoupling unitary
   𝒰 = exp(−i Σ_{x,i} β_i(x) φ_i(x)) as an MPO on a pseudo-1D MPS chain.

   ── Lattice ──────────────────────────────────────────────────────────────────
   A ladder of two rows (y ∈ {1,2}) and N+1 columns (x ∈ {1,…,N+1}):

       (1,2)──h──(2,2)──h──(3,2)── … ──(N+1,2)
         │         │         │              │           v = vertical link (i=2)
         v         v         v              v           h = horizontal link (i=1)
       (1,1)──h──(2,1)──h──(3,1)── … ──(N+1,1)

     • Vertices (matter, charge Q):  2(N+1)
     • Links    (gauge,  field φ):   horizontal 2N  +  vertical (N+1)  =  3N+1
     • There are N elementary plaquettes per row gap ⇒ "2×N" counting in the
       Bender–Zohar sense (the two-row strip has N plaquettes).

   Link convention: a link (x,y,i) is *based* at vertex (x,y) and points in
   direction i:
       i = 1 (horizontal): (x,y) → (x+1,y)      exists for x ∈ 1:N,  y ∈ {1,2}
       i = 2 (vertical):   (x,1) → (x,2)         stored canonically at (x,1,2),
                                                  exists for x ∈ 1:N+1

   ── MPS snake path ───────────────────────────────────────────────────────────
   All 5N+3 degrees of freedom (sites + links) are linearly ordered by a
   column-by-column boustrophedon snake (see `build_snake`).  Vertices and links
   each inherit a 1-based index from their order of appearance on the snake, so
   every matrix below is expressed "along the MPS path".

   ── Green's function ─────────────────────────────────────────────────────────
   The (negative) discrete Laplacian is the graph Laplacian  L = −∇² = Bᵀ B,
   where B is the signed gradient (incidence) matrix.  L is positive
   semidefinite with a single zero mode (the constant), so the finite-size
   Green's function is the Moore–Penrose pseudo-inverse  G = L⁺, i.e. the inverse
   on the charge-neutral subspace.  The longitudinal shift field is the gradient
   of G,  M = B G  ⇒  Mᵢ,ₓ,ᵧ = G(x+î,y) − G(x,y), and satisfies the discrete
   Gauss law  Bᵀ M = L G = I − 𝟙𝟙ᵀ/n_sites.
   ═══════════════════════════════════════════════════════════════════════════ =#

using LinearAlgebra
using Printf

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Geometry                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""One element of the MPS chain: a matter site (`kind=:site`, i=0) or a gauge
link (`kind=:link`, i ∈ {1,2})."""
struct ChainElement
    kind :: Symbol
    x    :: Int
    y    :: Int
    i    :: Int      # 0 for sites; 1 (horizontal) or 2 (vertical) for links
end

"""
Full ladder geometry and the MPS-path indexing derived from the snake.

Fields:
  N, n_sites=2(N+1), n_links=3N+1, n_chain=5N+3
  chain          :: Vector{ChainElement}                 # snake order, length n_chain
  vertex_id      :: Dict{Tuple{Int,Int},Int}             # (x,y)   → 1:n_sites
  vertex_xy      :: Vector{Tuple{Int,Int}}               # id      → (x,y)
  link_id        :: Dict{NTuple{3,Int},Int}              # (x,y,i) → 1:n_links
  link_def       :: Vector{NTuple{3,Int}}                # id      → (x,y,i)
  site_pos       :: Dict{Tuple{Int,Int},Int}             # (x,y)   → chain position
  link_pos       :: Dict{NTuple{3,Int},Int}              # (x,y,i) → chain position
"""
struct LadderGeometry
    N         :: Int
    n_sites   :: Int
    n_links   :: Int
    n_chain   :: Int
    chain     :: Vector{ChainElement}
    vertex_id :: Dict{Tuple{Int,Int},Int}
    vertex_xy :: Vector{Tuple{Int,Int}}
    link_id   :: Dict{NTuple{3,Int},Int}
    link_def  :: Vector{NTuple{3,Int}}
    site_pos  :: Dict{Tuple{Int,Int},Int}
    link_pos  :: Dict{NTuple{3,Int},Int}
end

"""
    build_snake(N) → Vector{ChainElement}

Column-by-column boustrophedon ordering of every site and link.  For column x
the local order is:  site(x,y₁), vertical-link(x), site(x,y₂), then (if x ≤ N)
the two horizontal links to column x+1 (nearer row first).  y₁,y₂ alternate
(1,2)/(2,1) so consecutive columns connect through the shared row.
"""
function build_snake(N::Int)
    chain = ChainElement[]
    for x in 1:N+1
        y1, y2 = isodd(x) ? (1, 2) : (2, 1)
        push!(chain, ChainElement(:site, x, y1, 0))
        push!(chain, ChainElement(:link, x, 1, 2))          # vertical link of column x
        push!(chain, ChainElement(:site, x, y2, 0))
        if x ≤ N
            push!(chain, ChainElement(:link, x, y2, 1))     # horizontal link, ending row
            push!(chain, ChainElement(:link, x, y1, 1))     # horizontal link, other row
        end
    end
    return chain
end

"""
    ladder_geometry(N) → LadderGeometry

Build the geometry and all MPS-path index maps for a 2×N-plaquette ladder.
"""
function ladder_geometry(N::Int)
    @assert N ≥ 1 "N must be ≥ 1"
    chain = build_snake(N)

    vertex_id = Dict{Tuple{Int,Int},Int}()
    vertex_xy = Tuple{Int,Int}[]
    link_id   = Dict{NTuple{3,Int},Int}()
    link_def  = NTuple{3,Int}[]
    site_pos  = Dict{Tuple{Int,Int},Int}()
    link_pos  = Dict{NTuple{3,Int},Int}()

    for (pos, e) in enumerate(chain)
        if e.kind == :site
            site_pos[(e.x, e.y)] = pos
            if !haskey(vertex_id, (e.x, e.y))
                push!(vertex_xy, (e.x, e.y))
                vertex_id[(e.x, e.y)] = length(vertex_xy)
            end
        else
            key = (e.x, e.y, e.i)
            link_pos[key] = pos
            if !haskey(link_id, key)
                push!(link_def, key)
                link_id[key] = length(link_def)
            end
        end
    end

    n_sites = 2 * (N + 1)
    n_links = 3 * N + 1
    n_chain = 5 * N + 3
    @assert length(vertex_xy) == n_sites "site count $(length(vertex_xy)) ≠ $n_sites"
    @assert length(link_def)  == n_links "link count $(length(link_def)) ≠ $n_links"
    @assert length(chain)     == n_chain "chain length $(length(chain)) ≠ $n_chain"

    return LadderGeometry(N, n_sites, n_links, n_chain, chain,
                          vertex_id, vertex_xy, link_id, link_def,
                          site_pos, link_pos)
end

"""The two vertices (head, tail) a link connects: (x,y,i) → ((x,y), (x+î))."""
function link_endpoints(x::Int, y::Int, i::Int)
    if i == 1
        return (x, y), (x + 1, y)          # horizontal
    elseif i == 2
        return (x, 1), (x, 2)              # vertical (canonical base at y=1)
    else
        error("direction i=$i not in {1,2}")
    end
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Gradient (incidence), Laplacian, Green's function, shift field          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    gradient_matrix(geo) → B   (n_links × n_sites)

Signed forward-difference (incidence) matrix:  for link ℓ from vertex a to b,
B[ℓ,a] = −1, B[ℓ,b] = +1, so (B f)_ℓ = f(b) − f(a) = Δ⁺ᵢ f on that link.
"""
function gradient_matrix(geo::LadderGeometry)
    B = zeros(Float64, geo.n_links, geo.n_sites)
    for (ℓ, (x, y, i)) in enumerate(geo.link_def)
        a, b = link_endpoints(x, y, i)
        B[ℓ, geo.vertex_id[a]] -= 1.0
        B[ℓ, geo.vertex_id[b]] += 1.0
    end
    return B
end

"""
    laplacian_matrix(geo) → L   (n_sites × n_sites)

Negative discrete Laplacian  L = −∇² = Bᵀ B = D − A (graph Laplacian) with open
(Neumann) boundaries.  Positive semidefinite; single zero mode (the constant).
"""
function laplacian_matrix(geo::LadderGeometry)
    B = gradient_matrix(geo)
    return Symmetric(B' * B)
end

"""
    greens_function(geo; tol) → G   (n_sites × n_sites)

Finite-size Green's function  G = (−∇²)⁺  (Moore–Penrose pseudo-inverse), i.e.
the inverse of −∇² on the charge-neutral subspace orthogonal to the constant
zero mode.  `tol` is the singular-value cutoff for the pseudo-inverse.
"""
function greens_function(geo::LadderGeometry; tol::Float64=1e-10)
    L = Matrix(laplacian_matrix(geo))
    return pinv(L; rtol=tol)
end

"""
    shift_field(geo; tol) → M   (n_links × n_sites)

Longitudinal shift-field tensor  Mᵢ,ₓ,ᵧ = Δ⁺ᵢ G(x,y) = G(x+î,y) − G(x,y),
flattened as M[link_id, source_vertex_id] = (B G)[ℓ,y].  Row ℓ is the field
induced on link ℓ by a unit charge placed at source vertex y.
"""
function shift_field(geo::LadderGeometry; tol::Float64=1e-10)
    B = gradient_matrix(geo)
    G = greens_function(geo; tol=tol)
    return B * G
end

"""Convenience accessor:  Mᵢ(x ; source) for link (x,y,i) and source vertex (sx,sy)."""
function shift_field_value(M::AbstractMatrix, geo::LadderGeometry,
                            x::Int, y::Int, i::Int, sx::Int, sy::Int)
    return M[geo.link_id[(x, y, i)], geo.vertex_id[(sx, sy)]]
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Verification / demo                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    verify_ladder(N; verbose=true) → Bool

Self-consistency checks for the geometry and Green's function:
  1. DoF counts: sites=2(N+1), links=3N+1, chain=5N+3.
  2. L symmetric PSD with exactly one zero eigenvalue; row sums zero.
  3. B·𝟙 = 0 (gradient of a constant vanishes).
  4. Projector identity  L·G = I − 𝟙𝟙ᵀ/n  (G is the pseudo-inverse).
  5. Discrete Gauss law  Bᵀ·M = I − 𝟙𝟙ᵀ/n  (divergence of the shift field).
"""
function verify_ladder(N::Int; verbose::Bool=true)
    geo = ladder_geometry(N)
    n   = geo.n_sites
    B   = gradient_matrix(geo)
    L   = Matrix(laplacian_matrix(geo))
    G   = greens_function(geo)
    M   = B * G
    P   = I(n) - fill(1.0 / n, n, n)            # projector off the constant mode

    evals    = sort(eigvals(Symmetric(L)))
    n_zero   = count(<(1e-9), abs.(evals))
    rowsum0  = maximum(abs, sum(L, dims=2))
    grad_c0  = maximum(abs, B * ones(n))
    proj_err = maximum(abs, L * G .- P)
    gauss_err = maximum(abs, B' * M .- P)
    sym_err  = maximum(abs, L .- L')

    ok_counts = geo.n_sites == 2(N+1) && geo.n_links == 3N+1 && geo.n_chain == 5N+3
    ok_zero   = n_zero == 1
    ok_psd    = evals[1] > -1e-9
    checks = (
        counts    = ok_counts,
        one_zero  = ok_zero,
        psd       = ok_psd,
        symmetric = sym_err  < 1e-12,
        rowsum    = rowsum0  < 1e-10,
        grad_const= grad_c0  < 1e-12,
        projector = proj_err < 1e-8,
        gauss_law = gauss_err < 1e-8,
    )

    if verbose
        println("─── ladder verification (N=$N) ───")
        @printf("  sites=%d  links=%d  chain=%d   (expect %d, %d, %d)\n",
                geo.n_sites, geo.n_links, geo.n_chain, 2(N+1), 3N+1, 5N+3)
        @printf("  L: zero modes=%d  λ_min=%.2e  rowsum=%.1e  sym=%.1e\n",
                n_zero, evals[1], rowsum0, sym_err)
        @printf("  B·𝟙=%.1e   L·G−P=%.1e   Bᵀ·M−P=%.1e\n",
                grad_c0, proj_err, gauss_err)
        for (k, v) in pairs(checks)
            println("    ", v ? "PASS " : "FAIL ", k)
        end
        println(all(values(checks)) ? "  ALL CHECKS PASSED" : "  SOME CHECKS FAILED")
    end
    return all(values(checks))
end

# Run a small demo / self-test when executed directly.
if abspath(PROGRAM_FILE) == @__FILE__
    for N in (1, 2, 3, 5)
        verify_ladder(N)
        println()
    end
end
