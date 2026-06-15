#= ═══════════════════════════════════════════════════════════════════════════════
   gauge_matter_decoupling_mpo.jl

   Stage 3 of the gauge–matter decoupling (Bender & Zohar, arXiv:2008.01349):
   compile the EXPONENT operator

       Ô = Σ_{x,i} β_i(x) φ_i(x),   β_i(x) = Σ_y M_{i,x,y} Q(y),

   i.e.   Ô = Σ_{ℓ∈links} Σ_{s∈vertices}  M[ℓ,s] · φ_ℓ · Q_s ,

   into a Matrix Product Operator along the snake MPS chain, using a finite-state
   automaton (FSA) whose virtual bond carries the accumulated, geometrically
   propagating "charge state".  The decoupling unitary 𝒰 = exp(−i Ô) is built
   from Ô in a later stage.

   ── Where the bond dimension comes from ──────────────────────────────────────
   Stage 2 showed that, per transverse rung mode, the shift field M is

     • MASSIVE  (antisymmetric, m²=2):  v_AA(x,x') = c · λ^{|x−x'|±…},  λ = 2−√3,
       with the half-integer antisymmetric extension  v(d) = −v(−1−d);
     • MASSLESS (symmetric,    m²=0):  v_SS(x,x') = p + q·x + r·x'   (affine ramp).

   Reconstructing the full kernel from the two diagonal channels (the cross
   channels vanish), for a HORIZONTAL link (x,yl,1) and source vertex (x',ys):

       M[link(x,yl,1), vert(x',ys)]
            =  ½ · v_SS(x,x')                       (symmetric, row-even)
            +  ½ · σ(yl)·σ(ys) · v_AA(x,x')         (antisymmetric, row-odd)

   with σ(1)=+1, σ(2)=−1.  Each separable channel becomes O(1) FSA bond states,
   so χ_MPO ∝ K (here K=1 massive exponential + the affine ramp).

   ── FSA conventions ──────────────────────────────────────────────────────────
   Standard lower-triangular MPO.  Bond-state 1 = "done" (accumulate), bond-state
   Dχ = "start".  Left boundary selects "start", right boundary selects "done".
   A geometric channel propagates a carrier whose self-loop multiplies by λ once
   per COLUMN step (a "tick" applied when leaving the last chain site of a
   column), so a carrier opened at column x' and closed at column x is weighted
   by λ^{x−x'}.  Both link←source and source←link orderings are carried (the
   kernel is two-sided), one carrier each.

   Requires: gauge_matter_ladder.jl, gauge_matter_exp_fit.jl
   ═══════════════════════════════════════════════════════════════════════════ =#

include(joinpath(@__DIR__, "gauge_matter_exp_fit.jl"))   # pulls in ladder too

using LinearAlgebra
using Printf

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Local physical operators                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    charge_operator(; centered=true) → Q   (2×2)

Matter charge operator on a vertex site (single hardcore mode: empty/occupied).
`Q = diag(0,1)`; when `centered`, the trace is removed (`Q → Q − ½·I`) so that
Frobenius extraction of MPO coefficients is contaminated only by genuine
two-body terms (identity-carrying bond states give tr(Q)=0)."""
function charge_operator(; centered::Bool=true)
    Q = Matrix(Diagonal(Float64[0, 1]))
    centered && (Q = Q - (tr(Q) / size(Q, 1)) * Matrix(I, size(Q)...))
    return Q
end

"""
    link_phase_operator(d; centered=true) → φ   (d×d, Hermitian)

Compact-U(1) gauge phase on a link site, truncated to a `d`-dimensional photon
cutoff.  In the electric-field basis E ∈ {−Λ,…,Λ} (d = 2Λ+1) the conjugate angle
is φ = ½ (U + U†) with U the unit lowering operator e^{iφ} (E → E−1); this is the
Hermitian "cosφ-type" generator that appears in the exponent.  Centered to be
traceless for clean coefficient extraction."""
function link_phase_operator(d::Int; centered::Bool=true)
    @assert d ≥ 2 "photon cutoff dimension d must be ≥ 2"
    U = diagm(-1 => ones(Float64, d - 1))      # lowering: |E⟩ ← |E+1⟩
    φ = 0.5 * (U + U')                          # Hermitian angle operator
    centered && (φ = φ - (tr(φ) / d) * Matrix(I, d, d))   # traceless (already is)
    return Matrix(φ)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Analytic channel reconstruction of M  (layer-1 self-test)               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

σrow(y::Int) = y == 1 ? 1.0 : -1.0             # antisymmetric-mode row sign

"""Parameters of the analytic horizontal-link kernel model (the MPO's exact
target).  Massive: amplitude `c`, rate `λ` (=2−√3).  Massless: continuous affine
background `a0 + q·x + r·x'` plus a Heaviside step `s·𝟙(x≥x')`."""
struct HModel
    c  :: Float64
    λ  :: Float64
    a0 :: Float64
    q  :: Float64
    r  :: Float64
    s  :: Float64
end

"""Extract the model parameters from the Stage-2 channel fits for a given N."""
function horizontal_model(geo::LadderGeometry; K::Int=3)
    M  = shift_field(geo)
    FA = fit_channel(geo, M, :A, :A; K=K)
    λ_anti = 2 - sqrt(3)
    kdom = argmin(abs.(FA.λ .- λ_anti))
    c = real(FA.c[kdom]); λ = real(FA.λ[kdom])
    a0, q, r, s, _ = symmetric_ramp_step(geo, M)
    return HModel(c, λ, a0, q, r, s)
end

"""Massive (antisymmetric) kernel with half-integer odd extension v(d)=−v(−1−d)."""
v_AA(m::HModel, x::Int, xp::Int) = (x - xp) ≥ 0 ? m.c * m.λ^(x - xp) :
                                                  -m.c * m.λ^(xp - x - 1)
"""Massless (symmetric) kernel: continuous affine background + source step."""
v_SS(m::HModel, x::Int, xp::Int) = m.a0 + m.q * x + m.r * xp + (x ≥ xp ? m.s : 0.0)

"""
    analytic_M(geo, m) → Mr   (n_links × n_sites)

Rebuild the horizontal-link block of the kernel purely from the model `m`:

    M[link(x,yl,1), vert(x',ys)] = ½ v_SS(x,x') + ½ σ(yl)σ(ys) v_AA(x,x').

This matrix is the EXACT target the FSA-compiled MPO must reproduce."""
function analytic_M(geo::LadderGeometry, m::HModel)
    Mr = zeros(Float64, geo.n_links, geo.n_sites)
    for x in 1:geo.N, yl in 1:2, xp in 1:geo.N + 1, ys in 1:2
        ℓ = geo.link_id[(x, yl, 1)]; s = geo.vertex_id[(xp, ys)]
        Mr[ℓ, s] = 0.5 * v_SS(m, x, xp) + 0.5 * σrow(yl) * σrow(ys) * v_AA(m, x, xp)
    end
    return Mr
end

"""Max abs error of the analytic model vs the exact shift field, over horizontal
links (the Stage-2 single-exponential TRUNCATION error, not an FSA error)."""
function analytic_truncation_error(geo::LadderGeometry, m::HModel)
    M = shift_field(geo); Mr = analytic_M(geo, m)
    err = 0.0
    for x in 1:geo.N, yl in 1:2, xp in 1:geo.N + 1, ys in 1:2
        ℓ = geo.link_id[(x, yl, 1)]; s = geo.vertex_id[(xp, ys)]
        err = max(err, abs(Mr[ℓ, s] - M[ℓ, s]))
    end
    return err
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FSA MPO compiler                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Bond layout (size Dχ):  index 1 = "done", indices 2:Dχ−? = carriers, last =
# "start".  We assemble W^[p] ∈ ℝ^{Dχ×Dχ} of d_p×d_p local operators.

"""A two-body channel for the FSA.

Couples an `open` operator placed at sites of `open_kind` (scalar weight
`open_w(x,y)`) to a `close` operator placed at sites of `close_kind` LATER on the
chain (weight `close_w(x,y)`).  The carrier self-loop multiplies by the constant
`sl` at every site (default 1).  Geometric distance dependence is NOT obtained by
"ticking": instead the weights carry absolute-position factors λ^{∓x} so that
open(x')·close(x) = λ^{x−x'} exactly, immune to within-column ordering.  Set
`open_kind=:vertex, close_kind=:link` for source→link and the reverse for
link→source."""
struct GeoChannel
    sl         :: Float64       # carrier self-loop multiplier (1 for our channels)
    open_kind  :: Symbol        # :vertex or :link
    open_op    :: Symbol        # :Q or :phi
    open_w     :: Function      # (x,y) -> scalar
    close_kind :: Symbol
    close_op   :: Symbol
    close_w    :: Function
end

"""
    build_exponent_mpo(geo, channels; d, Q=…, φ=…) → W::Vector{Array{Float64,4}}

Compile Ô = Σ_channels Σ_{open<close} open_w·close_w·λ^{Δcol} (op⊗op) into MPO
tensors.  Each `W[p]` has indices (left-bond, right-bond, ket, bra) with local
dimension d_p (= dim φ on link sites, = dim Q on vertex sites).  Bond dimension
Dχ = 2 + (#channels); virtual bonds pass the accumulated charge state."""
function build_exponent_mpo(geo::LadderGeometry, channels::Vector{GeoChannel};
                            d::Int=3,
                            Q::AbstractMatrix=charge_operator(),
                            φ::AbstractMatrix=link_phase_operator(d))
    nC = length(channels)
    Dχ = 2 + nC                                   # 1=done, 2:1+nC carriers, Dχ=start
    DONE, START = 1, Dχ
    carrier(k) = 1 + k

    locop(e, which) = which === :Q ? Q : φ
    locdim(e) = e.kind == :site ? size(Q, 1) : size(φ, 1)

    W = Vector{Array{Float64,4}}(undef, geo.n_chain)
    for (p, e) in enumerate(geo.chain)
        dp = locdim(e)
        Id = Matrix{Float64}(I, dp, dp)
        w = zeros(Float64, Dχ, Dχ, dp, dp)
        w[DONE, DONE, :, :]   .= Id               # already-placed identity
        w[START, START, :, :] .= Id               # not-yet-placed identity

        for (k, ch) in enumerate(channels)
            cα = carrier(k)
            w[cα, cα, :, :] .= ch.sl .* Id        # constant carrier self-loop

            this_kind = e.kind == :site ? :vertex : (e.i == 1 ? :hlink : :vlink)
            # OPEN: start → carrier, deposit the open operator with its weight.
            # Bond flows left→right (b₀=START … bₙ=DONE): open is W[START,cα].
            if this_kind == ch.open_kind
                op = locop(e, ch.open_op)
                w[START, cα, :, :] .+= ch.open_w(e.x, e.y) .* op
            end
            # CLOSE: carrier → done, deposit the close operator with its weight.
            if this_kind == ch.close_kind
                op = locop(e, ch.close_op)
                w[cα, DONE, :, :] .+= ch.close_w(e.x, e.y) .* op
            end
        end
        W[p] = w
    end
    return W
end

"""Left/right boundary bond vectors selecting START on the left, DONE on the
right (so a full path runs start → carrier(s) → done exactly once)."""
function mpo_boundaries(Dχ::Int)
    L = zeros(Float64, Dχ); L[Dχ] = 1.0          # START
    R = zeros(Float64, Dχ); R[1]  = 1.0          # DONE
    return L, R
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Coefficient extraction from the MPO  (layer-2 self-test)                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    mpo_coefficient(W, geo, ℓpos, spos; Q, φ) → scalar

Extract M̃[ℓ,s] = coefficient of (φ_ℓ Q_s) in Ô by contracting the MPO with the
traceless probes φ at chain position `ℓpos`, Q at `spos`, and the IDENTITY
elsewhere, then dividing by ‖φ‖²‖Q‖² and the trivial tr(I) on probe-free sites.
Because Q, φ are traceless, only the genuine φ_ℓQ_s term survives."""
function mpo_coefficient(W::Vector{Array{Float64,4}}, geo::LadderGeometry,
                         ℓpos::Int, spos::Int;
                         Q::AbstractMatrix=charge_operator(),
                         φ::AbstractMatrix=link_phase_operator(size(W[findfirst(e->e.kind==:link, geo.chain)],3)))
    Dχ = size(W[1], 1)
    L, R = mpo_boundaries(Dχ)
    row = reshape(copy(L), 1, Dχ)                 # 1×Dχ running bond covector
    prod_rest = 1.0                               # ∏ dim over identity-probe sites
    for (p, e) in enumerate(geo.chain)
        dp = size(W[p], 3)
        is_probe = (p == ℓpos) || (p == spos)
        probe = p == ℓpos ? φ :
                p == spos ? Q :
                Matrix{Float64}(I, dp, dp)
        is_probe || (prod_rest *= dp)             # Tr(I·I)=dp on probe-free sites
        # contract local operator against probe: T[a,b] = Σ_{ket,bra} W[a,b,ket,bra] probe[bra,ket]
        T = zeros(Float64, Dχ, Dχ)
        @inbounds for a in 1:Dχ, b in 1:Dχ
            T[a, b] = sum(W[p][a, b, :, :] .* permutedims(probe))
        end
        row = row * T
    end
    val = (row*R)[1]
    # divide by ‖φ‖²‖Q‖² and the trivial tr(I)=dim from every probe-free site
    return val / (tr(φ' * φ) * tr(Q' * Q) * prod_rest)
end

"""Reconstruct the full horizontal M̃[ℓ,s] from the assembled MPO and compare to
the target matrix `target` (default: the analytic model the MPO was built from,
so the error measures FSA-tensor correctness, not Stage-2 truncation); returns
(M̃, max_err)."""
function reconstruct_M_from_mpo(W::Vector{Array{Float64,4}}, geo::LadderGeometry,
                                target::AbstractMatrix;
                                Q::AbstractMatrix=charge_operator(),
                                φ::AbstractMatrix=link_phase_operator(size(W[2],3)))
    Mt = zeros(Float64, geo.n_links, geo.n_sites)
    max_err = 0.0
    for x in 1:geo.N, yl in 1:2, xp in 1:geo.N + 1, ys in 1:2
        ℓ = geo.link_id[(x, yl, 1)]; s = geo.vertex_id[(xp, ys)]
        ℓpos = geo.link_pos[(x, yl, 1)]; spos = geo.site_pos[(xp, ys)]
        v = mpo_coefficient(W, geo, ℓpos, spos; Q=Q, φ=φ)
        Mt[ℓ, s] = v
        max_err = max(max_err, abs(v - target[ℓ, s]))
    end
    return Mt, max_err
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Horizontal-channel assembly                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    horizontal_channels(m::HModel, x0) → Vector{GeoChannel}

Build the FSA channels reproducing the full horizontal-link kernel
    M[link(x,yl,1), vert(x',ys)] = ½ v_SS(x,x') + ½σ(yl)σ(ys)·v_AA(x,x'),
with v_SS = a0+q·x+r·x' + s·𝟙(x≥x') and v_AA = c·λ^{|…|} (half-integer odd).

Geometric distance is carried via absolute-position factors λ^{±(x−x0)} on the
open/close weights (x0 = lattice midpoint, for numerical balance) with carrier
self-loop sl=1; chain order alone selects the branch (vertex-before-link ⟺ x≥x').

Massive part (antisymmetric, decaying λ=2−√3), two orderings:
  • rightward (vertex opens → link closes):  +½σ(ys)·cσ(yl)·λ^{x−x'}  (x ≥ x');
  • leftward  (link opens → vertex closes):  −½σ(ys)·cσ(yl)·λ^{x'−x−1} (x < x').
Massless CONTINUOUS affine background (λ=1, both orderings, 3 pieces × 2 = 6) and
the massless STEP s·𝟙(x≥x') (one ordering, vertex-open/link-close)."""
function horizontal_channels(m::HModel, x0::Float64)
    c, λ, a0, q, r, s = m.c, m.λ, m.a0, m.q, m.r, m.s
    gp(x) = λ^(x - x0)        # grows toward large x
    gm(x) = λ^(-(x - x0))     # grows toward small x  (gp(x)*gm(x') = λ^{x−x'})

    chans = GeoChannel[]
    # ── massive (decaying), absolute-position factored ───────────────────────
    # rightward: source(vertex,x') opens → link(x) closes   ⇒ ½σ(ys)·cσ(yl)·λ^{x−x'}
    push!(chans, GeoChannel(1.0, :vertex, :Q,  (x, y) -> 0.5 * σrow(y) * gm(x),
                                 :hlink,  :phi,(x, y) -> c * σrow(y) * gp(x)))
    # leftward: link(x) opens → source(vertex,x') closes    ⇒ −½σ(ys)·cσ(yl)·λ^{x'−x−1}
    push!(chans, GeoChannel(1.0, :hlink,  :phi,(x, y) -> σrow(y) * gm(x),
                                 :vertex, :Q,  (x, y) -> -0.5 * c * σrow(y) * gp(x) / λ))

    # ── massless continuous affine (λ=1, sl=1), each piece × both orderings ──
    # ½a0 (Σφ)(ΣQ)
    push!(chans, GeoChannel(1.0, :vertex,:Q,  (x, y) -> 0.5, :hlink, :phi,(x, y) -> a0))
    push!(chans, GeoChannel(1.0, :hlink, :phi,(x, y) -> a0,  :vertex,:Q,  (x, y) -> 0.5))
    # ½q (Σ xφ)(ΣQ)
    push!(chans, GeoChannel(1.0, :vertex,:Q,  (x, y) -> 0.5,    :hlink, :phi,(x, y) -> q * x))
    push!(chans, GeoChannel(1.0, :hlink, :phi,(x, y) -> q * x,  :vertex,:Q,  (x, y) -> 0.5))
    # ½r (Σφ)(Σ xQ)
    push!(chans, GeoChannel(1.0, :vertex,:Q,  (x, y) -> 0.5 * r * x, :hlink, :phi,(x, y) -> 1.0))
    push!(chans, GeoChannel(1.0, :hlink, :phi,(x, y) -> 1.0, :vertex,:Q,  (x, y) -> 0.5 * r * x))

    # ── massless STEP s·𝟙(x≥x') (λ=1), one ordering: source opens → link closes
    push!(chans, GeoChannel(1.0, :vertex,:Q,  (x, y) -> 0.5 * s, :hlink, :phi,(x, y) -> 1.0))
    return chans
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Data-driven channels: geometric (massive) + affine ramp (massless)      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# A masked one-ordering exponential kernel is FULL rank, so SVD would inflate the
# bond.  The compact representation is semiseparable: each exponential generator
# μ_link^x · μ_src^{x'} is ONE geometric FSA channel (carrier sl=1, absolute
# weights).  We least-squares-fit each chain-ordered kernel block to the four
# generators {λ,1/λ}×{λ,1/λ} — bulk (λ^{|x−x'|}) plus both boundary reflections
# (λ^{x+x'}, λ^{−x−x'}) — capturing the finite-size kernel far better than a
# single bulk exponential.  All fits act on the RAW shift field, so they work at
# any N (no N≳12 bulk fit needed).

"""Least-squares-fit a chain-ordered kernel `Kmat[link_col,src_col]` to geometric
generators and push the resulting channels.  `after[li,si]` selects the block
where the link is chain-AFTER the source (source opens, link closes); the
complement is the reverse ordering.  `link_rowfac`/`src_rowfac` apply the
transverse row sign (σ for the antisymmetric mode, masks for a fixed source row);
weights are centered at `x0` for numerical balance."""
function geometric_kernel_channels!(chans::Vector{GeoChannel}, Kmat::AbstractMatrix,
                                    link_cols::Vector{Int}, src_cols::Vector{Int},
                                    after::AbstractMatrix{Bool},
                                    link_kind::Symbol, link_rowfac, src_rowfac,
                                    λ::Float64, x0::Float64; tol::Float64=1e-8)
    gens = [(a, b) for a in (λ, 1 / λ) for b in (λ, 1 / λ)]   # (μ_link, μ_src)
    xil = Dict(x => k for (k, x) in enumerate(link_cols))
    xis = Dict(x => k for (k, x) in enumerate(src_cols))
    for (mask, source_opens) in ((after, true), (.!after, false))
        rows = [(li, si) for li in eachindex(link_cols)
                         for si in eachindex(src_cols) if mask[li, si]]
        isempty(rows) && continue
        A = [gens[g][1]^(link_cols[li] - x0) * gens[g][2]^(src_cols[si] - x0)
             for (li, si) in rows, g in eachindex(gens)]
        b = [Kmat[li, si] for (li, si) in rows]
        # column-equilibrate: the reflection generators λ^{x+x'} span a huge
        # dynamic range (cond ~1e13 by N=12), so normalize columns before solving
        # and unscale the amplitudes — otherwise the LSQ loses all precision.
        scales = [norm(view(A, :, g)) for g in axes(A, 2)]
        scales = [s == 0 ? 1.0 : s for s in scales]
        a = ((A ./ scales') \ b) ./ scales
        for g in eachindex(gens)
            abs(a[g]) < tol && continue
            let μl = gens[g][1], μs = gens[g][2], ag = a[g],
                lf = link_rowfac, sf = src_rowfac, xil = xil, xis = xis,
                lk = link_kind, x0 = x0
                linkw = (x, y) -> (haskey(xil, x) ? lf(y) * μl^(x - x0) : 0.0)
                srcw  = (x, y) -> (haskey(xis, x) ? sf(y) * ag * μs^(x - x0) : 0.0)
                if source_opens
                    push!(chans, GeoChannel(1.0, :vertex, :Q, srcw, lk, :phi, linkw))
                else
                    push!(chans, GeoChannel(1.0, lk, :phi, linkw, :vertex, :Q, srcw))
                end
            end
        end
    end
    return chans
end

"""The seven massless (λ=1) affine-ramp + step channels for horizontal links,
reproducing ½(a0+q·x+r·x') for all pairs plus ½s·𝟙(x≥x')."""
function affine_step_channels!(chans::Vector{GeoChannel}, a0, q, r, s)
    push!(chans, GeoChannel(1.0, :vertex,:Q,  (x, y) -> 0.5, :hlink, :phi,(x, y) -> a0))
    push!(chans, GeoChannel(1.0, :hlink, :phi,(x, y) -> a0,  :vertex,:Q,  (x, y) -> 0.5))
    push!(chans, GeoChannel(1.0, :vertex,:Q,  (x, y) -> 0.5,    :hlink, :phi,(x, y) -> q * x))
    push!(chans, GeoChannel(1.0, :hlink, :phi,(x, y) -> q * x,  :vertex,:Q,  (x, y) -> 0.5))
    push!(chans, GeoChannel(1.0, :vertex,:Q,  (x, y) -> 0.5 * r * x, :hlink, :phi,(x, y) -> 1.0))
    push!(chans, GeoChannel(1.0, :hlink, :phi,(x, y) -> 1.0, :vertex,:Q,  (x, y) -> 0.5 * r * x))
    push!(chans, GeoChannel(1.0, :vertex,:Q,  (x, y) -> 0.5 * s, :hlink, :phi,(x, y) -> 1.0))
    return chans
end

"""
    ladder_channels(geo, M; λ, tol) → Vector{GeoChannel}

Full data-driven channel set reproducing the ENTIRE shift field M (horizontal
AND vertical links) from the raw matrix:
  • horizontal massive (antisymmetric): geometric generators, row factor σ;
  • horizontal massless (symmetric): affine ramp + step (a0,q,r,s);
  • vertical links: geometric generators per source row (the rung difference
    annihilates the massless mode, so vertical is purely massive).
`tol` drops negligible generator amplitudes (raise accuracy by lowering it)."""
function ladder_channels(geo::LadderGeometry, M::AbstractMatrix;
                         λ::Float64=2 - sqrt(3), tol::Float64=1e-8)
    chans = GeoChannel[]
    x0 = (geo.N + 2) / 2

    # horizontal massive (antisymmetric channel ½v_AA, row factor σ)
    hcols = collect(1:geo.N); scols = collect(1:geo.N + 1)
    after_h = [x ≥ xp for x in hcols, xp in scols]
    KA = zeros(length(hcols), length(scols))
    for (_, x, xp, v) in channel_pairs(geo, M, :A, :A)
        KA[x, xp] = 0.5 * v
    end
    geometric_kernel_channels!(chans, KA, hcols, scols, after_h, :hlink,
                               σrow, σrow, λ, x0; tol=tol)

    # horizontal massless (affine ramp + step), valid at any N
    a0, q, r, s, _ = symmetric_ramp_step(geo, M)
    affine_step_channels!(chans, a0, q, r, s)

    # vertical links (purely massive), per source row
    vcols = collect(1:geo.N + 1)
    for ys in 1:2
        KV = [M[geo.link_id[(x, 1, 2)], geo.vertex_id[(xp, ys)]] for x in vcols, xp in vcols]
        after_v = [geo.link_pos[(x, 1, 2)] > geo.site_pos[(xp, ys)] for x in vcols, xp in vcols]
        srcfac = let ys = ys; (y) -> (y == ys ? 1.0 : 0.0); end
        geometric_kernel_channels!(chans, KV, vcols, vcols, after_v, :vlink,
                                   (y) -> 1.0, srcfac, λ, x0; tol=tol)
    end
    return chans
end

"""Reconstruct M̃[ℓ,s] for ALL links (horizontal AND vertical) from the MPO and
compare to the exact shift field; returns (M̃, max_err)."""
function reconstruct_all_M_from_mpo(W::Vector{Array{Float64,4}}, geo::LadderGeometry;
                                    Q::AbstractMatrix=charge_operator(),
                                    φ::AbstractMatrix=link_phase_operator(size(W[2], 3)))
    M = shift_field(geo)
    Mt = zeros(Float64, geo.n_links, geo.n_sites)
    max_err = 0.0
    for (ℓ, key) in enumerate(geo.link_def)
        ℓpos = geo.link_pos[key]
        for (s, (xp, ys)) in enumerate(geo.vertex_xy)
            spos = geo.site_pos[(xp, ys)]
            v = real(mpo_coefficient(W, geo, ℓpos, spos; Q=Q, φ=φ))
            Mt[ℓ, s] = v
            max_err = max(max_err, abs(v - M[ℓ, s]))
        end
    end
    return Mt, max_err
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Self-test driver                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    validate_decoupling_mpo(N; d, K, tol) → Bool

Two-layer check on the horizontal-link exponent MPO:
  layer 1 — analytic channel reconstruction of M  (decomposition correctness);
  layer 2 — MPO coefficient extraction of M̃       (FSA-tensor correctness);
plus a report of the MPO bond dimension (should be 2 + #channels ∝ K)."""
function validate_decoupling_mpo(N::Int; d::Int=3, K::Int=3, tol_fsa::Float64=1e-9)
    geo = ladder_geometry(N)
    m   = horizontal_model(geo; K=K)
    Mr  = analytic_M(geo, m)                      # exact FSA target

    println("─── decoupling-MPO self-test (N=$N, d=$d) ───")
    @printf("  model: c=%+.5f λ=%.5f | a0=%+.4f q=%+.4f r=%+.4f s=%+.4f\n",
            m.c, m.λ, m.a0, m.q, m.r, m.s)

    # FSA: does the MPO reproduce its analytic target Mr exactly?
    x0 = (geo.N + 2) / 2                          # column midpoint (numerical balance)
    chans = horizontal_channels(m, x0)
    Q = charge_operator(); φ = link_phase_operator(d)
    W = build_exponent_mpo(geo, chans; d=d, Q=Q, φ=φ)
    Dχ = size(W[1], 1)
    _, err_fsa = reconstruct_M_from_mpo(W, geo, Mr; Q=Q, φ=φ)
    @printf("  FSA: MPO vs analytic target   max_err = %.2e   (Dχ=%d = 2+%d channels)\n",
            err_fsa, Dχ, length(chans))

    # Truncation: analytic model vs exact shift field (Stage-2 story).
    err_trunc = analytic_truncation_error(geo, m)
    @printf("  model vs exact shift field    max_err = %.2e   (single-exp truncation)\n",
            err_trunc)

    ok = err_fsa < tol_fsa
    println(ok ? "  PASS: FSA-compiled MPO reproduces the exponent kernel to machine precision" :
                 "  WARN: FSA error exceeds tol — check column-tick bookkeeping / signs")
    return ok
end

"""
    validate_full_mpo(N; d, tol, tol_kernel) → Bool

Validate the FULL data-driven exponent MPO (horizontal + vertical links) built by
`ladder_channels` directly from the exact shift field:
  • FSA + kernel accuracy — the contracted MPO reproduces the EXACT M over ALL
    links to `tol_kernel` (geometric generators capture the finite-size kernel;
    raise accuracy by lowering `tol`);
  • reports Dχ and a breakdown of horizontal vs vertical error.
This supersedes `validate_decoupling_mpo` (which was horizontal-only, K=1)."""
function validate_full_mpo(N::Int; d::Int=3, tol::Float64=1e-8, tol_kernel::Float64=1e-4)
    geo = ladder_geometry(N)
    M   = shift_field(geo)
    chans = ladder_channels(geo, M; tol=tol)
    Q = charge_operator(); φ = link_phase_operator(d)
    W = build_exponent_mpo(geo, chans; d=d, Q=Q, φ=φ)
    Dχ = size(W[1], 1)
    Mt, err = reconstruct_all_M_from_mpo(W, geo; Q=Q, φ=φ)

    # split error by link direction
    errh = 0.0; errv = 0.0
    for (ℓ, (x, y, i)) in enumerate(geo.link_def), s in 1:geo.n_sites
        e = abs(Mt[ℓ, s] - M[ℓ, s])
        i == 1 ? (errh = max(errh, e)) : (errv = max(errv, e))
    end

    println("─── full exponent MPO (N=$N, d=$d, gen-tol=$tol) ───")
    @printf("  Dχ=%d (= 2 + %d channels)   MPO vs exact M: max_err=%.2e  (h=%.2e, v=%.2e)\n",
            Dχ, length(chans), err, errh, errv)
    ok = err < tol_kernel
    println(ok ? "  PASS: full MPO (horizontal + vertical) reproduces M to tol_kernel=$tol_kernel" :
                 "  WARN: kernel error exceeds tol_kernel (lower gen-tol for more generators)")
    return ok
end

# Demo / self-test when run directly.
if abspath(PROGRAM_FILE) == @__FILE__
    for N in (8, 12)
        validate_full_mpo(N; d=3)
        println()
    end
end
