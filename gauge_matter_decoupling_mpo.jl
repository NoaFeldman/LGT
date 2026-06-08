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
    Q = Diagonal(Float64[0, 1]) |> Matrix
    centered && (Q .-= (tr(Q) / size(Q, 1)) * I)
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
    centered && (φ .-= (tr(φ) / d) * I)         # already traceless, kept for safety
    return Matrix(φ)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Chain bookkeeping: column index + per-column last site                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Column index of a chain element (the snake processes columns left→right)."""
column_of(e::ChainElement) = e.x

"""
    chain_columns(geo) → (col, is_col_last)

`col[p]`         = column of chain site p;
`is_col_last[p]` = true iff p is the final chain site of its column (the point at
                   which a carrier "ticks", i.e. multiplies by λ on the way out)."""
function chain_columns(geo::LadderGeometry)
    col = [column_of(e) for e in geo.chain]
    n = length(col)
    is_col_last = falses(n)
    for p in 1:n
        is_col_last[p] = (p == n) || (col[p+1] != col[p])
    end
    return col, is_col_last
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Analytic channel reconstruction of M  (layer-1 self-test)               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

σrow(y::Int) = y == 1 ? 1.0 : -1.0             # antisymmetric-mode row sign

"""
    reconstruct_M_horizontal(geo, fits, ramp) → (Mr, max_err)

Rebuild the horizontal-link block of the shift field purely from the Stage-2
channel parameters, using

    M[link(x,yl,1), vert(x',ys)] = ½ v_SS(x,x') + ½ σ(yl)σ(ys) v_AA(x,x'),

with v_SS the affine ramp (p,q,r) and v_AA the massive single-exponential kernel
(amplitude c, rate λ, half-integer antisymmetric extension v(d) = −v(−1−d)).
Compares to the exact `shift_field` on horizontal rows and returns the rebuilt
matrix plus the max abs error — this isolates the DECOMPOSITION from the MPO."""
function reconstruct_M_horizontal(geo::LadderGeometry,
                                  FA::ExpFit, ramp::NTuple{3,Float64})
    M = shift_field(geo)
    p, q, r = ramp
    λ_anti = 2 - sqrt(3)
    kdom = argmin(abs.(FA.λ .- λ_anti))
    c = real(FA.c[kdom]); λ = real(FA.λ[kdom])

    v_SS(x, xp) = p + q * x + r * xp
    Sd(d) = c * λ^d                                   # rightward (d ≥ 0)
    v_AA(x, xp) = (x - xp) ≥ 0 ? Sd(x - xp) : -Sd(-(x - xp) - 1)

    Mr = zeros(Float64, geo.n_links, geo.n_sites)
    max_err = 0.0
    for x in 1:geo.N, yl in 1:2, xp in 1:geo.N + 1, ys in 1:2
        ℓ = geo.link_id[(x, yl, 1)]
        s = geo.vertex_id[(xp, ys)]
        val = 0.5 * v_SS(x, xp) + 0.5 * σrow(yl) * σrow(ys) * v_AA(x, xp)
        Mr[ℓ, s] = val
        max_err = max(max_err, abs(val - M[ℓ, s]))
    end
    return Mr, max_err
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  FSA MPO compiler                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Bond layout (size Dχ):  index 1 = "done", indices 2:Dχ−? = carriers, last =
# "start".  We assemble W^[p] ∈ ℝ^{Dχ×Dχ} of d_p×d_p local operators.

"""A geometric two-body channel for the FSA.

Couples an `open` operator placed at sites of `open_kind` (with scalar weight
`open_w(x,y)`) to a `close` operator placed at sites of `close_kind` LATER on the
chain (weight `close_w(x,y)`), with a carrier that multiplies by `λ` once per
column tick between them.  Set `open_kind=:vertex, close_kind=:link` for the
source→link (rightward) ordering and the reverse for link→source (leftward)."""
struct GeoChannel
    λ        :: Float64
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
    _, is_col_last = chain_columns(geo)
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
            # carrier self-loop: tick by λ only when leaving the last site of a column
            w[cα, cα, :, :] .= (is_col_last[p] ? ch.λ : 1.0) .* Id

            this_kind = e.kind == :site ? :vertex : :link
            # OPEN: start → carrier, deposit the open operator with its weight
            if this_kind == ch.open_kind
                op = locop(e, ch.open_op)
                w[cα, START, :, :] .+= ch.open_w(e.x, e.y) .* op
            end
            # CLOSE: carrier → done, deposit the close operator with its weight
            if this_kind == ch.close_kind
                op = locop(e, ch.close_op)
                w[DONE, cα, :, :] .+= ch.close_w(e.x, e.y) .* op
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
the exact shift field; returns (M̃, max_err)."""
function reconstruct_M_from_mpo(W::Vector{Array{Float64,4}}, geo::LadderGeometry;
                                Q::AbstractMatrix=charge_operator(),
                                φ::AbstractMatrix=link_phase_operator(size(W[2],3)))
    M = shift_field(geo)
    Mt = zeros(Float64, geo.n_links, geo.n_sites)
    max_err = 0.0
    for x in 1:geo.N, yl in 1:2, xp in 1:geo.N + 1, ys in 1:2
        ℓ = geo.link_id[(x, yl, 1)]; s = geo.vertex_id[(xp, ys)]
        ℓpos = geo.link_pos[(x, yl, 1)]; spos = geo.site_pos[(xp, ys)]
        v = mpo_coefficient(W, geo, ℓpos, spos; Q=Q, φ=φ)
        Mt[ℓ, s] = v
        max_err = max(max_err, abs(v - M[ℓ, s]))
    end
    return Mt, max_err
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Horizontal-channel assembly                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    horizontal_channels(FA, ramp) → Vector{GeoChannel}

Build the FSA channels reproducing the full horizontal-link kernel
    M[link(x,yl,1), vert(x',ys)] = ½(p+q·x+r·x') + ½σ(yl)σ(ys)·c·λ^{|…|}.

Massive part (antisymmetric, decaying λ=2−√3), two chain orderings:
  • rightward (vertex opens → link closes):  +½σ(ys)·cσ(yl)·λ^{x−x'};
  • leftward  (link opens → vertex closes):  −½σ(ys)·cσ(yl)·λ^{x'−x−1}.

Massless ramp (symmetric, λ=1, couples EVERY pair in BOTH chain orders).  As a
rank-1 product of global one-body sums the affine kernel splits into three
pieces  ½p·(Σφ)(ΣQ) + ½q·(Σ xφ)(ΣQ) + ½r·(Σφ)(Σ xQ); φ (links) and Q (vertices)
live on disjoint sites, so each piece needs ONE carrier per chain ordering
(vertex-before-link and link-before-vertex), six in total.  A given (ℓ,s) pair
has a definite chain order, so the two orderings cover it exactly once."""
function horizontal_channels(FA::ExpFit, ramp::NTuple{3,Float64})
    p, q, r = ramp
    λ_anti = 2 - sqrt(3)
    kdom = argmin(abs.(FA.λ .- λ_anti))
    c = real(FA.c[kdom]); λ = real(FA.λ[kdom])

    chans = GeoChannel[]
    # ── massive (decaying) ───────────────────────────────────────────────────
    # rightward: source(vertex) opens → horizontal link closes  (x ≥ x')
    push!(chans, GeoChannel(λ, :vertex, :Q,  (x, y) -> 0.5 * σrow(y),
                                :link,   :phi,(x, y) -> c * σrow(y)))
    # leftward: horizontal link opens → source(vertex) closes    (x < x')
    push!(chans, GeoChannel(λ, :link,   :phi,(x, y) -> σrow(y),
                                :vertex, :Q,  (x, y) -> -0.5 * (c / λ) * σrow(y)))

    # ── massless affine ramp (λ=1), each rank-1 piece × both chain orderings ──
    # ½p (Σφ)(ΣQ)
    push!(chans, GeoChannel(1.0, :vertex,:Q,  (x, y) -> 0.5, :link,  :phi,(x, y) -> p))
    push!(chans, GeoChannel(1.0, :link,  :phi,(x, y) -> p,   :vertex,:Q,  (x, y) -> 0.5))
    # ½q (Σ xφ)(ΣQ)
    push!(chans, GeoChannel(1.0, :vertex,:Q,  (x, y) -> 0.5,    :link,  :phi,(x, y) -> q * x))
    push!(chans, GeoChannel(1.0, :link,  :phi,(x, y) -> q * x,  :vertex,:Q,  (x, y) -> 0.5))
    # ½r (Σφ)(Σ xQ)
    push!(chans, GeoChannel(1.0, :vertex,:Q,  (x, y) -> 0.5 * r * x, :link,  :phi,(x, y) -> 1.0))
    push!(chans, GeoChannel(1.0, :link,  :phi,(x, y) -> 1.0, :vertex,:Q,  (x, y) -> 0.5 * r * x))
    return chans
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
function validate_decoupling_mpo(N::Int; d::Int=3, K::Int=3, tol::Float64=1e-6)
    geo  = ladder_geometry(N)
    M    = shift_field(geo)
    FA   = fit_channel(geo, M, :A, :A; K=K)
    pqr  = symmetric_ramp(geo, M)
    ramp = (pqr[1], pqr[2], pqr[3])

    println("─── decoupling-MPO self-test (N=$N, d=$d) ───")

    # layer 1: analytic decomposition
    _, err1 = reconstruct_M_horizontal(geo, FA, ramp)
    @printf("  layer-1 analytic channel reconstruction:  max_err = %.2e\n", err1)

    # layer 2: MPO tensors
    chans = horizontal_channels(FA, ramp)
    Q = charge_operator(); φ = link_phase_operator(d)
    W = build_exponent_mpo(geo, chans; d=d, Q=Q, φ=φ)
    Dχ = size(W[1], 1)
    _, err2 = reconstruct_M_from_mpo(W, geo; Q=Q, φ=φ)
    @printf("  layer-2 MPO coefficient reconstruction:    max_err = %.2e\n", err2)
    @printf("  MPO bond dimension Dχ = %d   (= 2 + %d channels)\n", Dχ, length(chans))

    ok1 = err1 < tol
    ok2 = err2 < tol
    println(ok1 ? "  PASS: analytic channel decomposition reproduces M" :
                  "  WARN: channel decomposition error exceeds tol")
    println(ok2 ? "  PASS: MPO reproduces the exponent kernel M" :
                  "  WARN: MPO reconstruction error exceeds tol")
    return ok1 && ok2
end

# Demo / self-test when run directly.
if abspath(PROGRAM_FILE) == @__FILE__
    for N in (8, 12)
        validate_decoupling_mpo(N; d=3)
        println()
    end
end
