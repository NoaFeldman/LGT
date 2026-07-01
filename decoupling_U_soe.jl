#= ═══════════════════════════════════════════════════════════════════════════════
   decoupling_U_soe.jl

   SoE-approximated Bender–Zohar gauge–matter decoupling on the pseudo-1D (snake)
   MPS, applied to STATES.

   ── Why state-application, not operator-materialization ──────────────────────
   The decoupler 𝒰 = exp(−iÔ), Ô = Σ_{ℓ,n} M_{ℓ,n} φ_ℓ Q_n (M = ∇G), is a GLOBAL
   ENTANGLER: its operator bond grows with the Hilbert space, so materialising 𝒰
   as an MPO is only feasible on tiny lattices.  The Sum-of-Exponentials makes the
   low-rank object the EXPONENT Ô (and the kernel G), not 𝒰.  So we build the
   low-bond Ô and apply 𝒰|ψ⟩ = exp(−iÔ)|ψ⟩ by Krylov time-stepping the STATE
   (the state's entanglement is bounded) — this scales to 3×4 and beyond.

   ── The summed exponent (no per-channel blowup) ──────────────────────────────
   The SoE shift field M_SoE = Σ_{j=1}^K w_j ∇[e^{−γ_j(|Δx|+|Δy|)}] is BOUNDED even
   when individual w_j are huge (an ill-conditioned fit's large amplitudes cancel
   in the sum, since the sum reproduces ∇G).  So we build a single low-bond
   Ô_SoE from the summed M — no per-channel exponentials, no a-cap, no artifacts.
   Mathematically 𝒰 = Π_j exp(−iÔ_j) = exp(−iΣ_jÔ_j) (the terms commute), so the
   summed exponent IS the Form-A product, in its numerically-stable realisation.

   Decisions: (1) snake-MPS, column-major snake; (2) analytic SoE gradient;
   (3) Form A ≡ summed commuting exponent; (4) staggered Q, boundary folded in.

   Requires: mps_lgt.jl (MPS/MPO/HTerm/Krylov), lgt_greens_soe.jl (SoE + exact G)
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "mps_lgt.jl"))
include(joinpath(@__DIR__, "lgt_greens_soe.jl"))

using LinearAlgebra
using Printf

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
# ║  Shift-field kernels: exact and SoE (analytic gradient), summed            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

φ_op() = ComplexF64.(0.5 * (op_U_gauge(1) + op_U_gauge(1)'))     # ½(U+U†), dg=1
bg_stag(ix, iy) = isodd(ix + iy) ? 1.0 : 0.0

"""Exact shift field M_{ℓ,n} = G(a,n) − G(b,n) for link ℓ=(a→b)."""
function shift_exact(Gt, ix, iy, dir, jx, jy)
    bx, by = dir == :R ? (ix + 1, iy) : (ix, iy + 1)
    return Gt[ix, iy, jx, jy] - Gt[bx, by, jx, jy]
end

"""Summed SoE shift field M_SoE = Σ_j w_j ∇[e^{−γ_j(|Δx|+|Δy|)}] (bounded)."""
function shift_soe(soe, ix, iy, dir, jx, jy)
    s = 0.0
    for j in 1:soe.K
        w = soe.w[j]; γ = soe.γ[j]; ex(d) = exp(-γ * abs(d))
        s += dir == :R ? w * ex(iy - jy) * (ex(ix - jx) - ex(ix + 1 - jx)) :
                         w * ex(ix - jx) * (ex(iy - jy) - ex(iy + 1 - jy))
    end
    return s
end

near_boundary(nx, ny, x, y, bw) = x ≤ bw || x > nx - bw || y ≤ bw || y > ny - bw

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Exponent MPO  Ô = Σ M φ Q  from a shift-field function                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Assemble Ô = Σ_{ℓ,n} Mfun(ℓ,n) φ_ℓ Q_n on the snake (`pos`,`dims`); returns
(Ô::CMPO, Σ|M|).  Q_n = n_f − staggered background."""
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
    isempty(terms) && return _assemble_mpo(dims, [HTerm(0.0, Dict(1 => zeros(ComplexF64, dims[1], dims[1])))]), 0.0
    return _assemble_mpo(dims, terms), Mabs
end

# spectral-radius upper bound (1-norm): ‖Ô‖ ≤ Σ|M| ‖φ‖ ‖Q‖
_spectral_bound(Mabs) = Mabs * opnorm(Matrix(φ_op())) * opnorm(op_nf() - 0.5 * _Id(LGT_d_f))

"""
    build_exponents(nx, ny; K, bw, Nx_ref, Ny_ref)

Build the low-bond decoupling exponents on the column snake:
  • `Oexact`  — Ô from the exact ∇G;
  • `Obulk`   — Ô from the summed SoE ∇G (pure bulk approximation);
  • `Ofull`   — Ô_SoE with the exact field folded in within `bw` of the boundary.
Returns each with its spectral bound, plus the snake dims and the SoE kernel.
The SoE is fit on a larger reference lattice so K isn't capped by the target."""
function build_exponents(nx::Int, ny::Int; dg::Int=1, K::Int=10, bw::Int=1,
                         Nx_ref::Int=8, Ny_ref::Int=8)
    @assert dg == 1
    _, pos = column_snake(nx, ny); dims = col_node_dims(nx, ny, dg)
    soe = soe_approximation(nx, ny; K=K, refine=true,
                            Nx_ref=max(nx, Nx_ref), Ny_ref=max(ny, Ny_ref))
    _, Gt = generate_greens_function(nx, ny)

    Oexact, Me = build_O(nx, ny, dg, pos, dims, (a, b, d, c, e) -> shift_exact(Gt, a, b, d, c, e))
    Obulk,  Mb = build_O(nx, ny, dg, pos, dims, (a, b, d, c, e) -> shift_soe(soe, a, b, d, c, e))
    function Mfull(ix, iy, dir, jx, jy)
        bx, by = dir == :R ? (ix + 1, iy) : (ix, iy + 1)
        edge = near_boundary(nx, ny, ix, iy, bw) || near_boundary(nx, ny, bx, by, bw) ||
               near_boundary(nx, ny, jx, jy, bw)
        return edge ? shift_exact(Gt, ix, iy, dir, jx, jy) : shift_soe(soe, ix, iy, dir, jx, jy)
    end
    Ofull, Mf = build_O(nx, ny, dg, pos, dims, Mfull)

    return (Oexact=Oexact, a_exact=_spectral_bound(Me),
            Obulk=Obulk, a_bulk=_spectral_bound(Mb),
            Ofull=Ofull, a_full=_spectral_bound(Mf),
            dims=dims, soe=soe)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Apply the decoupler to a state:  𝒰|ψ⟩ = exp(−iÔ)|ψ⟩  (Krylov, sub-stepped)║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    decouple_state(O, a, ψ; Dmax, m) → exp(−iÔ)|ψ⟩

Apply the decoupler by Krylov time-stepping the STATE: split exp(−iÔ) into
`n ≈ ⌈a⌉` sub-steps exp(−iÔ/n) (each ‖Ô/n‖ ≲ 1, so a small Krylov space
converges).  Never materialises 𝒰."""
function decouple_state(O::CMPO, a::Float64, ψ::CMPS; Dmax::Int=64, m::Int=8)
    n = max(1, ceil(Int, a))
    φ = deepcopy(ψ); φ[1] ./= mps_norm(φ)
    for _ in 1:n
        φ = krylov_evolve_step(O, φ, 1.0 / n; m=m, Dmax=Dmax)
    end
    return φ
end

"""Random bond-1 product MPS with the given local dimensions."""
function random_product_mps(dims::AbstractVector{Int})
    ψ = CMPS(undef, length(dims))
    for (p, d) in enumerate(dims)
        v = randn(ComplexF64, d); v ./= norm(v); ψ[p] = reshape(v, 1, 1, d)
    end
    return ψ
end

mps_fidelity(φ1, φ2) = abs2(mps_overlap(φ1, φ2)) / (real(mps_overlap(φ1, φ1)) * real(mps_overlap(φ2, φ2)))

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Self-test: state-based, scales to 3×4 (no dense contraction)             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    validate_decouple(; nx, ny, Ks, bw, Dmax)

Apply the SoE and exact decouplers to a random product state and compare:
  • fidelity |⟨𝒰_SoE ψ | 𝒰_exact ψ⟩|²  (bulk-only and boundary-folded) vs K;
  • unitarity via norm preservation ‖𝒰_SoE ψ‖/‖ψ‖;
  • the exponent Ô bonds (low — the genuine SoE efficiency).
No dense contraction, so this runs at 3×4."""
function validate_decouple(; nx::Int=2, ny::Int=2, Ks=(2, 4, 6), bw::Int=1, Dmax::Int=64)
    println("─── SoE decoupling applied to a state ($(nx)×$(ny), column snake) ───")
    Random.seed!(1)
    dims = col_node_dims(nx, ny, 1)
    @printf("  Hilbert dim = %d   (state-based test — no dense contraction)\n", prod(dims))
    ψ = random_product_mps(dims)

    for K in Ks
        E = build_exponents(nx, ny; K=K, bw=bw)
        bd(O) = maximum(size(W, 2) for W in O)
        @printf("  K=%-2d:  Ô bonds  exact=%d  SoE-bulk=%d  SoE+bdry=%d   (a: ex=%.1f bulk=%.1f full=%.1f)\n",
                K, bd(E.Oexact), bd(E.Obulk), bd(E.Ofull), E.a_exact, E.a_bulk, E.a_full)
        φ_ex   = decouple_state(E.Oexact, E.a_exact, ψ; Dmax=Dmax)
        φ_bulk = decouple_state(E.Obulk,  E.a_bulk,  ψ; Dmax=Dmax)
        φ_full = decouple_state(E.Ofull,  E.a_full,  ψ; Dmax=Dmax)
        nψ = mps_norm(ψ)
        @printf("       fidelity vs exact:  bulk=%.6f  +bdry=%.6f   ‖𝒰ψ‖/‖ψ‖−1: %.2e\n",
                mps_fidelity(φ_bulk, φ_ex), mps_fidelity(φ_full, φ_ex),
                abs(mps_norm(φ_full) / nψ - 1))
        flush(stdout)
    end
    println("  (Ô is low-bond — that is the SoE efficiency; 𝒰 itself is a high-bond entangler)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_decouple(; nx=2, ny=2, Ks=(2, 4, 6))
    println()
    validate_decouple(; nx=3, ny=4, Ks=(2, 6), bw=1)   # full target — state-based, no dense
    println()
end
