#= ═══════════════════════════════════════════════════════════════════════════════
   matter_gauge_entanglement_pjw.jl

   Same pipeline as matter_gauge_entanglement.jl (matter vs dual-plaquette gauge
   half-system entanglement of the Bender–Zohar-decoupled 3×4 U(1) LGT ground
   state, over the gs_benchmark grid), but with a MAGNETIC PLAQUETTE TERM and
   JORDAN–WIGNER fermion strings in the Hamiltonian:

       H = Σ_i [ (g²/2)(E²_R+E²_U) + m(−1)^{ix+iy} n_f ]                (on-site)
          − t Σ_{hops} ( c†_i  [∏_{k between} (−1)^{n_f,k}]  U_ℓ  c_j + h.c. )
          − (1/2g²) Σ_plaq ( U†_b U_l U†_r U_t + h.c. )                 (magnetic)

   Differences from the base pipeline (everything downstream is identical):
     • JW strings — every hopping term carries a fermion-parity string
       ∏(−1)^{n_f} on the nodes lying STRICTLY BETWEEN its two endpoints in the
       column-snake chain order (vertical hops are chain-adjacent → no string;
       horizontal hops are long-range → real strings).  This turns the spinless
       hard-core convention of the base model into genuine lattice fermions.
     • Magnetic plaquette — the U(1) Wilson-loop term −(1/2g²)(W+W†) on every
       plaquette (a bond-1 product operator on the 3 corner nodes A,B,C).

   Gauss' law is UNCHANGED by both additions (the plaquette commutes with G_i; JW
   is a representation change of the fermions), so the staggered-sector penalty,
   the SoE decoupler and the dual-plaquette readout all carry over verbatim.

   The finite_ed.jl reference implements the base (no-plaquette / no-JW) model, so
   it is NOT a valid cross-check here.  Correctness is instead gated by:
     • a dense small-lattice self-test (this MPO vs exact diagonalisation of the
       same MPO) — run with:  PJW_SELFTEST=1 julia --project=. matter_gauge_entanglement_pjw.jl
     • the in-sector Gauss violation ⟨Σ(G−g)²⟩ ≈ 0 reported for every 3×4 point.

   Reuses the column-snake helpers, constants and readouts of the base worker.

   Usage:
       SLURM_ARRAY_TASK_ID=1 julia --project=. matter_gauge_entanglement_pjw.jl
       sbatch run_matter_gauge_entanglement_pjw.sh
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

# reuse the base pipeline's column-snake helpers, constants, grid and readouts
# (its `run_one` is guarded by @__FILE__ and does NOT auto-run on include)
include(joinpath(@__DIR__, "matter_gauge_entanglement.jl"))

using Printf, LinearAlgebra, CSV, DataFrames, Random

# The magnetic plaquette term makes the DMRG landscape rugged: the staggered
# start is the electric VACUUM, which traps 2-site DMRG because the plaquette
# term needs magnetic flux to act on.  Timing (diag_pjw_timing.jl) showed the
# cost is the local eigensolver on the d=18 sites, NOT the MPS bond (D=60 ≈ D=100),
# and that random full-bond restarts were what blew the runtime to 48h.  So we
# restart instead from cheap, low-bond, IN-SECTOR *flux-seeded* product states
# (staggered fermions + random source-free plaquette loops), and trim the solver.
const PJW_D     = 60       # DMRG bond (D=60 already matches D=100 on the 3×4 energy)
const PJW_NSW   = 6        # max sweeps (vacuum start converges in ~1–2; early-stops)
const PJW_NFLUX = 4        # flux-seeded restarts in addition to the staggered start
# NOTE: the local eigensolver uses dmrg_ground_state's DEFAULT settings
# (tol=1e-7, maxiter=60, krylovdim=12).  A custom "cheaper" set (krylovdim=10,
# maxiter=20) was tried and made a single sweep 50×+ SLOWER — a too-small Krylov
# subspace fails to converge and thrashes through restarts.  Do not shrink it.

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Hamiltonian with magnetic plaquette + Jordan–Wigner strings (column snake)║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""On-site + JW hopping + magnetic-plaquette terms of H on the COLUMN snake."""
function lgt_terms_plaqjw_cs(nx, ny, dg; g, t, m)
    chain, pos = column_snake(nx, ny)

    # fermion-parity operator (−1)^{n_f} embedded in each node (for the JW string)
    parity_at = Vector{Matrix{ComplexF64}}(undef, length(chain))
    for (p, (ix, iy)) in enumerate(chain)
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        parity_at[p] = ComplexF64.(embed_f_site(op_parity_f(), d_gR, d_gU))
    end

    terms = HTerm[]

    # ── on-site (diagonal): (g²/2)(E²_R+E²_U) + m(−1)^{ix+iy} n_f ─────────────
    for iy in 1:ny, ix in 1:nx
        H = ComplexF64.(H_onsite_site(ix, iy, nx, ny, dg; g=g, m=m))
        push!(terms, HTerm(1.0, Dict(pos[(ix, iy)] => H)))
    end

    # ── JW hopping: c†/c on the endpoints, parity on nodes strictly between ───
    function add_jw!(coef, pa, opa, pb, opb)
        lo, hi = minmax(pa, pb)
        d = Dict{Int,Matrix{ComplexF64}}(pa => opa, pb => opb)
        for p in lo+1:hi-1
            d[p] = parity_at[p]                       # (−1)^{n_f} string site
        end
        push!(terms, HTerm(coef, d))
    end

    for iy in 1:ny, ix in 1:nx-1                       # horizontal hops (long-range on snake)
        _, d_gR_R, d_gU_R = site_dims(ix+1, iy, nx, ny, dg)
        _, _, d_gU_L      = site_dims(ix,   iy, nx, ny, dg)
        AL = ComplexF64.(kron(op_cdag(), op_U_gauge(dg), _Id(d_gU_L)))   # c†⊗U_R⊗I on L
        BR = ComplexF64.(embed_f_site(op_c(), d_gR_R, d_gU_R))           # c on R
        pL, pR = pos[(ix, iy)], pos[(ix+1, iy)]
        add_jw!(-t, pL, AL,          pR, BR)
        add_jw!(-t, pL, Matrix(AL'), pR, Matrix(BR'))                    # h.c.
    end

    for iy in 1:ny-1, ix in 1:nx                       # vertical hops (chain-adjacent → no string)
        _, d_gR_U, d_gU_U = site_dims(ix, iy+1, nx, ny, dg)
        _, d_gR_D, _      = site_dims(ix, iy,   nx, ny, dg)
        AD = ComplexF64.(kron(op_cdag(), _Id(d_gR_D), op_U_gauge(dg)))   # c†⊗I⊗U_U on D
        BU = ComplexF64.(embed_f_site(op_c(), d_gR_U, d_gU_U))           # c on U
        pD, pU = pos[(ix, iy)], pos[(ix, iy+1)]
        add_jw!(-t, pD, AD,          pU, BU)
        add_jw!(-t, pD, Matrix(AD'), pU, Matrix(BU'))                    # h.c.
    end

    # ── magnetic plaquette: −(1/2g²)(W + W†),  W = U†_b U_l U†_r U_t ──────────
    invg2 = 1.0 / (2.0 * g^2)
    Ug = ComplexF64.(op_U_gauge(dg)); Ud = ComplexF64.(op_Udag_gauge(dg))
    for iy in 1:ny-1, ix in 1:nx-1
        A, B, C = (ix, iy), (ix+1, iy), (ix, iy+1)
        _, dA_R, dA_U = site_dims(A..., nx, ny, dg)
        _, dB_R, _    = site_dims(B..., nx, ny, dg)
        _, _, dC_U    = site_dims(C..., nx, ny, dg)
        opA = ComplexF64.(embed_R_site(Ud, dA_U) * embed_U_site(Ug, dA_R))  # I_f⊗U†_R⊗U_U
        opB = ComplexF64.(embed_U_site(Ud, dB_R))                           # I_f⊗I_R⊗U†_U
        opC = ComplexF64.(embed_R_site(Ug, dC_U))                           # I_f⊗U_R⊗I_U
        pA, pB, pC = pos[A], pos[B], pos[C]
        push!(terms, HTerm(-invg2, Dict(pA => opA,            pB => opB,            pC => opC)))
        push!(terms, HTerm(-invg2, Dict(pA => Matrix(opA'),   pB => Matrix(opB'),   pC => Matrix(opC'))))
    end

    return terms
end

"""Gauss-penalized plaquette+JW Hamiltonian MPO on the column snake."""
function build_penalized_H_plaqjw_cs(nx, ny, dg; g, t, m, gauss_g, Λ, ε::Float64=1e-12)
    dims = col_node_dims(nx, ny, dg)
    terms = vcat(lgt_terms_plaqjw_cs(nx, ny, dg; g=g, t=t, m=m),
                 gauss_penalty_terms_cs(nx, ny, dg, gauss_g, Λ))
    return _assemble_mpo(dims, terms; ε=ε)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Robust ground state: multi-start DMRG, keep the lowest in-sector result    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Cheap in-sector product start carrying magnetic flux: staggered fermions plus
a random set of source-free plaquette loops (a random height config → curl → link
E-field).  Being source-free, it stays in the staggered charge sector, and being
a product state it keeps early DMRG sweeps cheap — unlike a random full-bond MPS."""
function flux_seeded_mps(nx, ny, dg; seed::Int)
    Random.seed!(seed)
    chain, _ = column_snake(nx, ny)
    dims  = col_node_dims(nx, ny, dg)
    plaqs = plaquette_list(nx, ny)
    h = zeros(Int, nx - 1, ny - 1)
    for (px, py) in plaqs
        h[px, py] = rand(Bool) ? 1 : 0
    end
    ER, EU = height_to_E(h, nx, ny)                  # curl of the heights → link fields
    cfg = Int[]
    for (ix, iy) in chain
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        nf = isodd(ix + iy) ? 1 : 0
        eR = ix < nx ? ER[ix, iy] : 0
        eU = iy < ny ? EU[ix, iy] : 0
        push!(cfg, site_idx(nf, eR, eU, d_gR, d_gU, dg))
    end
    return product_mps(cfg, dims)
end

"""
    robust_ground_state(Hpen, dims, nx, ny, dg, gch; D, nsweeps, n_flux) → (E, ψ, viol)

DMRG from several cheap, in-sector product starts — the staggered (electric-
vacuum) state plus `n_flux` flux-seeded states — to escape the local minima the
plaquette term creates without paying for random full-bond restarts.  Returns the
LOWEST-energy state in the target Gauss sector (violation < 1e-3); if none reach
the sector, the overall lowest is returned.  Per-start energy/time is logged."""
function robust_ground_state(Hpen::CMPO, dims::Vector{Int}, nx, ny, dg,
                             gch::AbstractMatrix; D::Int, nsweeps::Int,
                             n_flux::Int=PJW_NFLUX, verbose::Bool=true)
    cands = Tuple{Float64,CMPS,Float64}[]
    function push_cand!(ψ0)
        t0 = time()
        E, ψ = dmrg_ground_state(Hpen, dims; D=D, nsweeps=nsweeps, verbose=false, ψ0=ψ0)
        push!(cands, (E, ψ, gauss_violation_cs(ψ, nx, ny, dg, gch)))
        return time() - t0
    end
    dt = Float64[]
    push!(dt, push_cand!(staggered_mps_cs(nx, ny, dg)))          # electric-vacuum start
    for k in 1:n_flux
        push!(dt, push_cand!(flux_seeded_mps(nx, ny, dg; seed=10k + 1)))  # flux-seeded starts
    end
    insec = [i for i in eachindex(cands) if cands[i][3] < 1e-3]
    pool  = isempty(insec) ? collect(eachindex(cands)) : insec
    best  = pool[argmin([cands[i][1] for i in pool])]
    if verbose
        for i in eachindex(cands)
            @printf("    start %d (%s): E_pen=%.8f  Gauss-viol=%.2e  %5.0fs%s\n",
                    i, i == 1 ? "vacuum" : "flux", cands[i][1], cands[i][3], dt[i],
                    i == best ? "   *" : "")
        end
    end
    return cands[best][1], cands[best][2], cands[best][3]
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Dense self-test: DMRG(penalized) vs exact diagonalisation of the MPO      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Validate the plaquette+JW MPO on a small lattice by comparing the penalized
DMRG ground-state energy to the lowest eigenvalue of the densely-contracted MPO."""
function selftest_plaqjw(; nx::Int=2, ny::Int=2, dg::Int=1, g::Float64=1.0,
                         t::Float64=1.0, m::Float64=0.5, Λ::Float64=20.0, D::Int=200)
    println("─── plaquette+JW MPO self-test ($(nx)×$(ny), g=$g, t=$t, m=$m) ───")
    dims = col_node_dims(nx, ny, dg)
    gch  = staggered_charges(nx, ny)
    Hpen = build_penalized_H_plaqjw_cs(nx, ny, dg; g=g, t=t, m=m, gauss_g=gch, Λ=Λ)
    ND   = prod(dims)
    @printf("  Hilbert dim = %d   penalized-H MPO max bond = %d   (DMRG D=%d)\n",
            ND, maximum(size(W, 2) for W in Hpen), D)

    Edmrg, ψ, _ = robust_ground_state(Hpen, dims, nx, ny, dg, gch;
                                      D=D, nsweeps=20, n_flux=6)
    Hbare = _assemble_mpo(dims, lgt_terms_plaqjw_cs(nx, ny, dg; g=g, t=t, m=m))
    Ebare = real(mpo_expect(Hbare, ψ)) / real(mps_overlap(ψ, ψ))
    viol  = gauss_violation_cs(ψ, nx, ny, dg, gch)
    @printf("  bare-H energy ⟨ψ|H|ψ⟩ = %.8f   Gauss violation = %.2e\n", Ebare, viol)

    if ND ≤ 4096
        Eexact = real(eigvals(Hermitian(Matrix(mpo_to_dense(Hpen))))[1])
        @printf("  DMRG E(penalized) = %.8f   dense lowest = %.8f   |Δ| = %.2e\n",
                Edmrg, Eexact, abs(Edmrg - Eexact))
        ok = abs(Edmrg - Eexact) < 1e-5 && viol < 1e-3
        println(ok ? "  PASS: DMRG reproduces the dense ground state of the plaquette+JW MPO" :
                     "  WARN: DMRG disagrees with dense diagonalisation / wrong sector")
        return ok
    else
        @printf("  DMRG E(penalized) = %.8f   (dense check skipped: Hilbert dim too large)\n", Edmrg)
        return viol < 1e-3
    end
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  One (m,g) point                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function run_one_pjw(task_id::Int)
    idx  = ((task_id - 1) % length(MG_GRID)) + 1
    p    = MG_GRID[idx]
    m, g = p.m, p.g
    nx, ny, dg = B_NX, B_NY, B_DG
    gch  = staggered_charges(nx, ny)

    @printf("=== Task %d: m=%.2f g=%.2f  (%d×%d staggered, dg=%d, PLAQUETTE+JW) ===\n",
            task_id, m, g, nx, ny, dg)
    flush(stdout)

    # ── 1. column-snake DMRG ground state of the plaquette+JW Hamiltonian ─────
    println("  [MPS] multi-start DMRG on the column snake (plaquette + JW) ...")
    dims = col_node_dims(nx, ny, dg)
    Hpen = build_penalized_H_plaqjw_cs(nx, ny, dg; g=g, t=B_THOP, m=m, gauss_g=gch, Λ=B_LAM)
    t1 = time()
    _, ψ, viol = robust_ground_state(Hpen, dims, nx, ny, dg, gch;
                                     D=PJW_D, nsweeps=PJW_NSW, n_flux=PJW_NFLUX)
    Hbare = _assemble_mpo(dims, lgt_terms_plaqjw_cs(nx, ny, dg; g=g, t=B_THOP, m=m))
    Emps  = real(mpo_expect(Hbare, ψ)) / real(mps_overlap(ψ, ψ))
    bond  = maximum(size(t, 2) for t in ψ)
    @printf("  [MPS] E = %.8f  bond=%d  Gauss-viol=%.2e  (%.1fs)  peak RSS=%.1f GB\n",
            Emps, bond, viol, time()-t1, Sys.maxrss()/2^30)
    flush(stdout)

    # standard (pre-decoupling) half-system entanglement of the full GS MPS
    S_pre = half_chain_entropy(ψ)
    @printf("  [ENT] S_pre (undecoupled full state) = %.6f nats\n", S_pre)

    # ── 2. decouple matter from gauge:  φ = 𝒰|ψ⟩ ──────────────────────────────
    println("  [DEC] Bender–Zohar SoE decoupling ..."); flush(stdout)
    t2 = time()
    Ex = build_exponents(nx, ny; dg=dg, K=B_K, bw=B_BW)
    φ  = decouple_state(Ex.Ofull, Ex.a_full, ψ; Dmax=B_DMAX)
    @printf("  [DEC] Ô bond=%d   ‖𝒰ψ‖/‖ψ‖−1=%.2e   (%.1fs)  peak RSS=%.1f GB\n",
            maximum(size(W, 2) for W in Ex.Ofull),
            abs(mps_norm(φ) / mps_norm(ψ) - 1), time()-t2, Sys.maxrss()/2^30)
    flush(stdout)

    # ── 3a. gauge: dual-plaquette half-system entanglement ────────────────────
    ρ, plaqs, srcfree_weight = plaquette_density_matrix(φ, nx, ny; dg=dg)
    Nplaq = length(plaqs)
    F = eigen(Hermitian(ρ))
    purity = real(F.values[end]); v = F.vectors[:, end]
    Aset = default_bipartition(nx, ny, plaqs)
    S_gauge = schmidt_entropy(schmidt_matrix(v, Nplaq, Aset))

    # ── 3b. matter: half-chain entanglement in the dominant source-free config ─
    chain, _ = column_snake(nx, ny)
    best_c = argmax(real.(diag(ρ))) - 1
    hbest  = config_to_height(best_c, plaqs, nx, ny)
    ERb, EUb = height_to_E(hbest, nx, ny)
    ψm = project_matter(φ, nx, ny, dg, chain, ERb, EUb)
    S_matter = half_chain_entropy(ψm)

    @printf("  [ENT] S_matter = %.6f  S_gauge = %.6f  (nats)   purity=%.4f srcfree=%.4f  peak RSS=%.1f GB\n",
            S_matter, S_gauge, purity, srcfree_weight, Sys.maxrss()/2^30)

    df = DataFrame(
        task=[task_id], m=[m], g=[g],
        E_mps=[Emps], gauss_viol=[viol], bond=[bond],
        S_pre=[S_pre], S_matter=[S_matter], S_gauge=[S_gauge],
        purity=[purity], srcfree_weight=[srcfree_weight],
        Nplaq=[Nplaq], best_config=[best_c],
    )
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    out = joinpath(results_dir, "mg_ent_pjw_task$(task_id).csv")
    CSV.write(out, df)
    println("  Saved: $out")
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    if get(ENV, "PJW_SELFTEST", "0") == "1" || (length(ARGS) ≥ 1 && ARGS[1] == "selftest")
        selftest_plaqjw(; nx=2, ny=2)
    else
        task_id = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))
        run_one_pjw(task_id)
    end
end
