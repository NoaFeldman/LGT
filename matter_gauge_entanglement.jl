#= ═══════════════════════════════════════════════════════════════════════════════
   matter_gauge_entanglement.jl

   For each (m,g) point of the gs_benchmark grid on the 3×4 U(1) LGT:

     1. Obtain the DMRG ground state of the ladder-like lattice, built DIRECTLY on
        the COLUMN-major snake (so it can feed the decoupler with no reordering —
        the row-snake state produced by mps_lgt/gs_benchmark_mps is in the WRONG
        node order for the decoupler; see decoupler-usage.md).  Energy is
        cross-checked against exact diagonalisation (finite_ed.jl) as a correctness
        gate — the GS energy is ordering-independent.

     2. Decouple matter from gauge with the Bender–Zohar SoE decoupler
        (decoupling_U_soe.jl):  φ = 𝒰|ψ⟩ = exp(−iÔ)|ψ⟩ .

     3. Read out the HALF-SYSTEM ENTANGLEMENT of BOTH sectors of the decoupled state:
          • matter — bipartite von-Neumann entropy of the matter wavefunction
            (in the dominant source-free gauge configuration) across the central
            bond of the column-snake matter MPS;
          • gauge  — half-system entanglement of the DUAL PLAQUETTE (height) model
            of the source-free gauge field (plaquette_entanglement.jl).

   SLURM array: task_id → (m,g) point.  One CSV row per task; aggregate + plot with
   matter_gauge_entanglement_collect.jl.

   Usage:
       SLURM_ARRAY_TASK_ID=1 julia --project=. matter_gauge_entanglement.jl
       sbatch run_matter_gauge_entanglement.sh
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

# plaquette_entanglement.jl → decoupling_U_soe.jl → mps_lgt.jl (+ u1 ops, greens SoE)
include(joinpath(@__DIR__, "plaquette_entanglement.jl"))
include(joinpath(@__DIR__, "finite_ed.jl"))            # ED reference (gauge sector)

using CSV, DataFrames, Printf, LinearAlgebra

# ── Same grid / parameters as gs_benchmark_mps.jl (the gs benchmark) ──────────
const MG_GRID = [
    (m=0.25, g=0.25), (m=0.25, g=0.50), (m=0.25, g=1.00),
    (m=0.25, g=2.00), (m=0.25, g=4.00),
    (m=0.50, g=0.25), (m=0.50, g=0.50), (m=0.50, g=2.00),
    (m=0.50, g=4.00),
]

const B_NX   = 3
const B_NY   = 4
const B_DG   = 1
const B_THOP = 1.0
const B_DMPS = 40        # DMRG bond dimension (gauge sector is small)
const B_NSW  = 8         # max DMRG sweeps (early-stops on convergence)
const B_LAM  = 5.0       # Gauss-law penalty (pins the staggered sector)

const B_K    = 2         # SoE decoupler terms (K=2 is enough — decoupler-usage.md)
const B_BW   = 1         # boundary shell width folded to the exact field
const B_DMAX = 64        # state bond kept during the Krylov decoupling

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Column-snake Hamiltonian (identical physics to mps_lgt's row-snake        ║
# ║  builders, re-laid on column_snake so the GS is decoupler-ready)           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""Every on-site + hopping term of H, placed on the COLUMN snake (cf. lgt_terms)."""
function lgt_terms_cs(nx, ny, dg; g, t, m)
    _, pos = column_snake(nx, ny)
    terms = HTerm[]
    for iy in 1:ny, ix in 1:nx                       # on-site (diagonal)
        H = ComplexF64.(H_onsite_site(ix, iy, nx, ny, dg; g=g, m=m))
        push!(terms, HTerm(1.0, Dict(pos[(ix, iy)] => H)))
    end
    for iy in 1:ny, ix in 1:nx-1                      # horizontal hopping
        _, d_gR_R, d_gU_R = site_dims(ix+1, iy, nx, ny, dg)
        _, _, d_gU_L      = site_dims(ix,   iy, nx, ny, dg)
        AL = ComplexF64.(kron(op_cdag(), op_U_gauge(dg), _Id(d_gU_L)))
        BR = ComplexF64.(embed_f_site(op_c(), d_gR_R, d_gU_R))
        pL, pR = pos[(ix, iy)], pos[(ix+1, iy)]
        push!(terms, HTerm(-t, Dict(pL => AL,          pR => BR)))
        push!(terms, HTerm(-t, Dict(pL => Matrix(AL'), pR => Matrix(BR'))))
    end
    for iy in 1:ny-1, ix in 1:nx                      # vertical hopping
        _, d_gR_U, d_gU_U = site_dims(ix, iy+1, nx, ny, dg)
        _, d_gR_D, _      = site_dims(ix, iy,   nx, ny, dg)
        AD = ComplexF64.(kron(op_cdag(), _Id(d_gR_D), op_U_gauge(dg)))
        BU = ComplexF64.(embed_f_site(op_c(), d_gR_U, d_gU_U))
        pD, pU = pos[(ix, iy)], pos[(ix, iy+1)]
        push!(terms, HTerm(-t, Dict(pD => AD,          pU => BU)))
        push!(terms, HTerm(-t, Dict(pD => Matrix(AD'), pU => Matrix(BU'))))
    end
    return terms
end

"""Gauss-law penalty terms Λ·Σ(G_i−g_i)² on the COLUMN snake (cf. gauss_penalty_terms)."""
function gauss_penalty_terms_cs(nx, ny, dg, g_charges::AbstractMatrix, Λ::Float64)
    _, pos = column_snake(nx, ny)
    terms = HTerm[]
    for iy in 1:ny, ix in 1:nx
        pieces = Dict{Int,Matrix{ComplexF64}}()
        add!(node, O) = (pieces[node] = haskey(pieces, node) ? pieces[node] .+ ComplexF64.(O) : ComplexF64.(O))
        if ix < nx
            _, _, d_gU = site_dims(ix, iy, nx, ny, dg);   add!(pos[(ix, iy)],   embed_R_site(op_E(dg), d_gU))
        end
        if ix > 1
            _, _, d_gU = site_dims(ix-1, iy, nx, ny, dg); add!(pos[(ix-1, iy)], -embed_R_site(op_E(dg), d_gU))
        end
        if iy < ny
            _, d_gR, _ = site_dims(ix, iy, nx, ny, dg);   add!(pos[(ix, iy)],   embed_U_site(op_E(dg), d_gR))
        end
        if iy > 1
            _, d_gR, _ = site_dims(ix, iy-1, nx, ny, dg); add!(pos[(ix, iy-1)], -embed_U_site(op_E(dg), d_gR))
        end
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg);    add!(pos[(ix, iy)], -embed_f_site(op_nf(), d_gR, d_gU))

        gi = Float64(g_charges[ix, iy])
        nodes = collect(keys(pieces))
        for a in nodes, b in nodes
            if a == b
                push!(terms, HTerm(Λ, Dict(a => pieces[a] * pieces[a])))
            else
                push!(terms, HTerm(Λ, Dict(a => pieces[a], b => pieces[b])))
            end
        end
        for a in nodes
            push!(terms, HTerm(-2 * Λ * gi, Dict(a => pieces[a])))
        end
    end
    return terms
end

"""Gauss-penalized Hamiltonian MPO on the column snake."""
function build_penalized_H_cs(nx, ny, dg; g, t, m, gauss_g, Λ, ε::Float64=1e-12)
    dims = col_node_dims(nx, ny, dg)
    terms = vcat(lgt_terms_cs(nx, ny, dg; g=g, t=t, m=m),
                 gauss_penalty_terms_cs(nx, ny, dg, gauss_g, Λ))
    return _assemble_mpo(dims, terms; ε=ε)
end

"""Staggered product state |n_f=1 on odd, 0 on even; E=0⟩ on the column snake."""
function staggered_mps_cs(nx, ny, dg)
    chain, _ = column_snake(nx, ny)
    dims = col_node_dims(nx, ny, dg)
    cfg = [begin _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
                 site_idx(isodd(ix + iy) ? 1 : 0, 0, 0, d_gR, d_gU, dg) end
           for (ix, iy) in chain]
    return product_mps(cfg, dims)
end

"""Gauss-law violation ⟨Σ(G_i−g_i)²⟩ of a column-snake MPS (≈0 ⇒ in target sector)."""
function gauss_violation_cs(ψ::CMPS, nx, ny, dg, g_charges::AbstractMatrix)
    dims = col_node_dims(nx, ny, dg)
    H = _assemble_mpo(dims, gauss_penalty_terms_cs(nx, ny, dg, g_charges, 1.0))
    shifted = real(mpo_expect(H, ψ)) / real(mps_overlap(ψ, ψ))
    return shifted + sum(abs2, g_charges)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Half-system (central-bond) entanglement entropy of an MPS                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    half_chain_entropy(ψ; cut) → S (nats)

Von-Neumann entanglement entropy across the bond between sites `cut` and `cut+1`
(default: the central bond).  Left-canonicalises the whole chain, then sweeps
right→left to `cut` so the reported singular values are the true Schmidt spectrum
of the bipartition."""
function half_chain_entropy(ψ::CMPS; cut::Int=length(ψ) ÷ 2)
    φ = deepcopy(ψ); φ[1] ./= mps_norm(φ)
    n = length(φ)
    for p in 1:n-1                                    # → fully left-canonical
        Dl, Dr, d = size(φ[p])
        F = svd(reshape(permutedims(φ[p], (1, 3, 2)), Dl * d, Dr)); r = length(F.S)
        φ[p] = permutedims(reshape(F.U, Dl, d, r), (1, 3, 2))
        _, Dr2, d2 = size(φ[p+1])
        φ[p+1] = reshape(Diagonal(F.S) * F.Vt * reshape(φ[p+1], Dr, Dr2 * d2), r, Dr2, d2)
    end
    S = Float64[]
    for p in n:-1:cut+1                               # ← right-canonical down to the cut
        Dl, Dr, d = size(φ[p])
        F = svd(reshape(permutedims(φ[p], (1, 3, 2)), Dl, d * Dr)); r = length(F.S)
        φ[p] = permutedims(reshape(F.Vt[1:r, :], r, d, Dr), (1, 3, 2))
        US = F.U * Diagonal(F.S)
        Dl0, _, d0 = size(φ[p-1])
        φ[p-1] = permutedims(reshape(reshape(permutedims(φ[p-1], (1, 3, 2)), Dl0 * d0, Dl) * US,
                                     Dl0, d0, r), (1, 3, 2))
        p == cut + 1 && (S = copy(F.S))               # Schmidt spectrum at the cut
    end
    prob = S .^ 2; prob ./= sum(prob)
    return -sum(x -> x > 0 ? x * log(x) : 0.0, prob)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  One (m,g) point                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function run_one(task_id::Int)
    idx  = ((task_id - 1) % length(MG_GRID)) + 1
    p    = MG_GRID[idx]
    m, g = p.m, p.g
    nx, ny, dg = B_NX, B_NY, B_DG
    gch  = staggered_charges(nx, ny)

    @printf("=== Task %d: m=%.2f g=%.2f  (%d×%d staggered, dg=%d) ===\n",
            task_id, m, g, nx, ny, dg)
    flush(stdout)

    # ── 1a. ED reference energy (ordering-independent correctness gate) ───────
    println("  [ED] building basis + Hamiltonian ...")
    states, key = build_basis(nx, ny, dg, Matrix{Int}(gch))
    Hed = build_hamiltonian(states, key, nx, ny, dg; g=g, t=B_THOP, m=m)
    t0 = time()
    Eed, _ = find_ground_state(Hed)
    @printf("  [ED] E0 = %.8f  (%d states, %.1fs)\n", Eed, length(states), time()-t0)

    # ── 1b. column-snake DMRG ground state (decoupler-ready node order) ───────
    println("  [MPS] DMRG on the column snake ...")
    dims = col_node_dims(nx, ny, dg)
    Hpen = build_penalized_H_cs(nx, ny, dg; g=g, t=B_THOP, m=m, gauss_g=gch, Λ=B_LAM)
    t1 = time()
    _, ψ = dmrg_ground_state(Hpen, dims; D=B_DMPS, nsweeps=B_NSW, verbose=true,
                             ψ0=staggered_mps_cs(nx, ny, dg))
    Hbare = _assemble_mpo(dims, lgt_terms_cs(nx, ny, dg; g=g, t=B_THOP, m=m))
    Emps  = real(mpo_expect(Hbare, ψ)) / real(mps_overlap(ψ, ψ))
    viol  = gauss_violation_cs(ψ, nx, ny, dg, gch)
    bond  = maximum(size(t, 2) for t in ψ)
    @printf("  [MPS] E = %.8f  bond=%d  Gauss-viol=%.2e  ΔE/|E|=%.2e  (%.1fs)\n",
            Emps, bond, viol, abs(Emps - Eed) / max(abs(Eed), 1e-12), time()-t1)

    # ── 2. decouple matter from gauge:  φ = 𝒰|ψ⟩ ──────────────────────────────
    println("  [DEC] Bender–Zohar SoE decoupling ...")
    t2 = time()
    Ex = build_exponents(nx, ny; dg=dg, K=B_K, bw=B_BW)
    φ  = decouple_state(Ex.Ofull, Ex.a_full, ψ; Dmax=B_DMAX)
    @printf("  [DEC] Ô bond=%d   ‖𝒰ψ‖/‖ψ‖−1=%.2e   (%.1fs)\n",
            maximum(size(W, 2) for W in Ex.Ofull),
            abs(mps_norm(φ) / mps_norm(ψ) - 1), time()-t2)

    # ── 3a. gauge: dual-plaquette half-system entanglement ────────────────────
    ρ, plaqs, srcfree_weight = plaquette_density_matrix(φ, nx, ny; dg=dg)
    Nplaq = length(plaqs)
    F = eigen(Hermitian(ρ))
    purity = real(F.values[end]); v = F.vectors[:, end]
    Aset = default_bipartition(nx, ny, plaqs)
    S_gauge = schmidt_entropy(schmidt_matrix(v, Nplaq, Aset))

    # ── 3b. matter: half-chain entanglement of the matter wavefunction ────────
    #        (in the dominant source-free gauge configuration)
    chain, _ = column_snake(nx, ny)
    best_c = argmax(real.(diag(ρ))) - 1                # config integer (max weight)
    hbest  = config_to_height(best_c, plaqs, nx, ny)
    ERb, EUb = height_to_E(hbest, nx, ny)
    ψm = project_matter(φ, nx, ny, dg, chain, ERb, EUb)
    S_matter = half_chain_entropy(ψm)

    @printf("  [ENT] S_matter = %.6f  S_gauge = %.6f  (nats)   purity=%.4f srcfree=%.4f\n",
            S_matter, S_gauge, purity, srcfree_weight)

    df = DataFrame(
        task=[task_id], m=[m], g=[g],
        E_ed=[Eed], E_mps=[Emps], dE_rel=[abs(Emps - Eed) / max(abs(Eed), 1e-12)],
        gauss_viol=[viol], bond=[bond],
        S_matter=[S_matter], S_gauge=[S_gauge],
        purity=[purity], srcfree_weight=[srcfree_weight],
        Nplaq=[Nplaq], best_config=[best_c],
    )
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    out = joinpath(results_dir, "mg_ent_task$(task_id).csv")
    CSV.write(out, df)
    println("  Saved: $out")
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    task_id = parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))
    run_one(task_id)
end
