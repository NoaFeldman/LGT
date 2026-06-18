#= ═══════════════════════════════════════════════════════════════════════════════
   decoupled_matter_benchmark.jl

   Wire the Green's-function Coulomb MPO (efficient_greens_mpo.jl) into a
   MATTER-ONLY DMRG solver and compare its ground state to the full gauged exact
   diagonalisation (finite_ed.jl), over the same 9 (m,g) points.

   ── Decoupled-matter model (gauge field integrated out) ──────────────────────
   Dropping the explicit gauge links and replacing the electric energy by the
   gauge-mediated Coulomb interaction gives a matter-only Hamiltonian on the
   Nx×Ny snake of dim-2 fermion sites:

       H_dec = − t Σ_⟨n,m⟩ (c†_n c_m + h.c.)            (bare hopping, no JW — ED convention)
              + m Σ_n (−1)^{x+y} n_f(n)                 (staggered mass)
              + (g²/2) Σ_{n,m} G(n,m) Q_n Q_m           (Coulomb; G = inverse 2D Laplacian)
              + Λ_N (Σ_n n_f − N_target)²               (pins the filling sector)

   with Q_n = n_f(n) − bg(n), bg staggered so the staggered vacuum is charge
   neutral (Q=0 ⇒ zero Coulomb), matching the gauged staggered sector.

   ── What this tests (and what it does NOT) ───────────────────────────────────
   The decoupling is APPROXIMATE for this model: (1) the gauged theory truncates
   E at dg=1 while G is untruncated; (2) 2D transverse gauge modes are dropped;
   (3) bare hopping replaces the fully transformed kinetic term.  So we compare
   matter OBSERVABLES (⟨n_f⟩, sublattice densities, CDW) — not absolute energies —
   to quantify how well the Coulomb-only decoupling reproduces the matter sector.

   Usage:
       julia --project=. decoupled_matter_benchmark.jl
       sbatch run_decoupled_matter.sh
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "mps_lgt.jl"))          # generic HTerm/MPO/DMRG/MPS toolkit
include(joinpath(@__DIR__, "lgt_greens_soe.jl"))   # generate_greens_function (exact G)
include(joinpath(@__DIR__, "finite_ed.jl"))        # gauged ED reference

using CSV, DataFrames, Printf, Statistics

const D_GRID = [
    (m=0.25, g=0.25), (m=0.25, g=0.50), (m=0.25, g=1.00),
    (m=0.25, g=2.00), (m=0.25, g=4.00),
    (m=0.50, g=0.25), (m=0.50, g=0.50), (m=0.50, g=2.00),
    (m=0.50, g=4.00),
]
const D_NX, D_NY, D_DG = 3, 4, 1
const D_THOP = 1.0
const D_BOND = 60
const D_NSW  = 12
const D_LAMN = 20.0      # filling-sector penalty

# ── matter operators (dim 2) ─────────────────────────────────────────────────
const _nf   = ComplexF64[0 0; 0 1]
const _c    = ComplexF64[0 1; 0 0]
const _cdag = ComplexF64[0 0; 1 0]
qop(bg::Float64) = _nf - bg * ComplexF64[1 0; 0 1]

bg_stag(ix, iy) = isodd(ix + iy) ? 1.0 : 0.0      # staggered background (vacuum: odd filled)

# ── build the decoupled matter-only Hamiltonian MPO ──────────────────────────
function decoupled_H(nx, ny; g, t, m, Gtens, N_target, ΛN)
    chain, pos = snake_nodes(nx, ny)
    N = nx * ny; dims = fill(2, N)
    terms = HTerm[]

    # staggered mass (1-site)
    for (ix, iy) in chain
        stag = iseven(ix + iy) ? 1.0 : -1.0
        push!(terms, HTerm(m * stag, Dict(pos[(ix, iy)] => copy(_nf))))
    end

    # bare hopping (2-site, no gauge link, no JW — matches ED/PEPS convention)
    for iy in 1:ny, ix in 1:nx-1
        a, b = pos[(ix, iy)], pos[(ix+1, iy)]
        push!(terms, HTerm(-t, Dict(a => copy(_cdag), b => copy(_c))))
        push!(terms, HTerm(-t, Dict(a => copy(_c),    b => copy(_cdag))))
    end
    for iy in 1:ny-1, ix in 1:nx
        a, b = pos[(ix, iy)], pos[(ix, iy+1)]
        push!(terms, HTerm(-t, Dict(a => copy(_cdag), b => copy(_c))))
        push!(terms, HTerm(-t, Dict(a => copy(_c),    b => copy(_cdag))))
    end

    # Coulomb interaction (g²/2) Σ G(n,m) Q_n Q_m
    for a in 1:N
        (ix, iy) = chain[a]; Qa = qop(bg_stag(ix, iy))
        push!(terms, HTerm(0.5 * g^2 * Gtens[ix, iy, ix, iy], Dict(a => Qa * Qa)))
        for b in a+1:N
            (jx, jy) = chain[b]; Qb = qop(bg_stag(jx, jy))
            push!(terms, HTerm(g^2 * Gtens[ix, iy, jx, jy], Dict(a => Qa, b => Qb)))
        end
    end

    # filling-sector penalty  ΛN (Σ n_f − N_target)²  = ΛN[Σ n_a n_b − 2N_t Σ n_a + N_t²]
    for a in 1:N
        push!(terms, HTerm(ΛN, Dict(a => _nf * _nf)))            # n_a² = n_a
        push!(terms, HTerm(-2 * ΛN * N_target, Dict(a => copy(_nf))))
        for b in a+1:N
            push!(terms, HTerm(2 * ΛN, Dict(a => copy(_nf), b => copy(_nf))))
        end
    end

    return _assemble_mpo(dims, terms), dims, chain, pos
end

"""Matter staggered product state (odd sites filled): index 2=occupied, 1=empty."""
function matter_staggered_mps(nx, ny)
    chain, _ = snake_nodes(nx, ny)
    cfg = [isodd(ix + iy) ? 2 : 1 for (ix, iy) in chain]
    return product_mps(cfg, fill(2, nx * ny))
end

function measure_matter(ψ, nx, ny, chain, pos)
    nf = zeros(Float64, nx, ny)
    for (ix, iy) in chain
        nf[ix, iy] = mps_local_expect(ψ, pos[(ix, iy)], _nf)
    end
    nf_even = mean(nf[ix, iy] for ix in 1:nx, iy in 1:ny if iseven(ix + iy))
    nf_odd  = mean(nf[ix, iy] for ix in 1:nx, iy in 1:ny if isodd(ix + iy))
    return (nf_mean=mean(nf), nf_even=nf_even, nf_odd=nf_odd, nf_grid=nf)
end

# ── one (m,g) point: gauged ED vs decoupled-matter DMRG ──────────────────────
function run_point(m, g; Gtens)
    nx, ny, dg = D_NX, D_NY, D_DG
    gch = [isodd(ix + iy) ? -1 : 0 for ix in 1:nx, iy in 1:ny]
    N_target = sum(gch .== -1)                       # = 6 (staggered half-filling)

    # gauged ED (matter observables of the full theory)
    states, key = build_basis(nx, ny, dg, Matrix{Int}(gch))
    Hed = build_hamiltonian(states, key, nx, ny, dg; g=g, t=D_THOP, m=m)
    _, ψed = find_ground_state(Hed)
    oe = measure_ED(ψed, states, nx, ny, dg, 0.0)

    # decoupled matter-only DMRG
    H, dims, chain, pos = decoupled_H(nx, ny; g=g, t=D_THOP, m=m,
                                      Gtens=Gtens, N_target=N_target, ΛN=D_LAMN)
    _, ψ = dmrg_ground_state(H, dims; D=D_BOND, nsweeps=D_NSW, verbose=false,
                             ψ0=matter_staggered_mps(nx, ny))
    od = measure_matter(ψ, nx, ny, chain, pos)
    nf_tot = sum(od.nf_grid)

    return (m=m, g=g,
            nf_mean_ed=oe.nf_mean, nf_mean_dec=od.nf_mean,
            nf_even_ed=oe.nf_even, nf_even_dec=od.nf_even,
            nf_odd_ed=oe.nf_odd,   nf_odd_dec=od.nf_odd,
            cdw_ed=oe.nf_even - oe.nf_odd, cdw_dec=od.nf_even - od.nf_odd,
            dnf=maximum(abs, [oe.nf_mean - od.nf_mean,
                              oe.nf_even - od.nf_even, oe.nf_odd - od.nf_odd]),
            nf_tot_dec=nf_tot)
end

function main()
    nx, ny = D_NX, D_NY
    println("Building exact Green's function ($(nx)×$(ny)) ...")
    _, Gtens = generate_greens_function(nx, ny)

    rows = NamedTuple[]
    @printf("\n  %-5s %-5s | %-8s %-8s | %-8s %-8s | %-8s %-8s | %-9s\n",
            "m", "g", "nf_ED", "nf_dec", "cdw_ED", "cdw_dec", "nfe_ED", "nfe_dec", "max|Δnf|")
    println("  " * "─"^92)
    for p in D_GRID
        r = run_point(p.m, p.g; Gtens=Gtens)
        push!(rows, r)
        @printf("  %-5.2f %-5.2f | %-8.4f %-8.4f | %-8.4f %-8.4f | %-8.4f %-8.4f | %-9.2e\n",
                r.m, r.g, r.nf_mean_ed, r.nf_mean_dec, r.cdw_ed, r.cdw_dec,
                r.nf_even_ed, r.nf_even_dec, r.dnf)
        flush(stdout)
    end

    df = DataFrame(rows)
    results_dir = joinpath(@__DIR__, "results"); mkpath(results_dir)
    out = joinpath(results_dir, "decoupled_matter_vs_ed.csv")
    CSV.write(out, df)
    @printf("\n  worst max|Δn_f| (decoupling discrepancy) = %.2e\n", maximum(df.dnf))
    println("  Saved: $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
