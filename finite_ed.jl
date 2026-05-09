# =============================================================================
#  finite_ed.jl
#
#  Exact diagonalization of the U(1) LGT on a finite nx×ny lattice for
#  comparison with finite PEPS quench results.
#
#  ── Hamiltonian ──────────────────────────────────────────────────────────────
#  H = H_onsite + H_hop_h + H_hop_v   (no magnetic plaquette term)
#  This matches finite_peps_quench.jl exactly: the PEPS Trotter sweep applies
#  only nearest-neighbour hopping + electric-field / mass on-site terms.
#
#  ── Gauge-invariant subspace ─────────────────────────────────────────────────
#  Gauss law at site (ix,iy):
#    G(ix,iy) = E_R(ix,iy) - E_R(ix-1,iy) + E_U(ix,iy) - E_U(ix,iy-1) - n_f(ix,iy)
#  (boundary link values are 0).  The Hamiltonian commutes with all G(ix,iy),
#  so the gauge-charge sector {g(ix,iy)} is conserved.
#
#  The sector for each quench is set by the PEPS initial state:
#    Quench A: n_f=0 everywhere then ψ† at (2,2) and U_R at (1,2) applied
#              → g(1,2)=+1, g(2,2)=-2, all others 0, N_f=1
#    Quench B / C: staggered initial state (n_f=0 even, n_f=1 odd, E=0)
#              → g_even=0, g_odd=-1, N_f=6
#
#  ── Parallelisation ──────────────────────────────────────────────────────────
#  Three quenches run as a SLURM array (task ids 1/2/3 → A/B/C).
#  Within each task, Julia threading is used by KrylovKit for matrix-vector
#  products.
#
#  ── Usage ────────────────────────────────────────────────────────────────────
#  Local:
#    julia --project=. --threads=8 finite_ed.jl 1   # Quench A
#    julia --project=. --threads=8 finite_ed.jl 2   # Quench B
#    julia --project=. --threads=8 finite_ed.jl 3   # Quench C
#  SLURM:
#    sbatch run_finite_ed.sh
# =============================================================================

ENV["GKSwstype"] = "nul"   # headless Plots

# ── Dependencies (loaded before any include so guards work) ──────────────────
const _LGT_HAMILTONIAN_LOADED = true
include(joinpath(@__DIR__, "u1_lgt_hamiltonian.jl"))

using SparseArrays
using KrylovKit
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using Printf
using Plots

default(fontfamily="Computer Modern", linewidth=2, framestyle=:box,
        grid=true, legend=:best, size=(800,500), dpi=200)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Parameters (mirror finite_peps_quench.jl)                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

const ED_NX      = 3
const ED_NY      = 4
const ED_DG      = 1
const ED_G_COUP  = 1.0
const ED_T_HOP   = 1.0
const ED_M_MASS  = 2.0

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  State encoding                                                         ║
# ║                                                                          ║
# ║  Each basis state: n_f[nx,ny]∈{0,1},  e_R[nx-1,ny]∈{-dg..dg},         ║
# ║                    e_U[nx,ny-1]∈{-dg..dg}                              ║
# ║                                                                          ║
# ║  Packed into Int64 for O(1) dictionary lookup.                          ║
# ║  Layout (column-major site indexing, iy outer):                         ║
# ║    bits  0..N_sites-1          : n_f  (1 bit each, 0 or 1)             ║
# ║    bits  N_sites..N_sites+2*Nh : e_R  (2 bits each, 0=−1,1=0,2=+1)    ║
# ║    bits  ..+2*Nv               : e_U  (2 bits each, 0=−1,1=0,2=+1)    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function _encode(n_f::Matrix{Int8}, e_R::Matrix{Int8}, e_U::Matrix{Int8},
                  nx::Int, ny::Int)
    key = Int64(0)
    offset = 0
    for iy in 1:ny, ix in 1:nx
        key |= Int64(n_f[ix,iy]) << offset
        offset += 1
    end
    for iy in 1:ny, ix in 1:nx-1
        key |= Int64(e_R[ix,iy] + 1) << offset
        offset += 2
    end
    for iy in 1:ny-1, ix in 1:nx
        key |= Int64(e_U[ix,iy] + 1) << offset
        offset += 2
    end
    return key
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Gauss-law basis enumeration                                            ║
# ║                                                                          ║
# ║  For a given gauge-charge sector g[nx,ny] (G_i eigenvalue at each site) ║
# ║  and gauge truncation dg, enumerate all product states                  ║
# ║    |n_f, {e_R}, {e_U}⟩                                                 ║
# ║  satisfying G(ix,iy)|ψ⟩ = g(ix,iy)|ψ⟩ at every site.                  ║
# ║                                                                          ║
# ║  Algorithm (row-by-row Gauss propagation):                              ║
# ║  1. Require N_f = -sum(g) and iterate over all C(N_sites, N_f) fermion  ║
# ║     configurations.                                                      ║
# ║  2. For each fermion config, compute required E divergences              ║
# ║     q(ix,iy) = n_f(ix,iy) + g(ix,iy).                                  ║
# ║  3. Row column sums S[iy] = cumsum_iy(sum_ix q[:,iy]) determine the     ║
# ║     required sum of E_U in each row.  Enumerate valid E_U assignments.  ║
# ║  4. For each E_U assignment, derive E_R by horizontal propagation and   ║
# ║     check the values stay within [-dg, dg].                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    build_basis(nx, ny, dg, g_charges) → (states, key_to_idx)

Enumerate all gauge-invariant basis states for the sector defined by
`g_charges[ix,iy]`.  Returns a vector of named tuples `(n_f, e_R, e_U)`
and a `Dict{Int64,Int}` mapping encoded state → 1-based index.
"""
function build_basis(nx::Int, ny::Int, dg::Int, g_charges::Matrix{Int})
    N_f_target = -sum(g_charges)
    @assert 0 ≤ N_f_target ≤ nx*ny "Inconsistent sector: N_f = $N_f_target"
    N_sites = nx * ny

    states   = NamedTuple{(:n_f,:e_R,:e_U), Tuple{Matrix{Int8},Matrix{Int8},Matrix{Int8}}}[]
    key_dict = Dict{Int64,Int}()

    n_f  = zeros(Int8, nx, ny)
    e_R  = zeros(Int8, nx-1, ny)
    e_U  = zeros(Int8, nx, ny-1)

    # --- enumerate all fermion bit-masks with exactly N_f_target ones --------
    for mask in 0:(2^N_sites - 1)
        count_ones(mask) == N_f_target || continue

        for iy in 1:ny, ix in 1:nx
            n_f[ix,iy] = Int8((mask >> ((iy-1)*nx + (ix-1))) & 1)
        end

        # Required electric-field divergences at each site
        q = n_f .+ Int8.(g_charges)   # q[ix,iy] = n_f + g

        # Row-sum of q → required cumulative E_U column sums
        R = [sum(q[:,iy]) for iy in 1:ny]       # R[iy] = Σ_ix q(ix,iy)
        S = cumsum(R)                             # S[iy] = Σ_{j≤iy} R[j]
        # Consistency: S[ny] should be 0 (guaranteed by N_f_target = -sum(g))
        S[ny] == 0 || continue

        # --- enumerate E_U row by row ----------------------------------------
        # For row iy (1..ny-1): e_U[1..nx, iy] must sum to S[iy],
        #   each value in -dg:dg.  Free variables: e_U[1..nx-1,iy];
        #   e_U[nx,iy] = S[iy] - sum(e_U[1..nx-1,iy]).
        _enumerate_eU!(states, key_dict, n_f, e_R, e_U, q, S, nx, ny, dg)
    end

    return states, key_dict
end

function _enumerate_eU!(states, key_dict,
                         n_f::Matrix{Int8}, e_R::Matrix{Int8}, e_U::Matrix{Int8},
                         q::Matrix{Int8}, S::Vector{Int},
                         nx::Int, ny::Int, dg::Int)
    # Recursively enumerate e_U row by row (iy = 1..ny-1).
    _eU_row!(states, key_dict, n_f, e_R, e_U, q, S, nx, ny, dg, 1)
end

function _eU_row!(states, key_dict,
                   n_f, e_R, e_U, q, S, nx, ny, dg, iy::Int)
    if iy == ny
        # All E_U rows fixed — derive E_R and check.
        _derive_and_check_eR!(states, key_dict, n_f, e_R, e_U, q, nx, ny, dg)
        return
    end
    # Enumerate e_U[1..nx-1,iy]; e_U[nx,iy] is determined.
    _eU_inner!(states, key_dict, n_f, e_R, e_U, q, S, nx, ny, dg, iy, 1, 0)
end

function _eU_inner!(states, key_dict,
                     n_f, e_R, e_U, q, S, nx, ny, dg,
                     iy::Int, ix::Int, partial_sum::Int)
    if ix == nx
        # Determine last element
        last_val = S[iy] - partial_sum
        if -dg ≤ last_val ≤ dg
            e_U[nx, iy] = Int8(last_val)
            _eU_row!(states, key_dict, n_f, e_R, e_U, q, S, nx, ny, dg, iy+1)
        end
        return
    end
    for v in -dg:dg
        e_U[ix, iy] = Int8(v)
        _eU_inner!(states, key_dict, n_f, e_R, e_U, q, S, nx, ny, dg,
                   iy, ix+1, partial_sum+v)
    end
end

function _derive_and_check_eR!(states, key_dict,
                                  n_f, e_R, e_U, q, nx, ny, dg)
    # For each row iy: E_R(ix,iy) = Σ_{j=1}^{ix} d(j,iy)
    # where d(ix,iy) = q(ix,iy) - E_U(ix,iy) + E_U(ix,iy-1)
    for iy in 1:ny
        acc = 0
        for ix in 1:nx
            eU_now  = (iy < ny) ? Int(e_U[ix,iy])   : 0
            eU_prev = (iy > 1)  ? Int(e_U[ix,iy-1]) : 0
            d = Int(q[ix,iy]) - eU_now + eU_prev
            acc += d
            if ix < nx
                if -dg ≤ acc ≤ dg
                    e_R[ix,iy] = Int8(acc)
                else
                    return   # invalid: abort this configuration
                end
            else
                # Rightmost column: acc must be 0 (Gauss at (nx,iy) closed)
                # [guaranteed by row-sum construction, but double-check]
                acc == 0 || return
            end
        end
    end
    # Valid state — store
    key = _encode(n_f, e_R, e_U, size(n_f,1), size(n_f,2))
    if !haskey(key_dict, key)
        push!(states, (n_f=copy(n_f), e_R=copy(e_R), e_U=copy(e_U)))
        key_dict[key] = length(states)
    end
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Sparse Hamiltonian construction                                        ║
# ║                                                                          ║
# ║  H = H_onsite_total + H_hop_h_total + H_hop_v_total                    ║
# ║                                                                          ║
# ║  On-site (diagonal):                                                    ║
# ║    H_os = Σ_{ix,iy} [(g²/2)(e_R²+e_U²) + m·(-1)^{ix+iy}·n_f]         ║
# ║                                                                          ║
# ║  Hopping (off-diagonal):                                                ║
# ║    Forward:  n_f_L: 0→1, n_f_R: 1→0, e_R: e→e+1  coeff = -t          ║
# ║    Backward: n_f_L: 1→0, n_f_R: 0→1, e_R: e→e-1  coeff = -t          ║
# ║  (same structure for vertical bonds, with e_U instead of e_R)          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
Build the sparse Hamiltonian in the gauge-invariant basis.
Returns a `SparseMatrixCSC{ComplexF64}`.
"""
function build_hamiltonian(states, key_dict,
                            nx::Int, ny::Int, dg::Int;
                            g::Float64, t::Float64, m::Float64)
    N = length(states)
    Is = Int[]; Js = Int[]; Vs = ComplexF64[]

    n_f  = zeros(Int8, nx, ny)
    e_R  = zeros(Int8, nx-1, ny)
    e_U  = zeros(Int8, nx, ny-1)

    for (idx, st) in enumerate(states)

        # ── Diagonal: on-site energy ─────────────────────────────────────
        E_diag = 0.0
        for iy in 1:ny, ix in 1:nx
            if ix < nx
                E_diag += (g^2 / 2) * Float64(st.e_R[ix,iy])^2
            end
            if iy < ny
                E_diag += (g^2 / 2) * Float64(st.e_U[ix,iy])^2
            end
            stag = iseven(ix + iy) ? 1 : -1
            E_diag += m * stag * Float64(st.n_f[ix,iy])
        end
        push!(Is, idx); push!(Js, idx); push!(Vs, E_diag)

        # ── Off-diagonal: horizontal hopping ─────────────────────────────
        for iy in 1:ny, ix in 1:nx-1
            e = st.e_R[ix,iy]
            nL = st.n_f[ix,   iy]
            nR = st.n_f[ix+1, iy]

            # Forward hop: L gains fermion, R loses fermion, link raises
            if nL == 0 && nR == 1 && e < dg
                copyto!(n_f, st.n_f); copyto!(e_R, st.e_R); copyto!(e_U, st.e_U)
                n_f[ix,   iy] = Int8(1)
                n_f[ix+1, iy] = Int8(0)
                e_R[ix,   iy] = Int8(e + 1)
                k = _encode(n_f, e_R, e_U, nx, ny)
                if haskey(key_dict, k)
                    jdx = key_dict[k]
                    push!(Is, jdx); push!(Js, idx); push!(Vs, ComplexF64(-t))
                    push!(Is, idx); push!(Js, jdx); push!(Vs, ComplexF64(-t))
                end
            end
        end

        # ── Off-diagonal: vertical hopping ───────────────────────────────
        for iy in 1:ny-1, ix in 1:nx
            e = st.e_U[ix,iy]
            nD = st.n_f[ix, iy  ]
            nU = st.n_f[ix, iy+1]

            # Forward hop: D gains fermion, U loses fermion, link raises
            if nD == 0 && nU == 1 && e < dg
                copyto!(n_f, st.n_f); copyto!(e_R, st.e_R); copyto!(e_U, st.e_U)
                n_f[ix, iy  ] = Int8(1)
                n_f[ix, iy+1] = Int8(0)
                e_U[ix, iy  ] = Int8(e + 1)
                k = _encode(n_f, e_R, e_U, nx, ny)
                if haskey(key_dict, k)
                    jdx = key_dict[k]
                    push!(Is, jdx); push!(Js, idx); push!(Vs, ComplexF64(-t))
                    push!(Is, idx); push!(Js, jdx); push!(Vs, ComplexF64(-t))
                end
            end
        end
    end

    return sparse(Is, Js, Vs, N, N)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Ground-state finder (imaginary-time KrylovKit eigsolve)               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    find_ground_state(H) → (E0, ψ0)

Lanczos ground-state eigensolver via KrylovKit.  Returns energy and state vector.
"""
function find_ground_state(H::SparseMatrixCSC{ComplexF64})
    N = size(H, 1)
    v0 = normalize!(randn(ComplexF64, N))
    vals, vecs, info = eigsolve(H, v0, 1, :SR;
                                 ishermitian=true, krylovdim=min(50, N),
                                 maxiter=300, tol=1e-10)
    @assert info.converged ≥ 1 "Ground-state eigsolve did not converge"
    return real(vals[1]), vecs[1]
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ITE ground-state preparation (for quenches B and C)                   ║
# ║                                                                          ║
# ║  Build H at initial parameters, find GS, return state vector.          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function ite_ground_state_ED(states, key_dict, nx, ny, dg;
                               g, t, m)
    println("    Building H at g=$g  t=$t  m=$m ...")
    H = build_hamiltonian(states, key_dict, nx, ny, dg; g=g, t=t, m=m)
    println("    Sparse H: $(size(H,1))×$(size(H,2))  nnz=$(nnz(H))")
    println("    Solving for ground state ...")
    E0, ψ0 = find_ground_state(H)
    @printf("    E0 = %.8f\n", E0)
    return E0, ψ0
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Observables                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    measure_ED(ψ, states, nx, ny, dg, t_now) → NamedTuple

Compute site-resolved ⟨n_f⟩ and ⟨E²⟩ from the exact state vector.
"""
function measure_ED(ψ::Vector{ComplexF64}, states, nx::Int, ny::Int, dg::Int,
                    t_now::Float64)
    nf_grid  = zeros(Float64, nx, ny)
    E2_grid  = zeros(Float64, nx, ny)
    ρ = abs2.(ψ)   # probabilities (diagonal of density matrix)

    for (idx, st) in enumerate(states)
        p = ρ[idx]
        for iy in 1:ny, ix in 1:nx
            nf_grid[ix,iy]  += p * st.n_f[ix,iy]
            e2 = 0.0
            if ix < nx;  e2 += Float64(st.e_R[ix,iy])^2; end
            if iy < ny;  e2 += Float64(st.e_U[ix,iy])^2; end
            E2_grid[ix,iy]  += p * e2
        end
    end

    nf_mean  = mean(nf_grid)
    nf_even  = mean(nf_grid[ix,iy] for ix in 1:nx, iy in 1:ny if iseven(ix+iy))
    nf_odd   = mean(nf_grid[ix,iy] for ix in 1:nx, iy in 1:ny if isodd(ix+iy))
    E2_mean  = mean(E2_grid)

    return (t=t_now, nf_mean=nf_mean, nf_even=nf_even, nf_odd=nf_odd,
            E2_mean=E2_mean, nf_grid=nf_grid, E2_grid=E2_grid)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Real-time evolution loop                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    evolve_ED!(ψ, H, states, nx, ny, dg; dt, n_steps, measure_every)

Evolve `ψ` under H in real time using KrylovKit.exponentiate.
Returns vector of measurement named tuples.
"""
function evolve_ED!(ψ::Vector{ComplexF64}, H::SparseMatrixCSC{ComplexF64},
                    states, nx::Int, ny::Int, dg::Int;
                    dt::Float64, n_steps::Int, measure_every::Int=10)
    data = NamedTuple[]
    push!(data, measure_ED(ψ, states, nx, ny, dg, 0.0))

    println("  step  |    t        ⟨n_f⟩    ⟨n_f⟩_e  ⟨n_f⟩_o   ⟨E²⟩")
    println("  ──────┼─────────────────────────────────────────────────────")
    @printf("  %4d  | %6.3f    %.4f   %.4f   %.4f   %.4f\n",
            0, 0.0, data[1].nf_mean, data[1].nf_even, data[1].nf_odd, data[1].E2_mean)

    for step in 1:n_steps
        ψ, info = exponentiate(H, -im*dt, ψ;
                                krylovdim=min(30, length(ψ)),
                                maxiter=100, tol=1e-10, ishermitian=true)
        ψ ./= norm(ψ)   # renormalize for numerical stability
        t_now = Float64(step) * dt
        obs = measure_ED(ψ, states, nx, ny, dg, t_now)
        push!(data, obs)

        if step % measure_every == 0 || step == n_steps
            @printf("  %4d  | %6.3f    %.4f   %.4f   %.4f   %.4f\n",
                    step, t_now, obs.nf_mean, obs.nf_even, obs.nf_odd, obs.E2_mean)
        end
    end
    return data
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Output helpers                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function data_to_df_ED(data)
    DataFrame(
        t        = [d.t        for d in data],
        nf_mean  = [d.nf_mean  for d in data],
        nf_even  = [d.nf_even  for d in data],
        nf_odd   = [d.nf_odd   for d in data],
        E2_mean  = [d.E2_mean  for d in data],
    )
end

"""Find PEPS CSV: check results_dir first, then @__DIR__."""
function _find_peps_csv(name::String, results_dir::String)
    p1 = joinpath(results_dir, name)
    isfile(p1) && return p1
    p2 = joinpath(@__DIR__, name)
    isfile(p2) && return p2
    return p1   # return preferred path even if missing (warning printed later)
end

function plot_ed_vs_peps(df_ed::DataFrame, df_peps_path::String;
                          title::String, prefix::String)
    fig = plot(layout=(2,2), size=(1200,800), plot_title=title)

    plot!(fig[1], df_ed.t, df_ed.nf_mean,  label="ED",   xlabel="t", ylabel="⟨n_f⟩")
    plot!(fig[2], df_ed.t, df_ed.E2_mean,  label="ED",   xlabel="t", ylabel="⟨E²⟩")
    plot!(fig[3], df_ed.t, df_ed.nf_even,  label="ED (even)", xlabel="t", ylabel="⟨n_f⟩ sublattice")
    plot!(fig[3], df_ed.t, df_ed.nf_odd,   label="ED (odd)",  xlabel="t", linestyle=:dash)
    plot!(fig[4], df_ed.t, df_ed.nf_even .- df_ed.nf_odd, label="ED",
          xlabel="t", ylabel="⟨n_f⟩_e − ⟨n_f⟩_o")

    if isfile(df_peps_path)
        df_p = CSV.read(df_peps_path, DataFrame)
        plot!(fig[1], df_p.t, df_p.nf_mean,  label="PEPS",  linestyle=:dash)
        plot!(fig[2], df_p.t, df_p.E2_mean,  label="PEPS",  linestyle=:dash)
        plot!(fig[3], df_p.t, df_p.nf_even,  label="PEPS (even)", linestyle=:dot)
        plot!(fig[3], df_p.t, df_p.nf_odd,   label="PEPS (odd)",  linestyle=:dot)
        plot!(fig[4], df_p.t, df_p.nf_even .- df_p.nf_odd, label="PEPS", linestyle=:dash)
    else
        @warn "PEPS data not found at $df_peps_path — skipping overlay"
    end

    savefig(fig, "$(prefix)_ed_vs_peps.png")
    println("  Saved: $(prefix)_ed_vs_peps.png")
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Quench A: String Breaking                                              ║
# ║                                                                          ║
# ║  Initial state: vacuum (n_f=0, E=0) → apply U_R at (1,iy_str) and      ║
# ║  ψ† at (2,iy_str).  Evolve in real time.                               ║
# ║                                                                          ║
# ║  Gauge sector: g(1,iy_str)=+1, g(2,iy_str)=-2, all others 0.          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function run_quench_A_ED(;
        nx=ED_NX, ny=ED_NY, dg=ED_DG,
        g::Float64=ED_G_COUP, t::Float64=ED_T_HOP, m::Float64=ED_M_MASS,
        dt::Float64=0.02, n_steps::Int=50,
        results_dir::String="results",
        label::String="finite_ed_quench_A")

    println("=" ^ 70)
    println("  ED QUENCH A: String Breaking")
    println("  Lattice: $(nx)×$(ny)  dg=$dg  g=$g  t=$t  m=$m")
    println("  dt=$dt  steps=$n_steps")
    println("=" ^ 70)

    iy_str   = div(ny, 2)
    ix_left  = 1
    ix_right = 2

    # Gauge sector
    g_charges = zeros(Int, nx, ny)
    g_charges[ix_left,  iy_str] = +1
    g_charges[ix_right, iy_str] = -2

    println("\n  Building gauge-invariant basis  (sector: g($(ix_left),$(iy_str))=+1, g($(ix_right),$(iy_str))=-2)...")
    states, key_dict = build_basis(nx, ny, dg, g_charges)
    println("  Basis size: $(length(states)) states")

    # Build Hamiltonian
    println("\n  Building sparse Hamiltonian...")
    H = build_hamiltonian(states, key_dict, nx, ny, dg; g=g, t=t, m=m)
    println("  H: $(size(H,1))×$(size(H,2))  nnz=$(nnz(H))")

    # Construct initial state: vacuum + U_R(1,iy_str) + ψ†(2,iy_str)
    # In the gauged basis this is the unique state n_f[2,iy_str]=1, e_R[1,iy_str]=+1, all others 0
    n_f0 = zeros(Int8, nx, ny)
    n_f0[ix_right, iy_str] = Int8(1)
    e_R0 = zeros(Int8, nx-1, ny)
    e_R0[ix_left, iy_str] = Int8(1)
    e_U0 = zeros(Int8, nx, ny-1)
    k0 = _encode(n_f0, e_R0, e_U0, nx, ny)
    @assert haskey(key_dict, k0) "Initial state not found in basis!"
    ψ0 = zeros(ComplexF64, length(states))
    ψ0[key_dict[k0]] = 1.0

    println("\n  Evolving...")
    data = evolve_ED!(ψ0, H, states, nx, ny, dg; dt=dt, n_steps=n_steps)

    mkpath(results_dir)
    df = data_to_df_ED(data)
    CSV.write(joinpath(results_dir, "$(label)_data.csv"), df)
    println("\n  Saved: $(results_dir)/$(label)_data.csv")

    peps_csv = _find_peps_csv("finite_peps_quench_A_data.csv", results_dir)
    plot_ed_vs_peps(df, peps_csv;
                    title="Quench A: String Breaking — ED vs PEPS  ($(nx)×$(ny))",
                    prefix=joinpath(results_dir, label))
    return data
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Quench B: Mass Quench                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function run_quench_B_ED(;
        nx=ED_NX, ny=ED_NY, dg=ED_DG,
        g::Float64=ED_G_COUP, t::Float64=ED_T_HOP,
        m_init::Float64=5.0, m_final::Float64=0.1,
        dt::Float64=0.02, n_steps::Int=50,
        results_dir::String="results",
        label::String="finite_ed_quench_B")

    println("=" ^ 70)
    println("  ED QUENCH B: Mass Quench  m = $m_init → $m_final")
    println("  Lattice: $(nx)×$(ny)  dg=$dg  g=$g  t=$t")
    println("  dt=$dt  steps=$n_steps")
    println("=" ^ 70)

    # Gauge sector: staggered (g_odd = -1, g_even = 0)
    g_charges = zeros(Int, nx, ny)
    for iy in 1:ny, ix in 1:nx
        isodd(ix + iy) && (g_charges[ix,iy] = -1)
    end

    println("\n  Building gauge-invariant basis  (staggered sector, N_f=$(sum(g_charges .== -1)))...")
    states, key_dict = build_basis(nx, ny, dg, g_charges)
    println("  Basis size: $(length(states)) states")

    println("\n  Ground state at m = $m_init ...")
    E0, ψ0 = ite_ground_state_ED(states, key_dict, nx, ny, dg;
                                   g=g, t=t, m=m_init)

    println("\n  Building H at m = $m_final ...")
    H_final = build_hamiltonian(states, key_dict, nx, ny, dg; g=g, t=t, m=m_final)
    println("  H_final: $(size(H_final,1))×$(size(H_final,2))  nnz=$(nnz(H_final))")

    println("\n  Evolving...")
    ψ = copy(ψ0)
    data = evolve_ED!(ψ, H_final, states, nx, ny, dg; dt=dt, n_steps=n_steps)

    mkpath(results_dir)
    df = data_to_df_ED(data)
    CSV.write(joinpath(results_dir, "$(label)_data.csv"), df)
    println("\n  Saved: $(results_dir)/$(label)_data.csv")

    peps_csv = _find_peps_csv("finite_peps_quench_B_data.csv", results_dir)
    plot_ed_vs_peps(df, peps_csv;
                    title="Quench B: Mass Quench m=$m_init→$m_final — ED vs PEPS  ($(nx)×$(ny))",
                    prefix=joinpath(results_dir, label))
    return data
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Quench C: Coupling Quench                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function run_quench_C_ED(;
        nx=ED_NX, ny=ED_NY, dg=ED_DG,
        t::Float64=ED_T_HOP, m::Float64=ED_M_MASS,
        g_init::Float64=2.0, g_final::Float64=0.5,
        dt::Float64=0.02, n_steps::Int=50,
        results_dir::String="results",
        label::String="finite_ed_quench_C")

    println("=" ^ 70)
    println("  ED QUENCH C: Coupling Quench  g = $g_init → $g_final")
    println("  Lattice: $(nx)×$(ny)  dg=$dg  t=$t  m=$m")
    println("  dt=$dt  steps=$n_steps")
    println("=" ^ 70)

    # Gauge sector: staggered
    g_charges = zeros(Int, nx, ny)
    for iy in 1:ny, ix in 1:nx
        isodd(ix + iy) && (g_charges[ix,iy] = -1)
    end

    println("\n  Building gauge-invariant basis  (staggered sector)...")
    states, key_dict = build_basis(nx, ny, dg, g_charges)
    println("  Basis size: $(length(states)) states")

    println("\n  Ground state at g = $g_init ...")
    E0, ψ0 = ite_ground_state_ED(states, key_dict, nx, ny, dg;
                                   g=g_init, t=t, m=m)

    println("\n  Building H at g = $g_final ...")
    H_final = build_hamiltonian(states, key_dict, nx, ny, dg; g=g_final, t=t, m=m)

    println("\n  Evolving...")
    ψ = copy(ψ0)
    data = evolve_ED!(ψ, H_final, states, nx, ny, dg; dt=dt, n_steps=n_steps)

    mkpath(results_dir)
    df = data_to_df_ED(data)
    CSV.write(joinpath(results_dir, "$(label)_data.csv"), df)
    println("\n  Saved: $(results_dir)/$(label)_data.csv")

    peps_csv = _find_peps_csv("finite_peps_quench_C_data.csv", results_dir)
    plot_ed_vs_peps(df, peps_csv;
                    title="Quench C: Coupling Quench g=$g_init→$g_final — ED vs PEPS  ($(nx)×$(ny))",
                    prefix=joinpath(results_dir, label))
    return data
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function main()
    quench_id = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 0

    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)

    if quench_id == 1 || quench_id == 0
        println("\n" * "=" ^ 70)
        println("  Running ED Quench A (String Breaking)")
        println("=" ^ 70)
        run_quench_A_ED(results_dir=results_dir)
    end

    if quench_id == 2 || quench_id == 0
        println("\n" * "=" ^ 70)
        println("  Running ED Quench B (Mass Quench)")
        println("=" ^ 70)
        run_quench_B_ED(results_dir=results_dir)
    end

    if quench_id == 3 || quench_id == 0
        println("\n" * "=" ^ 70)
        println("  Running ED Quench C (Coupling Quench)")
        println("=" ^ 70)
        run_quench_C_ED(results_dir=results_dir)
    end

    if quench_id == 0
        println("\n  All ED quenches complete.  Results in: $results_dir")
    end
end

main()
