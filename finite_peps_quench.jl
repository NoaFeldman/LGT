# =============================================================================
#  finite_peps_quench.jl
#
#  Quench benchmark for the finite PEPS code on a 3×4 lattice
#  (2×3 plaquettes, open boundary conditions).
#
#  Three quench protocols (mirroring benchmark_quenches.jl but adapted to the
#  site-local finite PEPS with boundary-aware physical spaces):
#
#    A. String Breaking:  vacuum + electric-flux string → real-time evolution
#    B. Mass Quench:      ITE to GS at m_init, then evolve with m_final
#    C. Coupling Quench:  ITE to GS at g_init, then evolve with g_final
#
#  Observables:
#    • ⟨n_f(ix,iy)⟩     site-resolved fermion density
#    • ⟨E²(ix,iy)⟩      site-resolved electric-field energy (right + up links)
#    • S_vN(bond)        bond entanglement entropy from Vidal weights
#    • total energy      via simple-update two-site expectation values
#
#  Output per quench:
#    finite_peps_quench_{A|B|C}_data.csv     time-series aggregate scalars
#    finite_peps_quench_{A|B|C}_panels.png   4-panel observable figure
#    finite_peps_quench_{A|B|C}_final.txt    site-resolved snapshot at t_final
#
#  Usage (local):
#    julia --project=. finite_peps_quench.jl
#
#  Usage (SLURM):
#    sbatch run_finite_peps_quench.sh
# =============================================================================

# GR headless backend — must be set before Plots is loaded
ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "finite_peps_ground_state.jl"))

using CSV
using DataFrames
using Statistics
using Printf

default(
    fontfamily = "Computer Modern",
    linewidth  = 2,
    framestyle = :box,
    grid       = true,
    legend     = :best,
    size       = (800, 500),
    dpi        = 200,
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Benchmark parameters                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

const QNX      = 3        # lattice columns  → QNX-1 = 2 horizontal plaquettes
const QNY      = 4        # lattice rows     → QNY-1 = 3 vertical plaquettes
const QDG      = 1        # gauge truncation |e| ≤ 1
const QD_BOND  = 2        # initial virtual bond dimension
const QD_MAX   = 6        # max bond dimension during quench

const QG_COUP  = 1.0      # default coupling g
const QT_HOP   = 1.0      # default hopping t
const QM_MASS  = 2.0      # default / initial staggered mass

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Real-time Trotter step                                                 ║
# ║                                                                          ║
# ║  Second-order Suzuki-Trotter with complex step τ = i·dt:               ║
# ║    exp(-i·dt·H) ≈ exp(-i·dt/2·H_h) exp(-i·dt·H_v) exp(-i·dt/2·H_h)   ║
# ║                                                                          ║
# ║  Reuses update_bond_h! / update_bond_v! from finite_peps_ground_state.jl║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    realtime_trotter_step!(peps, nx, ny, dg; g, t_hop, m, dt, D_trunc)

One second-order Trotter step for real-time (Hamiltonian) evolution.
Gate: exp(-i·dt·H) via h-half → v-full → h-half sweep.
"""
function realtime_trotter_step!(peps::FinitePEPS, nx::Int, ny::Int, dg::Int;
                                 g::Float64, t_hop::Float64, m::Float64,
                                 dt::Float64, D_trunc::Int)
    τ_h = im * dt / 2
    τ_v = im * dt

    # Horizontal half-step  (left → right)
    for iy in 1:ny, ix in 1:nx-1
        Hh   = H_merged_h_site(ix, iy, nx, ny, dg; g=g, t=t_hop, m=m)
        gate = exp(-τ_h .* Hh)
        update_bond_h!(peps, ix, iy, gate, D_trunc)
    end

    # Vertical full-step  (bottom → top)
    for iy in 1:ny-1, ix in 1:nx
        Hv   = H_merged_v_site(ix, iy, nx, ny, dg; g=g, t=t_hop, m=m)
        gate = exp(-τ_v .* Hv)
        update_bond_v!(peps, ix, iy, gate, D_trunc)
    end

    # Horizontal half-step  (right → left, for symmetry)
    for iy in 1:ny, ix in nx-1:-1:1
        Hh   = H_merged_h_site(ix, iy, nx, ny, dg; g=g, t=t_hop, m=m)
        gate = exp(-τ_h .* Hh)
        update_bond_h!(peps, ix, iy, gate, D_trunc)
    end

    return nothing
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Observables                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
    bond_entropy(λ) → S_vN

Von Neumann entanglement entropy from a vector of Vidal weights:
    S = -Σ_i p_i log p_i,   p_i = λ_i² / Σ_j λ_j².
"""
function bond_entropy(λ::Vector{Float64})
    p = λ .^ 2
    Z = sum(p)
    Z < 1e-30 && return 0.0
    p ./= Z
    return -sum(x > 1e-30 ? x * log(x) : 0.0 for x in p)
end

"""
    measure_nf_grid(peps, nx, ny, dg) → Matrix{Float64}

(nx × ny) matrix of ⟨n_f(ix,iy)⟩.
"""
function measure_nf_grid(peps::FinitePEPS, nx::Int, ny::Int, dg::Int)
    grid = zeros(nx, ny)
    for iy in 1:ny, ix in 1:nx
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        grid[ix, iy] = expect_site(peps, ix, iy, embed_f_site(op_nf(), d_gR, d_gU))
    end
    return grid
end

"""
    measure_E2_grid(peps, nx, ny, dg) → Matrix{Float64}

(nx × ny) matrix of ⟨E_R² + E_U²⟩ at each site (only existing links counted).
"""
function measure_E2_grid(peps::FinitePEPS, nx::Int, ny::Int, dg::Int)
    grid = zeros(nx, ny)
    for iy in 1:ny, ix in 1:nx
        d_phys, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        O = zeros(ComplexF64, d_phys, d_phys)
        if ix < nx
            O .+= embed_R_site(op_E2(dg), d_gU)
        end
        if iy < ny
            O .+= embed_U_site(op_E2(dg), d_gR)
        end
        grid[ix, iy] = expect_site(peps, ix, iy, O)
    end
    return grid
end

"""
    measure_all_finite(peps, nx, ny, dg, t) → NamedTuple

Aggregate observables at time t.
"""
function measure_all_finite(peps::FinitePEPS, nx::Int, ny::Int, dg::Int, t::Float64)
    nf = measure_nf_grid(peps, nx, ny, dg)
    E2 = measure_E2_grid(peps, nx, ny, dg)

    even_vals = [nf[ix, iy] for iy in 1:ny for ix in 1:nx if  iseven(ix + iy)]
    odd_vals  = [nf[ix, iy] for iy in 1:ny for ix in 1:nx if  isodd(ix + iy)]

    S_h_vals = [bond_entropy(peps.λh[ix, iy]) for iy in 1:ny  for ix in 1:nx-1]
    S_v_vals = [bond_entropy(peps.λv[ix, iy]) for iy in 1:ny-1 for ix in 1:nx]

    S_h_mean = isempty(S_h_vals) ? 0.0 : mean(S_h_vals)
    S_v_mean = isempty(S_v_vals) ? 0.0 : mean(S_v_vals)

    return (
        t        = t,
        nf_mean  = mean(nf),
        nf_even  = isempty(even_vals) ? 0.0 : mean(even_vals),
        nf_odd   = isempty(odd_vals)  ? 0.0 : mean(odd_vals),
        E2_mean  = mean(E2),
        S_h_mean = S_h_mean,
        S_v_mean = S_v_mean,
        S_mean   = (S_h_mean + S_v_mean) / 2,
        nf_grid  = copy(nf),
        E2_grid  = copy(E2),
    )
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Helpers                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

"""
Run imaginary-time evolution to prepare a finite PEPS ground state.
"""
function ite_ground_state(nx::Int, ny::Int, dg::Int, D_bond::Int, D_max::Int;
                           g::Float64, t_hop::Float64, m::Float64,
                           τ_ite::Float64, n_ite::Int, verbose::Bool=true)
    peps = init_finite_peps(nx, ny, dg, D_bond)

    for step in 1:n_ite
        trotter_step!(peps, nx, ny, dg; g=g, t_hop=t_hop, m=m,
                      τ=τ_ite, D_trunc=D_max)

        if verbose && (step % 100 == 0 || step == n_ite)
            nf_val = mean_nf(peps, nx, ny, dg)
            D_now  = maximum(length(peps.λh[ix, iy])
                             for iy in 1:ny for ix in 1:nx-1; init=1)
            @printf("    ITE step %4d / %d:  ⟨n_f⟩ = %.5f   D = %d\n",
                    step, n_ite, nf_val, D_now)
        end
    end

    return peps
end

"""
Apply a local operator O (d_phys × d_phys matrix) to the physical index of
tensor (ix, iy) in-place.
"""
function apply_site_op!(peps::FinitePEPS, ix::Int, iy::Int, O::AbstractMatrix)
    T   = peps.tensors[ix, iy]
    dp  = size(T, 1)
    new = reshape(O * reshape(T, dp, :), size(T))
    peps.tensors[ix, iy] = new
    return nothing
end

"""
Serialise time-series data to a DataFrame (scalar columns only).
"""
function data_to_df(data::Vector)
    DataFrame(
        t        = [d.t        for d in data],
        nf_mean  = [d.nf_mean  for d in data],
        nf_even  = [d.nf_even  for d in data],
        nf_odd   = [d.nf_odd   for d in data],
        E2_mean  = [d.E2_mean  for d in data],
        S_h_mean = [d.S_h_mean for d in data],
        S_v_mean = [d.S_v_mean for d in data],
        S_mean   = [d.S_mean   for d in data],
    )
end

"""
Print a site-resolved grid to stdout (and optionally write to file).
"""
function print_grid(grid::Matrix, label::String; io=stdout)
    nx, ny = size(grid)
    println(io, "  $label  (rows = iy top→bottom, cols = ix left→right):")
    for iy in ny:-1:1
        print(io, "    iy=$iy:  ")
        for ix in 1:nx
            @printf(io, "%8.4f ", grid[ix, iy])
        end
        println(io)
    end
end

"""
Save a site-resolved snapshot to a text file.
"""
function save_final_snapshot(peps::FinitePEPS, nx::Int, ny::Int, dg::Int,
                              t_final::Float64, filename::String)
    nf = measure_nf_grid(peps, nx, ny, dg)
    E2 = measure_E2_grid(peps, nx, ny, dg)
    open(filename, "w") do io
        println(io, "Finite PEPS quench — final snapshot at t = $t_final")
        println(io, "Lattice: $(nx)×$(ny)  dg=$(dg)\n")
        print_grid(nf, "⟨n_f⟩"; io=io)
        println(io)
        print_grid(E2, "⟨E²⟩"; io=io)
    end
    println("  Saved: $filename")
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Plotting                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function plot_quench(df::DataFrame; title::String, prefix::String)
    t = df.t

    p1 = plot(t, df.nf_even, label="⟨n_f⟩ even sites", xlabel="t", ylabel="⟨n_f⟩",
              title="(a) Fermion Density")
    plot!(p1, t, df.nf_odd,  label="⟨n_f⟩ odd sites",  linestyle=:dash)
    plot!(p1, t, df.nf_mean, label="lattice mean",      color=:black, lw=2.5)
    hline!(p1, [0.5]; color=:gray, linestyle=:dot, label="0.5")

    p2 = plot(t, df.E2_mean, label="⟨E²⟩", xlabel="t", ylabel="⟨E²⟩",
              title="(b) Electric Energy", color=:red, lw=2.5)

    p3 = plot(t, df.S_h_mean, label="S_h (horizontal bonds)", xlabel="t", ylabel="S_vN",
              title="(c) Bond Entanglement Entropy")
    plot!(p3, t, df.S_v_mean, label="S_v (vertical bonds)", linestyle=:dash)
    plot!(p3, t, df.S_mean,   label="mean",                 color=:black, lw=2.5)

    cdw = df.nf_even .- df.nf_odd
    p4 = plot(t, cdw, label="CDW = ⟨n_f⟩_e − ⟨n_f⟩_o",
              xlabel="t", ylabel="CDW order", title="(d) CDW Order Parameter",
              color=:purple, lw=2.5)
    hline!(p4, [0.0]; color=:gray, linestyle=:dot, label="")

    fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800), plot_title=title)
    savefig(fig, "$(prefix)_panels.png")
    println("  Saved: $(prefix)_panels.png")
    return fig
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Evolution loop (shared by all three quenches)                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function evolve!(peps::FinitePEPS, nx::Int, ny::Int, dg::Int;
                 g::Float64, t_hop::Float64, m::Float64,
                 dt::Float64, n_steps::Int, D_trunc::Int,
                 measure_every::Int = 10)

    data = NamedTuple[]
    push!(data, measure_all_finite(peps, nx, ny, dg, 0.0))

    println("  step  |    t        ⟨n_f⟩    ⟨n_f⟩_e  ⟨n_f⟩_o   ⟨E²⟩     S_mean   D")
    println("  ──────┼───────────────────────────────────────────────────────────────────")
    @printf("  %4d  | %6.3f    %.4f   %.4f   %.4f   %.4f   %.4f   %d\n",
            0, 0.0, data[1].nf_mean, data[1].nf_even, data[1].nf_odd,
            data[1].E2_mean, data[1].S_mean, QD_BOND)

    for step in 1:n_steps
        realtime_trotter_step!(peps, nx, ny, dg;
                               g=g, t_hop=t_hop, m=m, dt=dt, D_trunc=D_trunc)
        t_now = Float64(step) * dt
        obs   = measure_all_finite(peps, nx, ny, dg, t_now)
        push!(data, obs)

        if step % measure_every == 0 || step == n_steps
            D_now = maximum(length(peps.λh[ix, iy])
                            for iy in 1:ny for ix in 1:nx-1; init=1)
            @printf("  %4d  | %6.3f    %.4f   %.4f   %.4f   %.4f   %.4f   %d\n",
                    step, t_now, obs.nf_mean, obs.nf_even, obs.nf_odd,
                    obs.E2_mean, obs.S_mean, D_now)
        end
    end

    return data
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Quench A: String Breaking                                              ║
# ║                                                                          ║
# ║  Initial state: vacuum (n_f = 0 everywhere, all gauge links e = 0).    ║
# ║  Perturbation: create a fermion at site (2, 2) and raise the gauge link ║
# ║  on the (1,2)–(2,2) bond by applying U_R at site (1,2).               ║
# ║  This seeds a meson: charged site connected to the other by a flux     ║
# ║  string. Evolve in real time and observe string dynamics.              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function run_quench_A(;
        nx = QNX, ny = QNY, dg = QDG, D_bond = QD_BOND, D_max = QD_MAX,
        g::Float64 = QG_COUP, t_hop::Float64 = QT_HOP, m::Float64 = QM_MASS,
        dt::Float64 = 0.02, n_steps::Int = 200,
        label::String = "finite_peps_quench_A")

    println("=" ^ 70)
    println("  QUENCH A: String Breaking")
    println("  Lattice: $(nx)×$(ny)  dg=$dg  D_bond=$D_bond  D_max=$D_max")
    println("  g=$g  t=$t_hop  m=$m  |  dt=$dt  steps=$n_steps")
    println("=" ^ 70)

    # ── Vacuum: all sites n_f = 0, all gauge links e = 0 ─────────────────────
    peps = init_finite_peps(nx, ny, dg, D_bond)
    for iy in 1:ny, ix in 1:nx
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        d_phys = LGT_d_f * d_gR * d_gU
        T = peps.tensors[ix, iy]
        arr = zeros(ComplexF64, d_phys, size(T, 2), size(T, 3), size(T, 4), size(T, 5))
        arr[site_idx(0, 0, 0, d_gR, d_gU, dg), 1, 1, 1, 1] = 1.0
        peps.tensors[ix, iy] = arr
    end

    # ── Create flux string on bond (1, iy_str) – (2, iy_str) ─────────────────
    iy_str   = div(ny, 2)           # row 2 on 4-row lattice
    ix_left  = 1                    # left end of string
    ix_right = 2                    # right end (fermion site)

    _, d_gR_l, d_gU_l = site_dims(ix_left, iy_str, nx, ny, dg)
    Ur_site = embed_R_site(op_U_gauge(dg), d_gU_l)    # U on right-gauge DoF of (1, iy_str)
    apply_site_op!(peps, ix_left, iy_str, Ur_site)

    _, d_gR_r, d_gU_r = site_dims(ix_right, iy_str, nx, ny, dg)
    cdag_site = embed_f_site(op_cdag(), d_gR_r, d_gU_r)
    apply_site_op!(peps, ix_right, iy_str, cdag_site)

    println("\n  String created: U_R on ($(ix_left),$(iy_str)),  ψ† on ($(ix_right),$(iy_str))")
    println("\n  Evolving...")

    data = evolve!(peps, nx, ny, dg;
                   g=g, t_hop=t_hop, m=m, dt=dt, n_steps=n_steps, D_trunc=D_max)

    df = data_to_df(data)
    CSV.write("$(label)_data.csv", df)
    println("\n  Saved: $(label)_data.csv")

    plot_quench(df;
                title="Quench A: String Breaking  ($(nx)×$(ny), m=$m)",
                prefix=label)
    save_final_snapshot(peps, nx, ny, dg, data[end].t, "$(label)_final.txt")

    return data
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Quench B: Mass Quench  (m_init → m_final)                              ║
# ║                                                                          ║
# ║  Prepare GS at large m_init (CDW / charge-density wave).               ║
# ║  Quench to m_final → observe plasma oscillations between matter         ║
# ║  and gauge sectors.                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function run_quench_B(;
        nx = QNX, ny = QNY, dg = QDG, D_bond = QD_BOND, D_max = QD_MAX,
        g::Float64 = QG_COUP, t_hop::Float64 = QT_HOP,
        m_init::Float64 = 5.0, m_final::Float64 = 0.1,
        τ_ite::Float64 = 0.05, n_ite::Int = 300,
        dt::Float64 = 0.02, n_steps::Int = 200,
        label::String = "finite_peps_quench_B")

    println("=" ^ 70)
    println("  QUENCH B: Mass Quench  m = $m_init → $m_final")
    println("  Lattice: $(nx)×$(ny)  dg=$dg  D_bond=$D_bond  D_max=$D_max")
    println("  g=$g  t=$t_hop  |  ITE: τ=$τ_ite n=$n_ite  |  RT: dt=$dt steps=$n_steps")
    println("=" ^ 70)

    println("\n  Preparing ground state at m = $m_init  ($n_ite ITE steps)...")
    peps = ite_ground_state(nx, ny, dg, D_bond, D_max;
                             g=g, t_hop=t_hop, m=m_init,
                             τ_ite=τ_ite, n_ite=n_ite)

    obs0 = measure_all_finite(peps, nx, ny, dg, 0.0)
    @printf("\n  GS at m=$m_init:  ⟨n_f⟩=%.4f  ⟨n_f⟩_e=%.4f  ⟨n_f⟩_o=%.4f  ⟨E²⟩=%.4f\n",
            obs0.nf_mean, obs0.nf_even, obs0.nf_odd, obs0.E2_mean)

    println("\n  Evolving with m = $m_final...")
    data = evolve!(peps, nx, ny, dg;
                   g=g, t_hop=t_hop, m=m_final, dt=dt, n_steps=n_steps, D_trunc=D_max)

    df = data_to_df(data)
    CSV.write("$(label)_data.csv", df)
    println("\n  Saved: $(label)_data.csv")

    plot_quench(df;
                title="Quench B: Mass Quench m=$m_init → $m_final  ($(nx)×$(ny))",
                prefix=label)
    save_final_snapshot(peps, nx, ny, dg, data[end].t, "$(label)_final.txt")

    return data
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Quench C: Coupling Quench  (g_init → g_final)                          ║
# ║                                                                          ║
# ║  Prepare GS at strong coupling g_init (electric energy dominates,      ║
# ║  gauge field nearly frozen near e = 0).  Quench to weak coupling       ║
# ║  g_final → plaquette term becomes important, magnetic flux loops        ║
# ║  develop, entanglement grows.                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function run_quench_C(;
        nx = QNX, ny = QNY, dg = QDG, D_bond = QD_BOND, D_max = QD_MAX,
        t_hop::Float64 = QT_HOP, m::Float64 = QM_MASS,
        g_init::Float64 = 2.0, g_final::Float64 = 0.5,
        τ_ite::Float64 = 0.05, n_ite::Int = 300,
        dt::Float64 = 0.02, n_steps::Int = 200,
        label::String = "finite_peps_quench_C")

    println("=" ^ 70)
    println("  QUENCH C: Coupling Quench  g = $g_init → $g_final")
    println("  Lattice: $(nx)×$(ny)  dg=$dg  D_bond=$D_bond  D_max=$D_max")
    println("  t=$t_hop  m=$m  |  ITE: τ=$τ_ite n=$n_ite  |  RT: dt=$dt steps=$n_steps")
    println("=" ^ 70)

    println("\n  Preparing ground state at g = $g_init  ($n_ite ITE steps)...")
    peps = ite_ground_state(nx, ny, dg, D_bond, D_max;
                             g=g_init, t_hop=t_hop, m=m,
                             τ_ite=τ_ite, n_ite=n_ite)

    obs0 = measure_all_finite(peps, nx, ny, dg, 0.0)
    @printf("\n  GS at g=$g_init:  ⟨n_f⟩=%.4f  ⟨E²⟩=%.4f  S_mean=%.4f\n",
            obs0.nf_mean, obs0.E2_mean, obs0.S_mean)

    println("\n  Evolving with g = $g_final...")
    data = evolve!(peps, nx, ny, dg;
                   g=g_final, t_hop=t_hop, m=m, dt=dt, n_steps=n_steps, D_trunc=D_max)

    df = data_to_df(data)
    CSV.write("$(label)_data.csv", df)
    println("\n  Saved: $(label)_data.csv")

    plot_quench(df;
                title="Quench C: Coupling Quench g=$g_init → $g_final  ($(nx)×$(ny))",
                prefix=label)
    save_final_snapshot(peps, nx, ny, dg, data[end].t, "$(label)_final.txt")

    return data
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Summary comparison figure                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function plot_summary(data_A, data_B, data_C; prefix="finite_peps_quench_summary")
    df_A = data_to_df(data_A)
    df_B = data_to_df(data_B)
    df_C = data_to_df(data_C)

    p1 = plot(df_A.t, df_A.nf_mean, label="String Breaking", xlabel="t", ylabel="⟨n_f⟩",
              title="(a) Mean Fermion Density")
    plot!(p1, df_B.t, df_B.nf_mean, label="Mass Quench",    linestyle=:dash)
    plot!(p1, df_C.t, df_C.nf_mean, label="Coupling Quench",linestyle=:dot)

    p2 = plot(df_A.t, df_A.E2_mean, label="String Breaking", xlabel="t", ylabel="⟨E²⟩",
              title="(b) Electric Energy")
    plot!(p2, df_B.t, df_B.E2_mean, label="Mass Quench",    linestyle=:dash)
    plot!(p2, df_C.t, df_C.E2_mean, label="Coupling Quench",linestyle=:dot)

    p3 = plot(df_A.t, df_A.S_mean, label="String Breaking", xlabel="t", ylabel="S_vN",
              title="(c) Entanglement Entropy")
    plot!(p3, df_B.t, df_B.S_mean, label="Mass Quench",    linestyle=:dash)
    plot!(p3, df_C.t, df_C.S_mean, label="Coupling Quench",linestyle=:dot)

    p4 = plot(df_A.t, df_A.nf_even .- df_A.nf_odd, label="String Breaking",
              xlabel="t", ylabel="⟨n_f⟩_e − ⟨n_f⟩_o", title="(d) CDW Order Parameter")
    plot!(p4, df_B.t, df_B.nf_even .- df_B.nf_odd, label="Mass Quench",    linestyle=:dash)
    plot!(p4, df_C.t, df_C.nf_even .- df_C.nf_odd, label="Coupling Quench",linestyle=:dot)
    hline!(p4, [0.0]; color=:gray, linestyle=:dot, label="")

    fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800),
               plot_title="Finite PEPS Quenches — $(QNX)×$(QNY) lattice ($(QNX-1)×$(QNY-1) plaquettes)")
    savefig(fig, "$(prefix).png")
    println("  Saved: $(prefix).png")
    return fig
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function main()
    println("\n" * "=" ^ 70)
    println("  Finite PEPS Quench Benchmark")
    println("  Lattice: $(QNX)×$(QNY)  →  $(QNX-1)×$(QNY-1) plaquettes")
    println("  dg=$(QDG)  (link dim $(gauge_dim(QDG)))  D_bond=$(QD_BOND)  D_max=$(QD_MAX)")
    println("=" ^ 70 * "\n")

    data_A = run_quench_A()
    println()

    data_B = run_quench_B()
    println()

    data_C = run_quench_C()
    println()

    println("  Generating summary comparison figure...")
    plot_summary(data_A, data_B, data_C)

    println("\n" * "=" ^ 70)
    println("  DONE.  Output files:")
    for tag in ["A", "B", "C"]
        println("    finite_peps_quench_$(tag)_data.csv")
        println("    finite_peps_quench_$(tag)_panels.png")
        println("    finite_peps_quench_$(tag)_final.txt")
    end
    println("    finite_peps_quench_summary.png")
    println("=" ^ 70)
end

main()
