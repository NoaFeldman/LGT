# =============================================================================
#  Finite PEPS Quench Benchmark — 3×4 lattice (2×3 plaquettes)
#
#  Mirrors the panel/summary plot style of benchmark_ipeps.jl.
#
#  Three quench protocols:
#    A. String Breaking / Confinement → Deconfinement  (g²=10 → 0.5)
#    B. Mass Quench                                    (m=5  → 0.1)
#    C. Coupling Quench                                (g²=4 → 0.5)
#
#  Observables:
#    • ⟨n_f⟩  per even/odd sublattice
#    • ⟨E_R⟩, ⟨E_U⟩  averaged per even/odd sublattice
#    • ⟨□⟩     mean-field plaquette approximation, averaged over 6 plaquettes
#    • S_bond  mean entanglement entropy from bond weights
# =============================================================================

include(joinpath(@__DIR__, "finite_peps_ground_state.jl"))

using Pkg
for pkg in ["Plots", "CSV", "DataFrames"]
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg)
end
using Plots, CSV, DataFrames, Printf

# Suppress display on headless runs
ENV["GKSwstype"] = "100"

default(fontfamily = "sans-serif", linewidth = 2, framestyle = :box,
        grid = true, legend = :best, size = (900, 600), dpi = 200)

# ─── Lattice parameters ────────────────────────────────────────────────────
const BNX    = 3     # columns  (2×3 plaquettes → 3×4 nodes)
const BNY    = 4     # rows
const BDG    = 1     # gauge truncation
const BD_ITE = 2     # starting bond dim for ITE
const BD_MAX = 4     # max bond dim

# ─── Evolution parameters ─────────────────────────────────────────────────
const TAU_ITE  = 0.05    # imaginary-time step
const N_ITE    = 200     # ITE steps for ground-state prep
const DT_RT    = 0.10    # real-time step
const N_RT     = 60      # real-time steps (T = 6.0)
const MEAS_INT = 3       # measure every N_RT steps

# =============================================================================
#  Complex-valued single-site expectation value  (needed for U, U† operators)
# =============================================================================
function expect_site_c(peps::FinitePEPS, ix::Int, iy::Int,
                        O_mat::AbstractMatrix)
    T  = peps.tensors[ix, iy]
    Tw = copy(T)
    dp = size(T, 1)

    λ_l = (ix > 1)       ? peps.λh[ix-1, iy] : ones(1)
    λ_r = (ix < peps.nx) ? peps.λh[ix,   iy] : ones(1)
    λ_u = (iy < peps.ny) ? peps.λv[ix,   iy] : ones(1)
    λ_d = (iy > 1)       ? peps.λv[ix, iy-1] : ones(1)

    for p in 1:dp, l in axes(Tw,2), r in axes(Tw,3),
                    u in axes(Tw,4), d in axes(Tw,5)
        Tw[p, l, r, u, d] *= λ_l[l] * λ_r[r] * λ_u[u] * λ_d[d]
    end

    T_flat = reshape(Tw, dp, :)
    num = tr(T_flat' * O_mat * T_flat)
    den = real(tr(T_flat' * T_flat))
    return den > 1e-15 ? num / den : zero(ComplexF64)
end

# =============================================================================
#  Observable measurement
# =============================================================================
function finite_peps_measure(peps::FinitePEPS, nx::Int, ny::Int, dg::Int)
    nf_even = 0.0;  nf_odd  = 0.0
    Er_even = 0.0;  Er_odd  = 0.0
    Eu_even = 0.0;  Eu_odd  = 0.0
    n_even  = 0;    n_odd   = 0
    n_Er_even = 0;  n_Er_odd = 0
    n_Eu_even = 0;  n_Eu_odd = 0

    for iy in 1:ny, ix in 1:nx
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        even = iseven(ix + iy)

        # Fermion density
        nf_val = expect_site(peps, ix, iy, embed_f_site(op_nf(), d_gR, d_gU))
        if even; nf_even += nf_val; n_even += 1
        else;    nf_odd  += nf_val; n_odd  += 1; end

        # Electric field on right link (only if link exists)
        if ix < nx
            Er_val = expect_site(peps, ix, iy, embed_R_site(op_E(dg), d_gU))
            if even; Er_even += Er_val; n_Er_even += 1
            else;    Er_odd  += Er_val; n_Er_odd  += 1; end
        end

        # Electric field on up link (only if link exists)
        if iy < ny
            Eu_val = expect_site(peps, ix, iy, embed_U_site(op_E(dg), d_gR))
            if even; Eu_even += Eu_val; n_Eu_even += 1
            else;    Eu_odd  += Eu_val; n_Eu_odd  += 1; end
        end
    end

    nf_even /= max(1, n_even);  nf_odd /= max(1, n_odd)
    Er_even /= max(1, n_Er_even); Er_odd /= max(1, n_Er_odd)
    Eu_even /= max(1, n_Eu_even); Eu_odd /= max(1, n_Eu_odd)
    E_flux_avg = (Er_even + Er_odd + Eu_even + Eu_odd) / 4

    # ── Mean-field plaquette (2×3 = 6 plaquettes) ─────────────────────────
    # Plaquette at lower-left (ix,iy) involves nodes (ix,iy), (ix+1,iy), (ix,iy+1)
    # W = I_f⊗U†_R⊗U_U  on (ix,iy)  ×  I_f⊗I_R⊗U†_U  on (ix+1,iy)  ×  I_f⊗U_R⊗I_U  on (ix,iy+1)
    gd = gauge_dim(dg)
    Ug = op_U_gauge(dg)
    Ud = op_Udag_gauge(dg)

    plaq_sum = 0.0
    n_plaq   = 0
    for iy in 1:ny-1, ix in 1:nx-1
        _, d_gR_A, d_gU_A = site_dims(ix,   iy,   nx, ny, dg)
        _, d_gR_B, d_gU_B = site_dims(ix+1, iy,   nx, ny, dg)
        _, d_gR_C, d_gU_C = site_dims(ix,   iy+1, nx, ny, dg)

        O_A = kron(_Id(LGT_d_f), Ud, Ug)                       # (ix,iy):   U†_R ⊗ U_U
        O_B = kron(_Id(LGT_d_f), _Id(d_gR_B), Ud)              # (ix+1,iy): U†_U
        O_C = kron(_Id(LGT_d_f), Ug,           _Id(d_gU_C))    # (ix,iy+1): U_R

        W_A = expect_site_c(peps, ix,   iy,   O_A)
        W_B = expect_site_c(peps, ix+1, iy,   O_B)
        W_C = expect_site_c(peps, ix,   iy+1, O_C)

        plaq_sum += 2 * real(W_A * W_B * W_C)
        n_plaq   += 1
    end
    plaq = plaq_sum / max(1, n_plaq)

    # ── Mean entanglement entropy from bond weights ────────────────────────
    S_sum = 0.0;  n_S = 0
    for iy in 1:ny, ix in 1:nx-1
        λ = peps.λh[ix, iy]
        p = λ.^2;  p ./= max(1e-30, sum(p))
        S_sum += -sum(x -> x > 1e-30 ? x * log(x) : 0.0, p)
        n_S += 1
    end
    for iy in 1:ny-1, ix in 1:nx
        λ = peps.λv[ix, iy]
        p = λ.^2;  p ./= max(1e-30, sum(p))
        S_sum += -sum(x -> x > 1e-30 ? x * log(x) : 0.0, p)
        n_S += 1
    end
    S_ent = S_sum / max(1, n_S)

    return (nf_A = nf_even, nf_B = nf_odd,
            Er_A = Er_even, Er_B = Er_odd,
            Eu_A = Eu_even, Eu_B = Eu_odd,
            E_flux_avg = E_flux_avg,
            plaq = plaq,
            S_ent = S_ent)
end

# =============================================================================
#  Gate pre-computation
# =============================================================================

"""
Pre-compute all 2-site gate matrices for the full lattice.
Returns (gates_h, gates_v) where
  gates_h[ix,iy]  = exp(-factor_h * H_merged_h_site(ix,iy,...))
  gates_v[ix,iy]  = exp(-factor_v * H_merged_v_site(ix,iy,...))
"""
function precompute_gates(nx, ny, dg; g, t, m,
                          factor_h::Number, factor_v::Number)
    gates_h = Matrix{Matrix{ComplexF64}}(undef, nx-1, ny)
    gates_v = Matrix{Matrix{ComplexF64}}(undef, nx,   ny-1)
    for iy in 1:ny, ix in 1:nx-1
        H = H_merged_h_site(ix, iy, nx, ny, dg; g=g, t=t, m=m)
        gates_h[ix, iy] = exp(factor_h .* H)
    end
    for iy in 1:ny-1, ix in 1:nx
        H = H_merged_v_site(ix, iy, nx, ny, dg; g=g, t=t, m=m)
        gates_v[ix, iy] = exp(factor_v .* H)
    end
    return gates_h, gates_v
end

# =============================================================================
#  Trotter step using pre-computed gates
#  (imaginary time:  factor = -τ, real time: factor = -im*dt)
# =============================================================================

"""
One 2nd-order Trotter step with pre-computed gate matrices.
gates_h_half: exp(-(τ/2 or im*dt/2)*H) for each horizontal bond
gates_h_half_back: same (reversed sweep uses same gates for imaginary time;
                   for real time with non-Hermitian split we keep separate)
gates_v:      exp(-(τ or im*dt)*H) for each vertical bond
"""
function trotter_step_gates!(peps::FinitePEPS, nx::Int, ny::Int,
                              gates_h_half::Matrix,
                              gates_v::Matrix,
                              D_trunc::Int)
    # Forward horizontal half-step
    for iy in 1:ny, ix in 1:nx-1
        update_bond_h!(peps, ix, iy, gates_h_half[ix, iy], D_trunc)
    end
    # Vertical full-step
    for iy in 1:ny-1, ix in 1:nx
        update_bond_v!(peps, ix, iy, gates_v[ix, iy], D_trunc)
    end
    # Backward horizontal half-step (adjoint for real time, same for ITE)
    for iy in 1:ny, ix in nx-1:-1:1
        update_bond_h!(peps, ix, iy, gates_h_half[ix, iy]', D_trunc)
    end
    return nothing
end

# =============================================================================
#  Ground-state preparation via imaginary-time evolution
# =============================================================================

function prepare_ground_state_finite(;
        nx=BNX, ny=BNY, dg=BDG,
        D_init=BD_ITE, D_max=BD_MAX,
        g::Float64, t::Float64, m::Float64,
        n_ite=N_ITE, tau=TAU_ITE)

    peps = init_finite_peps(nx, ny, dg, D_init)

    # Build imaginary-time gates once
    gh, gv = precompute_gates(nx, ny, dg; g=g, t=t, m=m,
                               factor_h = -(tau/2),
                               factor_v = -tau)

    for _ in 1:n_ite
        trotter_step_gates!(peps, nx, ny, gh, gv, D_max)
    end
    return peps
end

# =============================================================================
#  Real-time quench evolution with measurement
# =============================================================================

function evolve_and_measure_finite!(peps::FinitePEPS, nx, ny, dg;
                                    g::Float64, t::Float64, m::Float64,
                                    D_max=BD_MAX, dt=DT_RT, n_steps=N_RT,
                                    meas_interval=MEAS_INT, label="")
    # Build real-time gates once
    gh, gv = precompute_gates(nx, ny, dg; g=g, t=t, m=m,
                               factor_h = -(im * dt/2),
                               factor_v = -(im * dt))

    ts   = Float64[0.0]
    meas = [finite_peps_measure(peps, nx, ny, dg)]
    m0 = meas[1]
    @printf("    t=0.00: nf_A=%.4f nf_B=%.4f Er_A=%.4f □=%.4f S=%.4f\n",
            m0.nf_A, m0.nf_B, m0.Er_A, m0.plaq, m0.S_ent)

    t_wall = time()
    for step in 1:n_steps
        trotter_step_gates!(peps, nx, ny, gh, gv, D_max)

        if step % meas_interval == 0
            t_now = step * dt
            push!(ts, t_now)
            ms = finite_peps_measure(peps, nx, ny, dg)
            push!(meas, ms)
            if step % (meas_interval * 5) == 0
                elapsed = time() - t_wall
                @printf("    t=%.2f: nf_A=%.4f nf_B=%.4f Er_A=%.4f □=%.4f S=%.4f  [%.1fs]\n",
                        t_now, ms.nf_A, ms.nf_B, ms.Er_A, ms.plaq, ms.S_ent, elapsed)
            end
        end
    end

    return (t          = ts,
            nf_A       = [m.nf_A       for m in meas],
            nf_B       = [m.nf_B       for m in meas],
            Er_A       = [m.Er_A       for m in meas],
            Er_B       = [m.Er_B       for m in meas],
            Eu_A       = [m.Eu_A       for m in meas],
            Eu_B       = [m.Eu_B       for m in meas],
            E_flux_avg = [m.E_flux_avg for m in meas],
            plaq       = [m.plaq       for m in meas],
            S_ent      = [m.S_ent      for m in meas])
end

function data_to_df(data)
    DataFrame(
        t          = data.t,
        nf_A       = data.nf_A,
        nf_B       = data.nf_B,
        Er_A       = data.Er_A,
        Er_B       = data.Er_B,
        Eu_A       = data.Eu_A,
        Eu_B       = data.Eu_B,
        E_flux_avg = data.E_flux_avg,
        plaq       = data.plaq,
        S_ent      = data.S_ent,
    )
end

# =============================================================================
#  Quench A: String Breaking  g²=10 → 0.5
# =============================================================================
function run_quench_A(; t_hop=1.0, m_mass=0.5,
                        g2_conf=10.0, g2_deconf=0.5,
                        nx=BNX, ny=BNY, dg=BDG, D_max=BD_MAX)
    g_conf   = sqrt(g2_conf)
    g_deconf = sqrt(g2_deconf)

    println("\n" * "="^70)
    println("  QUENCH A: String Breaking  g²=$g2_conf → $g2_deconf")
    println("  Lattice $(nx)×$(ny),  D_max=$D_max,  dt=$DT_RT,  T=$(N_RT*DT_RT)")
    println("="^70)

    println("  Preparing ground state (g²=$g2_conf) ...")
    t0 = time()
    peps = prepare_ground_state_finite(nx=nx, ny=ny, dg=dg, D_max=D_max,
                                       g=g_conf, t=t_hop, m=m_mass)
    m0 = finite_peps_measure(peps, nx, ny, dg)
    @printf("  GS (%.1fs): nf_A=%.4f nf_B=%.4f Er_A=%.4f □=%.4f S=%.4f\n",
            time()-t0, m0.nf_A, m0.nf_B, m0.Er_A, m0.plaq, m0.S_ent)

    println("  Real-time evolution (g²=$g2_deconf) ...")
    data = evolve_and_measure_finite!(peps, nx, ny, dg;
                                      g=g_deconf, t=t_hop, m=m_mass,
                                      D_max=D_max, label="A")
    return data
end

# =============================================================================
#  Quench B: Mass Quench  m=5 → 0.1
# =============================================================================
function run_quench_B(; t_hop=1.0, m_init=5.0, m_final=0.1, g2=1.0,
                        nx=BNX, ny=BNY, dg=BDG, D_max=BD_MAX)
    g = sqrt(g2)

    println("\n" * "="^70)
    println("  QUENCH B: Mass Quench  m=$m_init → $m_final")
    println("  Lattice $(nx)×$(ny),  D_max=$D_max,  dt=$DT_RT,  T=$(N_RT*DT_RT)")
    println("="^70)

    println("  Preparing ground state (m=$m_init) ...")
    t0 = time()
    peps = prepare_ground_state_finite(nx=nx, ny=ny, dg=dg, D_max=D_max,
                                       g=g, t=t_hop, m=m_init)
    m0 = finite_peps_measure(peps, nx, ny, dg)
    @printf("  GS (%.1fs): nf_A=%.4f nf_B=%.4f Er_A=%.4f □=%.4f S=%.4f\n",
            time()-t0, m0.nf_A, m0.nf_B, m0.Er_A, m0.plaq, m0.S_ent)

    println("  Real-time evolution (m=$m_final) ...")
    data = evolve_and_measure_finite!(peps, nx, ny, dg;
                                      g=g, t=t_hop, m=m_final,
                                      D_max=D_max, label="B")
    return data
end

# =============================================================================
#  Quench C: Coupling Quench  g²=4 → 0.5
# =============================================================================
function run_quench_C(; t_hop=1.0, m_mass=0.5,
                        g2_init=4.0, g2_final=0.5,
                        nx=BNX, ny=BNY, dg=BDG, D_max=BD_MAX)
    g_init  = sqrt(g2_init)
    g_final = sqrt(g2_final)

    println("\n" * "="^70)
    println("  QUENCH C: Coupling Quench  g²=$g2_init → $g2_final")
    println("  Lattice $(nx)×$(ny),  D_max=$D_max,  dt=$DT_RT,  T=$(N_RT*DT_RT)")
    println("="^70)

    println("  Preparing ground state (g²=$g2_init) ...")
    t0 = time()
    peps = prepare_ground_state_finite(nx=nx, ny=ny, dg=dg, D_max=D_max,
                                       g=g_init, t=t_hop, m=m_mass)
    m0 = finite_peps_measure(peps, nx, ny, dg)
    @printf("  GS (%.1fs): nf_A=%.4f nf_B=%.4f Er_A=%.4f □=%.4f S=%.4f\n",
            time()-t0, m0.nf_A, m0.nf_B, m0.Er_A, m0.plaq, m0.S_ent)

    println("  Real-time evolution (g²=$g2_final) ...")
    data = evolve_and_measure_finite!(peps, nx, ny, dg;
                                      g=g_final, t=t_hop, m=m_mass,
                                      D_max=D_max, label="C")
    return data
end

# =============================================================================
#  4-panel plot per quench
# =============================================================================
function plot_finite_peps_panels(data, title_str, prefix)
    t = data.t

    p1 = plot(t, data.nf_A; label="⟨n_f⟩ even", xlabel="t", ylabel="⟨n_f⟩",
              title="(a) Fermion Density", color=:blue, lw=2)
    plot!(p1, t, data.nf_B; label="⟨n_f⟩ odd", color=:red, ls=:dash, lw=2)
    hline!(p1, [0.5]; label="0.5", ls=:dot, color=:gray, lw=1)

    p2 = plot(t, data.Er_A; label="⟨E_R⟩ even", xlabel="t", ylabel="⟨E⟩",
              title="(b) Electric Flux", color=:blue, lw=2)
    plot!(p2, t, data.Er_B; label="⟨E_R⟩ odd",  color=:red,    ls=:dash, lw=2)
    plot!(p2, t, data.Eu_A; label="⟨E_U⟩ even", color=:cyan,   ls=:dot,  lw=2)
    plot!(p2, t, data.Eu_B; label="⟨E_U⟩ odd",  color=:orange, ls=:dashdot, lw=2)

    p3 = plot(t, data.plaq; label="⟨□⟩ (MF avg)", xlabel="t", ylabel="⟨□⟩",
              title="(c) Plaquette", color=:purple, lw=2.5)

    p4 = plot(t, data.S_ent; label="S_bond (avg)", xlabel="t", ylabel="S_vN",
              title="(d) Bond Entanglement Entropy", color=:black, lw=2.5)

    fig = plot(p1, p2, p3, p4; layout=(2,2), size=(1200, 800),
               plot_title=title_str * "  (finite PEPS $(BNX)×$(BNY), D=$BD_MAX)")
    fname = "finite_peps_$(prefix)_panels.png"
    savefig(fig, joinpath(@__DIR__, fname))
    println("  Saved: $fname")
    return fig
end

# =============================================================================
#  6-panel summary plot
# =============================================================================
function plot_finite_peps_summary(dA, dB, dC)
    # (1) Quench A: fermion density
    p1 = plot(dA.t, dA.nf_A; label="⟨n_f⟩ even", xlabel="t", ylabel="⟨n_f⟩",
              title="A: String Breaking", color=:blue, lw=2)
    plot!(p1, dA.t, dA.nf_B; label="⟨n_f⟩ odd", color=:red, ls=:dash, lw=2)
    hline!(p1, [0.5]; ls=:dot, color=:gray, lw=1, label="")

    # (2) Quench B: CDW order parameter
    cdw = dB.nf_A .- dB.nf_B
    p2 = plot(dB.t, cdw; label="CDW = nf_even − nf_odd", xlabel="t", ylabel="CDW",
              title="B: Mass Quench", color=:black, lw=2)
    plot!(p2, dB.t, dB.nf_A; label="nf_even", color=:blue, ls=:dash, alpha=0.5, lw=1.5)
    plot!(p2, dB.t, dB.nf_B; label="nf_odd",  color=:red,  ls=:dash, alpha=0.5, lw=1.5)

    # (3) Quench C: plaquette + flux
    p3 = plot(dC.t, dC.plaq; label="⟨□⟩", xlabel="t", ylabel="mixed",
              title="C: Coupling Quench", color=:purple, lw=2)
    plot!(p3, dC.t, dC.E_flux_avg; label="⟨E⟩ avg", color=:red, ls=:dash, lw=2)

    # (4) Entanglement entropy comparison
    p4 = plot(dA.t, dA.S_ent; label="A: String", xlabel="t", ylabel="S_bond",
              title="Bond Entanglement Entropy", color=:blue, lw=2)
    plot!(p4, dB.t, dB.S_ent; label="B: Mass",     color=:red,   ls=:dash, lw=2)
    plot!(p4, dC.t, dC.S_ent; label="C: Coupling", color=:green, ls=:dot,  lw=2)

    # (5) Electric flux comparison
    p5 = plot(dA.t, dA.E_flux_avg; label="A", xlabel="t", ylabel="⟨E⟩ avg",
              title="Electric Flux Comparison", color=:blue, lw=2)
    plot!(p5, dB.t, dB.E_flux_avg; label="B", color=:red,   ls=:dash, lw=2)
    plot!(p5, dC.t, dC.E_flux_avg; label="C", color=:green, ls=:dot,  lw=2)

    # (6) Plaquette comparison
    p6 = plot(dA.t, dA.plaq; label="A", xlabel="t", ylabel="⟨□⟩",
              title="Plaquette Comparison", color=:blue, lw=2)
    plot!(p6, dB.t, dB.plaq; label="B", color=:red,   ls=:dash, lw=2)
    plot!(p6, dC.t, dC.plaq; label="C", color=:green, ls=:dot,  lw=2)

    fig = plot(p1, p2, p3, p4, p5, p6; layout=(2,3), size=(1500, 800),
               plot_title="Finite PEPS Quench Benchmark — U(1) LGT  $(BNX)×$(BNY)  D=$BD_MAX  dg=$BDG")
    fname = "finite_peps_quench_summary.png"
    savefig(fig, joinpath(@__DIR__, fname))
    println("  Saved: $fname")
    return fig
end

# =============================================================================
#  Spatial density map at end of each quench
# =============================================================================
function plot_density_map(peps::FinitePEPS, nx, ny, dg, title_str, fname)
    dens = [begin
        _, d_gR, d_gU = site_dims(ix, iy, nx, ny, dg)
        expect_site(peps, ix, iy, embed_f_site(op_nf(), d_gR, d_gU))
    end for ix in 1:nx, iy in 1:ny]

    fig = heatmap(1:nx, 1:ny, dens';
                  xlabel="ix", ylabel="iy", title=title_str,
                  color=:viridis, clim=(0,1),
                  aspect_ratio=:equal, size=(500, 600))
    savefig(fig, joinpath(@__DIR__, fname))
    println("  Saved: $fname")
    return fig
end

# =============================================================================
#  Main
# =============================================================================
function run_all_benchmarks()
    println("="^70)
    println("  FINITE PEPS QUENCH BENCHMARK — U(1) LGT")
    println("  Lattice: $(BNX)×$(BNY)  ($(BNX-1)×$(BNY-1) plaquettes)")
    @printf("  dg=%d  (link dim=%d),  D: %d → %d\n",
            BDG, gauge_dim(BDG), BD_ITE, BD_MAX)
    println("  ITE: $(N_ITE) steps × τ=$(TAU_ITE)  →  imaginary T=$(N_ITE*TAU_ITE)")
    println("  RT:  $(N_RT) steps × dt=$(DT_RT)     →  real T=$(N_RT*DT_RT)")
    println("  Measure every $MEAS_INT RT steps")
    println("="^70)

    t_total = time()

    # ── Quench A ─────────────────────────────────────────────────────────────
    dA = run_quench_A()
    dfA = data_to_df(dA)
    CSV.write(joinpath(@__DIR__, "finite_peps_quench_A_data.csv"), dfA)
    plot_finite_peps_panels(dA, "Quench A: String Breaking (g²=10→0.5)", "quench_A")
    println("  Quench A done in $(round(time()-t_total, digits=1))s\n")

    # ── Quench B ─────────────────────────────────────────────────────────────
    t1 = time()
    dB = run_quench_B()
    dfB = data_to_df(dB)
    CSV.write(joinpath(@__DIR__, "finite_peps_quench_B_data.csv"), dfB)
    plot_finite_peps_panels(dB, "Quench B: Mass Quench (m=5→0.1)", "quench_B")
    println("  Quench B done in $(round(time()-t1, digits=1))s\n")

    # ── Quench C ─────────────────────────────────────────────────────────────
    t2 = time()
    dC = run_quench_C()
    dfC = data_to_df(dC)
    CSV.write(joinpath(@__DIR__, "finite_peps_quench_C_data.csv"), dfC)
    plot_finite_peps_panels(dC, "Quench C: Coupling Quench (g²=4→0.5)", "quench_C")
    println("  Quench C done in $(round(time()-t2, digits=1))s\n")

    # ── Summary plot ──────────────────────────────────────────────────────────
    plot_finite_peps_summary(dA, dB, dC)

    println("\n" * "="^70)
    @printf("  ALL BENCHMARKS COMPLETE  (total: %.1fs)\n", time()-t_total)
    println("  Output files:")
    println("    finite_peps_quench_A_panels.png  /  finite_peps_quench_A_data.csv")
    println("    finite_peps_quench_B_panels.png  /  finite_peps_quench_B_data.csv")
    println("    finite_peps_quench_C_panels.png  /  finite_peps_quench_C_data.csv")
    println("    finite_peps_quench_summary.png")
    println("="^70)

    return dA, dB, dC
end

run_all_benchmarks()
