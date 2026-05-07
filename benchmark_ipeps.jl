# =============================================================================
#  iPEPS Quench Benchmark for 2+1D U(1) LGT
#
#  Uses the iPEPS simple-update code from tensorkit_tst.jl to produce
#  quench dynamics plots matching the exact-diag benchmarks in benchmark_ed.jl.
#
#  Three quench protocols:
#    A. String Breaking / Confinement→Deconfinement  (g²=10 → 0.5)
#    B. Mass Quench                                  (m=5 → 0.1)
#    C. Coupling Quench                              (g²=4 → 0.5)
#
#  Observables: fermion density ⟨n_f⟩, electric flux ⟨E_r⟩, plaquette ⟨□⟩,
#               bond entanglement entropy S_vN.
#
#  Note: D_max=4 with 4-site plaquette gates is very slow. This script uses
#  bond-only Hamiltonians (hopping + mass + electric) without plaquette gates
#  to keep runtime practical. The plaquette observable is still measured.
# =============================================================================

include("tensorkit_tst.jl")

using Pkg
for pkg in ["Plots", "CSV", "DataFrames"]
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg)
end
using Plots, CSV, DataFrames

default(fontfamily="sans-serif", linewidth=2, framestyle=:box,
        grid=true, legend=:best, size=(800,500), dpi=200)

# ─── Tunable parameters ─────────────────────────────────────────────────────
const DT       = D_max    # bond truncation (D_max=4 from tensorkit_tst.jl)
const N_ITE    = 80       # ITE steps for ground-state prep
const TAU_ITE  = 0.05     # ITE time step
const DT_RT    = 0.10     # real-time Trotter step
const N_RT     = 60       # real-time steps  (total time = N_RT * DT_RT = 6.0)
const MEAS_INT = 3        # measure every N steps

# =============================================================================
#  Measurement function: extract all observables from the iPEPS state
# =============================================================================
function ipeps_measure(peps::CheckerboardiPEPS)
    A_arr = convert(Array, peps.A)
    B_arr = convert(Array, peps.B)
    λh = peps.λh;  λv = peps.λv

    nf_op = fermion_n()
    Er_op = electric_field_right()
    Eu_op = electric_field_up()

    nf_A = expect_onsite(A_arr, λh, λv, nf_op)
    nf_B = expect_onsite(B_arr, λh, λv, nf_op)
    Er_A = expect_onsite(A_arr, λh, λv, Er_op)
    Er_B = expect_onsite(B_arr, λh, λv, Er_op)
    Eu_A = expect_onsite(A_arr, λh, λv, Eu_op)
    Eu_B = expect_onsite(B_arr, λh, λv, Eu_op)

    # Plaquette from two-site horizontal correlator ⟨U_br_A ⊗ U_bu_B + h.c.⟩
    Ur_op = gauge_U_right()
    Uu_op = gauge_U_up()
    plaq_val = real(expect_twosite_h(peps, Ur_op, Uu_op) +
                    expect_twosite_h(peps, Ur_op', Uu_op'))

    # Bond entanglement entropy from λh
    p_h = λh.^2
    s = sum(p_h)
    if s > 1e-30
        p_h ./= s
        S_ent = -sum(x -> x > 1e-30 ? x * log(x) : 0.0, p_h)
    else
        S_ent = 0.0
    end

    return (nf_A=nf_A, nf_B=nf_B, Er_A=Er_A, Er_B=Er_B,
            Eu_A=Eu_A, Eu_B=Eu_B,
            E_flux_avg=(Er_A + Er_B + Eu_A + Eu_B)/4,
            plaq=plaq_val, S_ent=S_ent)
end

# =============================================================================
#  Ground-state preparation via imaginary-time evolution (ITE)
# =============================================================================
function prepare_ground_state(;
    t_hop=1.0, m_mass=0.5, g2=1.0,
    n_ite=N_ITE, tau_ite=TAU_ITE, D_trunc=DT,
    init_nf_A=0, init_nf_B=1,      # anti-CDW default (has hopping dynamics)
)
    V_b = ℂ^2   # start at D=2, will grow via SVD up to D_trunc
    peps = init_checkerboard(V_phys, V_b;
                             nf_A=init_nf_A, nbr_A=0, nbu_A=0,
                             nf_B=init_nf_B, nbr_B=0, nbu_B=0)

    H_h = build_horizontal_hamiltonian(t=t_hop, m=m_mass, g2=g2, sign_L=1)
    H_v = build_vertical_hamiltonian(t=t_hop, m=m_mass, g2=g2, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h')
    H_v .= 0.5 .* (H_v .+ H_v')

    for step in 1:n_ite
        trotter_step!(peps, H_h, H_v, tau_ite, D_trunc)
    end

    return peps
end

# =============================================================================
#  Real-time evolution with observable measurement
# =============================================================================
function evolve_and_measure!(peps, H_h, H_v, dt, n_steps, D_trunc;
                              meas_interval=MEAS_INT, label="")
    ts    = Float64[0.0]
    meas  = [ipeps_measure(peps)]

    m0 = meas[1]
    @printf("    t=0.00: nf_A=%.4f nf_B=%.4f Er_A=%.4f □=%.4f S=%.4f\n",
            m0.nf_A, m0.nf_B, m0.Er_A, m0.plaq, m0.S_ent)

    t_wall = time()
    for step in 1:n_steps
        realtime_trotter_step!(peps, H_h, H_v, dt, D_trunc)

        if step % meas_interval == 0
            t_now = step * dt
            push!(ts, t_now)
            m = ipeps_measure(peps)
            push!(meas, m)
            if step % (meas_interval * 5) == 0
                elapsed = time() - t_wall
                @printf("    t=%.2f: nf_A=%.4f nf_B=%.4f Er_A=%.4f □=%.4f S=%.4f  [%.1fs]\n",
                        t_now, m.nf_A, m.nf_B, m.Er_A, m.plaq, m.S_ent, elapsed)
            end
        end
    end

    return (t=ts,
            nf_A   = [m.nf_A for m in meas],
            nf_B   = [m.nf_B for m in meas],
            Er_A   = [m.Er_A for m in meas],
            Er_B   = [m.Er_B for m in meas],
            Eu_A   = [m.Eu_A for m in meas],
            Eu_B   = [m.Eu_B for m in meas],
            E_flux_avg = [m.E_flux_avg for m in meas],
            plaq   = [m.plaq for m in meas],
            S_ent  = [m.S_ent for m in meas])
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
#  Quench A: String Breaking / Confinement → Deconfinement
#
#  GS at g²=10 (confining) → evolve with g²=0.5 (deconfining)
# =============================================================================
function run_quench_A_ipeps(; t_hop=1.0, m_mass=0.5,
                              g2_conf=10.0, g2_deconf=0.5,
                              dt=DT_RT, n_steps=N_RT, D_trunc=DT)
    println("\n" * "="^65)
    println("  iPEPS QUENCH A: String Breaking  g²=$g2_conf → $g2_deconf")
    println("  D=$D_trunc, dt=$dt, steps=$n_steps, T=$(n_steps*dt)")
    println("="^65)

    println("  Preparing ground state at g²=$g2_conf ...")
    peps = prepare_ground_state(t_hop=t_hop, m_mass=m_mass, g2=g2_conf, D_trunc=D_trunc)
    m0 = ipeps_measure(peps)
    @printf("  GS: nf_A=%.4f nf_B=%.4f Er_A=%.4f □=%.4f S=%.4f\n",
            m0.nf_A, m0.nf_B, m0.Er_A, m0.plaq, m0.S_ent)

    println("  Evolving with g²=$g2_deconf ...")
    H_h = build_horizontal_hamiltonian(t=t_hop, m=m_mass, g2=g2_deconf, sign_L=1)
    H_v = build_vertical_hamiltonian(t=t_hop, m=m_mass, g2=g2_deconf, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h'); H_v .= 0.5 .* (H_v .+ H_v')

    return evolve_and_measure!(peps, H_h, H_v, dt, n_steps, D_trunc; label="A")
end

# =============================================================================
#  Quench B: Mass Quench  m=m_init → m=m_final
# =============================================================================
function run_quench_B_ipeps(; t_hop=1.0, m_init=5.0, m_final=0.1, g2=1.0,
                              dt=DT_RT, n_steps=N_RT, D_trunc=DT)
    println("\n" * "="^65)
    println("  iPEPS QUENCH B: Mass Quench  m=$m_init → $m_final")
    println("  D=$D_trunc, dt=$dt, steps=$n_steps, T=$(n_steps*dt)")
    println("="^65)

    println("  Preparing ground state at m=$m_init ...")
    peps = prepare_ground_state(t_hop=t_hop, m_mass=m_init, g2=g2, D_trunc=D_trunc)
    m0 = ipeps_measure(peps)
    @printf("  GS: nf_A=%.4f nf_B=%.4f Er_A=%.4f □=%.4f S=%.4f\n",
            m0.nf_A, m0.nf_B, m0.Er_A, m0.plaq, m0.S_ent)

    println("  Evolving with m=$m_final ...")
    H_h = build_horizontal_hamiltonian(t=t_hop, m=m_final, g2=g2, sign_L=1)
    H_v = build_vertical_hamiltonian(t=t_hop, m=m_final, g2=g2, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h'); H_v .= 0.5 .* (H_v .+ H_v')

    return evolve_and_measure!(peps, H_h, H_v, dt, n_steps, D_trunc; label="B")
end

# =============================================================================
#  Quench C: Coupling Quench  g²=g2_init → g²=g2_final
# =============================================================================
function run_quench_C_ipeps(; t_hop=1.0, m_mass=0.5,
                              g2_init=4.0, g2_final=0.5,
                              dt=DT_RT, n_steps=N_RT, D_trunc=DT)
    println("\n" * "="^65)
    println("  iPEPS QUENCH C: Coupling Quench  g²=$g2_init → $g2_final")
    println("  D=$D_trunc, dt=$dt, steps=$n_steps, T=$(n_steps*dt)")
    println("="^65)

    println("  Preparing ground state at g²=$g2_init ...")
    peps = prepare_ground_state(t_hop=t_hop, m_mass=m_mass, g2=g2_init, D_trunc=D_trunc)
    m0 = ipeps_measure(peps)
    @printf("  GS: nf_A=%.4f nf_B=%.4f Er_A=%.4f □=%.4f S=%.4f\n",
            m0.nf_A, m0.nf_B, m0.Er_A, m0.plaq, m0.S_ent)

    println("  Evolving with g²=$g2_final ...")
    H_h = build_horizontal_hamiltonian(t=t_hop, m=m_mass, g2=g2_final, sign_L=1)
    H_v = build_vertical_hamiltonian(t=t_hop, m=m_mass, g2=g2_final, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h'); H_v .= 0.5 .* (H_v .+ H_v')

    return evolve_and_measure!(peps, H_h, H_v, dt, n_steps, D_trunc; label="C")
end

# =============================================================================
#  Plotting
# =============================================================================

function plot_ipeps_panels(data, title_str, prefix)
    t = data.t

    p1 = plot(t, data.nf_A, label="⟨n_f⟩_A", xlabel="t", ylabel="⟨n_f⟩",
              title="(a) Fermion Density", color=:blue)
    plot!(p1, t, data.nf_B, label="⟨n_f⟩_B", color=:red, ls=:dash)

    p2 = plot(t, data.Er_A, label="⟨E_r⟩_A", xlabel="t", ylabel="⟨E⟩",
              title="(b) Electric Flux", color=:blue, lw=2)
    plot!(p2, t, data.Er_B, label="⟨E_r⟩_B", color=:red, ls=:dash)
    plot!(p2, t, data.Eu_A, label="⟨E_u⟩_A", color=:cyan, ls=:dot)
    plot!(p2, t, data.Eu_B, label="⟨E_u⟩_B", color=:orange, ls=:dashdot)

    p3 = plot(t, data.plaq, label="⟨□⟩", xlabel="t", ylabel="⟨□⟩",
              title="(c) Plaquette Expectation", color=:purple, lw=2.5)

    p4 = plot(t, data.S_ent, label="S_bond", xlabel="t", ylabel="S_vN",
              title="(d) Bond Entanglement Entropy", color=:black, lw=2.5)

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800),
              plot_title=title_str * " (iPEPS D=$DT)")
    savefig(fig, "ipeps_$(prefix)_panels.png")
    println("  Saved: ipeps_$(prefix)_panels.png")
    return fig
end

function plot_ipeps_summary(dA, dB, dC)
    # A: fermion density
    p1 = plot(dA.t, dA.nf_A, label="nf_A", xlabel="t", ylabel="⟨n_f⟩",
              title="A: String Breaking", color=:blue, lw=2)
    plot!(p1, dA.t, dA.nf_B, label="nf_B", color=:red, ls=:dash)

    # B: CDW order parameter
    cdw = dB.nf_A .- dB.nf_B
    p2 = plot(dB.t, cdw, label="nf_A - nf_B", xlabel="t", ylabel="CDW",
              title="B: Mass Quench (CDW)", color=:black, lw=2)
    plot!(p2, dB.t, dB.nf_A, label="nf_A", color=:blue, ls=:dash, alpha=0.5)
    plot!(p2, dB.t, dB.nf_B, label="nf_B", color=:red, ls=:dash, alpha=0.5)

    # C: Plaquette + Electric flux
    p3 = plot(dC.t, dC.plaq, label="⟨□⟩", xlabel="t", ylabel="⟨□⟩",
              title="C: Coupling Quench", color=:purple, lw=2)
    plot!(p3, dC.t, dC.E_flux_avg, label="⟨E⟩", color=:red, ls=:dash)

    # Entanglement entropy
    p4 = plot(dA.t, dA.S_ent, label="A: String", xlabel="t", ylabel="S_vN",
              title="Entanglement Entropy", color=:blue, lw=2)
    plot!(p4, dB.t, dB.S_ent, label="B: Mass", color=:red, ls=:dash)
    plot!(p4, dC.t, dC.S_ent, label="C: Coupling", color=:green, ls=:dot)

    # Electric flux comparison
    p5 = plot(dA.t, dA.E_flux_avg, label="A", xlabel="t", ylabel="⟨E⟩ avg",
              title="Electric Flux Comparison", color=:blue, lw=2)
    plot!(p5, dB.t, dB.E_flux_avg, label="B", color=:red, ls=:dash)
    plot!(p5, dC.t, dC.E_flux_avg, label="C", color=:green, ls=:dot)

    # Plaquette comparison
    p6 = plot(dA.t, dA.plaq, label="A", xlabel="t", ylabel="⟨□⟩",
              title="Plaquette Comparison", color=:blue, lw=2)
    plot!(p6, dB.t, dB.plaq, label="B", color=:red, ls=:dash)
    plot!(p6, dC.t, dC.plaq, label="C", color=:green, ls=:dot)

    fig = plot(p1, p2, p3, p4, p5, p6, layout=(2,3), size=(1500, 800),
              plot_title="iPEPS Quench Benchmark — U(1) LGT (D=$DT, d_b=$d_b)")
    savefig(fig, "ipeps_quench_summary.png")
    println("  Saved: ipeps_quench_summary.png")
    return fig
end

# =============================================================================
#  Main
# =============================================================================
function run_all_ipeps()
    println("="^65)
    println("  iPEPS QUENCH BENCHMARK: 2+1D U(1) LGT")
    println("  d=$d, d_b=$d_b, D=$DT")
    println("  ITE: $(N_ITE) steps × τ=$(TAU_ITE)")
    println("  RT:  $(N_RT) steps × dt=$(DT_RT)  (T=$(N_RT*DT_RT))")
    println("  Measure every $MEAS_INT steps")
    println("  NOTE: no plaquette gate in Trotter (bond-only H)")
    println("="^65)

    t0 = time()

    # ── Quench A ──────────────────────────────────────────────────────────────
    dA = run_quench_A_ipeps()
    dfA = data_to_df(dA)
    CSV.write("ipeps_quench_A_data.csv", dfA)
    plot_ipeps_panels(dA, "Quench A: String Breaking (g²=10→0.5)", "quench_A")
    println("  Quench A done in $(round(time()-t0, digits=1))s\n")

    # ── Quench B ──────────────────────────────────────────────────────────────
    t1 = time()
    dB = run_quench_B_ipeps()
    dfB = data_to_df(dB)
    CSV.write("ipeps_quench_B_data.csv", dfB)
    plot_ipeps_panels(dB, "Quench B: Mass Quench (m=5→0.1)", "quench_B")
    println("  Quench B done in $(round(time()-t1, digits=1))s\n")

    # ── Quench C ──────────────────────────────────────────────────────────────
    t2 = time()
    dC = run_quench_C_ipeps()
    dfC = data_to_df(dC)
    CSV.write("ipeps_quench_C_data.csv", dfC)
    plot_ipeps_panels(dC, "Quench C: Coupling Quench (g²=4→0.5)", "quench_C")
    println("  Quench C done in $(round(time()-t2, digits=1))s\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    plot_ipeps_summary(dA, dB, dC)

    println("\n" * "="^65)
    println("  ALL iPEPS BENCHMARKS COMPLETE  (total: $(round(time()-t0, digits=1))s)")
    println("  Output:")
    println("    ipeps_quench_A_panels.png, ipeps_quench_A_data.csv")
    println("    ipeps_quench_B_panels.png, ipeps_quench_B_data.csv")
    println("    ipeps_quench_C_panels.png, ipeps_quench_C_data.csv")
    println("    ipeps_quench_summary.png")
    println("="^65)
end

run_all_ipeps()
