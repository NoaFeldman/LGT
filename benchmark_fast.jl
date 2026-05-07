# =============================================================================
#  Fast Benchmark: Quench Dynamics in 2+1D U(1) LGT (iPEPS simple update)
#
#  Three quench protocols:
#    A. String Breaking  (vacuum + flux string → evolve)
#    B. Mass Quench      (GS at m=5 → m=0.1)
#    C. Coupling Quench  (GS at g²=4 → g²=0.5)
#
#  Observables: charge density, ⟨E²⟩, ⟨□⟩, bond entropy S_vN
#  Uses D_trunc=2, 50 steps, measure every 5 steps for speed.
# =============================================================================

include("tensorkit_tst.jl")

using Pkg
for pkg in ["Plots", "CSV", "DataFrames"]
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg)
end
using Plots, CSV, DataFrames

default(fontfamily="sans-serif", linewidth=2, framestyle=:box,
        grid=true, legend=:best, size=(800,500), dpi=150)

# ─── Parameters ──────────────────────────────────────────────────────────────
const DT       = 2       # bond truncation for speed
const N_STEPS  = 50      # total evolution steps
const DT_STEP  = 0.05    # time step
const MEAS_INT = 5       # measure every N steps
const N_ITE    = 100     # ITE steps for ground-state prep
const TAU_ITE  = 0.05    # ITE time step

# ─── Lightweight observables (skip Gauss law during evolution) ───────────────

function measure_lite(peps::CheckerboardiPEPS, t_now::Float64)
    A_arr = convert(Array, peps.A)
    B_arr = convert(Array, peps.B)

    nf = fermion_n()
    nf_A = expect_onsite(A_arr, peps.λh, peps.λv, nf)
    nf_B = expect_onsite(B_arr, peps.λh, peps.λv, nf)

    Er = electric_field_right(); Eu = electric_field_up()
    E2_op = Er^2 + Eu^2
    E2_A = expect_onsite(A_arr, peps.λh, peps.λv, E2_op)
    E2_B = expect_onsite(B_arr, peps.λh, peps.λv, E2_op)

    Id = Matrix{ComplexF64}(I, d, d)
    Ur = gauge_U_right(); Uu = gauge_U_up()
    plaq = real(expect_twosite_h(peps, Ur, Uu)) +
           real(expect_twosite_h(peps, Ur', Uu'))

    # Bond entropy
    p_h = peps.λh .^ 2; p_h ./= max(sum(p_h), 1e-30)
    S_h = -sum(x -> x > 1e-30 ? x * log(x) : 0.0, p_h)
    p_v = peps.λv .^ 2; p_v ./= max(sum(p_v), 1e-30)
    S_v = -sum(x -> x > 1e-30 ? x * log(x) : 0.0, p_v)

    return (t=t_now, Q_A=nf_A, Q_B=nf_B-1.0, nf_A=nf_A, nf_B=nf_B,
            E2_A=E2_A, E2_B=E2_B, E2_avg=(E2_A+E2_B)/2,
            plaq=plaq, S_h=S_h, S_v=S_v, S_avg=(S_h+S_v)/2)
end

function to_df(data)
    DataFrame(
        t=[d.t for d in data], Q_A=[d.Q_A for d in data], Q_B=[d.Q_B for d in data],
        nf_A=[d.nf_A for d in data], nf_B=[d.nf_B for d in data],
        E2_A=[d.E2_A for d in data], E2_B=[d.E2_B for d in data],
        E2_avg=[d.E2_avg for d in data], plaq=[d.plaq for d in data],
        S_h=[d.S_h for d in data], S_v=[d.S_v for d in data],
        S_avg=[d.S_avg for d in data])
end

# ─── Ground state via short ITE ─────────────────────────────────────────────

function quick_ground_state(; m_mass, g2, n_ite=N_ITE)
    H_h = build_horizontal_hamiltonian(t=1.0, m=m_mass, g2=g2, sign_L=1)
    H_v = build_vertical_hamiltonian(t=1.0, m=m_mass, g2=g2, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h'); H_v .= 0.5 .* (H_v .+ H_v')
    H_plaq_4 = build_plaquette_4site(g2=g2)
    G_plaq_4 = plaquette_gate(H_plaq_4, -TAU_ITE)

    peps = init_checkerboard(V_phys, V_bond;
                             nf_A=1, nbr_A=0, nbu_A=0,
                             nf_B=0, nbr_B=0, nbu_B=0)
    for step in 1:n_ite
        trotter_step!(peps, H_h, H_v, TAU_ITE, DT; G_plaq_4site=G_plaq_4)
        if step % 25 == 0
            A_arr = convert(Array, peps.A)
            nfA = expect_onsite(A_arr, peps.λh, peps.λv, fermion_n())
            @printf("    ITE %3d: ⟨nf⟩_A=%.4f  λh=%s\n", step, nfA,
                    string(round.(peps.λh[1:min(2,end)], digits=4)))
        end
    end
    return peps
end

# ─── Realtime evolution helper ───────────────────────────────────────────────

function evolve_and_measure!(peps, H_h, H_v, G_plaq_4, dt, n_steps, label)
    data = [measure_lite(peps, 0.0)]
    println("  t=0: Q_A=$(round(data[1].Q_A,digits=4)), E²=$(round(data[1].E2_avg,digits=4)), □=$(round(data[1].plaq,digits=4))")
    println("  step |    t     Q_A      Q_B     ⟨E²⟩     ⟨□⟩     S_avg")
    println("  ─────┼──────────────────────────────────────────────────")
    for step in 1:n_steps
        realtime_trotter_step!(peps, H_h, H_v, dt, DT; G_plaq_4site=G_plaq_4)
        if step % MEAS_INT == 0 || step == 1
            obs = measure_lite(peps, step * dt)
            push!(data, obs)
            @printf("  %4d | %5.2f  %+.4f  %+.4f  %.4f  %+.4f  %.4f\n",
                    step, obs.t, obs.Q_A, obs.Q_B, obs.E2_avg, obs.plaq, obs.S_avg)
        end
    end
    return data
end

# =============================================================================
#  QUENCH A: String Breaking
# =============================================================================
function run_A()
    println("\n" * "="^60)
    println("  QUENCH A: String Breaking")
    println("  dt=$(DT_STEP), steps=$(N_STEPS), D=$(DT)")
    println("="^60)

    peps = init_checkerboard(V_phys, V_bond;
                             nf_A=0, nbr_A=0, nbu_A=0,
                             nf_B=0, nbr_B=0, nbu_B=0)

    # Create flux string: U_right on A, ψ† on B
    Ur = gauge_U_right(); cd = fermion_c()'
    A_arr = convert(Array, peps.A)
    B_arr = convert(Array, peps.B)
    dp = size(A_arr,1); Db = size(A_arr,2)

    A_mat = reshape(A_arr, dp, :)
    A_arr_new = reshape(Ur * A_mat, size(A_arr))
    B_mat = reshape(B_arr, dp, :)
    B_arr_new = reshape(cd * B_mat, size(B_arr))
    nrm = max(maximum(abs.(A_arr_new)), maximum(abs.(B_arr_new)), 1e-15)
    V_b = ℂ^Db
    peps.A = TensorMap(A_arr_new ./ sqrt(nrm), V_phys, V_b ⊗ V_b ⊗ V_b ⊗ V_b)
    peps.B = TensorMap(B_arr_new ./ sqrt(nrm), V_phys, V_b ⊗ V_b ⊗ V_b ⊗ V_b)

    m, g2 = 0.5, 1.0
    H_h = build_horizontal_hamiltonian(t=1.0, m=m, g2=g2, sign_L=1)
    H_v = build_vertical_hamiltonian(t=1.0, m=m, g2=g2, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h'); H_v .= 0.5 .* (H_v .+ H_v')
    G_plaq_4 = plaquette_gate(build_plaquette_4site(g2=g2), -im * DT_STEP)

    return evolve_and_measure!(peps, H_h, H_v, G_plaq_4, DT_STEP, N_STEPS, "A")
end

# =============================================================================
#  QUENCH B: Mass Quench (m=5 → m=0.1)
# =============================================================================
function run_B()
    println("\n" * "="^60)
    println("  QUENCH B: Mass Quench m=5.0 → m=0.1")
    println("  ITE=$(N_ITE) steps, then dt=$(DT_STEP), steps=$(N_STEPS), D=$(DT)")
    println("="^60)

    println("  Preparing ground state at m=5.0...")
    peps = quick_ground_state(m_mass=5.0, g2=1.0)

    m_f, g2 = 0.1, 1.0
    H_h = build_horizontal_hamiltonian(t=1.0, m=m_f, g2=g2, sign_L=1)
    H_v = build_vertical_hamiltonian(t=1.0, m=m_f, g2=g2, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h'); H_v .= 0.5 .* (H_v .+ H_v')
    G_plaq_4 = plaquette_gate(build_plaquette_4site(g2=g2), -im * DT_STEP)

    println("  Quenching to m=$m_f ...")
    return evolve_and_measure!(peps, H_h, H_v, G_plaq_4, DT_STEP, N_STEPS, "B")
end

# =============================================================================
#  QUENCH C: Coupling Quench (g²=4 → g²=0.5)
# =============================================================================
function run_C()
    println("\n" * "="^60)
    println("  QUENCH C: Coupling Quench g²=4.0 → g²=0.5")
    println("  ITE=$(N_ITE) steps, then dt=$(DT_STEP), steps=$(N_STEPS), D=$(DT)")
    println("="^60)

    println("  Preparing ground state at g²=4.0...")
    peps = quick_ground_state(m_mass=0.5, g2=4.0)

    g2_f = 0.5
    H_h = build_horizontal_hamiltonian(t=1.0, m=0.5, g2=g2_f, sign_L=1)
    H_v = build_vertical_hamiltonian(t=1.0, m=0.5, g2=g2_f, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h'); H_v .= 0.5 .* (H_v .+ H_v')
    G_plaq_4 = plaquette_gate(build_plaquette_4site(g2=g2_f), -im * DT_STEP)

    println("  Quenching to g²=$g2_f ...")
    return evolve_and_measure!(peps, H_h, H_v, G_plaq_4, DT_STEP, N_STEPS, "C")
end

# =============================================================================
#  Plotting
# =============================================================================

function make_4panel(df, title_str, prefix)
    t = df.t
    p1 = plot(t, df.Q_A, label="Q_A", xlabel="t", ylabel="⟨Q⟩", title="(a) Charge")
    plot!(p1, t, df.Q_B, label="Q_B", ls=:dash)
    hline!(p1, [0.0], c=:gray, ls=:dot, label="")

    p2 = plot(t, df.E2_A, label="E²_A", xlabel="t", ylabel="⟨E²⟩", title="(b) Electric Energy")
    plot!(p2, t, df.E2_B, label="E²_B", ls=:dash)
    plot!(p2, t, df.E2_avg, label="avg", c=:black, lw=2.5)

    p3 = plot(t, df.plaq, label="⟨□⟩", xlabel="t", ylabel="⟨□⟩", title="(c) Plaquette", c=:purple)

    p4 = plot(t, df.S_h, label="S_h", xlabel="t", ylabel="S_vN", title="(d) Entanglement")
    plot!(p4, t, df.S_v, label="S_v", ls=:dash)
    plot!(p4, t, df.S_avg, label="S_avg", c=:black, lw=2.5)

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1100,750),
              plot_title=title_str)
    savefig(fig, "$(prefix)_panels.png")
    println("  → Saved $(prefix)_panels.png")
    return fig
end

function make_summary(dA, dB, dC)
    p1 = plot(dA.t, dA.Q_A, label="A: Q_A", xlabel="t", ylabel="⟨Q⟩", title="Charge Density")
    plot!(p1, dB.t, dB.Q_A, label="B: Q_A", ls=:dash)
    plot!(p1, dC.t, dC.Q_A, label="C: Q_A", ls=:dot)

    p2 = plot(dA.t, dA.E2_avg, label="A", xlabel="t", ylabel="⟨E²⟩", title="Electric Energy")
    plot!(p2, dB.t, dB.E2_avg, label="B", ls=:dash)
    plot!(p2, dC.t, dC.E2_avg, label="C", ls=:dot)

    p3 = plot(dA.t, dA.plaq, label="A", xlabel="t", ylabel="⟨□⟩", title="Plaquette", c=:purple)
    plot!(p3, dB.t, dB.plaq, label="B", ls=:dash, c=:orange)
    plot!(p3, dC.t, dC.plaq, label="C", ls=:dot, c=:green)

    p4 = plot(dA.t, dA.S_avg, label="A", xlabel="t", ylabel="S_vN", title="Entanglement")
    plot!(p4, dB.t, dB.S_avg, label="B", ls=:dash)
    plot!(p4, dC.t, dC.S_avg, label="C", ls=:dot)

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1100,750),
              plot_title="Quench Comparison — 2+1D U(1) LGT iPEPS (D=$DT)")
    savefig(fig, "quench_summary.png")
    println("  → Saved quench_summary.png")
    return fig
end

# =============================================================================
#  Main
# =============================================================================
function run_benchmarks()
    println("="^60)
    println("  iPEPS BENCHMARK: Quench Dynamics in 2+1D U(1) LGT")
    println("  d=$d, D_trunc=$DT, dt=$(DT_STEP), steps=$(N_STEPS)")
    println("="^60)

    t0 = time()

    println("\n--- Running Quench A ---")
    dataA = run_A()
    dfA = to_df(dataA)
    CSV.write("quench_A_data.csv", dfA)
    make_4panel(dfA, "Quench A: String Breaking", "quench_A")
    println("  Quench A done in $(round(time()-t0, digits=1))s\n")

    t1 = time()
    println("--- Running Quench B ---")
    dataB = run_B()
    dfB = to_df(dataB)
    CSV.write("quench_B_data.csv", dfB)
    make_4panel(dfB, "Quench B: Mass Quench m=5→0.1", "quench_B")
    println("  Quench B done in $(round(time()-t1, digits=1))s\n")

    t2 = time()
    println("--- Running Quench C ---")
    dataC = run_C()
    dfC = to_df(dataC)
    CSV.write("quench_C_data.csv", dfC)
    make_4panel(dfC, "Quench C: Coupling g²=4→0.5", "quench_C")
    println("  Quench C done in $(round(time()-t2, digits=1))s\n")

    make_summary(dfA, dfB, dfC)

    println("\n" * "="^60)
    println("  ALL BENCHMARKS COMPLETE")
    println("  Total wall time: $(round(time()-t0, digits=1))s")
    println("  Output files:")
    println("    quench_A_data.csv, quench_A_panels.png")
    println("    quench_B_data.csv, quench_B_panels.png")
    println("    quench_C_data.csv, quench_C_panels.png")
    println("    quench_summary.png")
    println("="^60)
end

run_benchmarks()
