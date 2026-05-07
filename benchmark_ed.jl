# =============================================================================
#  Benchmark Quench Dynamics for 2+1D U(1) LGT
#
#  Uses EXACT diagonalization on a 2-site cluster (A–B NN pair)
#  to show real dynamics without iPEPS truncation artifacts.
#
#  The full 2-site Hilbert space has dimension d² = 64 (d=8 per site).
#  Time evolution: |ψ(t)⟩ = exp(-i H t) |ψ(0)⟩  (exact, no Trotter error).
#
#  Also runs iPEPS simple-update at D=4 for comparison on Quench A.
#
#  Quenches:
#    A. String Breaking: vacuum + flux string → evolve with full H
#    B. Mass Quench: GS at m=5 → evolve with m=0.1
#    C. Coupling Quench: GS at g²=4 → evolve with g²=0.5
# =============================================================================

include("tensorkit_tst.jl")

using Pkg
for pkg in ["Plots", "CSV", "DataFrames"]
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg)
end
using Plots, CSV, DataFrames

default(fontfamily="sans-serif", linewidth=2, framestyle=:box,
        grid=true, legend=:best, size=(800,500), dpi=200)

# =============================================================================
#  Exact 2-site Hamiltonian (FULL on-site terms, not 1/4 share)
#
#  For an isolated A–B dimer the Hamiltonian is:
#    H_dimer = H_mass_full + H_electric_full + H_hopping + H_plaquette_piece
#
#  On-site terms get their FULL weight (not split over 4 bonds):
#    H_mass     = m [(-1)^A n_f_A ⊗ I + (-1)^B I ⊗ n_f_B]
#    H_electric = (g²/2) [E²_total_A ⊗ I + I ⊗ E²_total_B]
#    H_hop      = -t [ψ†_A U_br_A ⊗ ψ_B + h.c.]
#    H_plaq_h   = -(1/2g²) [U_br_A ⊗ U_bu_B + h.c.]
# =============================================================================

"""
Build the FULL two-site dimer Hamiltonian (horizontal A–B pair).
On-site terms are NOT divided by 4 (this is the complete isolated dimer).
"""
function build_dimer_hamiltonian(;
    t_hop  = 1.0,
    m_mass = 0.5,
    g2     = 1.0,
    sign_A = 1,     # (-1)^{x+y} for site A
    include_plaquette = true,
)
    Id = Matrix{ComplexF64}(I, d, d)
    c  = fermion_c()
    cd = c'
    nf = fermion_n()
    Er = electric_field_right()
    Eu = electric_field_up()
    Ur = gauge_U_right()
    Uu = gauge_U_up()

    sign_B = -sign_A
    H = zeros(ComplexF64, d^2, d^2)

    # ── Full staggered mass ──────────────────────────────────────────────────
    H .+= m_mass * sign_A .* kron(nf, Id)
    H .+= m_mass * sign_B .* kron(Id, nf)

    # ── Full electric energy: g²/2 · (E_r² + E_u²) per site ─────────────────
    E2_site = Er^2 + Eu^2
    H .+= (g2 / 2) .* kron(E2_site, Id)
    H .+= (g2 / 2) .* kron(Id, E2_site)

    # ── Hopping across horizontal bond: ψ†_A U_br_A ψ_B + h.c. ─────────────
    H .+= -t_hop .* kron(cd * Ur, c)
    H .+= -t_hop .* kron(c * Ur', cd)

    # ── Plaquette piece on this bond ─────────────────────────────────────────
    if include_plaquette
        coeff = -1.0 / (2.0 * g2)
        H .+= coeff .* (kron(Ur, Uu) .+ kron(Ur', Uu'))
    end

    H .= 0.5 .* (H .+ H')
    return H
end

# =============================================================================
#  Exact time evolution on 2-site cluster
# =============================================================================

"""
Exact real-time evolution: |ψ(t)⟩ = exp(-i H t) |ψ(0)⟩.
Precomputes the eigendecomposition for efficiency.
"""
struct ExactEvolver
    evals::Vector{Float64}
    evecs::Matrix{ComplexF64}
    evecs_inv::Matrix{ComplexF64}
end

function ExactEvolver(H::AbstractMatrix)
    F = eigen(Hermitian(H))
    return ExactEvolver(F.values, F.vectors, F.vectors')
end

function evolve(ev::ExactEvolver, ψ::Vector{ComplexF64}, t::Float64)
    coeffs = ev.evecs_inv * ψ
    phases = exp.(-im .* ev.evals .* t)
    return ev.evecs * (phases .* coeffs)
end

"""Find the ground state of H."""
function ground_state(H::AbstractMatrix)
    F = eigen(Hermitian(H))
    return F.vectors[:, 1], F.values[1]
end

# =============================================================================
#  Observable measurement on exact state vector
# =============================================================================

function exact_expect(ψ::Vector{ComplexF64}, O::AbstractMatrix)
    return real(dot(ψ, O * ψ))
end

"""Build standard two-site observables as d²×d² operators."""
function build_observables()
    Id = Matrix{ComplexF64}(I, d, d)
    nf = fermion_n()
    Er = electric_field_right()
    Eu = electric_field_up()
    Ur = gauge_U_right()
    Uu = gauge_U_up()

    return (
        nf_A   = kron(nf, Id),
        nf_B   = kron(Id, nf),
        Er_A   = kron(Er, Id),          # electric flux on A's right link
        Er_B   = kron(Id, Er),          # electric flux on B's right link
        Eu_A   = kron(Eu, Id),          # electric flux on A's up link
        Eu_B   = kron(Id, Eu),          # electric flux on B's up link
        plaq_h = kron(Ur, Uu) + kron(Ur', Uu'),
    )
end

function exact_measure_all(ψ, obs, t_now)
    nf_A  = exact_expect(ψ, obs.nf_A)
    nf_B  = exact_expect(ψ, obs.nf_B)
    Er_A  = exact_expect(ψ, obs.Er_A)
    Er_B  = exact_expect(ψ, obs.Er_B)
    Eu_A  = exact_expect(ψ, obs.Eu_A)
    Eu_B  = exact_expect(ψ, obs.Eu_B)
    plaq  = exact_expect(ψ, obs.plaq_h)

    # Entanglement entropy from SVD of bipartite state
    ψ_mat = reshape(ψ, d, d)
    S_vals = svd(ψ_mat).S
    p = S_vals.^2
    p ./= sum(p)
    S_ent = -sum(x -> x > 1e-30 ? x * log(x) : 0.0, p)

    return (
        t = t_now,
        nf_A = nf_A, nf_B = nf_B,
        Er_A = Er_A, Er_B = Er_B,
        Eu_A = Eu_A, Eu_B = Eu_B,
        E_flux_avg = (Er_A + Er_B + Eu_A + Eu_B) / 4,
        plaq = plaq,
        S_ent = S_ent,
    )
end

function obs_to_df(data)
    DataFrame(
        t      = [r.t for r in data],
        nf_A   = [r.nf_A for r in data],
        nf_B   = [r.nf_B for r in data],
        Er_A   = [r.Er_A for r in data],
        Er_B   = [r.Er_B for r in data],
        Eu_A   = [r.Eu_A for r in data],
        Eu_B   = [r.Eu_B for r in data],
        E_flux_avg = [r.E_flux_avg for r in data],
        plaq   = [r.plaq for r in data],
        S_ent  = [r.S_ent for r in data],
    )
end

# =============================================================================
#  Quench A: String Breaking / Confinement-Deconfinement
#
#  Initial state: ground state of H with strong coupling g²=10 (confining)
#  Quench: suddenly reduce coupling to g²=0.5 (deconfining regime)
#  This releases the strongly-confined charge-anticharge pair into
#  a regime where they can propagate, breaking the electric flux string.
# =============================================================================
function run_quench_A(; t_hop=1.0, m_mass=0.5, g2_conf=10.0, g2_deconf=0.5,
                       T_final=8.0, n_pts=400)
    println("\n" * "="^65)
    println("  QUENCH A: String Breaking (Exact 2-site)")
    println("  t=$t_hop, m=$m_mass, g²: $g2_conf → $g2_deconf, T=$T_final")
    println("="^65)

    # Ground state in confining regime
    H_conf = build_dimer_hamiltonian(t_hop=t_hop, m_mass=m_mass, g2=g2_conf)
    ψ0, E0 = ground_state(H_conf)
    @printf("  Ground state energy (g²=%g): E0 = %.6f\n", g2_conf, E0)

    obs = build_observables()
    m0 = exact_measure_all(ψ0, obs, 0.0)
    @printf("  t=0: nf_A=%.4f, nf_B=%.4f, ⟨Er⟩_A=%.4f, ⟨□⟩=%.6f, S=%.4f\n",
            m0.nf_A, m0.nf_B, m0.Er_A, m0.plaq, m0.S_ent)

    # Evolve with deconfining Hamiltonian
    H_deconf = build_dimer_hamiltonian(t_hop=t_hop, m_mass=m_mass, g2=g2_deconf)
    ev = ExactEvolver(H_deconf)

    times = range(0, T_final, length=n_pts+1)
    data = [exact_measure_all(evolve(ev, ψ0, t), obs, t) for t in times]

    println("  Evolution complete.")
    return data
end

# =============================================================================
#  Quench B: Mass Quench (m → m')
# =============================================================================
function run_quench_B(; t_hop=1.0, m_init=5.0, m_final=0.1, g2=1.0,
                       T_final=8.0, n_pts=400)
    println("\n" * "="^65)
    println("  QUENCH B: Mass Quench  m=$m_init → $m_final (Exact 2-site)")
    println("  t=$t_hop, g²=$g2, T=$T_final")
    println("="^65)

    H_init = build_dimer_hamiltonian(t_hop=t_hop, m_mass=m_init, g2=g2)
    ψ0, E0 = ground_state(H_init)
    @printf("  Ground state energy (m=%g): E0 = %.6f\n", m_init, E0)

    obs = build_observables()
    m0 = exact_measure_all(ψ0, obs, 0.0)
    @printf("  t=0: nf_A=%.4f, nf_B=%.4f, ⟨Er⟩_A=%.4f, S=%.4f\n",
            m0.nf_A, m0.nf_B, m0.Er_A, m0.S_ent)

    H_final = build_dimer_hamiltonian(t_hop=t_hop, m_mass=m_final, g2=g2)
    ev = ExactEvolver(H_final)

    times = range(0, T_final, length=n_pts+1)
    data = [exact_measure_all(evolve(ev, ψ0, t), obs, t) for t in times]

    println("  Evolution complete.")
    return data
end

# =============================================================================
#  Quench C: Coupling Quench (g² → g'²)
# =============================================================================
function run_quench_C(; t_hop=1.0, m_mass=0.5, g2_init=4.0, g2_final=0.5,
                       T_final=8.0, n_pts=400)
    println("\n" * "="^65)
    println("  QUENCH C: Coupling Quench  g²=$g2_init → $g2_final (Exact 2-site)")
    println("  t=$t_hop, m=$m_mass, T=$T_final")
    println("="^65)

    H_init = build_dimer_hamiltonian(t_hop=t_hop, m_mass=m_mass, g2=g2_init)
    ψ0, E0 = ground_state(H_init)
    @printf("  Ground state energy (g²=%g): E0 = %.6f\n", g2_init, E0)

    obs = build_observables()
    m0 = exact_measure_all(ψ0, obs, 0.0)
    @printf("  t=0: nf_A=%.4f, nf_B=%.4f, ⟨Er⟩_A=%.4f, ⟨□⟩=%.6f, S=%.4f\n",
            m0.nf_A, m0.nf_B, m0.Er_A, m0.plaq, m0.S_ent)

    H_final = build_dimer_hamiltonian(t_hop=t_hop, m_mass=m_mass, g2=g2_final)
    ev = ExactEvolver(H_final)

    times = range(0, T_final, length=n_pts+1)
    data = [exact_measure_all(evolve(ev, ψ0, t), obs, t) for t in times]

    println("  Evolution complete.")
    return data
end

# =============================================================================
#  iPEPS comparison for Quench A
# =============================================================================
function run_quench_A_ipeps(; t_hop=1.0, m_mass=0.5, g2=1.0,
                             dt=0.02, n_steps=150, D_trunc=D_max)
    println("\n  iPEPS simple-update (D=$D_trunc) for Quench A...")

    peps = init_checkerboard(V_phys, V_bond;
                             nf_A=0, nbr_A=0, nbu_A=0,
                             nf_B=0, nbr_B=0, nbu_B=0)

    Ur_op = gauge_U_right(); cd_op = fermion_c()'
    A_arr = convert(Array, peps.A); B_arr = convert(Array, peps.B)
    dp = size(A_arr,1); Db = size(A_arr,2)
    A_new = reshape(Ur_op * reshape(A_arr, dp, :), size(A_arr))
    B_new = reshape(cd_op * reshape(B_arr, dp, :), size(B_arr))
    nrm = max(maximum(abs.(A_new)), maximum(abs.(B_new)), 1e-15)
    V_b = ℂ^Db
    peps.A = TensorMap(A_new./sqrt(nrm), V_phys, V_b⊗V_b⊗V_b⊗V_b)
    peps.B = TensorMap(B_new./sqrt(nrm), V_phys, V_b⊗V_b⊗V_b⊗V_b)

    H_h = build_horizontal_hamiltonian(t=t_hop, m=m_mass, g2=g2, sign_L=1)
    H_v = build_vertical_hamiltonian(t=t_hop, m=m_mass, g2=g2, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h'); H_v .= 0.5 .* (H_v .+ H_v')
    G_plaq_4 = plaquette_gate(build_plaquette_4site(g2=g2), -im*dt)

    nf_op = fermion_n()
    Er = electric_field_right(); Eu = electric_field_up()
    E2_op = Er^2 + Eu^2

    ts = Float64[0.0]
    nf_As = Float64[expect_onsite(convert(Array, peps.A), peps.λh, peps.λv, nf_op)]
    nf_Bs = Float64[expect_onsite(convert(Array, peps.B), peps.λh, peps.λv, nf_op)]
    E2s = Float64[(expect_onsite(convert(Array, peps.A), peps.λh, peps.λv, E2_op) +
                   expect_onsite(convert(Array, peps.B), peps.λh, peps.λv, E2_op)) / 2]

    p_h = peps.λh.^2; p_h ./= max(sum(p_h), 1e-30)
    Ss = Float64[-sum(x -> x > 1e-30 ? x*log(x) : 0.0, p_h)]

    for step in 1:n_steps
        realtime_trotter_step!(peps, H_h, H_v, dt, D_trunc; G_plaq_4site=G_plaq_4)
        if step % 5 == 0
            t_now = step * dt
            push!(ts, t_now)
            A_arr = convert(Array, peps.A); B_arr = convert(Array, peps.B)
            push!(nf_As, expect_onsite(A_arr, peps.λh, peps.λv, nf_op))
            push!(nf_Bs, expect_onsite(B_arr, peps.λh, peps.λv, nf_op))
            e2a = expect_onsite(A_arr, peps.λh, peps.λv, E2_op)
            e2b = expect_onsite(B_arr, peps.λh, peps.λv, E2_op)
            push!(E2s, (e2a + e2b) / 2)
            p_h = peps.λh.^2; p_h ./= max(sum(p_h), 1e-30)
            push!(Ss, -sum(x -> x > 1e-30 ? x*log(x) : 0.0, p_h))
            if step % 25 == 0
                @printf("    step %4d  t=%.2f  nf_A=%.4f  nf_B=%.4f  E²=%.4f  S=%.4f\n",
                        step, t_now, nf_As[end], nf_Bs[end], E2s[end], Ss[end])
            end
        end
    end

    return (t=ts, nf_A=nf_As, nf_B=nf_Bs, E2=E2s, S=Ss)
end

# =============================================================================
#  Plotting
# =============================================================================

function plot_quench_panels(df, title_str, prefix;
                            ipeps_data=nothing, ipeps_label="iPEPS D=$(D_max)")
    t = df.t

    p1 = plot(t, df.nf_A, label="⟨n_f⟩_A (exact)", xlabel="t", ylabel="⟨n_f⟩",
              title="(a) Fermion Density", color=:blue)
    plot!(p1, t, df.nf_B, label="⟨n_f⟩_B (exact)", color=:red, ls=:dash)
    if ipeps_data !== nothing
        plot!(p1, ipeps_data.t, ipeps_data.nf_A, label="nf_A ($ipeps_label)",
              color=:blue, ls=:dot, lw=1.5, marker=:circle, ms=2, alpha=0.6)
        plot!(p1, ipeps_data.t, ipeps_data.nf_B, label="nf_B ($ipeps_label)",
              color=:red, ls=:dot, lw=1.5, marker=:diamond, ms=2, alpha=0.6)
    end

    p2 = plot(t, df.Er_A, label="⟨E_r⟩_A (exact)", xlabel="t", ylabel="⟨E⟩",
              title="(b) Electric Flux", color=:blue, lw=2)
    plot!(p2, t, df.Er_B, label="⟨E_r⟩_B", color=:red, ls=:dash)
    plot!(p2, t, df.Eu_A, label="⟨E_u⟩_A", color=:cyan, ls=:dot)
    plot!(p2, t, df.Eu_B, label="⟨E_u⟩_B", color=:orange, ls=:dashdot)

    p3 = plot(t, df.plaq, label="⟨□⟩ (exact)", xlabel="t", ylabel="⟨□⟩",
              title="(c) Plaquette Expectation", color=:purple, lw=2.5)

    p4 = plot(t, df.S_ent, label="S_ent (exact)", xlabel="t", ylabel="S_vN",
              title="(d) Entanglement Entropy", color=:black, lw=2.5)

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800),
              plot_title=title_str)
    savefig(fig, "$(prefix)_panels.png")
    println("  Saved: $(prefix)_panels.png")
    return fig
end

function plot_summary(dfA, dfB, dfC)
    # A: fermion density oscillation (string breaking signature)
    p1 = plot(dfA.t, dfA.nf_A, label="nf_A", xlabel="t", ylabel="⟨n_f⟩",
              title="A: String Breaking", color=:blue, lw=2)
    plot!(p1, dfA.t, dfA.nf_B, label="nf_B", color=:red, ls=:dash)

    # B: CDW order parameter
    cdw = dfB.nf_A .- dfB.nf_B
    p2 = plot(dfB.t, cdw, label="nf_A - nf_B", xlabel="t", ylabel="CDW",
              title="B: Mass Quench (CDW)", color=:black, lw=2)
    plot!(p2, dfB.t, dfB.nf_A, label="nf_A", color=:blue, ls=:dash, alpha=0.5)
    plot!(p2, dfB.t, dfB.nf_B, label="nf_B", color=:red, ls=:dash, alpha=0.5)

    # C: Plaquette growth
    p3 = plot(dfC.t, dfC.plaq, label="⟨□⟩", xlabel="t", ylabel="⟨□⟩",
              title="C: Coupling Quench (Plaq)", color=:purple, lw=2)
    plot!(p3, dfC.t, dfC.E_flux_avg, label="⟨E⟩", color=:red, ls=:dash)

    # Entanglement entropy comparison
    p4 = plot(dfA.t, dfA.S_ent, label="A: String", xlabel="t", ylabel="S_vN",
              title="Entanglement Entropy", color=:blue, lw=2)
    plot!(p4, dfB.t, dfB.S_ent, label="B: Mass", color=:red, ls=:dash)
    plot!(p4, dfC.t, dfC.S_ent, label="C: Coupling", color=:green, ls=:dot)

    # E flux comparison (linear scale)
    p5 = plot(dfA.t, dfA.E_flux_avg, label="A", xlabel="t", ylabel="⟨E⟩ avg",
              title="Electric Flux Comparison", color=:blue, lw=2)
    plot!(p5, dfB.t, dfB.E_flux_avg, label="B", color=:red, ls=:dash)
    plot!(p5, dfC.t, dfC.E_flux_avg, label="C", color=:green, ls=:dot)

    # Plaquette comparison
    p6 = plot(dfA.t, dfA.plaq, label="A", xlabel="t", ylabel="⟨□⟩",
              title="Plaquette Comparison", color=:blue, lw=2)
    plot!(p6, dfB.t, dfB.plaq, label="B", color=:red, ls=:dash)
    plot!(p6, dfC.t, dfC.plaq, label="C", color=:green, ls=:dot)

    fig = plot(p1, p2, p3, p4, p5, p6, layout=(2,3), size=(1500, 800),
              plot_title="Quench Benchmark — 2+1D U(1) LGT (Exact 2-site, d=$d, d_b=$d_b)")
    savefig(fig, "quench_summary.png")
    println("  Saved: quench_summary.png")
    return fig
end

# =============================================================================
#  Main
# =============================================================================
function run_all()
    println("="^65)
    println("  QUENCH BENCHMARK: 2+1D U(1) LGT")
    println("  d_f=$d_f, d_b=$d_b, d=$d per site, d²=$(d^2) two-site space")
    println("  Method: exact diagonalization (2-site cluster)")
    println("="^65)

    t0 = time()

    # ── Quench A: String Breaking ─────────────────────────────────────────────
    dataA = run_quench_A(t_hop=1.0, m_mass=0.5, g2_conf=10.0, g2_deconf=0.5,
                         T_final=8.0, n_pts=400)
    dfA = obs_to_df(dataA)
    CSV.write("quench_A_data.csv", dfA)

    plot_quench_panels(dfA, "Quench A: String Breaking (t=1, m=0.5, g²=1)",
                       "quench_A")
    println("  Quench A done in $(round(time()-t0, digits=1))s")

    # ── Quench B: Mass Quench ─────────────────────────────────────────────────
    t1 = time()
    dataB = run_quench_B(t_hop=1.0, m_init=5.0, m_final=0.1, g2=1.0,
                         T_final=8.0, n_pts=400)
    dfB = obs_to_df(dataB)
    CSV.write("quench_B_data.csv", dfB)
    plot_quench_panels(dfB, "Quench B: Mass Quench m=5 → 0.1", "quench_B")
    println("  Quench B done in $(round(time()-t1, digits=1))s")

    # ── Quench C: Coupling Quench ─────────────────────────────────────────────
    t2 = time()
    dataC = run_quench_C(t_hop=1.0, m_mass=0.5, g2_init=4.0, g2_final=0.5,
                         T_final=8.0, n_pts=400)
    dfC = obs_to_df(dataC)
    CSV.write("quench_C_data.csv", dfC)
    plot_quench_panels(dfC, "Quench C: Coupling Quench g²=4 → 0.5", "quench_C")
    println("  Quench C done in $(round(time()-t2, digits=1))s")

    # ── Summary figure ────────────────────────────────────────────────────────
    plot_summary(dfA, dfB, dfC)

    println("\n" * "="^65)
    println("  ALL BENCHMARKS COMPLETE  (total: $(round(time()-t0, digits=1))s)")
    println("  Output files:")
    println("    quench_A_panels.png, quench_A_data.csv")
    println("    quench_B_panels.png, quench_B_data.csv")
    println("    quench_C_panels.png, quench_C_data.csv")
    println("    quench_summary.png")
    println("="^65)
end

run_all()
