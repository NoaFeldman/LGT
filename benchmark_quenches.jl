# =============================================================================
#  Benchmark Quench Simulations for 2+1D U(1) Lattice Gauge Theory
#
#  Three quench protocols with observable tracking:
#    A. Vacuum Decay / String Breaking
#    B. Mass Quench  (m → m')
#    C. Coupling Quench (g → g')
#
#  Observables:
#    1. Local charge density  ⟨Q_i(t)⟩
#    2. Electric energy density  ⟨E²(t)⟩
#    3. Plaquette expectation value  ⟨□(t)⟩
#    4. Bond entanglement entropy  S_vN(t)
#
#  Generates CSV data files and plots via Plots.jl.
# =============================================================================

# ── Load the main iPEPS LGT code ─────────────────────────────────────────────
include("tensorkit_tst.jl")

using Pkg
for pkg in ["Plots", "CSV", "DataFrames"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

using Plots
using CSV
using DataFrames

# Make plots look clean
default(
    fontfamily  = "Computer Modern",
    linewidth   = 2,
    framestyle  = :box,
    grid        = true,
    legend      = :best,
    size        = (800, 500),
    dpi         = 200,
)

# =============================================================================
#  Additional Observable Functions
# =============================================================================

"""
Electric energy density: ⟨E²⟩ = ⟨E_right² + E_up²⟩ per site.
Averaged over A and B sublattices.
"""
function measure_electric_energy(peps::CheckerboardiPEPS)
    Er = electric_field_right()
    Eu = electric_field_up()
    E2_op = Er^2 + Eu^2

    A_arr = convert(Array, peps.A)
    B_arr = convert(Array, peps.B)
    E2_A = expect_onsite(A_arr, peps.λh, peps.λv, E2_op)
    E2_B = expect_onsite(B_arr, peps.λh, peps.λv, E2_op)
    return (E2_A = E2_A, E2_B = E2_B, E2_avg = (E2_A + E2_B) / 2)
end

"""
Plaquette expectation value: ⟨□⟩ = ⟨U_bottom U_right U†_top U†_left + h.c.⟩.
Measured on the horizontal bond as a proxy (the two-site piece of the plaquette).
"""
function measure_plaquette(peps::CheckerboardiPEPS)
    Ur = gauge_U_right()
    Uu = gauge_U_up()

    # Horizontal plaquette piece: ⟨U_br(A) ⊗ U_bu(B) + h.c.⟩
    plaq_h = real(expect_twosite_h(peps, Ur, Uu)) +
             real(expect_twosite_h(peps, Ur', Uu'))

    # Vertical plaquette piece: ⟨U†_bu(A) ⊗ U†_br(B) + h.c.⟩
    plaq_v = real(expect_twosite_v(peps, Uu', Ur')) +
             real(expect_twosite_v(peps, Uu, Ur))

    return (plaq_h = plaq_h, plaq_v = plaq_v, plaq_avg = (plaq_h + plaq_v) / 2)
end

"""
Charge density: ⟨Q_i⟩ = ⟨n_f⟩ - staggered_offset.
  A sites (even): q_background = 0  → Q_A = ⟨n_f⟩_A
  B sites (odd):  q_background = 1  → Q_B = ⟨n_f⟩_B - 1
"""
function measure_charge_density(peps::CheckerboardiPEPS)
    nf = fermion_n()
    A_arr = convert(Array, peps.A)
    B_arr = convert(Array, peps.B)
    nf_A = expect_onsite(A_arr, peps.λh, peps.λv, nf)
    nf_B = expect_onsite(B_arr, peps.λh, peps.λv, nf)
    Q_A = nf_A           # staggered offset for even site = 0
    Q_B = nf_B - 1.0     # staggered offset for odd site = 1
    return (Q_A = Q_A, Q_B = Q_B, nf_A = nf_A, nf_B = nf_B)
end

"""
Bond entanglement entropy from the Vidal singular values λ.
  S_vN = -Σ_i p_i log(p_i),  where p_i = λ_i² / Σ λ_j².
This is the entanglement entropy of the bipartition across the bond.
"""
function bond_entropy(λ::Vector{Float64})
    p = λ .^ 2
    p_sum = sum(p)
    if p_sum < 1e-30
        return 0.0
    end
    p ./= p_sum
    S = 0.0
    for pi in p
        if pi > 1e-30
            S -= pi * log(pi)
        end
    end
    return S
end

"""
Entanglement entropies on horizontal and vertical bonds.
"""
function measure_entanglement(peps::CheckerboardiPEPS)
    S_h = bond_entropy(peps.λh)
    S_v = bond_entropy(peps.λv)
    return (S_h = S_h, S_v = S_v, S_avg = (S_h + S_v) / 2)
end

"""
Measure all observables and return a NamedTuple.
"""
function measure_all(peps::CheckerboardiPEPS, t::Float64)
    ch = measure_charge_density(peps)
    el = measure_electric_energy(peps)
    pl = measure_plaquette(peps)
    en = measure_entanglement(peps)
    gl = measure_gauss_law(peps)

    return (
        t      = t,
        Q_A    = ch.Q_A,
        Q_B    = ch.Q_B,
        nf_A   = ch.nf_A,
        nf_B   = ch.nf_B,
        E2_A   = el.E2_A,
        E2_B   = el.E2_B,
        E2_avg = el.E2_avg,
        plaq_h = pl.plaq_h,
        plaq_v = pl.plaq_v,
        plaq_avg = pl.plaq_avg,
        S_h    = en.S_h,
        S_v    = en.S_v,
        S_avg  = en.S_avg,
        G_avg  = gl.avg_G,
        G_var  = gl.var_G,
    )
end

# =============================================================================
#  Helper: prepare iPEPS ground state via imaginary-time evolution
# =============================================================================
"""
    prepare_ground_state(; t_hop, m_mass, g2, τ_ite, n_ite, D_trunc,
                           nf_A, nbr_A, nbu_A, nf_B, nbr_B, nbu_B)

Run imaginary-time evolution to approximate the ground state.
Returns the converged `CheckerboardiPEPS`.
"""
function prepare_ground_state(;
    t_hop   = 1.0,
    m_mass  = 0.5,
    g2      = 1.0,
    τ_ite   = 0.05,
    n_ite   = 200,
    D_trunc = D_max,
    nf_A    = 1,  nbr_A = 0,  nbu_A = 0,
    nf_B    = 0,  nbr_B = 0,  nbu_B = 0,
    verbose = true,
)
    H_h = build_horizontal_hamiltonian(t=t_hop, m=m_mass, g2=g2, sign_L=1)
    H_v = build_vertical_hamiltonian(t=t_hop, m=m_mass, g2=g2, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h')
    H_v .= 0.5 .* (H_v .+ H_v')

    H_plaq_4 = build_plaquette_4site(g2=g2)
    G_plaq_4 = plaquette_gate(H_plaq_4, -τ_ite)

    peps = init_checkerboard(V_phys, V_bond;
                             nf_A=nf_A, nbr_A=nbr_A, nbu_A=nbu_A,
                             nf_B=nf_B, nbr_B=nbr_B, nbu_B=nbu_B)

    nf_mat = fermion_n()
    for step in 1:n_ite
        trotter_step!(peps, H_h, H_v, τ_ite, D_trunc; G_plaq_4site=G_plaq_4)

        if verbose && (step % 50 == 0 || step == 1)
            A_arr = convert(Array, peps.A)
            B_arr = convert(Array, peps.B)
            nf_A_val = expect_onsite(A_arr, peps.λh, peps.λv, nf_mat)
            nf_B_val = expect_onsite(B_arr, peps.λh, peps.λv, nf_mat)
            @printf("    ITE step %4d: ⟨nf⟩_A=%.4f  ⟨nf⟩_B=%.4f  λh=%s\n",
                    step, nf_A_val, nf_B_val,
                    string(round.(peps.λh[1:min(3,end)], digits=4)))
        end
    end

    return peps
end

# =============================================================================
#  Helper: apply a local operator on the A-sublattice tensor
# =============================================================================
"""
Apply a single-site operator O (d×d matrix) on the A tensor of the iPEPS.
This modifies the tensor in-place: T'[p',l,r,u,d] = O[p',p] T[p,l,r,u,d].
"""
function apply_onsite_operator!(peps::CheckerboardiPEPS, O::AbstractMatrix;
                                sublattice::Symbol=:A)
    if sublattice == :A
        T_arr = convert(Array, peps.A)
    else
        T_arr = convert(Array, peps.B)
    end
    dp = size(T_arr, 1)
    D_bond = size(T_arr, 2)
    T_mat = reshape(T_arr, dp, :)
    T_new = O * T_mat
    T_arr_new = reshape(T_new, size(T_arr))

    V_b = ℂ^D_bond
    if sublattice == :A
        peps.A = TensorMap(T_arr_new, V_phys, V_b ⊗ V_b ⊗ V_b ⊗ V_b)
    else
        peps.B = TensorMap(T_arr_new, V_phys, V_b ⊗ V_b ⊗ V_b ⊗ V_b)
    end
    return nothing
end

# =============================================================================
#  Quench A:  Vacuum Decay / String Breaking
#
#  Initial state: bare vacuum (all empty fermions, zero flux)
#  Action: Create a particle on B (odd site) and apply U_right on A to
#          create an electric flux string on the A–B horizontal bond.
#          This mimics ψ†_B · U_{br,A} applied to vacuum.
#  Evolve with the full Hamiltonian and watch string breaking.
# =============================================================================
function run_quench_A_string_breaking(;
    t_hop   = 1.0,
    m_mass  = 0.5,
    g2      = 1.0,
    dt      = 0.02,
    n_steps = 200,
    D_trunc = D_max,
)
    println("=" ^ 72)
    println("  QUENCH A: Vacuum Decay / String Breaking")
    println("  dt=$dt, steps=$n_steps, D_trunc=$D_trunc")
    println("  t=$t_hop, m=$m_mass, g²=$g2")
    println("=" ^ 72)

    # ── Initial state: bare vacuum |0,0,0⟩ on all sites ──────────────────────
    peps = init_checkerboard(V_phys, V_bond;
                             nf_A=0, nbr_A=0, nbu_A=0,
                             nf_B=0, nbr_B=0, nbu_B=0)

    # ── Create string: apply ψ†_B · U_{br,A} ─────────────────────────────────
    # On A: apply U_right (creates flux +1 on horizontal link)
    # On B: apply ψ† (creates fermion)
    # This creates state: |0,1,0⟩_A ⊗ |1,0,0⟩_B with a flux string.
    println("  Creating string: ψ†_B · U_{br,A} on vacuum...")
    Ur = gauge_U_right()
    cd = fermion_c()'
    apply_onsite_operator!(peps, Ur; sublattice=:A)
    apply_onsite_operator!(peps, cd; sublattice=:B)

    # Normalise after operator application
    A_arr = convert(Array, peps.A)
    B_arr = convert(Array, peps.B)
    nrm = max(maximum(abs.(A_arr)), maximum(abs.(B_arr)), 1e-15)
    D_bond = size(A_arr, 2)
    V_b = ℂ^D_bond
    peps.A = TensorMap(A_arr ./ sqrt(nrm), V_phys, V_b ⊗ V_b ⊗ V_b ⊗ V_b)
    peps.B = TensorMap(B_arr ./ sqrt(nrm), V_phys, V_b ⊗ V_b ⊗ V_b ⊗ V_b)

    # ── Build Hamiltonians ────────────────────────────────────────────────────
    H_h = build_horizontal_hamiltonian(t=t_hop, m=m_mass, g2=g2, sign_L=1)
    H_v = build_vertical_hamiltonian(t=t_hop, m=m_mass, g2=g2, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h')
    H_v .= 0.5 .* (H_v .+ H_v')
    H_plaq_4 = build_plaquette_4site(g2=g2)
    G_plaq_4 = plaquette_gate(H_plaq_4, -im * dt)

    # ── Measure initial state ─────────────────────────────────────────────────
    data = [measure_all(peps, 0.0)]
    println("  t=0: Q_A=$(round(data[1].Q_A, digits=4)), " *
            "Q_B=$(round(data[1].Q_B, digits=4)), " *
            "⟨E²⟩=$(round(data[1].E2_avg, digits=4)), " *
            "⟨□⟩=$(round(data[1].plaq_avg, digits=4))")

    # ── Real-time evolution ───────────────────────────────────────────────────
    println("\n  Evolving...")
    println("  step  |    t       Q_A       Q_B      ⟨E²⟩      ⟨□⟩       S_avg     ΔG²")
    println("  ──────┼────────────────────────────────────────────────────────────────────")
    for step in 1:n_steps
        realtime_trotter_step!(peps, H_h, H_v, dt, D_trunc; G_plaq_4site=G_plaq_4)
        t_now = step * dt
        obs = measure_all(peps, t_now)
        push!(data, obs)

        if step % 10 == 0 || step == 1
            @printf("  %4d  | %6.2f  %+.4f   %+.4f   %.4f   %+.4f   %.4f   %.2e\n",
                    step, t_now, obs.Q_A, obs.Q_B, obs.E2_avg,
                    obs.plaq_avg, obs.S_avg, obs.G_var)
        end
    end

    return data
end

# =============================================================================
#  Quench B:  Mass Quench  (m_init → m_final)
#
#  Initial state: ground state at large mass m_init (fermions localized, CDW).
#  Quench: suddenly switch to small mass m_final.
#  Dynamics: plasma oscillations between matter and gauge sectors.
# =============================================================================
function run_quench_B_mass(;
    t_hop    = 1.0,
    m_init   = 5.0,     # large initial mass → CDW ground state
    m_final  = 0.1,     # quench to small mass
    g2       = 1.0,
    dt       = 0.02,
    n_steps  = 200,
    D_trunc  = D_max,
    n_ite    = 200,      # imaginary-time steps for ground state preparation
    τ_ite    = 0.05,
)
    println("=" ^ 72)
    println("  QUENCH B: Mass Quench  m=$m_init → m=$m_final")
    println("  dt=$dt, steps=$n_steps, D_trunc=$D_trunc")
    println("  t=$t_hop, g²=$g2")
    println("=" ^ 72)

    # ── Prepare ground state at large mass ────────────────────────────────────
    println("\n  Preparing ground state at m=$m_init via ITE ($n_ite steps)...")
    peps = prepare_ground_state(
        t_hop=t_hop, m_mass=m_init, g2=g2,
        τ_ite=τ_ite, n_ite=n_ite, D_trunc=D_trunc,
        nf_A=1, nbr_A=0, nbu_A=0,
        nf_B=0, nbr_B=0, nbu_B=0,
    )

    # ── Build quenched Hamiltonians  (m_final) ────────────────────────────────
    H_h = build_horizontal_hamiltonian(t=t_hop, m=m_final, g2=g2, sign_L=1)
    H_v = build_vertical_hamiltonian(t=t_hop, m=m_final, g2=g2, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h')
    H_v .= 0.5 .* (H_v .+ H_v')
    H_plaq_4 = build_plaquette_4site(g2=g2)
    G_plaq_4 = plaquette_gate(H_plaq_4, -im * dt)

    # ── Measure initial state ─────────────────────────────────────────────────
    data = [measure_all(peps, 0.0)]
    println("\n  Initial (GS at m=$m_init):")
    println("    Q_A=$(round(data[1].Q_A, digits=4)), Q_B=$(round(data[1].Q_B, digits=4))")
    println("    ⟨E²⟩=$(round(data[1].E2_avg, digits=4)), ⟨□⟩=$(round(data[1].plaq_avg, digits=4))")

    # ── Real-time evolution with quenched mass ────────────────────────────────
    println("\n  Evolving with m=$m_final...")
    println("  step  |    t       nf_A      nf_B     ⟨E²⟩      ⟨□⟩       S_avg     ΔG²")
    println("  ──────┼────────────────────────────────────────────────────────────────────")
    for step in 1:n_steps
        realtime_trotter_step!(peps, H_h, H_v, dt, D_trunc; G_plaq_4site=G_plaq_4)
        t_now = step * dt
        obs = measure_all(peps, t_now)
        push!(data, obs)

        if step % 10 == 0 || step == 1
            @printf("  %4d  | %6.2f  %.4f   %.4f   %.4f   %+.4f   %.4f   %.2e\n",
                    step, t_now, obs.nf_A, obs.nf_B, obs.E2_avg,
                    obs.plaq_avg, obs.S_avg, obs.G_var)
        end
    end

    return data
end

# =============================================================================
#  Quench C:  Coupling Quench  (g² → g'²)
#
#  Initial state: ground state at strong coupling g²_init (large electric cost).
#  Quench: lower to g²_final → plaquette term becomes important.
#  Dynamics: emergence of magnetic flux loops, entanglement spreading.
# =============================================================================
function run_quench_C_coupling(;
    t_hop     = 1.0,
    m_mass    = 0.5,
    g2_init   = 4.0,     # strong coupling: electric dominates
    g2_final  = 0.5,     # weak coupling: magnetic plaquette dominates
    dt        = 0.02,
    n_steps   = 200,
    D_trunc   = D_max,
    n_ite     = 200,
    τ_ite     = 0.05,
)
    println("=" ^ 72)
    println("  QUENCH C: Coupling Quench  g²=$g2_init → g²=$g2_final")
    println("  dt=$dt, steps=$n_steps, D_trunc=$D_trunc")
    println("  t=$t_hop, m=$m_mass")
    println("=" ^ 72)

    # ── Prepare ground state at strong coupling ───────────────────────────────
    println("\n  Preparing ground state at g²=$g2_init via ITE ($n_ite steps)...")
    peps = prepare_ground_state(
        t_hop=t_hop, m_mass=m_mass, g2=g2_init,
        τ_ite=τ_ite, n_ite=n_ite, D_trunc=D_trunc,
        nf_A=1, nbr_A=0, nbu_A=0,
        nf_B=0, nbr_B=0, nbu_B=0,
    )

    # ── Build quenched Hamiltonians (g²_final) ────────────────────────────────
    H_h = build_horizontal_hamiltonian(t=t_hop, m=m_mass, g2=g2_final, sign_L=1)
    H_v = build_vertical_hamiltonian(t=t_hop, m=m_mass, g2=g2_final, sign_D=1)
    H_h .= 0.5 .* (H_h .+ H_h')
    H_v .= 0.5 .* (H_v .+ H_v')
    H_plaq_4 = build_plaquette_4site(g2=g2_final)
    G_plaq_4 = plaquette_gate(H_plaq_4, -im * dt)

    # ── Measure initial state ─────────────────────────────────────────────────
    data = [measure_all(peps, 0.0)]
    println("\n  Initial (GS at g²=$g2_init):")
    println("    ⟨E²⟩=$(round(data[1].E2_avg, digits=4)), ⟨□⟩=$(round(data[1].plaq_avg, digits=4))")
    println("    S_h=$(round(data[1].S_h, digits=4)), S_v=$(round(data[1].S_v, digits=4))")

    # ── Real-time evolution with quenched coupling ────────────────────────────
    println("\n  Evolving with g²=$g2_final...")
    println("  step  |    t       nf_A      nf_B     ⟨E²⟩      ⟨□⟩       S_avg     ΔG²")
    println("  ──────┼────────────────────────────────────────────────────────────────────")
    for step in 1:n_steps
        realtime_trotter_step!(peps, H_h, H_v, dt, D_trunc; G_plaq_4site=G_plaq_4)
        t_now = step * dt
        obs = measure_all(peps, t_now)
        push!(data, obs)

        if step % 10 == 0 || step == 1
            @printf("  %4d  | %6.2f  %.4f   %.4f   %.4f   %+.4f   %.4f   %.2e\n",
                    step, t_now, obs.nf_A, obs.nf_B, obs.E2_avg,
                    obs.plaq_avg, obs.S_avg, obs.G_var)
        end
    end

    return data
end

# =============================================================================
#  Plotting Functions
# =============================================================================

function data_to_dataframe(data::Vector, label::String)
    df = DataFrame(
        t        = [d.t for d in data],
        Q_A      = [d.Q_A for d in data],
        Q_B      = [d.Q_B for d in data],
        nf_A     = [d.nf_A for d in data],
        nf_B     = [d.nf_B for d in data],
        E2_A     = [d.E2_A for d in data],
        E2_B     = [d.E2_B for d in data],
        E2_avg   = [d.E2_avg for d in data],
        plaq_h   = [d.plaq_h for d in data],
        plaq_v   = [d.plaq_v for d in data],
        plaq_avg = [d.plaq_avg for d in data],
        S_h      = [d.S_h for d in data],
        S_v      = [d.S_v for d in data],
        S_avg    = [d.S_avg for d in data],
        G_avg    = [d.G_avg for d in data],
        G_var    = [d.G_var for d in data],
    )
    return df
end

"""
    plot_quench_A(data)

Generate 4-panel figure for the string-breaking quench.
"""
function plot_quench_A(data; save_prefix="quench_A")
    df = data_to_dataframe(data, "A")
    CSV.write("$(save_prefix)_data.csv", df)

    t = df.t

    # Panel 1: Charge density
    p1 = plot(t, df.Q_A, label="Q_A (even)", xlabel="t", ylabel="⟨Q⟩",
              title="(a) Charge Density — String Breaking")
    plot!(p1, t, df.Q_B, label="Q_B (odd)", linestyle=:dash)
    hline!(p1, [0.0], color=:gray, linestyle=:dot, label="")

    # Panel 2: Electric energy density
    p2 = plot(t, df.E2_A, label="⟨E²⟩_A", xlabel="t", ylabel="⟨E²⟩",
              title="(b) Electric Energy Density")
    plot!(p2, t, df.E2_B, label="⟨E²⟩_B", linestyle=:dash)
    plot!(p2, t, df.E2_avg, label="average", color=:black, linewidth=2.5)

    # Panel 3: Plaquette expectation
    p3 = plot(t, df.plaq_avg, label="⟨□⟩", xlabel="t", ylabel="⟨□⟩",
              title="(c) Plaquette Expectation", color=:purple)
    plot!(p3, t, df.plaq_h, label="horizontal", linestyle=:dash, color=:blue)
    plot!(p3, t, df.plaq_v, label="vertical", linestyle=:dot, color=:red)

    # Panel 4: Entanglement entropy
    p4 = plot(t, df.S_h, label="S_h (horizontal)", xlabel="t", ylabel="S_vN",
              title="(d) Bond Entanglement Entropy")
    plot!(p4, t, df.S_v, label="S_v (vertical)", linestyle=:dash)
    plot!(p4, t, df.S_avg, label="S_avg", color=:black, linewidth=2.5)

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800),
              plot_title="Quench A: Vacuum Decay / String Breaking")
    savefig(fig, "$(save_prefix)_panels.png")
    println("  Saved: $(save_prefix)_panels.png")

    # Gauss law violation panel (separate)
    p_gl = plot(t, abs.(df.G_var), label="|ΔG²|", xlabel="t", ylabel="|ΔG²|",
                title="Gauss Law Violation — Quench A", yscale=:log10,
                ylims=(1e-16, 1.0), color=:red)
    savefig(p_gl, "$(save_prefix)_gauss_law.png")
    println("  Saved: $(save_prefix)_gauss_law.png")

    return fig
end

"""
    plot_quench_B(data; m_init, m_final)

Generate 4-panel figure for the mass quench.
"""
function plot_quench_B(data; m_init=5.0, m_final=0.1, save_prefix="quench_B")
    df = data_to_dataframe(data, "B")
    CSV.write("$(save_prefix)_data.csv", df)

    t = df.t

    # Panel 1: Fermion density (plasma oscillations)
    p1 = plot(t, df.nf_A, label="⟨n_f⟩_A (even)", xlabel="t", ylabel="⟨n_f⟩",
              title="(a) Fermion Density — m=$m_init→$m_final")
    plot!(p1, t, df.nf_B, label="⟨n_f⟩_B (odd)", linestyle=:dash)
    # CDW order parameter: (nf_A - nf_B)
    cdw = df.nf_A .- df.nf_B
    plot!(p1, t, cdw, label="CDW: nf_A - nf_B", color=:black, linewidth=2.5)

    # Panel 2: Electric energy density (gauge sector response)
    p2 = plot(t, df.E2_avg, label="⟨E²⟩", xlabel="t", ylabel="⟨E²⟩",
              title="(b) Electric Energy Density", color=:red, linewidth=2.5)
    plot!(p2, t, df.E2_A, label="⟨E²⟩_A", linestyle=:dash, color=:blue)
    plot!(p2, t, df.E2_B, label="⟨E²⟩_B", linestyle=:dot, color=:green)

    # Panel 3: Plaquette (magnetic coherence development)
    p3 = plot(t, df.plaq_avg, label="⟨□⟩ avg", xlabel="t", ylabel="⟨□⟩",
              title="(c) Plaquette (Magnetic Coherence)", color=:purple,
              linewidth=2.5)

    # Panel 4: Entanglement entropy growth
    p4 = plot(t, df.S_avg, label="S_avg", xlabel="t", ylabel="S_vN",
              title="(d) Entanglement Entropy Growth", color=:black,
              linewidth=2.5)
    plot!(p4, t, df.S_h, label="S_h", linestyle=:dash)
    plot!(p4, t, df.S_v, label="S_v", linestyle=:dot)

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800),
              plot_title="Quench B: Mass Quench m=$m_init → $m_final")
    savefig(fig, "$(save_prefix)_panels.png")
    println("  Saved: $(save_prefix)_panels.png")

    # Gauss law
    p_gl = plot(t, abs.(df.G_var), label="|ΔG²|", xlabel="t", ylabel="|ΔG²|",
                title="Gauss Law Violation — Quench B", yscale=:log10,
                ylims=(1e-16, 1.0), color=:red)
    savefig(p_gl, "$(save_prefix)_gauss_law.png")
    println("  Saved: $(save_prefix)_gauss_law.png")

    return fig
end

"""
    plot_quench_C(data; g2_init, g2_final)

Generate 4-panel figure for the coupling quench.
"""
function plot_quench_C(data; g2_init=4.0, g2_final=0.5, save_prefix="quench_C")
    df = data_to_dataframe(data, "C")
    CSV.write("$(save_prefix)_data.csv", df)

    t = df.t

    # Panel 1: Plaquette expectation (magnetic flux loops emerging)
    p1 = plot(t, df.plaq_avg, label="⟨□⟩ avg", xlabel="t", ylabel="⟨□⟩",
              title="(a) Plaquette — g²=$g2_init→$g2_final", color=:purple,
              linewidth=2.5)
    plot!(p1, t, df.plaq_h, label="⟨□⟩_h", linestyle=:dash, color=:blue)
    plot!(p1, t, df.plaq_v, label="⟨□⟩_v", linestyle=:dot, color=:red)

    # Panel 2: Electric energy (should decrease as magnetic grows)
    p2 = plot(t, df.E2_avg, label="⟨E²⟩ avg", xlabel="t", ylabel="⟨E²⟩",
              title="(b) Electric Energy Density", color=:red, linewidth=2.5)

    # Panel 3: Charge density
    p3 = plot(t, df.Q_A, label="Q_A", xlabel="t", ylabel="⟨Q⟩",
              title="(c) Charge Density")
    plot!(p3, t, df.Q_B, label="Q_B", linestyle=:dash)

    # Panel 4: Entanglement entropy (spreading)
    p4 = plot(t, df.S_avg, label="S_avg", xlabel="t", ylabel="S_vN",
              title="(d) Entanglement Entropy Spreading", color=:black,
              linewidth=2.5)
    plot!(p4, t, df.S_h, label="S_h", linestyle=:dash)
    plot!(p4, t, df.S_v, label="S_v", linestyle=:dot)

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800),
              plot_title="Quench C: Coupling Quench g²=$g2_init → $g2_final")
    savefig(fig, "$(save_prefix)_panels.png")
    println("  Saved: $(save_prefix)_panels.png")

    # Gauss law
    p_gl = plot(t, abs.(df.G_var), label="|ΔG²|", xlabel="t", ylabel="|ΔG²|",
                title="Gauss Law Violation — Quench C", yscale=:log10,
                ylims=(1e-16, 1.0), color=:red)
    savefig(p_gl, "$(save_prefix)_gauss_law.png")
    println("  Saved: $(save_prefix)_gauss_law.png")

    return fig
end

# =============================================================================
#  Summary comparison figure: overlay key observables from all three quenches
# =============================================================================
function plot_summary(data_A, data_B, data_C; save_prefix="quench_summary")
    df_A = data_to_dataframe(data_A, "A")
    df_B = data_to_dataframe(data_B, "B")
    df_C = data_to_dataframe(data_C, "C")

    # Panel 1: Electric energy comparison
    p1 = plot(df_A.t, df_A.E2_avg, label="String Breaking", xlabel="t",
              ylabel="⟨E²⟩", title="(a) Electric Energy Density")
    plot!(p1, df_B.t, df_B.E2_avg, label="Mass Quench", linestyle=:dash)
    plot!(p1, df_C.t, df_C.E2_avg, label="Coupling Quench", linestyle=:dot)

    # Panel 2: Plaquette comparison
    p2 = plot(df_A.t, df_A.plaq_avg, label="String Breaking", xlabel="t",
              ylabel="⟨□⟩", title="(b) Plaquette Expectation")
    plot!(p2, df_B.t, df_B.plaq_avg, label="Mass Quench", linestyle=:dash)
    plot!(p2, df_C.t, df_C.plaq_avg, label="Coupling Quench", linestyle=:dot)

    # Panel 3: Entanglement entropy comparison
    p3 = plot(df_A.t, df_A.S_avg, label="String Breaking", xlabel="t",
              ylabel="S_vN", title="(c) Entanglement Entropy")
    plot!(p3, df_B.t, df_B.S_avg, label="Mass Quench", linestyle=:dash)
    plot!(p3, df_C.t, df_C.S_avg, label="Coupling Quench", linestyle=:dot)

    # Panel 4: Gauss law violation comparison
    p4 = plot(df_A.t, abs.(df_A.G_var) .+ 1e-16, label="String Breaking",
              xlabel="t", ylabel="|ΔG²|", title="(d) Gauss Law Violation",
              yscale=:log10, ylims=(1e-16, 1.0))
    plot!(p4, df_B.t, abs.(df_B.G_var) .+ 1e-16, label="Mass Quench",
          linestyle=:dash)
    plot!(p4, df_C.t, abs.(df_C.G_var) .+ 1e-16, label="Coupling Quench",
          linestyle=:dot)

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800),
              plot_title="Summary: Three Quench Protocols for 2+1D U(1) LGT")
    savefig(fig, "$(save_prefix).png")
    println("  Saved: $(save_prefix).png")

    return fig
end

# =============================================================================
#  Main benchmark runner
# =============================================================================
function run_all_benchmarks()
    println("\n" * "=" ^ 72)
    println("  iPEPS BENCHMARK: Quench Dynamics in 2+1D U(1) LGT")
    println("  d_f=$d_f, d_b=$d_b, d=$d, D_max=$D_max")
    println("=" ^ 72 * "\n")

    # Common parameters
    dt_rt   = 0.02
    n_rt    = 200
    D_tr    = D_max

    # ── Quench A: String Breaking ─────────────────────────────────────────────
    data_A = run_quench_A_string_breaking(
        t_hop=1.0, m_mass=0.5, g2=1.0,
        dt=dt_rt, n_steps=n_rt, D_trunc=D_tr)
    fig_A = plot_quench_A(data_A)

    println()

    # ── Quench B: Mass Quench ─────────────────────────────────────────────────
    data_B = run_quench_B_mass(
        t_hop=1.0, m_init=5.0, m_final=0.1, g2=1.0,
        dt=dt_rt, n_steps=n_rt, D_trunc=D_tr,
        n_ite=200, τ_ite=0.05)
    fig_B = plot_quench_B(data_B; m_init=5.0, m_final=0.1)

    println()

    # ── Quench C: Coupling Quench ─────────────────────────────────────────────
    data_C = run_quench_C_coupling(
        t_hop=1.0, m_mass=0.5, g2_init=4.0, g2_final=0.5,
        dt=dt_rt, n_steps=n_rt, D_trunc=D_tr,
        n_ite=200, τ_ite=0.05)
    fig_C = plot_quench_C(data_C; g2_init=4.0, g2_final=0.5)

    println()

    # ── Summary figure ────────────────────────────────────────────────────────
    println("  Generating summary comparison figure...")
    fig_summary = plot_summary(data_A, data_B, data_C)

    println("\n" * "=" ^ 72)
    println("  BENCHMARK COMPLETE")
    println("  Output files:")
    println("    quench_A_panels.png, quench_A_gauss_law.png, quench_A_data.csv")
    println("    quench_B_panels.png, quench_B_gauss_law.png, quench_B_data.csv")
    println("    quench_C_panels.png, quench_C_gauss_law.png, quench_C_data.csv")
    println("    quench_summary.png")
    println("=" ^ 72)

    return (data_A=data_A, data_B=data_B, data_C=data_C)
end

# ── Run ───────────────────────────────────────────────────────────────────────
run_all_benchmarks()
