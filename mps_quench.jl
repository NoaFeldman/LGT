#= ═══════════════════════════════════════════════════════════════════════════════
   mps_quench.jl

   Real-time quench benchmark of the snake-MPS solver (mps_lgt.jl) against exact
   diagonalisation (finite_ed.jl) on the 3×4 U(1) LGT, for the SAME three
   protocols as finite_peps_quench.jl / finite_ed.jl:

     A. String breaking : vacuum + electric string → evolve (g=1,t=1,m=2)
     B. Mass quench      : GS at m_init=5 → evolve with m_final=0.1
     C. Coupling quench  : GS at g_init=2 → evolve with g_final=0.5

   Both ED and MPS are run here with IDENTICAL dt / n_steps, so the time grids
   match exactly for overlay.  MPS time stepping uses the global-Krylov
   (Lanczos) stepper.  Ground states (B,C) come from Gauss-penalized DMRG (MPS)
   and find_ground_state (ED), both in the staggered sector.

   SLURM array: task 1/2/3 → quench A/B/C.

   Usage:
       julia --project=. mps_quench.jl 1          # quench A
       sbatch run_mps_quench.sh
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "mps_lgt.jl"))
include(joinpath(@__DIR__, "finite_ed.jl"))

using CSV, DataFrames, Printf

const Q_NX, Q_NY, Q_DG = 3, 4, 1
const Q_THOP   = 1.0
const Q_DT     = 0.05
const Q_NSTEPS = 40        # t_final = 2.0
const Q_DMAX   = 120       # evolution bond cap
const Q_KRYLOV = 8
const Q_DGS    = 80        # DMRG bond (B,C initial state)
const Q_NSW    = 10
const Q_LAM    = 30.0

# ── MPS real-time loop ───────────────────────────────────────────────────────
function evolve_mps(ψ0::CMPS, H::CMPO; dt::Float64, n_steps::Int, Dmax::Int)
    ψ = deepcopy(ψ0); ψ[1] ./= mps_norm(ψ)
    data = Any[measure_mps(ψ, Q_NX, Q_NY, Q_DG; t_now=0.0)]
    @printf("  step  t       ⟨n_f⟩    ⟨n_f⟩_e  ⟨n_f⟩_o   ⟨E²⟩    bond\n")
    @printf("  %4d  %5.2f   %.4f   %.4f   %.4f   %.4f   %d\n",
            0, 0.0, data[1].nf_mean, data[1].nf_even, data[1].nf_odd, data[1].E2_mean,
            maximum(size(t, 2) for t in ψ))
    for step in 1:n_steps
        ψ = krylov_evolve_step(H, ψ, dt; m=Q_KRYLOV, Dmax=Dmax)
        obs = measure_mps(ψ, Q_NX, Q_NY, Q_DG; t_now=step * dt)
        push!(data, obs)
        if step % 5 == 0 || step == n_steps
            @printf("  %4d  %5.2f   %.4f   %.4f   %.4f   %.4f   %d\n",
                    step, step * dt, obs.nf_mean, obs.nf_even, obs.nf_odd, obs.E2_mean,
                    maximum(size(t, 2) for t in ψ))
        end
        flush(stdout)
    end
    return data
end

function write_mps_csv(data, label, results_dir)
    df = DataFrame(
        t       = [d.t        for d in data],
        nf_mean = [d.nf_mean  for d in data],
        nf_even = [d.nf_even  for d in data],
        nf_odd  = [d.nf_odd   for d in data],
        E2_mean = [d.E2_mean  for d in data],
    )
    out = joinpath(results_dir, "$(label)_data.csv")
    CSV.write(out, df)
    println("  Saved: $out")
end

# ── Quench A: string breaking ────────────────────────────────────────────────
function run_quench_A_mps(results_dir)
    println("\n" * "="^70 * "\n  MPS QUENCH A: String Breaking\n" * "="^70)
    g, m = 1.0, 2.0
    ψ0 = string_breaking_mps(Q_NX, Q_NY, Q_DG)
    H  = build_H_mpo(Q_NX, Q_NY, Q_DG; g=g, t=Q_THOP, m=m)
    data = evolve_mps(ψ0, H; dt=Q_DT, n_steps=Q_NSTEPS, Dmax=Q_DMAX)
    write_mps_csv(data, "mps_quench_A", results_dir)
end

# ── Quench B: mass quench ────────────────────────────────────────────────────
function run_quench_B_mps(results_dir; m_init=5.0, m_final=0.1)
    println("\n" * "="^70 * "\n  MPS QUENCH B: Mass Quench  m=$m_init→$m_final\n" * "="^70)
    g = 1.0
    gch = staggered_charges(Q_NX, Q_NY)
    dims = node_dims(Q_NX, Q_NY, Q_DG)
    println("  preparing GS at m=$m_init (penalized DMRG) ...")
    Hpen = build_penalized_H(Q_NX, Q_NY, Q_DG; g=g, t=Q_THOP, m=m_init, gauss_g=gch, Λ=Q_LAM)
    _, ψ0 = dmrg_ground_state(Hpen, dims; D=Q_DGS, nsweeps=Q_NSW, verbose=true)
    H = build_H_mpo(Q_NX, Q_NY, Q_DG; g=g, t=Q_THOP, m=m_final)
    data = evolve_mps(ψ0, H; dt=Q_DT, n_steps=Q_NSTEPS, Dmax=Q_DMAX)
    write_mps_csv(data, "mps_quench_B", results_dir)
end

# ── Quench C: coupling quench ────────────────────────────────────────────────
function run_quench_C_mps(results_dir; g_init=2.0, g_final=0.5)
    println("\n" * "="^70 * "\n  MPS QUENCH C: Coupling Quench  g=$g_init→$g_final\n" * "="^70)
    m = 2.0
    gch = staggered_charges(Q_NX, Q_NY)
    dims = node_dims(Q_NX, Q_NY, Q_DG)
    println("  preparing GS at g=$g_init (penalized DMRG) ...")
    Hpen = build_penalized_H(Q_NX, Q_NY, Q_DG; g=g_init, t=Q_THOP, m=m, gauss_g=gch, Λ=Q_LAM)
    _, ψ0 = dmrg_ground_state(Hpen, dims; D=Q_DGS, nsweeps=Q_NSW, verbose=true)
    H = build_H_mpo(Q_NX, Q_NY, Q_DG; g=g_final, t=Q_THOP, m=m)
    data = evolve_mps(ψ0, H; dt=Q_DT, n_steps=Q_NSTEPS, Dmax=Q_DMAX)
    write_mps_csv(data, "mps_quench_C", results_dir)
end

# ── Main: also run the aligned ED reference for the same quench ──────────────
function main()
    quench_id = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) :
                parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1"))
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)

    if quench_id == 1
        run_quench_A_mps(results_dir)
        run_quench_A_ED(; dt=Q_DT, n_steps=Q_NSTEPS, results_dir=results_dir)
    elseif quench_id == 2
        run_quench_B_mps(results_dir; m_init=5.0, m_final=0.1)
        run_quench_B_ED(; m_init=5.0, m_final=0.1, dt=Q_DT, n_steps=Q_NSTEPS, results_dir=results_dir)
    elseif quench_id == 3
        run_quench_C_mps(results_dir; g_init=2.0, g_final=0.5)
        run_quench_C_ED(; g_init=2.0, g_final=0.5, dt=Q_DT, n_steps=Q_NSTEPS, results_dir=results_dir)
    else
        error("quench_id must be 1 (A), 2 (B), or 3 (C)")
    end
    println("\n  Done.  MPS + ED CSVs written to $results_dir")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
