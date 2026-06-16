#= ═══════════════════════════════════════════════════════════════════════════════
   plot_ed_vs_mps.jl

   Figures comparing the snake-MPS solver to exact diagonalisation on the 3×4
   U(1) LGT, mirroring plot_ed_peps_comparison.jl but for MPS vs ED:

     • Ground state  : reads results/gs_mps_summary.csv (from
                       gs_benchmark_mps_collect.jl) → energy + observable panels
                       over the 9 (m,g) points.
     • Quenches A/B/C: overlays results/finite_ed_quench_X_data.csv and
                       results/mps_quench_X_data.csv → 4-panel time series.

   Usage:
       julia --project=. plot_ed_vs_mps.jl            # uses results/
       julia --project=. plot_ed_vs_mps.jl /path/dir
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

using CSV, DataFrames, Plots, Printf

default(fontfamily="Computer Modern", linewidth=2, framestyle=:box,
        grid=true, legend=:best, size=(1200, 800), dpi=200)

results_dir = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "results")

load_if(path) = isfile(path) ? CSV.read(path, DataFrame) : (@warn("missing: $path"); nothing)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Ground-state benchmark                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function plot_gs()
    df = load_if(joinpath(results_dir, "gs_mps_summary.csv"))
    isnothing(df) && (println("  (no GS summary — run gs_benchmark_mps_collect.jl)"); return)
    sort!(df, [:m, :g])
    ms = unique(df.m)
    cols = [:steelblue, :seagreen, :darkorange, :purple]

    p1 = plot(title="(a) Ground-state energy", xlabel="g", ylabel="E", legend=:topright)
    p2 = plot(title="(b) Relative energy error", xlabel="g", ylabel="|E_mps − E_ed|/|E_ed|",
              yscale=:log10)
    p3 = plot(title="(c) Mean fermion density", xlabel="g", ylabel="⟨n_f⟩")
    p4 = plot(title="(d) Mean electric energy ⟨E²⟩", xlabel="g", ylabel="⟨E²⟩")
    for (i, mval) in enumerate(ms)
        sub = df[df.m .== mval, :]
        c = cols[mod1(i, length(cols))]
        plot!(p1, sub.g, sub.E_ed,  label="ED m=$mval",  color=c, marker=:circle)
        plot!(p1, sub.g, sub.E_mps, label="MPS m=$mval", color=c, ls=:dash, marker=:x)
        plot!(p2, sub.g, max.(sub.dE_rel, 1e-16), label="m=$mval", color=c, marker=:circle)
        plot!(p3, sub.g, sub.nf_mean_ed,  label="ED m=$mval",  color=c, marker=:circle)
        plot!(p3, sub.g, sub.nf_mean_mps, label="MPS m=$mval", color=c, ls=:dash, marker=:x)
        plot!(p4, sub.g, sub.E2_mean_ed,  label="ED m=$mval",  color=c, marker=:circle)
        plot!(p4, sub.g, sub.E2_mean_mps, label="MPS m=$mval", color=c, ls=:dash, marker=:x)
    end
    fig = plot(p1, p2, p3, p4, layout=(2, 2), plot_title="Ground state — MPS vs ED (3×4)")
    out = joinpath(results_dir, "gs_mps_vs_ed.png")
    savefig(fig, out); println("  Saved: $out")
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Quench overlays                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function plot_quench(tag, name)
    df_ed  = load_if(joinpath(results_dir, "finite_ed_quench_$(tag)_data.csv"))
    df_mps = load_if(joinpath(results_dir, "mps_quench_$(tag)_data.csv"))
    (isnothing(df_ed) && isnothing(df_mps)) && return

    p1 = plot(title="(a) Mean fermion density", xlabel="t", ylabel="⟨n_f⟩")
    p2 = plot(title="(b) Mean electric energy ⟨E²⟩", xlabel="t", ylabel="⟨E²⟩")
    p3 = plot(title="(c) CDW order ⟨n_f⟩_e − ⟨n_f⟩_o", xlabel="t", ylabel="⟨n_f⟩_e − ⟨n_f⟩_o")
    p4 = plot(title="(d) Sublattice fermion densities", xlabel="t", ylabel="⟨n_f⟩")
    if !isnothing(df_mps)
        plot!(p1, df_mps.t, df_mps.nf_mean, label="MPS", color=:steelblue)
        plot!(p2, df_mps.t, df_mps.E2_mean, label="MPS", color=:steelblue)
        plot!(p3, df_mps.t, df_mps.nf_even .- df_mps.nf_odd, label="MPS", color=:steelblue)
        plot!(p4, df_mps.t, df_mps.nf_even, label="MPS even", color=:steelblue)
        plot!(p4, df_mps.t, df_mps.nf_odd,  label="MPS odd",  color=:steelblue, ls=:dash)
    end
    if !isnothing(df_ed)
        plot!(p1, df_ed.t, df_ed.nf_mean, label="ED", color=:firebrick, ls=:dash)
        plot!(p2, df_ed.t, df_ed.E2_mean, label="ED", color=:firebrick, ls=:dash)
        plot!(p3, df_ed.t, df_ed.nf_even .- df_ed.nf_odd, label="ED", color=:firebrick, ls=:dash)
        plot!(p4, df_ed.t, df_ed.nf_even, label="ED even", color=:firebrick)
        plot!(p4, df_ed.t, df_ed.nf_odd,  label="ED odd",  color=:firebrick, ls=:dash)
    end
    hline!(p1, [0.5]; color=:gray, ls=:dot, label="")
    hline!(p3, [0.0]; color=:gray, ls=:dot, label="")
    fig = plot(p1, p2, p3, p4, layout=(2, 2),
               plot_title="Quench $tag: $name — MPS vs ED (3×4)")
    out = joinpath(results_dir, "quench_$(tag)_mps_vs_ed.png")
    savefig(fig, out); println("  Saved: $out")
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Driver                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

println("─"^60 * "\n  Ground-state benchmark\n" * "─"^60)
plot_gs()
for (tag, name) in [("A", "String Breaking"), ("B", "Mass Quench"), ("C", "Coupling Quench")]
    println("─"^60 * "\n  Quench $tag: $name\n" * "─"^60)
    plot_quench(tag, name)
end
println("\nDone.")
