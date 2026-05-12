# =============================================================================
#  gs_benchmark_collect.jl
#
#  Collect per-task CSVs from gs_benchmark.jl into one aggregate table and
#  produce summary plots: PEPS vs ED observables, and absolute differences,
#  versus mass m at fixed coupling g and vice versa.
#
#  Usage:
#      julia --project=. gs_benchmark_collect.jl [results/gs_bench]
# =============================================================================

ENV["GKSwstype"] = "nul"

using CSV, DataFrames, Statistics, Printf, Plots

default(fontfamily="Computer Modern", linewidth=2, framestyle=:box,
        grid=true, legend=:best, size=(800,500), dpi=200)

function _gather(dir::String)
    files = sort(filter(f -> startswith(f, "gs_benchmark_") && endswith(f, ".csv"),
                        readdir(dir)))
    isempty(files) && error("No per-task CSVs found in $dir")
    df = mapreduce(f -> CSV.read(joinpath(dir, f), DataFrame), vcat, files)
    sort!(df, [:m, :g])
    return df
end

function _plot_vs_m(df::DataFrame, prefix::String)
    g_vals = sort(unique(df.g))
    palette = cgrad(:viridis, length(g_vals); categorical=true)

    p1 = plot(title="(a) ⟨n_f⟩_even − ⟨n_f⟩_odd  (CDW)",
              xlabel="m", ylabel="CDW", xscale=:log10)
    p2 = plot(title="(b) ⟨E²⟩",
              xlabel="m", ylabel="⟨E²⟩", xscale=:log10)
    p3 = plot(title="(c) |Δ(CDW)|  PEPS−ED",
              xlabel="m", ylabel="|Δ|", xscale=:log10, yscale=:log10)
    p4 = plot(title="(d) |Δ⟨E²⟩|  PEPS−ED",
              xlabel="m", ylabel="|Δ|", xscale=:log10, yscale=:log10)

    for (i, gval) in enumerate(g_vals)
        sub = sort(df[df.g .== gval, :], :m)
        c   = palette[i]
        peps_cdw = sub.peps_nf_even .- sub.peps_nf_odd
        ed_cdw   = sub.ed_nf_even   .- sub.ed_nf_odd
        plot!(p1, sub.m, peps_cdw, label="PEPS g=$gval", color=c, lw=2)
        plot!(p1, sub.m, ed_cdw,   label="ED g=$gval",   color=c, lw=2, linestyle=:dash)
        plot!(p2, sub.m, sub.peps_E2_mean, label="PEPS g=$gval", color=c, lw=2)
        plot!(p2, sub.m, sub.ed_E2_mean,   label="ED g=$gval",   color=c, lw=2, linestyle=:dash)
        plot!(p3, sub.m, abs.(sub.d_cdw) .+ 1e-16, label="g=$gval", color=c, lw=2, marker=:circle)
        plot!(p4, sub.m, abs.(sub.d_E2_mean) .+ 1e-16, label="g=$gval", color=c, lw=2, marker=:circle)
    end

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
    out = prefix * "_vs_m.png"
    savefig(fig, out)
    println("  Saved: $out")
end

function _plot_vs_g(df::DataFrame, prefix::String)
    m_vals = sort(unique(df.m))
    palette = cgrad(:plasma, length(m_vals); categorical=true)

    p1 = plot(title="(a) ⟨n_f⟩_even − ⟨n_f⟩_odd  (CDW)",
              xlabel="g", ylabel="CDW", xscale=:log10)
    p2 = plot(title="(b) ⟨E²⟩",
              xlabel="g", ylabel="⟨E²⟩", xscale=:log10)
    p3 = plot(title="(c) |Δ(CDW)|  PEPS−ED",
              xlabel="g", ylabel="|Δ|", xscale=:log10, yscale=:log10)
    p4 = plot(title="(d) |Δ⟨E²⟩|  PEPS−ED",
              xlabel="g", ylabel="|Δ|", xscale=:log10, yscale=:log10)

    for (i, mval) in enumerate(m_vals)
        sub = sort(df[df.m .== mval, :], :g)
        c   = palette[i]
        peps_cdw = sub.peps_nf_even .- sub.peps_nf_odd
        ed_cdw   = sub.ed_nf_even   .- sub.ed_nf_odd
        plot!(p1, sub.g, peps_cdw, label="PEPS m=$mval", color=c, lw=2)
        plot!(p1, sub.g, ed_cdw,   label="ED m=$mval",   color=c, lw=2, linestyle=:dash)
        plot!(p2, sub.g, sub.peps_E2_mean, label="PEPS m=$mval", color=c, lw=2)
        plot!(p2, sub.g, sub.ed_E2_mean,   label="ED m=$mval",   color=c, lw=2, linestyle=:dash)
        plot!(p3, sub.g, abs.(sub.d_cdw) .+ 1e-16, label="m=$mval", color=c, lw=2, marker=:circle)
        plot!(p4, sub.g, abs.(sub.d_E2_mean) .+ 1e-16, label="m=$mval", color=c, lw=2, marker=:circle)
    end

    fig = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
    out = prefix * "_vs_g.png"
    savefig(fig, out)
    println("  Saved: $out")
end

function _plot_heatmap(df::DataFrame, prefix::String)
    m_vals = sort(unique(df.m))
    g_vals = sort(unique(df.g))
    Mcdw   = fill(NaN, length(m_vals), length(g_vals))
    Me2    = fill(NaN, length(m_vals), length(g_vals))
    for row in eachrow(df)
        i = findfirst(==(row.m), m_vals)
        j = findfirst(==(row.g), g_vals)
        Mcdw[i,j] = abs(row.d_cdw)
        Me2[i,j]  = abs(row.d_E2_mean)
    end
    Mcdw_log = log10.(Mcdw .+ 1e-16)
    Me2_log  = log10.(Me2  .+ 1e-16)
    h1 = heatmap(g_vals, m_vals, Mcdw_log, xlabel="g", ylabel="m",
                 title="log10 |Δ(CDW)|  PEPS−ED", c=:viridis)
    h2 = heatmap(g_vals, m_vals, Me2_log,  xlabel="g", ylabel="m",
                 title="log10 |Δ⟨E²⟩|  PEPS−ED", c=:viridis)
    fig = plot(h1, h2, layout=(1,2), size=(1400, 500))
    out = prefix * "_heatmap.png"
    savefig(fig, out)
    println("  Saved: $out")
end

function main()
    dir = length(ARGS) >= 1 ? ARGS[1] :
          joinpath(@__DIR__, "results", "gs_bench")
    println("Collecting from: $dir")
    df = _gather(dir)
    println("  $(nrow(df)) rows, $(length(unique(df.m))) masses × $(length(unique(df.g))) couplings")

    out_csv = joinpath(dir, "gs_benchmark_summary.csv")
    CSV.write(out_csv, df)
    println("  Saved aggregate: $out_csv")

    prefix = joinpath(dir, "gs_benchmark")
    _plot_vs_m(df, prefix)
    _plot_vs_g(df, prefix)
    _plot_heatmap(df, prefix)

    # Brief textual summary
    println("\n  Worst-case mismatches:")
    @printf("    max |Δ(CDW)|    = %.3e  at  m=%g  g=%g\n",
            maximum(abs, df.d_cdw),
            df.m[argmax(abs.(df.d_cdw))], df.g[argmax(abs.(df.d_cdw))])
    @printf("    max |Δ⟨E²⟩|    = %.3e  at  m=%g  g=%g\n",
            maximum(abs, df.d_E2_mean),
            df.m[argmax(abs.(df.d_E2_mean))], df.g[argmax(abs.(df.d_E2_mean))])
end

main()
