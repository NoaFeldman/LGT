#= ═══════════════════════════════════════════════════════════════════════════════
   matter_gauge_entanglement_collect.jl

   Aggregate the per-task CSVs from matter_gauge_entanglement.jl into one summary
   table and plot, for every (m,g) point of the gs_benchmark grid:

     • the MATTER half-system entanglement, and
     • the GAUGE dual-plaquette half-system entanglement,

   of the Bender–Zohar-decoupled 3×4 U(1) LGT ground state, as a function of the
   gauge coupling g (one curve per fermion mass m).  Two diagnostic panels report
   the decoupling quality (plaquette purity and source-free weight).

   Usage:
       julia --project=. matter_gauge_entanglement_collect.jl            # uses results/
       julia --project=. matter_gauge_entanglement_collect.jl /path/dir
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

using CSV, DataFrames, Printf, Plots

default(fontfamily="Computer Modern", linewidth=2, framestyle=:box,
        grid=true, legend=:best, size=(1200, 900), dpi=200)

results_dir = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "results")

rows = DataFrame[]
for f in sort(readdir(results_dir; join=true))
    occursin(r"mg_ent_task\d+\.csv$", f) || continue
    push!(rows, CSV.read(f, DataFrame))
end
isempty(rows) && error("No mg_ent_task*.csv files found in $results_dir")

df = sort(vcat(rows...), [:m, :g])
out_csv = joinpath(results_dir, "mg_ent_summary.csv")
CSV.write(out_csv, df)
println("Saved: $out_csv\n")

# ── console table ─────────────────────────────────────────────────────────────
@printf("  %-5s %-5s | %-10s %-10s | %-8s %-9s | %-9s %-6s\n",
        "m", "g", "S_matter", "S_gauge", "purity", "srcfree", "ΔE/|E|", "bond")
println("  " * "─"^78)
for r in eachrow(df)
    @printf("  %-5.2f %-5.2f | %-10.5f %-10.5f | %-8.4f %-9.4f | %-9.1e %-6d\n",
            r.m, r.g, r.S_matter, r.S_gauge, r.purity, r.srcfree_weight, r.dE_rel, r.bond)
end
@printf("\n  worst ΔE/|E| = %.2e   min srcfree weight = %.4f   min purity = %.4f\n",
        maximum(df.dE_rel), minimum(df.srcfree_weight), minimum(df.purity))

# ── figure ────────────────────────────────────────────────────────────────────
ms   = unique(df.m)
cols = [:steelblue, :darkorange, :seagreen, :purple]

p1 = plot(title="(a) Matter half-system entanglement", xlabel="g", ylabel="S_matter (nats)")
p2 = plot(title="(b) Gauge dual-plaquette half-system entanglement", xlabel="g", ylabel="S_gauge (nats)")
p3 = plot(title="(c) Plaquette purity (decoupling quality)", xlabel="g", ylabel="leading ρ_plaq eigenvalue")
p4 = plot(title="(d) Source-free weight (decoupling quality)", xlabel="g", ylabel="tr ρ")
for (i, mval) in enumerate(ms)
    sub = sort(df[df.m .== mval, :], :g)
    c = cols[mod1(i, length(cols))]
    plot!(p1, sub.g, sub.S_matter,        label="m=$mval", color=c, marker=:circle)
    plot!(p2, sub.g, sub.S_gauge,         label="m=$mval", color=c, marker=:diamond)
    plot!(p3, sub.g, sub.purity,          label="m=$mval", color=c, marker=:circle)
    plot!(p4, sub.g, sub.srcfree_weight,  label="m=$mval", color=c, marker=:circle)
end

fig = plot(p1, p2, p3, p4, layout=(2, 2),
           plot_title="Decoupled 3×4 U(1) LGT — matter vs dual-plaquette gauge entanglement")
out_png = joinpath(results_dir, "mg_entanglement.png")
savefig(fig, out_png)
println("\n  Saved: $out_png")
