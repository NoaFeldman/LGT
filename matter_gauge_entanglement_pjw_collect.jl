#= ═══════════════════════════════════════════════════════════════════════════════
   matter_gauge_entanglement_pjw_collect.jl

   Aggregate the per-task CSVs from matter_gauge_entanglement_pjw.jl and produce
   the SAME figure as mg_entanglement.png (matter + dual-plaquette gauge
   half-system entanglement, with the pre-decoupling reference and the decoupling-
   quality diagnostics), for the plaquette+JW Hamiltonian.

   Usage:
       julia --project=. matter_gauge_entanglement_pjw_collect.jl            # uses results/
       julia --project=. matter_gauge_entanglement_pjw_collect.jl /path/dir
   ═══════════════════════════════════════════════════════════════════════════ =#

ENV["GKSwstype"] = "nul"

using CSV, DataFrames, Printf, Plots

default(fontfamily="Computer Modern", linewidth=2, framestyle=:box,
        grid=true, legend=:best, size=(1200, 900), dpi=200)

results_dir = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "results")

rows = DataFrame[]
for f in sort(readdir(results_dir; join=true))
    occursin(r"mg_ent_pjw_task\d+\.csv$", f) || continue
    push!(rows, CSV.read(f, DataFrame))
end
isempty(rows) && error("No mg_ent_pjw_task*.csv files found in $results_dir")

df = sort(vcat(rows...), [:m, :g])
out_csv = joinpath(results_dir, "mg_ent_pjw_summary.csv")
CSV.write(out_csv, df)
println("Saved: $out_csv\n")

# ── console table ─────────────────────────────────────────────────────────────
@printf("  %-5s %-5s | %-9s %-9s %-9s | %-8s %-9s | %-9s\n",
        "m", "g", "S_pre", "S_matter", "S_gauge", "purity", "srcfree", "Gviol")
println("  " * "─"^80)
for r in eachrow(df)
    @printf("  %-5.2f %-5.2f | %-9.5f %-9.5f %-9.5f | %-8.4f %-9.4f | %-9.1e\n",
            r.m, r.g, r.S_pre, r.S_matter, r.S_gauge, r.purity, r.srcfree_weight, r.gauss_viol)
end
@printf("\n  max Gauss violation = %.2e   min srcfree weight = %.4f   min purity = %.4f\n",
        maximum(df.gauss_viol), minimum(df.srcfree_weight), minimum(df.purity))

# ── figure (same layout as mg_entanglement.png) ───────────────────────────────
ms   = unique(df.m)
cols = [:steelblue, :darkorange, :seagreen, :purple]

p1 = plot(title="(a) Matter half-system entanglement", xlabel="g", ylabel="S (nats)")
p2 = plot(title="(b) Gauge dual-plaquette half-system entanglement", xlabel="g", ylabel="S (nats)")
p3 = plot(title="(c) Plaquette purity (decoupling quality)", xlabel="g", ylabel="leading ρ_plaq eigenvalue")
p4 = plot(title="(d) Source-free weight (decoupling quality)", xlabel="g", ylabel="tr ρ")
for (i, mval) in enumerate(ms)
    sub = sort(df[df.m .== mval, :], :g)
    c = cols[mod1(i, length(cols))]
    plot!(p1, sub.g, sub.S_pre, label="m=$mval (pre)", color=c, ls=:dash, marker=:utriangle, alpha=0.6)
    plot!(p2, sub.g, sub.S_pre, label="m=$mval (pre)", color=c, ls=:dash, marker=:utriangle, alpha=0.6)
    plot!(p1, sub.g, sub.S_matter,       label="m=$mval", color=c, marker=:circle)
    plot!(p2, sub.g, sub.S_gauge,        label="m=$mval", color=c, marker=:diamond)
    plot!(p3, sub.g, sub.purity,         label="m=$mval", color=c, marker=:circle)
    plot!(p4, sub.g, sub.srcfree_weight, label="m=$mval", color=c, marker=:circle)
end

fig = plot(p1, p2, p3, p4, layout=(2, 2),
           plot_title="Decoupled 3×4 U(1) LGT (plaquette + JW) — matter vs dual-plaquette gauge entanglement")
out_png = joinpath(results_dir, "mg_entanglement_pjw.png")
savefig(fig, out_png)
println("\n  Saved: $out_png")
