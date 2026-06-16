#= ═══════════════════════════════════════════════════════════════════════════════
   gs_benchmark_mps_collect.jl

   Aggregate the per-task CSVs from gs_benchmark_mps.jl into one summary table
   and print an ED-vs-MPS comparison.

   Usage:
       julia --project=. gs_benchmark_mps_collect.jl            # uses results/
       julia --project=. gs_benchmark_mps_collect.jl /path/dir
   ═══════════════════════════════════════════════════════════════════════════ =#

using CSV, DataFrames, Printf

results_dir = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "results")

rows = DataFrame[]
for f in sort(readdir(results_dir; join=true))
    occursin(r"gs_mps_task\d+\.csv$", f) || continue
    push!(rows, CSV.read(f, DataFrame))
end
isempty(rows) && error("No gs_mps_task*.csv files found in $results_dir")

df = sort(vcat(rows...), [:m, :g])
out = joinpath(results_dir, "gs_mps_summary.csv")
CSV.write(out, df)
println("Saved: $out\n")

@printf("  %-5s %-5s | %-12s %-12s %-10s | %-8s %-8s | %-9s %-6s\n",
        "m", "g", "E_ed", "E_mps", "ΔE/|E|", "nf_ed", "nf_mps", "Gviol", "bond")
println("  " * "─"^92)
for r in eachrow(df)
    @printf("  %-5.2f %-5.2f | %-12.6f %-12.6f %-10.2e | %-8.4f %-8.4f | %-9.1e %-6d\n",
            r.m, r.g, r.E_ed, r.E_mps, r.dE_rel,
            r.nf_mean_ed, r.nf_mean_mps, r.gauss_viol, r.bond)
end
@printf("\n  worst ΔE/|E| = %.2e   max Gauss violation = %.2e\n",
        maximum(df.dE_rel), maximum(df.gauss_viol))
