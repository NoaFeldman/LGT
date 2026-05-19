# =============================================================================
#  plot_ed_peps_comparison.jl
#
#  Overlay ED and PEPS quench results on the same figures.
#  Reads CSVs from the results/ directory (or the directory passed as ARGS[1]).
#
#  Usage:
#    julia --project=. plot_ed_peps_comparison.jl            # uses results/
#    julia --project=. plot_ed_peps_comparison.jl /path/dir  # explicit dir
# =============================================================================

ENV["GKSwstype"] = "nul"

using CSV, DataFrames, Plots, Printf

default(fontfamily="Computer Modern", linewidth=2, framestyle=:box,
        grid=true, legend=:topright, size=(1200, 800), dpi=200)

results_dir = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "results")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Helpers                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

function load_if_exists(path::String)
    if isfile(path)
        return CSV.read(path, DataFrame)
    else
        @warn "File not found, skipping: $path"
        return nothing
    end
end

"""
Four-panel comparison for one quench.
Panels: (a) ⟨n_f⟩ mean, (b) ⟨E²⟩ mean, (c) CDW order, (d) sublattice densities.
ED has no entanglement entropy so the entropy panel is omitted here.
"""
function plot_comparison(df_peps, df_ed;
                          quench_label::String,
                          prefix::String)

    has_peps = !isnothing(df_peps)
    has_ed   = !isnothing(df_ed)

    # ── (a) Mean fermion density ──────────────────────────────────────────
    p1 = plot(title="(a) Mean Fermion Density", xlabel="t", ylabel="⟨n_f⟩")
    if has_peps
        plot!(p1, df_peps.t, df_peps.nf_mean,
              label="PEPS", color=:steelblue, lw=2.5)
    end
    if has_ed
        plot!(p1, df_ed.t, df_ed.nf_mean,
              label="ED", color=:firebrick, lw=2, linestyle=:dash)
    end
    hline!(p1, [0.5]; color=:gray, linestyle=:dot, label="")

    # ── (b) Electric energy ───────────────────────────────────────────────
    p2 = plot(title="(b) Mean Electric Energy ⟨E²⟩", xlabel="t", ylabel="⟨E²⟩")
    if has_peps
        plot!(p2, df_peps.t, df_peps.E2_mean,
              label="PEPS", color=:steelblue, lw=2.5)
    end
    if has_ed
        plot!(p2, df_ed.t, df_ed.E2_mean,
              label="ED", color=:firebrick, lw=2, linestyle=:dash)
    end

    # ── (c) CDW order parameter ───────────────────────────────────────────
    p3 = plot(title="(c) CDW Order  ⟨n_f⟩_even − ⟨n_f⟩_odd", xlabel="t",
              ylabel="⟨n_f⟩_e − ⟨n_f⟩_o")
    if has_peps
        cdw_p = df_peps.nf_even .- df_peps.nf_odd
        plot!(p3, df_peps.t, cdw_p, label="PEPS", color=:steelblue, lw=2.5)
    end
    if has_ed
        cdw_e = df_ed.nf_even .- df_ed.nf_odd
        plot!(p3, df_ed.t, cdw_e,   label="ED",   color=:firebrick, lw=2, linestyle=:dash)
    end
    hline!(p3, [0.0]; color=:gray, linestyle=:dot, label="")

    # ── (d) Sublattice densities ──────────────────────────────────────────
    p4 = plot(title="(d) Sublattice Fermion Densities", xlabel="t", ylabel="⟨n_f⟩")
    if has_peps
        plot!(p4, df_peps.t, df_peps.nf_even, label="PEPS even",
              color=:steelblue, lw=2)
        plot!(p4, df_peps.t, df_peps.nf_odd,  label="PEPS odd",
              color=:steelblue, lw=2, linestyle=:dash)
    end
    if has_ed
        plot!(p4, df_ed.t, df_ed.nf_even, label="ED even",
              color=:firebrick, lw=2)
        plot!(p4, df_ed.t, df_ed.nf_odd,  label="ED odd",
              color=:firebrick, lw=2, linestyle=:dash)
    end

    fig = plot(p1, p2, p3, p4, layout=(2, 2),
               plot_title="$quench_label — ED vs PEPS comparison",
               size=(1200, 800))

    out = "$(prefix)_ed_vs_peps.png"
    savefig(fig, out)
    println("  Saved: $out")
    return fig
end

"""
Entropy panel (PEPS only — ED has no bond entropy).
Shown separately so it isn't omitted.
"""
function plot_entropy(df_peps; quench_label::String, prefix::String)
    isnothing(df_peps) && return
    hascol(df, c) = c ∈ names(df)
    (hascol(df_peps, "S_mean") && hascol(df_peps, "S_h_mean") &&
     hascol(df_peps, "S_v_mean")) || return

    p = plot(df_peps.t, df_peps.S_h_mean, label="S_h (horizontal)",
             xlabel="t", ylabel="S_vN",
             title="$quench_label — PEPS Bond Entanglement Entropy")
    plot!(p, df_peps.t, df_peps.S_v_mean, label="S_v (vertical)", linestyle=:dash)
    plot!(p, df_peps.t, df_peps.S_mean,   label="mean", color=:black, lw=2.5)

    out = "$(prefix)_entropy.png"
    savefig(p, out)
    println("  Saved: $out")
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Per-quench plots                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

for (tag, name) in [("A", "String Breaking"),
                    ("B", "Mass Quench"),
                    ("C", "Coupling Quench")]

    println("\n" * "─" ^ 60)
    println("  Quench $tag: $name")
    println("─" ^ 60)

    # Look for CSVs in results_dir first, then fall back to @__DIR__
    function find_csv(fname)
        p = joinpath(results_dir, fname)
        isfile(p) && return p
        p2 = joinpath(@__DIR__, fname)
        isfile(p2) && return p2
        return p  # missing — load_if_exists will warn
    end

    df_peps = load_if_exists(find_csv("finite_peps_quench_$(tag)_data.csv"))
    df_ed   = load_if_exists(find_csv("finite_ed_quench_$(tag)_data.csv"))

    prefix = joinpath(results_dir, "quench_$(tag)")
    plot_comparison(df_peps, df_ed;
                    quench_label = "Quench $tag: $name",
                    prefix       = prefix)
    plot_entropy(df_peps; quench_label="Quench $tag: $name", prefix=prefix)
end

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Summary: all three quenches, one observable each                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

println("\n" * "─" ^ 60)
println("  Summary figure")
println("─" ^ 60)

tags        = ["A", "B", "C"]
quench_names = ["String Breaking", "Mass Quench", "Coupling Quench"]
colors_peps = [:steelblue, :seagreen, :darkorange]
colors_ed   = [:firebrick,  :purple,   :saddlebrown]

function find_csv_summary(fname)
    p = joinpath(results_dir, fname)
    isfile(p) && return p
    joinpath(@__DIR__, fname)
end

s1 = plot(title="(a) Mean Fermion Density", xlabel="t", ylabel="⟨n_f⟩")
s2 = plot(title="(b) Mean Electric Energy", xlabel="t", ylabel="⟨E²⟩")
s3 = plot(title="(c) CDW Order", xlabel="t", ylabel="⟨n_f⟩_e − ⟨n_f⟩_o")

for (i, (tag, name)) in enumerate(zip(tags, quench_names))
    df_p = load_if_exists(find_csv_summary("finite_peps_quench_$(tag)_data.csv"))
    df_e = load_if_exists(find_csv_summary("finite_ed_quench_$(tag)_data.csv"))

    if !isnothing(df_p)
        plot!(s1, df_p.t, df_p.nf_mean, label="PEPS $tag",
              color=colors_peps[i], lw=2)
        plot!(s2, df_p.t, df_p.E2_mean, label="PEPS $tag",
              color=colors_peps[i], lw=2)
        plot!(s3, df_p.t, df_p.nf_even .- df_p.nf_odd, label="PEPS $tag",
              color=colors_peps[i], lw=2)
    end
    if !isnothing(df_e)
        plot!(s1, df_e.t, df_e.nf_mean, label="ED $tag",
              color=colors_ed[i], lw=2, linestyle=:dash)
        plot!(s2, df_e.t, df_e.E2_mean, label="ED $tag",
              color=colors_ed[i], lw=2, linestyle=:dash)
        plot!(s3, df_e.t, df_e.nf_even .- df_e.nf_odd, label="ED $tag",
              color=colors_ed[i], lw=2, linestyle=:dash)
    end
end
hline!(s1, [0.5]; color=:gray, linestyle=:dot, label="")
hline!(s3, [0.0]; color=:gray, linestyle=:dot, label="")

fig_sum = plot(s1, s2, s3, layout=(1,3), size=(1600,500),
               plot_title="ED vs PEPS — all quenches")
out_sum = joinpath(results_dir, "summary_ed_vs_peps.png")
savefig(fig_sum, out_sum)
println("  Saved: $out_sum")

println("\nDone.")
