"""
Plot PEPS vs ED ground-state benchmark — large-D results.
Optionally overlays old (small-D) results for comparison.

Usage:
    python plot_gs_bench_largeD.py
    python plot_gs_bench_largeD.py results/gs_bench_largeD
"""
import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

# Directories
new_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "results", "gs_bench_largeD")
old_dir = os.path.join(script_dir, "results", "gs_bench")

def load_bench(d):
    files = sorted(glob.glob(os.path.join(d, "gs_benchmark_*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.sort_values(["m", "g"], inplace=True)
    return df

df_new = load_bench(new_dir)
df_old = load_bench(old_dir)

if df_new is None and df_old is None:
    print(f"No CSV files found in {new_dir} or {old_dir}")
    sys.exit(1)

# Use whichever is available; prefer new
df_main = df_new if df_new is not None else df_old
label_main = f"PEPS D_max={int(df_main['D_max'].iloc[0])}" if df_main is not None else "PEPS"

g_vals = sorted(df_main["g"].unique())
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(g_vals)))

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
Dinfo = f"D_bond={int(df_main['D_bond'].iloc[0])}, D_max={int(df_main['D_max'].iloc[0])}, n_ite={int(df_main['n_ite'].iloc[0])}"
fig.suptitle(f"PEPS vs ED Ground-State Benchmark  (3×4, dg=1)\n{Dinfo}", fontsize=13, fontweight="bold")

# ── Row 1: Observables vs m ──

# (a) Mean fermion density
ax = axes[0, 0]
for gv, c in zip(g_vals, colors):
    sub = df_main[df_main["g"] == gv].sort_values("m")
    ax.plot(sub["m"], sub["ed_nf_mean"], "s-", color=c, alpha=0.7, label=f"ED g={gv}")
    ax.plot(sub["m"], sub["peps_nf_mean"], "o--", color=c, label=f"PEPS g={gv}")
    if df_old is not None and df_new is not None:
        sub_old = df_old[df_old["g"] == gv].sort_values("m")
        if len(sub_old):
            ax.plot(sub_old["m"], sub_old["peps_nf_mean"], "x:", color=c, alpha=0.4, ms=6)
ax.set_xlabel("m"); ax.set_ylabel("⟨n_f⟩"); ax.set_title("(a) Mean Fermion Density")
ax.set_xscale("log"); ax.legend(fontsize=5.5, ncol=2)

# (b) CDW order
ax = axes[0, 1]
for gv, c in zip(g_vals, colors):
    sub = df_main[df_main["g"] == gv].sort_values("m")
    ed_cdw = sub["ed_nf_even"] - sub["ed_nf_odd"]
    peps_cdw = sub["peps_nf_even"] - sub["peps_nf_odd"]
    ax.plot(sub["m"], ed_cdw, "s-", color=c, alpha=0.7, label=f"ED g={gv}")
    ax.plot(sub["m"], peps_cdw, "o--", color=c, label=f"PEPS g={gv}")
    if df_old is not None and df_new is not None:
        sub_old = df_old[df_old["g"] == gv].sort_values("m")
        if len(sub_old):
            old_cdw = sub_old["peps_nf_even"] - sub_old["peps_nf_odd"]
            ax.plot(sub_old["m"], old_cdw, "x:", color=c, alpha=0.4, ms=6)
ax.set_xlabel("m"); ax.set_ylabel("CDW"); ax.set_title("(b) CDW Order  ⟨n_f⟩_e − ⟨n_f⟩_o")
ax.set_xscale("log"); ax.legend(fontsize=5.5, ncol=2)

# (c) Electric energy
ax = axes[0, 2]
for gv, c in zip(g_vals, colors):
    sub = df_main[df_main["g"] == gv].sort_values("m")
    ax.plot(sub["m"], sub["ed_E2_mean"], "s-", color=c, alpha=0.7, label=f"ED g={gv}")
    ax.plot(sub["m"], sub["peps_E2_mean"], "o--", color=c, label=f"PEPS g={gv}")
    if df_old is not None and df_new is not None:
        sub_old = df_old[df_old["g"] == gv].sort_values("m")
        if len(sub_old):
            ax.plot(sub_old["m"], sub_old["peps_E2_mean"], "x:", color=c, alpha=0.4, ms=6)
ax.set_xlabel("m"); ax.set_ylabel("⟨E²⟩"); ax.set_title("(c) Mean Electric Energy")
ax.set_xscale("log"); ax.legend(fontsize=5.5, ncol=2)

# ── Row 2: Differences and scatter ──

# (d) Scatter PEPS vs ED for E2
ax = axes[1, 0]
sc = ax.scatter(df_main["ed_E2_mean"], df_main["peps_E2_mean"],
                c=np.log10(df_main["g"]), cmap="viridis", edgecolors="k", s=60, zorder=3)
lims = [0, max(df_main["ed_E2_mean"].max(), df_main["peps_E2_mean"].max()) * 1.1]
ax.plot(lims, lims, "k--", alpha=0.4, label="y=x")
plt.colorbar(sc, ax=ax, label="log10(g)")
ax.set_xlabel("ED ⟨E²⟩"); ax.set_ylabel("PEPS ⟨E²⟩"); ax.set_title("(d) ⟨E²⟩: PEPS vs ED")
ax.legend()

# (e) |Δ(CDW)| vs m
ax = axes[1, 1]
for gv, c in zip(g_vals, colors):
    sub = df_main[df_main["g"] == gv].sort_values("m")
    ax.plot(sub["m"], sub["d_cdw"].abs() + 1e-16, "o-", color=c, label=f"g={gv}")
    if df_old is not None and df_new is not None:
        sub_old = df_old[df_old["g"] == gv].sort_values("m")
        if len(sub_old):
            ax.plot(sub_old["m"], sub_old["d_cdw"].abs() + 1e-16, "x:", color=c, alpha=0.4, ms=6)
ax.set_xlabel("m"); ax.set_ylabel("|Δ(CDW)|"); ax.set_title("(e) |PEPS − ED| CDW")
ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=7)

# (f) |Δ(E²)| vs m
ax = axes[1, 2]
for gv, c in zip(g_vals, colors):
    sub = df_main[df_main["g"] == gv].sort_values("m")
    ax.plot(sub["m"], sub["d_E2_mean"].abs() + 1e-16, "o-", color=c, label=f"g={gv}")
    if df_old is not None and df_new is not None:
        sub_old = df_old[df_old["g"] == gv].sort_values("m")
        if len(sub_old):
            ax.plot(sub_old["m"], sub_old["d_E2_mean"].abs() + 1e-16, "x:", color=c, alpha=0.4, ms=6)
ax.set_xlabel("m"); ax.set_ylabel("|Δ⟨E²⟩|"); ax.set_title("(f) |PEPS − ED| Electric Energy")
ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=7)

plt.tight_layout()
out_path = os.path.join(new_dir, "peps_vs_ed_largeD.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.show()
