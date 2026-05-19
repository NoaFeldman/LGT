"""
Plot PEPS vs ED ground-state benchmark from gs_bench CSV files.
"""
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Gather all per-task CSVs
bench_dir = os.path.join(os.path.dirname(__file__), "results", "gs_bench")
files = sorted(glob.glob(os.path.join(bench_dir, "gs_benchmark_*.csv")))
if not files:
    raise FileNotFoundError(f"No gs_benchmark_*.csv found in {bench_dir}")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.sort_values(["m", "g"], inplace=True)
print(f"Loaded {len(df)} data points from {len(files)} files")
print(df[["task_id", "m", "g", "peps_nf_mean", "ed_nf_mean", "peps_E2_mean", "ed_E2_mean"]].to_string(index=False))

g_vals = sorted(df["g"].unique())
m_vals = sorted(df["m"].unique())

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("PEPS vs ED Ground-State Benchmark  (3×4 lattice, dg=1)", fontsize=14, fontweight="bold")

# ── Row 1: Observables vs m, colored by g ──
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(g_vals)))

# (a) Mean fermion density
ax = axes[0, 0]
for gv, c in zip(g_vals, colors):
    sub = df[df["g"] == gv].sort_values("m")
    ax.plot(sub["m"], sub["peps_nf_mean"], "o-", color=c, label=f"PEPS g={gv}")
    ax.plot(sub["m"], sub["ed_nf_mean"], "s--", color=c, alpha=0.7, label=f"ED g={gv}")
ax.set_xlabel("m"); ax.set_ylabel("⟨n_f⟩"); ax.set_title("(a) Mean Fermion Density")
ax.set_xscale("log"); ax.legend(fontsize=6, ncol=2)

# (b) CDW order
ax = axes[0, 1]
for gv, c in zip(g_vals, colors):
    sub = df[df["g"] == gv].sort_values("m")
    peps_cdw = sub["peps_nf_even"] - sub["peps_nf_odd"]
    ed_cdw = sub["ed_nf_even"] - sub["ed_nf_odd"]
    ax.plot(sub["m"], peps_cdw, "o-", color=c, label=f"PEPS g={gv}")
    ax.plot(sub["m"], ed_cdw, "s--", color=c, alpha=0.7, label=f"ED g={gv}")
ax.set_xlabel("m"); ax.set_ylabel("⟨n_f⟩_even − ⟨n_f⟩_odd"); ax.set_title("(b) CDW Order")
ax.set_xscale("log"); ax.legend(fontsize=6, ncol=2)

# (c) Electric energy
ax = axes[0, 2]
for gv, c in zip(g_vals, colors):
    sub = df[df["g"] == gv].sort_values("m")
    ax.plot(sub["m"], sub["peps_E2_mean"], "o-", color=c, label=f"PEPS g={gv}")
    ax.plot(sub["m"], sub["ed_E2_mean"], "s--", color=c, alpha=0.7, label=f"ED g={gv}")
ax.set_xlabel("m"); ax.set_ylabel("⟨E²⟩"); ax.set_title("(c) Mean Electric Energy")
ax.set_xscale("log"); ax.legend(fontsize=6, ncol=2)

# ── Row 2: Scatter PEPS vs ED (1:1 comparison) + differences ──

# (d) Scatter: PEPS vs ED nf_mean
ax = axes[1, 0]
ax.scatter(df["ed_nf_mean"], df["peps_nf_mean"], c=df["g"], cmap="viridis", edgecolors="k", s=60, zorder=3)
lims = [min(df["ed_nf_mean"].min(), df["peps_nf_mean"].min()) - 0.02,
        max(df["ed_nf_mean"].max(), df["peps_nf_mean"].max()) + 0.02]
ax.plot(lims, lims, "k--", alpha=0.4, label="y=x")
ax.set_xlabel("ED ⟨n_f⟩"); ax.set_ylabel("PEPS ⟨n_f⟩"); ax.set_title("(d) nf: PEPS vs ED")
ax.legend()

# (e) |Δ(CDW)| vs m
ax = axes[1, 1]
for gv, c in zip(g_vals, colors):
    sub = df[df["g"] == gv].sort_values("m")
    ax.plot(sub["m"], sub["d_cdw"].abs(), "o-", color=c, label=f"g={gv}")
ax.set_xlabel("m"); ax.set_ylabel("|Δ(CDW)|"); ax.set_title("(e) |PEPS − ED| CDW")
ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=7)

# (f) |Δ(E²)| vs m
ax = axes[1, 2]
for gv, c in zip(g_vals, colors):
    sub = df[df["g"] == gv].sort_values("m")
    ax.plot(sub["m"], sub["d_E2_mean"].abs(), "o-", color=c, label=f"g={gv}")
ax.set_xlabel("m"); ax.set_ylabel("|Δ⟨E²⟩|"); ax.set_title("(f) |PEPS − ED| Electric Energy")
ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=7)

plt.tight_layout()
out_path = os.path.join(bench_dir, "peps_vs_ed_comparison.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.show()
