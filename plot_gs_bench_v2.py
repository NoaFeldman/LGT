"""Plot gs_bench_v2 results: PEPS (full-update) vs ED."""
import glob, os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datadir = os.path.join(os.path.dirname(__file__), "results", "gs_bench_v2")
files = sorted(glob.glob(os.path.join(datadir, "gs_benchmark_*.csv")))
if not files:
    print(f"No CSV files found in {datadir}"); sys.exit(1)

df = pd.concat([pd.read_csv(f) for f in files]).sort_values(["m", "g"])
print(df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for mi, (m_val, grp) in enumerate(df.groupby("m")):
    g = grp["g"].values
    c = f"C{mi}"

    # nf_even
    axes[0].plot(g, grp["ed_nf_even"],  "o-", color=c, label=f"ED m={m_val}")
    axes[0].plot(g, grp["peps_nf_even"],"s--",color=c, label=f"PEPS m={m_val}")

    # E2_mean
    axes[1].plot(g, grp["ed_E2_mean"],  "o-", color=c, label=f"ED m={m_val}")
    axes[1].plot(g, grp["peps_E2_mean"],"s--",color=c, label=f"PEPS m={m_val}")

    # Relative error
    err_nf = np.abs(grp["d_nf_even"].values)
    err_E2 = np.abs(grp["d_E2_mean"].values)
    axes[2].semilogy(g, err_nf, "o-",  color=c, label=f"|Δnf| m={m_val}")
    axes[2].semilogy(g, err_E2, "s--", color=c, label=f"|ΔE²| m={m_val}")

axes[0].set(xlabel="g", ylabel="⟨n_f⟩ even sites", title="Fermion density (even)")
axes[1].set(xlabel="g", ylabel="⟨E²⟩ mean", title="Electric field energy")
axes[2].set(xlabel="g", ylabel="Absolute error", title="PEPS vs ED error")
for ax in axes:
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle("Full-Update PEPS vs ED  (3×4, dg=1, D_max=12, τ-annealed)", fontsize=13)
fig.tight_layout()
out = os.path.join(datadir, "peps_vs_ed_v2.png")
fig.savefig(out, dpi=200)
print(f"\nSaved: {out}")
plt.show()
