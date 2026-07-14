"""Figure for the P/D disaggregation E2E validation (Experiment C)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
FIGDIR = BASE / "figures"
FIGDIR.mkdir(exist_ok=True)


def load(run_name):
    p = RUNS / run_name / "pareto.csv"
    with open(p) as f:
        r = list(csv.DictReader(f))[0]
    g = lambda k: float(r[k])
    return {"ttft": g("ttft_ms"), "tpot": g("tpot_ms"), "itl": g("itl_p99_ms"),
            "throughput": g("throughput_toks_s"), "toks_per_wh": g("toks_per_wh")}


pd_ = load("pd_1p1d")
cb = load("combined_2rep")
labels = ["1P+1D\n(disaggregated)", "2 replicas\n(combined, DP)"]
colors = ["#C44E52", "#4C72B0"]

metrics = [
    ("throughput", "Throughput (tok/s)", "higher better"),
    ("toks_per_wh", "Energy eff (toks/Wh)", "higher better"),
    ("ttft", "TTFT mean (ms)", "lower better"),
    ("itl", "ITL p99 (ms)", "lower better"),
]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
for ax, (key, title, note) in zip(axes, metrics):
    vals = [pd_[key], cb[key]]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.set_title(f"{title}\n({note})", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
fig.suptitle("Experiment C — P/D disaggregation vs combined replicas "
             "(2x A6000, Llama-3.1-8B, 30 req)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIGDIR / "fig3_pd_vs_combined.png", dpi=130)
print("wrote fig3_pd_vs_combined.png")
print("P/D     :", pd_)
print("combined:", cb)
