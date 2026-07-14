"""Figure for the A40+RNGD 70B planner run (Experiment D)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RUN = BASE / "runs" / "a40_rngd_70b"
FIGDIR = BASE / "figures"
FIGDIR.mkdir(exist_ok=True)

# run_id -> readable label (from the known layouts)
LABEL = {
    "cand_0baf3f021ddb84e7": "A40x2+RNGDx2\n(3600W)",
    "cand_d90f2d27d675a9b2": "A40x1+RNGDx2\n(2400W)",
    "cand_b947de80bf8841a4": "RNGDx2\n(1200W)",
    "cand_c2a5f4c05289f378": "RNGDx1\n(600W)",
}

rows = list(csv.DictReader(open(RUN / "pareto.csv")))
rows = [r for r in rows if r["status"] == "ok"]
num = lambda s: float(s)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))

# Left: Pareto scatter throughput vs toks/Wh
for r in rows:
    on_p = r["on_pareto"].lower() == "true"
    axL.scatter(num(r["throughput_toks_s"]), num(r["toks_per_wh"]),
                s=190 if on_p else 90,
                c="#C44E52" if on_p else "#4C72B0",
                marker="*" if on_p else "o",
                edgecolors="#333", linewidths=1, zorder=3)
    axL.annotate(LABEL.get(r["run_id"], r["run_id"][:8]).replace("\n", " "),
                 (num(r["throughput_toks_s"]), num(r["toks_per_wh"])),
                 textcoords="offset points", xytext=(8, 6), fontsize=8)
axL.set_xlabel("Throughput (tok/s, higher better)")
axL.set_ylabel("Energy efficiency (toks/Wh, higher better)")
axL.set_title("Pareto: throughput vs energy (★ = Pareto front)", fontsize=11, fontweight="bold")
axL.grid(alpha=0.3)

# Right: throughput bars, colored by hardware mix
order = ["cand_c2a5f4c05289f378", "cand_b947de80bf8841a4",
         "cand_d90f2d27d675a9b2", "cand_0baf3f021ddb84e7"]
by_id = {r["run_id"]: r for r in rows}
labels = [LABEL[i] for i in order if i in by_id]
thr = [num(by_id[i]["throughput_toks_s"]) for i in order if i in by_id]
colors = ["#55A868" if "RNGD" in LABEL[i] and "A40" not in LABEL[i] else "#DD8452"
          for i in order if i in by_id]
bars = axR.bar(labels, thr, color=colors, width=0.6)
for b, v in zip(bars, thr):
    axR.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
axR.set_ylabel("Throughput (tok/s)")
axR.set_title("Adding A40 (orange) HURTS throughput", fontsize=11, fontweight="bold")
axR.grid(axis="y", alpha=0.3)

fig.suptitle("Experiment D — A40x8 + RNGDx8 (IB 200Gb), Llama-3.1-70B, TP4, 16 req",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGDIR / "fig4_a40_rngd_70b.png", dpi=130)
print("wrote fig4_a40_rngd_70b.png")
