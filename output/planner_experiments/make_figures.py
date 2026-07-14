"""Generate figures for REPORT_MILP_MaxFlow.md from the experiment pareto.csv files."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
FIGDIR = BASE / "figures"
FIGDIR.mkdir(exist_ok=True)

# muted, print-friendly palette
C = {"throughput": "#4C72B0", "energy": "#55A868", "ttft": "#C44E52", "tpot": "#8172B3"}


def load(run_name):
    p = RUNS / run_name / "pareto.csv"
    if not p.is_file():
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    r = rows[0]  # top_k=1 -> single row
    def fnum(k):
        try:
            return float(r[k])
        except (ValueError, KeyError):
            return float("nan")
    return {
        "ttft": fnum("ttft_ms"), "tpot": fnum("tpot_ms"), "itl": fnum("itl_p99_ms"),
        "throughput": fnum("throughput_toks_s"), "toks_per_wh": fnum("toks_per_wh"),
        "status": r.get("status", ""),
    }


# ---------------------------------------------------------------------------
# Figure 1: hardware comparison (Experiment A)
# ---------------------------------------------------------------------------
hw_order = ["A5000", "A6000", "H100"]
hwd = {hw: load(f"hw_{hw}") for hw in hw_order}
hwd = {k: v for k, v in hwd.items() if v}

if hwd:
    labels = list(hwd.keys())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, title, unit, col) in zip(axes, [
        ("throughput", "Throughput", "tok/s", C["throughput"]),
        ("toks_per_wh", "Energy efficiency", "toks/Wh", C["energy"]),
        ("ttft", "TTFT (mean)", "ms", C["ttft"]),
    ]):
        vals = [hwd[l][key] for l in labels]
        bars = ax.bar(labels, vals, color=col, width=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Experiment A — single-instance hardware comparison "
                 "(Llama-3.1-8B, TP1, 30 req)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGDIR / "fig1_hardware.png", dpi=130)
    print("wrote fig1_hardware.png", {k: hwd[k] for k in labels})

# ---------------------------------------------------------------------------
# Figure 2: TP scaling on A6000 (Experiment B)
# ---------------------------------------------------------------------------
tps = [1, 2, 4]
tpd = {tp: load(f"tp_A6000_tp{tp}") for tp in tps}
tpd = {k: v for k, v in tpd.items() if v}

if tpd:
    xs = list(tpd.keys())
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
    # left: throughput (bars) + toks/Wh (line, twin axis)
    thr = [tpd[t]["throughput"] for t in xs]
    eff = [tpd[t]["toks_per_wh"] for t in xs]
    xlbl = [f"TP{t}" for t in xs]
    b = axL.bar(xlbl, thr, color=C["throughput"], width=0.5, label="Throughput")
    axL.set_ylabel("Throughput (tok/s)", color=C["throughput"], fontsize=9)
    axL.grid(axis="y", alpha=0.3)
    for bar, v in zip(b, thr):
        axL.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax2 = axL.twinx()
    ax2.plot(xlbl, eff, color=C["energy"], marker="o", lw=2, label="toks/Wh")
    ax2.set_ylabel("Energy eff (toks/Wh)", color=C["energy"], fontsize=9)
    axL.set_title("Throughput & energy efficiency vs TP", fontsize=11, fontweight="bold")

    # right: latency (TTFT + TPOT)
    ttft = [tpd[t]["ttft"] for t in xs]
    tpot = [tpd[t]["tpot"] for t in xs]
    axR.plot(xlbl, ttft, color=C["ttft"], marker="s", lw=2, label="TTFT (mean)")
    axR.plot(xlbl, tpot, color=C["tpot"], marker="^", lw=2, label="TPOT (mean)")
    axR.set_ylabel("Latency (ms)", fontsize=9)
    axR.set_title("Latency vs TP", fontsize=11, fontweight="bold")
    axR.grid(axis="y", alpha=0.3)
    axR.legend(fontsize=8)
    fig.suptitle("Experiment B — TP scaling on A6000 (Llama-3.1-8B, 30 req)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGDIR / "fig2_tp_scaling.png", dpi=130)
    print("wrote fig2_tp_scaling.png", {k: tpd[k] for k in xs})

print("done")
