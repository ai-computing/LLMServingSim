"""Extrapolate tp4/tp8 profiles from existing tp1 + tp2 measurements.

When actual GPU hardware is unavailable, this script derives tp4 (and tp8)
layer latency and attention profiles by extrapolating the observed tp1→tp2
scaling ratios.

Validated on H100/Llama-3.1-70B (only model with measured tp4 ground truth):
  - Mean absolute error: ~5.4% across all layers
  - Largest errors: rope / attn layers (~14%) — overestimate latency slightly
  - Smallest errors: compute-bound proj layers (~4-5%)

NOTE: These are approximations. Run the real profiler when hardware is
available (see profile_layers.sh / profile_attn.sh / build_predictor.sh).

Usage:
  # Dry-run — show which combos would be generated:
  python extrapolate_tp_profile.py --dry-run

  # Generate tp4 + tp8 for all hw/model combos that have tp1 + tp2:
  python extrapolate_tp_profile.py --target-tp 4 8

  # Limit to a specific hw/model:
  python extrapolate_tp_profile.py --hw A6000 --model meta-llama/Llama-3.1-8B --target-tp 4 8

  # Generate tp2 for hardware that only has tp1, using a reference hw's ratios:
  python extrapolate_tp_profile.py --from-tp 1 --ref-hw A6000 --hw RNGD \\
    --model meta-llama/Llama-3.1-8B --target-tp 2

After generation, rebuild the attention predictor for each combo:
  cd llm_profile
  python -m profiler.predictor.main \\
    --model meta-llama/Llama-3.1-8B --hardware A6000 \\
    --tp-size "1, 2, 4, 8" \\
    --kv-granularity 64 --chunk-granularity 32 \\
    --max-len 2048 --max-batch 256
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
from pathlib import Path


# Non-TP-parallel layers: their latency doesn't depend on TP count.
# (embedding and layernorm are replicated on every rank, not sharded.)
FIXED_LAYERS = {
    "embedding",
    "input_layernorm",
    "post_layernorm",
    "final_layernorm",
}

PERF_MODELS = Path(__file__).parent / "perf_models"


# ---------------------------------------------------------------------------
# layers.csv

def _load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _avg_by_layer(rows: list[dict]) -> dict[str, float]:
    d: dict[str, list[float]] = {}
    for r in rows:
        d.setdefault(r["layer_name"], []).append(float(r["latency(ns)"]))
    return {k: statistics.mean(v) for k, v in d.items()}


def _extrapolate_layers(
    src_dir: Path, tp1_avg: dict[str, float], tp2_avg: dict[str, float],
    target_tp: int, out_dir: Path, overwrite: bool,
) -> bool:
    """Write layers.csv for target_tp using tp1/tp2 scaling ratio."""
    out_path = out_dir / "layers.csv"
    if out_path.exists() and not overwrite:
        print(f"    [skip] {out_path} already exists")
        return False

    # Per-layer ratio: tp1 / tp2.  For FIXED layers force ratio = 1.0.
    ratios: dict[str, float] = {}
    for layer, lat1 in tp1_avg.items():
        if layer not in tp2_avg or tp2_avg[layer] <= 0:
            continue
        ratios[layer] = 1.0 if layer in FIXED_LAYERS else lat1 / tp2_avg[layer]

    # Number of TP doublings from tp2 to target_tp
    steps = int(math.log2(target_tp / 2))

    tp2_rows = _load_rows(src_dir / "tp2" / "layers.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_name", "input", "kv_cache", "tp_size", "latency(ns)"])
        for row in tp2_rows:
            layer = row["layer_name"]
            lat = float(row["latency(ns)"])
            r = ratios.get(layer, 1.0)
            for _ in range(steps):
                lat /= r
            writer.writerow([layer, row["input"], row["kv_cache"], target_tp, round(lat)])

    print(f"    [wrote] {out_path}")
    return True


# ---------------------------------------------------------------------------
# attention.csv

def _extrapolate_attention(
    src_dir: Path, target_tp: int, out_dir: Path, overwrite: bool,
) -> bool:
    """Write attention.csv for target_tp.

    Attention latency barely changes with TP in practice:
      - Prefill: ratio ≈ 1.009 per TP doubling (FlashAttention, chunk-compute bound)
      - Decode at small batch (≤8): ratio ≈ 1.01-1.03 (memory-BW bound)
      - Decode at larger batch:     ratio ≈ 1.2-1.4  (becoming compute bound)

    Strategy: copy tp2 values verbatim, just update num_tensor_parallel_workers.
    For large-batch decode this slightly over-estimates latency — conservative
    and safe for a simulation (better to over-estimate than under-estimate).
    """
    src = src_dir / "tp2" / "attention.csv"
    if not src.exists():
        return False
    out_path = out_dir / "attention.csv"
    if out_path.exists() and not overwrite:
        print(f"    [skip] {out_path} already exists")
        return False

    rows = _load_rows(src)
    if not rows:
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if "num_tensor_parallel_workers" in row:
                row["num_tensor_parallel_workers"] = target_tp
            writer.writerow(row)

    print(f"    [wrote] {out_path}")
    return True


# ---------------------------------------------------------------------------
# predictions/  — copy from tp2 as placeholder until build_predictor re-runs

def _copy_predictions(src_dir: Path, target_tp: int, out_dir: Path, overwrite: bool):
    src_pred = src_dir / "tp2" / "predictions"
    if not src_pred.exists():
        return
    dst_pred = out_dir / "predictions"
    dst_pred.mkdir(parents=True, exist_ok=True)
    for csv_file in src_pred.glob("*.csv"):
        dst = dst_pred / csv_file.name
        if dst.exists() and not overwrite:
            print(f"    [skip] {dst} already exists")
            continue
        shutil.copy2(csv_file, dst)
        print(f"    [copied] {dst}  ← rerun build_predictor.sh to update")


# ---------------------------------------------------------------------------
# Cross-hardware tp2-from-tp1 extrapolation

def _find_combos_tp1only(hw_filter: str | None, model_filter: str | None):
    """Return hw/model combos that have tp1 but NOT tp2 (candidates for cross-hw extrap)."""
    combos = []
    for hw_dir in sorted(PERF_MODELS.iterdir()):
        if not hw_dir.is_dir():
            continue
        if hw_filter and hw_dir.name != hw_filter:
            continue
        for vendor_dir in sorted(hw_dir.iterdir()):
            if not vendor_dir.is_dir():
                continue
            for model_dir in sorted(vendor_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model_path = f"{vendor_dir.name}/{model_dir.name}"
                if model_filter and model_path != model_filter:
                    continue
                has_tp1 = (model_dir / "tp1" / "layers.csv").exists()
                has_tp2 = (model_dir / "tp2" / "layers.csv").exists()
                if has_tp1 and not has_tp2:
                    combos.append((hw_dir.name, model_path, model_dir))
    return combos


def _extrapolate_layers_cross_hw(
    src_dir: Path,
    ref_hw: str,
    model: str,
    target_tp: int,
    out_dir: Path,
    overwrite: bool,
) -> bool:
    """Generate tpN layers.csv from tp1 using a reference hardware's tp1/tp2 ratios.

    For each layer the scaling ratio is:  ref_tp2_lat / ref_tp1_lat
    (i.e. how much latency shrinks per TP doubling on the reference hardware).
    The ratio is applied `log2(target_tp)` times to the source tp1 latencies.
    FIXED_LAYERS always keep ratio=1.0 regardless of reference hardware.
    Minimum output latency is 1 (avoid zeroing out placeholder 1-ns rows).
    """
    out_path = out_dir / "layers.csv"
    if out_path.exists() and not overwrite:
        print(f"    [skip] {out_path} already exists")
        return False

    ref_dir = PERF_MODELS / ref_hw / model
    ref_tp1_path = ref_dir / "tp1" / "layers.csv"
    ref_tp2_path = ref_dir / "tp2" / "layers.csv"
    if not ref_tp1_path.exists() or not ref_tp2_path.exists():
        print(f"    [error] reference hw '{ref_hw}' missing tp1 or tp2 for {model}")
        return False

    ref_tp1_avg = _avg_by_layer(_load_rows(ref_tp1_path))
    ref_tp2_avg = _avg_by_layer(_load_rows(ref_tp2_path))

    # Per-layer tp2/tp1 ratio from the reference hardware (shrink factor per doubling)
    ratios: dict[str, float] = {}
    for layer, lat1 in ref_tp1_avg.items():
        if layer in FIXED_LAYERS:
            ratios[layer] = 1.0
        elif lat1 > 0 and layer in ref_tp2_avg and ref_tp2_avg[layer] > 0:
            ratios[layer] = ref_tp2_avg[layer] / ref_tp1_avg[layer]
        else:
            ratios[layer] = 0.5  # fallback: assume ideal halving

    steps = int(math.log2(target_tp))  # doublings from tp1

    tp1_rows = _load_rows(src_dir / "tp1" / "layers.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_name", "input", "kv_cache", "tp_size", "latency(ns)"])
        for row in tp1_rows:
            layer = row["layer_name"]
            lat = float(row["latency(ns)"])
            r = ratios.get(layer, 0.5)
            for _ in range(steps):
                lat *= r
            writer.writerow([layer, row["input"], row["kv_cache"],
                              target_tp, max(1, round(lat))])

    print(f"    [wrote] {out_path}  (ref ratios from {ref_hw})")
    return True


def _extrapolate_attention_from_tp1(
    src_dir: Path, target_tp: int, out_dir: Path, overwrite: bool,
) -> bool:
    """Copy attention.csv from tp1 with num_tensor_parallel_workers updated."""
    src = src_dir / "tp1" / "attention.csv"
    if not src.exists():
        return False
    out_path = out_dir / "attention.csv"
    if out_path.exists() and not overwrite:
        print(f"    [skip] {out_path} already exists")
        return False

    rows = _load_rows(src)
    if not rows:
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if "num_tensor_parallel_workers" in row:
                row["num_tensor_parallel_workers"] = target_tp
            writer.writerow(row)

    print(f"    [wrote] {out_path}  (copied from tp1, workers={target_tp})")
    return True


def _copy_predictions_from_tp1(src_dir: Path, out_dir: Path, overwrite: bool):
    """Copy CSV predictions from tp1 as placeholder (pkl excluded — needs rebuild)."""
    src_pred = src_dir / "tp1" / "predictions"
    if not src_pred.exists():
        return
    dst_pred = out_dir / "predictions"
    dst_pred.mkdir(parents=True, exist_ok=True)
    for csv_file in src_pred.glob("*.csv"):
        dst = dst_pred / csv_file.name
        if dst.exists() and not overwrite:
            print(f"    [skip] {dst} already exists")
            continue
        shutil.copy2(csv_file, dst)
        print(f"    [copied] {dst}  ← rerun build_predictor.sh to update")


# ---------------------------------------------------------------------------
# Main

def _find_combos(hw_filter: str | None, model_filter: str | None):
    combos = []
    for hw_dir in sorted(PERF_MODELS.iterdir()):
        if not hw_dir.is_dir():
            continue
        if hw_filter and hw_dir.name != hw_filter:
            continue
        for vendor_dir in sorted(hw_dir.iterdir()):
            if not vendor_dir.is_dir():
                continue
            for model_dir in sorted(vendor_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model_path = f"{vendor_dir.name}/{model_dir.name}"
                if model_filter and model_path != model_filter:
                    continue
                has_tp1 = (model_dir / "tp1" / "layers.csv").exists()
                has_tp2 = (model_dir / "tp2" / "layers.csv").exists()
                if has_tp1 and has_tp2:
                    combos.append((hw_dir.name, model_path, model_dir))
    return combos


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-tp", nargs="+", type=int, default=[4, 8],
                        help="TP sizes to generate (must be powers of 2)")
    parser.add_argument("--from-tp", type=int, default=2, choices=[1, 2],
                        help="Source TP size to extrapolate from (default: 2). "
                             "Use 1 with --ref-hw to generate tp2 without measured tp2 data.")
    parser.add_argument("--ref-hw", default=None,
                        help="Reference hardware whose tp1/tp2 ratios are used when --from-tp 1 "
                             "(e.g. A6000). Must have both tp1 and tp2 for the same model.")
    parser.add_argument("--hw", default=None, help="Limit to this hardware (e.g. RNGD)")
    parser.add_argument("--model", default=None,
                        help="Limit to this model (e.g. meta-llama/Llama-3.1-8B)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without writing")
    args = parser.parse_args()

    # --from-tp 1 mode: cross-hardware tp2 extrapolation
    if args.from_tp == 1:
        if not args.ref_hw:
            parser.error("--from-tp 1 requires --ref-hw <hardware>")
        for tp in args.target_tp:
            if tp < 2 or (tp & (tp - 1)) != 0:
                parser.error(f"--target-tp values must be powers of 2 >= 2, got {tp}")

        combos = _find_combos_tp1only(args.hw, args.model)
        if not combos:
            print("No hw/model combinations found with tp1 only (tp2 already exists or tp1 missing).")
            return

        for hw, model, model_dir in combos:
            print(f"\n{'='*60}")
            print(f"  HW: {hw}  Model: {model}  (ref: {args.ref_hw})")

            if args.dry_run:
                for tp in args.target_tp:
                    out_dir = model_dir / f"tp{tp}"
                    exists = (out_dir / "layers.csv").exists()
                    tag = "[exists]" if exists else "[would create]"
                    print(f"    {tag} {out_dir}/layers.csv")
                continue

            for tp in args.target_tp:
                out_dir = model_dir / f"tp{tp}"
                print(f"\n  → tp{tp}  (cross-hw from tp1 via {args.ref_hw}):")
                _extrapolate_layers_cross_hw(
                    model_dir, args.ref_hw, model, tp, out_dir, args.overwrite
                )
                _extrapolate_attention_from_tp1(model_dir, tp, out_dir, args.overwrite)
                _copy_predictions_from_tp1(model_dir, out_dir, args.overwrite)

    # --from-tp 2 mode (original): extrapolate tp4/tp8 from tp1+tp2
    else:
        for tp in args.target_tp:
            if tp <= 2 or (tp & (tp - 1)) != 0:
                parser.error(f"--target-tp values must be powers of 2 > 2, got {tp}")

        combos = _find_combos(args.hw, args.model)
        if not combos:
            print("No hw/model combinations found with both tp1 and tp2 profiles.")
            return

        for hw, model, model_dir in combos:
            print(f"\n{'='*60}")
            print(f"  HW: {hw}  Model: {model}")

            if args.dry_run:
                for tp in args.target_tp:
                    out_dir = model_dir / f"tp{tp}"
                    exists = (out_dir / "layers.csv").exists()
                    tag = "[exists]" if exists else "[would create]"
                    print(f"    {tag} {out_dir}/layers.csv")
                continue

            tp1_rows = _load_rows(model_dir / "tp1" / "layers.csv")
            tp2_rows = _load_rows(model_dir / "tp2" / "layers.csv")
            tp1_avg = _avg_by_layer(tp1_rows)
            tp2_avg = _avg_by_layer(tp2_rows)

            for tp in args.target_tp:
                out_dir = model_dir / f"tp{tp}"
                print(f"\n  → tp{tp}:")
                _extrapolate_layers(model_dir, tp1_avg, tp2_avg, tp, out_dir, args.overwrite)
                _extrapolate_attention(model_dir, tp, out_dir, args.overwrite)
                _copy_predictions(model_dir, tp, out_dir, args.overwrite)

    if not args.dry_run:
        print(f"\n{'='*60}")
        print("Done. Next step: rerun build_predictor.sh for each hw/model")
        print("to update predictions/ from the extrapolated attention.csv.")
        print("Example:")
        print("  cd llm_profile")
        print('  python -m profiler.predictor.main \\')
        print('    --model meta-llama/Llama-3.1-8B --hardware RNGD \\')
        print('    --tp-size "1, 2" \\')
        print('    --kv-granularity 64 --chunk-granularity 32 \\')
        print('    --max-len 2048 --max-batch 256')


if __name__ == "__main__":
    main()
