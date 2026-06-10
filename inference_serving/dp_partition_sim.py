"""
Method 2 DP Simulation: Request Partition

Splits the request stream into dp_count partitions and runs each through an
independent single-instance simulation.  The wall-clock is the MAX of the
per-partition simulation times, matching real DP behavior where all GPUs work
in parallel.

Accuracy advantages over native multi-instance ASTRA-Sim run:
  1. KV cache pressure is correct — each instance sees N/dp requests.
  2. Batch sizes and scheduler decisions reflect the actual per-GPU queue size.
  3. Load imbalance from request length variance emerges naturally.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY   = REPO_ROOT / "main.py"
FREQ      = 1_000_000_000  # 1 GHz, matches simulator constant


# ── Dataset partitioning ──────────────────────────────────────────────────────

def split_jsonl_dataset(
    dataset_path: str,
    n_parts: int,
    num_req: int,
) -> list[str]:
    """
    Round-robin split of JSONL into n_parts temp files under dataset/.
    Returns paths RELATIVE to repo root (so Router's '../' prefix works).
    """
    full_path = REPO_ROOT / dataset_path
    rows: list[dict] = []
    with open(full_path) as f:
        for i, line in enumerate(f):
            if i >= num_req:
                break
            rows.append(json.loads(line.strip()))

    parts: list[list[dict]] = [[] for _ in range(n_parts)]
    for i, row in enumerate(rows):
        parts[i % n_parts].append(row)

    uid = uuid.uuid4().hex[:8]
    out_dir = REPO_ROOT / "dataset"
    out_dir.mkdir(exist_ok=True)

    rel_paths: list[str] = []
    for i, part in enumerate(parts):
        fname = out_dir / f"_dp_{uid}_part{i}.jsonl"
        with open(fname, "w") as f:
            for row in part:
                f.write(json.dumps(row) + "\n")
        rel_paths.append(str(fname.relative_to(REPO_ROOT)))

    return rel_paths


# ── Config manipulation ───────────────────────────────────────────────────────

def make_single_instance_config(dp_config_path: str, instance_idx: int = 0) -> str:
    """
    Derive a single-instance cluster config from a DP config.
    Writes to cluster_config/ and returns a path RELATIVE to repo root.
    """
    full_path = REPO_ROOT / dp_config_path
    with open(full_path) as f:
        cfg = json.load(f)

    # Find the node + local index that holds instance_idx
    inst_counter = 0
    target_node: dict | None = None
    local_idx = 0
    for node in cfg["nodes"]:
        n = node["num_instances"]
        if inst_counter + n > instance_idx:
            local_idx = instance_idx - inst_counter
            target_node = node
            break
        inst_counter += n

    if target_node is None:
        raise ValueError(f"instance_idx={instance_idx} out of range in {dp_config_path}")

    new_node = {k: v for k, v in target_node.items() if k not in ("instances", "num_instances")}
    new_node["num_instances"] = 1
    new_node["instances"] = [target_node["instances"][local_idx]]

    single_cfg = {k: v for k, v in cfg.items() if k != "nodes"}
    single_cfg["nodes"] = [new_node]

    uid = uuid.uuid4().hex[:8]
    out_path = REPO_ROOT / "cluster_config" / f"_dp_single_{uid}.json"
    with open(out_path, "w") as f:
        json.dump(single_cfg, f, indent=2)

    return str(out_path.relative_to(REPO_ROOT))


# ── Running a single partition simulation ────────────────────────────────────

def _parse_ns(stdout: str) -> int:
    m = re.search(r"Total clocks \(ns\):\s+(\d+)", stdout)
    if m:
        return int(m.group(1))
    m = re.search(r"Total latency \(s\):\s+([\d.]+)", stdout)
    if m:
        return int(float(m.group(1)) * FREQ)
    raise ValueError("Cannot parse simulation time from stdout")


def _parse_gen_tokens(stdout: str) -> int:
    m = re.search(r"Total generated tokens:\s+(\d+)", stdout)
    return int(m.group(1)) if m else 0


def run_partition(
    single_config_path: str,
    dataset_rel_path: str,
    num_req: int,
    output_csv: str,
    extra_args: list[str],
    verbose: bool = False,
) -> dict[str, Any]:
    """Run main.py for one partition. Returns latency + token counts."""
    cmd = [
        sys.executable, str(MAIN_PY),
        "--cluster-config", single_config_path,
        "--dataset",        dataset_rel_path,
        "--num-req",        str(num_req),
        "--output",         output_csv,
    ] + extra_args

    result = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=True,
        cwd=str(REPO_ROOT),
    )

    if result.returncode != 0:
        snippet = (result.stderr or "")[-2000:]
        raise RuntimeError(
            f"Partition simulation exited {result.returncode}:\n{snippet}"
        )

    stdout = result.stdout if not verbose else ""
    return {
        "total_latency_ns":  _parse_ns(stdout),
        "total_gen_tokens":  _parse_gen_tokens(stdout),
        "output_csv":        output_csv,
        "stdout":            stdout,
    }


# ── Merging partition results ─────────────────────────────────────────────────

def merge_results(
    partition_results: list[dict[str, Any]],
    merged_csv: str | None,
) -> dict[str, Any]:
    """
    Combine N partition runs into DP-level metrics.
    wall_clock = max(latencies)  — parallel GPU assumption
    """
    wall_ns = max(r["total_latency_ns"] for r in partition_results)
    total_tokens = sum(r["total_gen_tokens"] for r in partition_results)
    wall_s = wall_ns / FREQ
    throughput = total_tokens / wall_s if wall_s > 0 else 0.0

    if merged_csv:
        dfs = []
        for r in partition_results:
            try:
                dfs.append(pd.read_csv(r["output_csv"]))
            except Exception:
                pass
        if dfs:
            pd.concat(dfs, ignore_index=True).to_csv(merged_csv, index=False)

    ttft: dict[str, float] = {}
    tpot: dict[str, float] = {}
    try:
        src = merged_csv or partition_results[0]["output_csv"]
        df = pd.read_csv(src)
        if "TTFT" in df.columns:
            ms = df["TTFT"].dropna() / 1e6
            ttft = {"mean": ms.mean(), "p50": ms.quantile(0.5), "p99": ms.quantile(0.99)}
        if "TPOT" in df.columns:
            ms = df["TPOT"].dropna() / 1e6
            tpot = {"mean": ms.mean(), "p50": ms.quantile(0.5), "p99": ms.quantile(0.99)}
    except Exception:
        pass

    return {
        "wall_clock_s":          wall_s,
        "wall_clock_ns":         wall_ns,
        "total_gen_tokens":      total_tokens,
        "throughput_tok_s":      throughput,
        "partition_latencies_s": [r["total_latency_ns"] / FREQ for r in partition_results],
        "ttft_ms":               ttft,
        "tpot_ms":               tpot,
    }


# ── Top-level orchestration ───────────────────────────────────────────────────

def run_dp_partition_sim(
    dp_config_path: str,
    dp_count: int,
    dataset_path: str,
    num_req: int,
    output_csv: str | None = None,
    extra_args: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Full Method-2 DP simulation pipeline:
      1. Split dataset into dp_count round-robin partitions.
      2. Derive single-instance config from first instance of dp_config.
      3. Run each partition sequentially through main.py.
      4. Merge: wall_clock = max(partition simulation times).
    """
    extra_args = extra_args or []

    print(f"\n[DP Partition Sim]  dp_count={dp_count}, num_req={num_req}")
    print(f"  config : {dp_config_path}")
    print(f"  dataset: {dataset_path}")

    # 1. Split
    part_datasets = split_jsonl_dataset(dataset_path, dp_count, num_req)
    part_sizes = [sum(1 for _ in open(REPO_ROOT / p)) for p in part_datasets]
    print(f"  partition sizes: {part_sizes}")

    # 2. Single-instance config (reused for all partitions)
    single_cfg = make_single_instance_config(dp_config_path, instance_idx=0)
    print(f"  single-instance config: {single_cfg}")

    # 3. Run each partition
    partition_results: list[dict[str, Any]] = []
    part_csvs: list[str] = []
    try:
        for i in range(dp_count):
            base = output_csv.replace(".csv", "") if output_csv else f"output/_dp_part"
            part_csv = f"{base}_part{i}.csv"
            part_csvs.append(part_csv)

            print(f"\n  [Partition {i}] {part_sizes[i]} reqs → {part_csv}")
            r = run_partition(
                single_config_path=single_cfg,
                dataset_rel_path=part_datasets[i],
                num_req=part_sizes[i],
                output_csv=part_csv,
                extra_args=extra_args,
                verbose=verbose,
            )
            lat_s = r["total_latency_ns"] / FREQ
            print(f"    latency: {lat_s:.3f} s   gen_tokens: {r['total_gen_tokens']}")
            partition_results.append(r)

    finally:
        # Clean up temp partition dataset files
        for p in part_datasets:
            try:
                os.unlink(REPO_ROOT / p)
            except OSError:
                pass
        # Clean up temp single-instance config
        try:
            os.unlink(REPO_ROOT / single_cfg)
        except OSError:
            pass

    # 4. Merge
    metrics = merge_results(partition_results, output_csv)

    print(f"\n  ── DP Partition Result ──")
    print(f"  Partition latencies (s) : {[f'{x:.3f}' for x in metrics['partition_latencies_s']]}")
    print(f"  Wall-clock (max)    (s) : {metrics['wall_clock_s']:.3f}")
    print(f"  Total gen tokens        : {metrics['total_gen_tokens']}")
    print(f"  Throughput (tok/s)      : {metrics['throughput_tok_s']:.1f}")
    if metrics["ttft_ms"]:
        t = metrics["ttft_ms"]
        print(f"  TTFT mean/p50/p99 (ms) : {t['mean']:.1f} / {t['p50']:.1f} / {t['p99']:.1f}")
    if metrics["tpot_ms"]:
        t = metrics["tpot_ms"]
        print(f"  TPOT mean/p50/p99 (ms) : {t['mean']:.2f} / {t['p50']:.2f} / {t['p99']:.2f}")

    return metrics
