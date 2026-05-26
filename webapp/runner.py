"""Asyncio sweep orchestrator.

Manages a pool of subprocess simulator runs (Semaphore-bound to MAX_CONCURRENT),
maintains per-sweep status JSON, and broadcasts events to subscribed SSE
client queues.

Status JSON shape (atomically written to <sweep_dir>/status.json):
{
  "sweep_id": "...",
  "created_at": "ISO-8601",
  "state": "running" | "done" | "failed" | "cancelled",
  "configs": {
     "<label>": {"state": "queued|running|done|failed|cancelled",
                 "elapsed_s": float, "metrics": {...}}
  }
}
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from .cluster_builder import ConfigSpec, build_cluster_json
from .config import (
    CONFIG_TIMEOUT_S,
    CPU_MEM_DEFAULT,
    LINK_BW_DEFAULT,
    LINK_LATENCY_DEFAULT,
    MAIN_PY,
    MAX_CONCURRENT,
    OUTPUT_DIR,
    REPO_ROOT,
    SIM_ENV,
)
from .parser import extract_error_excerpt, is_successful, parse_run

# In-memory event queues: sweep_id -> list[asyncio.Queue]
_event_queues: dict[str, list[asyncio.Queue]] = {}
# Running process handles: sweep_id -> {label: Process}
_running_procs: dict[str, dict[str, asyncio.subprocess.Process]] = {}
# Cancellation flags: sweep_id -> bool
_cancel_flags: dict[str, bool] = {}
# Async lock per sweep_id for status file writes (created lazily).
_status_locks: dict[str, asyncio.Lock] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_path(sweep_dir: Path) -> Path:
    return sweep_dir / "status.json"


def _read_status(sweep_dir: Path) -> dict:
    p = _status_path(sweep_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_status_atomic(sweep_dir: Path, status: dict) -> None:
    """Write status.json atomically: write to .tmp then rename."""
    sweep_dir.mkdir(parents=True, exist_ok=True)
    tmp = sweep_dir / "status.json.tmp"
    final = _status_path(sweep_dir)
    tmp.write_text(json.dumps(status, indent=2, default=str))
    os.replace(tmp, final)


def _get_lock(sweep_id: str) -> asyncio.Lock:
    lock = _status_locks.get(sweep_id)
    if lock is None:
        lock = asyncio.Lock()
        _status_locks[sweep_id] = lock
    return lock


async def _broadcast(sweep_id: str, event: dict) -> None:
    """Push an event onto every subscribed queue for this sweep."""
    queues = _event_queues.get(sweep_id, [])
    for q in list(queues):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            # Drop if a slow consumer fills its queue.
            pass


async def _update_config_state(
    sweep_id: str,
    sweep_dir: Path,
    label: str,
    updates: dict,
) -> None:
    """Merge `updates` into status.configs[label] and broadcast event."""
    async with _get_lock(sweep_id):
        status = _read_status(sweep_dir)
        cfgs = status.setdefault("configs", {})
        entry = cfgs.setdefault(label, {})
        entry.update(updates)
        _write_status_atomic(sweep_dir, status)

    event = {"label": label, **updates}
    await _broadcast(sweep_id, event)


_TERMINAL_STATES = ("done", "failed", "cancelled")


async def _update_sweep_state(
    sweep_id: str,
    sweep_dir: Path,
    state: str,
) -> None:
    """Set the top-level sweep state and broadcast.

    On entering a terminal state we also stamp ``finished_at`` so the
    progress UI can show accurate total-elapsed time on reload, instead
    of treating "now - created_at" as the duration.
    """
    finished_at: str | None = None
    async with _get_lock(sweep_id):
        status = _read_status(sweep_dir)
        status["state"] = state
        if state in _TERMINAL_STATES and not status.get("finished_at"):
            finished_at = _now_iso()
            status["finished_at"] = finished_at
        _write_status_atomic(sweep_dir, status)

    event = {"sweep_state": state}
    if finished_at is not None:
        event["finished_at"] = finished_at
    await _broadcast(sweep_id, event)


def _cleanup_pid_artifacts(pid: int) -> None:
    """Remove ASTRA-Sim trace/workload files left by a finished main.py PID.

    trace_generator.py / utils.py namespace per-process artifacts under
    `pidNNNN_` to avoid concurrent-write races (see utils.py:PID_TAG). This
    runs after each per-config subprocess exits so the shared
    `astra-sim/inputs/{trace,workload}/` dirs don't accumulate stale files
    across many sweeps. Best-effort: errors are swallowed since the run
    has already completed and partial cleanup is harmless.
    """
    tag = f"pid{pid}_"
    astra_inputs = REPO_ROOT / "astra-sim" / "inputs"
    for sub in ("trace", "workload"):
        root = astra_inputs / sub
        if not root.is_dir():
            continue
        # rglob walks all depths; covers trace/{hw}/{model}/pidN_*.txt,
        # trace/pidN_event_handler.txt, workload/{hw}/{model}/pidN_*/...
        for path in root.rglob(f"{tag}*"):
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass


async def _run_one_config(
    sweep_id: str,
    sweep_dir: Path,
    config_path: Path,
    log_path: Path,
    csv_path: Path,
    label: str,
    workload: dict,
    sem: asyncio.Semaphore,
) -> None:
    """Run a single config simulation under the concurrency semaphore."""
    if _cancel_flags.get(sweep_id):
        await _update_config_state(
            sweep_id, sweep_dir, label,
            {"state": "cancelled", "elapsed_s": 0.0},
        )
        return

    async with sem:
        if _cancel_flags.get(sweep_id):
            await _update_config_state(
                sweep_id, sweep_dir, label,
                {"state": "cancelled", "elapsed_s": 0.0},
            )
            return

        start = time.monotonic()
        await _update_config_state(
            sweep_id, sweep_dir, label,
            {"state": "running", "elapsed_s": 0.0},
        )

        # Cluster config path is relative to repo root (config_builder.py
        # prepends "../" itself when entering the astra-sim CWD).
        try:
            cluster_arg = str(config_path.relative_to(REPO_ROOT))
        except ValueError:
            cluster_arg = str(config_path)

        try:
            dataset_arg = str(Path(workload["dataset"]))
        except KeyError:
            await _update_config_state(
                sweep_id, sweep_dir, label,
                {"state": "failed", "elapsed_s": 0.0,
                 "error": "missing 'dataset' in workload"},
            )
            return
        num_req = int(workload.get("num_req", 100))

        try:
            csv_arg = str(csv_path.relative_to(REPO_ROOT))
        except ValueError:
            csv_arg = str(csv_path)

        cmd = [
            "python3", str(MAIN_PY),
            "--cluster-config", cluster_arg,
            "--fp", "16",
            "--block-size", "16",
            "--dataset", dataset_arg,
            "--output", csv_arg,
            "--num-req", str(num_req),
            "--log-interval", "1.0",
            "--log-level", "WARNING",
        ]

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "wb")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(REPO_ROOT),
                env=SIM_ENV,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,  # own process group for clean signal
            )
        except Exception as exc:
            log_file.close()
            await _update_config_state(
                sweep_id, sweep_dir, label,
                {"state": "failed",
                 "elapsed_s": time.monotonic() - start,
                 "error": f"failed to launch subprocess: {exc!r}"},
            )
            return

        _running_procs.setdefault(sweep_id, {})[label] = proc

        # Per-sweep timeout override from workload.timeout_s (set by the
        # "Timeout (s)" input next to Run Sweep). Falls back to the global
        # CONFIG_TIMEOUT_S default when unset / invalid.
        try:
            timeout_s = int(workload.get("timeout_s") or CONFIG_TIMEOUT_S)
            if timeout_s < 10:
                timeout_s = CONFIG_TIMEOUT_S
        except (TypeError, ValueError):
            timeout_s = CONFIG_TIMEOUT_S

        timed_out = False
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    await proc.wait()
                except Exception:
                    pass
        finally:
            log_file.close()
            _running_procs.get(sweep_id, {}).pop(label, None)
            # Subprocess is finished — sweep up its PID-tagged trace/workload
            # files so the shared astra-sim/inputs/ dirs stay tidy across runs.
            _cleanup_pid_artifacts(proc.pid)

        elapsed = time.monotonic() - start

        if _cancel_flags.get(sweep_id) and proc.returncode != 0:
            await _update_config_state(
                sweep_id, sweep_dir, label,
                {"state": "cancelled", "elapsed_s": elapsed},
            )
            return

        if timed_out:
            await _update_config_state(
                sweep_id, sweep_dir, label,
                {"state": "failed", "elapsed_s": elapsed,
                 "error": f"timeout after {timeout_s}s"},
            )
            return

        if proc.returncode != 0 or not is_successful(log_path):
            # Pull a concise reason out of the log so the progress UI's
            # "last log line" cell shows something more useful than blank —
            # mirrors what the timeout path already does via the "error" field.
            excerpt = extract_error_excerpt(log_path)
            error_msg = f"exit code {proc.returncode}"
            if excerpt:
                error_msg = f"{error_msg}: {excerpt}"
            await _update_config_state(
                sweep_id, sweep_dir, label,
                {"state": "failed", "elapsed_s": elapsed,
                 "returncode": proc.returncode,
                 "error": error_msg},
            )
            return

        metrics = parse_run(log_path, csv_path)
        # Drop large per-request lists from status.json to keep it small;
        # they remain available via parse_run() on the saved files.
        status_metrics = {
            k: v for k, v in metrics.items()
            if k not in ("ttft_values_ms", "itl_values_ms")
        }
        await _update_config_state(
            sweep_id, sweep_dir, label,
            {"state": "done", "elapsed_s": elapsed, "metrics": status_metrics},
        )


async def run_sweep(
    sweep_id: str,
    configs: list[ConfigSpec],
    scenario_json: dict,
    sweep_dir: Path,
    workload: dict,
) -> None:
    """Top-level sweep coroutine.

    Writes scenario.json, per-config cluster JSONs, then schedules each
    config to run with a Semaphore(MAX_CONCURRENT) gate. Updates status.json
    and pushes events to subscribers as configs progress.
    """
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "configs").mkdir(parents=True, exist_ok=True)
    (sweep_dir / "runs").mkdir(parents=True, exist_ok=True)

    (sweep_dir / "scenario.json").write_text(
        json.dumps(scenario_json, indent=2, default=str)
    )

    # Initial status with all configs queued.
    initial: dict = {
        "sweep_id": sweep_id,
        "created_at": _now_iso(),
        "state": "running",
        "configs": {c.label: {"state": "queued"} for c in configs},
    }
    async with _get_lock(sweep_id):
        _write_status_atomic(sweep_dir, initial)

    cpu_mem = workload.get("cpu_mem", CPU_MEM_DEFAULT)
    link_bw = int(workload.get("link_bw", LINK_BW_DEFAULT))
    link_latency = int(workload.get("link_latency", LINK_LATENCY_DEFAULT))
    # Power modeling: when the scenario was loaded from a cluster config that
    # had a `power` block, the JS captures it and forwards it here. Applied
    # uniformly to every generated node so each variant runs power simulation.
    power_template = workload.get("power_template") or None

    config_paths: dict[str, Path] = {}
    for spec in configs:
        cluster_json = build_cluster_json(
            spec, cpu_mem, link_bw, link_latency,
            power_template=power_template,
        )
        cfg_path = sweep_dir / "configs" / f"{spec.label}.json"
        cfg_path.write_text(json.dumps(cluster_json, indent=4))
        config_paths[spec.label] = cfg_path

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    _cancel_flags[sweep_id] = False
    _running_procs.setdefault(sweep_id, {})

    tasks = []
    for spec in configs:
        log_path = sweep_dir / "runs" / f"{spec.label}.log"
        csv_path = sweep_dir / "runs" / f"{spec.label}.csv"
        tasks.append(asyncio.create_task(_run_one_config(
            sweep_id=sweep_id,
            sweep_dir=sweep_dir,
            config_path=config_paths[spec.label],
            log_path=log_path,
            csv_path=csv_path,
            label=spec.label,
            workload=workload,
            sem=sem,
        )))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # Decide final sweep state from per-config states.
    final_status = _read_status(sweep_dir)
    cfgs = final_status.get("configs", {})
    states = [c.get("state") for c in cfgs.values()]
    if _cancel_flags.get(sweep_id):
        new_state = "cancelled"
    elif any(s == "failed" for s in states):
        new_state = "failed"
    elif all(s == "done" for s in states):
        new_state = "done"
    else:
        new_state = "failed"

    await _update_sweep_state(sweep_id, sweep_dir, new_state)
    _running_procs.pop(sweep_id, None)
    _cancel_flags.pop(sweep_id, None)


async def cancel_sweep(sweep_id: str) -> None:
    """Set cancel flag and SIGTERM all running subprocesses for this sweep."""
    _cancel_flags[sweep_id] = True
    procs = _running_procs.get(sweep_id, {})
    for label, proc in list(procs.items()):
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass


def subscribe_events(sweep_id: str) -> asyncio.Queue:
    """Return a fresh queue that receives future events for the sweep."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
    _event_queues.setdefault(sweep_id, []).append(queue)
    return queue


def unsubscribe_events(sweep_id: str, queue: asyncio.Queue) -> None:
    """Detach a previously-subscribed queue."""
    queues = _event_queues.get(sweep_id)
    if not queues:
        return
    try:
        queues.remove(queue)
    except ValueError:
        pass
    if not queues:
        _event_queues.pop(sweep_id, None)


def get_status(sweep_id: str) -> dict:
    """Read status.json from the sweep directory; {} if missing."""
    sweep_dir = OUTPUT_DIR / sweep_id
    return _read_status(sweep_dir)


def list_sweeps() -> list[dict]:
    """Return summary records for every sweep on disk, newest first."""
    out: list[dict] = []
    if not OUTPUT_DIR.is_dir():
        return out
    for entry in OUTPUT_DIR.iterdir():
        if not entry.is_dir():
            continue
        status = _read_status(entry)
        if not status:
            continue
        out.append({
            "sweep_id":     status.get("sweep_id", entry.name),
            "created_at":   status.get("created_at"),
            "state":        status.get("state", "unknown"),
            "config_count": len(status.get("configs", {})),
        })
    out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return out
