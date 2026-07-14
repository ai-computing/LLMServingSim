"""Stage 1: structural allocation via CP-SAT (Google OR-Tools).

We enumerate feasible instance *templates* -- (hardware, tp, role) triples whose
memory fits -- and let the solver choose how many of each to deploy, subject to
per-hardware device availability and (when P/D disaggregation is on) a
prefill/decode balance window. The objective is a coarse linear throughput proxy
minus a power penalty; Stage 2 (simulation) re-ranks the Top-K it returns.

The proxy is deliberately transparent (documented constants) rather than
precise: the plan delegates accuracy to the Stage-2 simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model

from .graph_model import device_inventory
from .spec_schema import PlannerSpec
from .types import Allocation, Device, Instance
from .utils import (
    estimate_kv_bytes_per_token,
    estimate_weight_bytes,
    get_logger,
    load_model_config,
)

log = get_logger("planner.milp")

# Coarse per-hardware relative compute throughput (unitless; A6000 = 1.0 baseline)
# and active power (W). Only used for the Stage-1 proxy ordering; refined by sim.
_HW_REL_THROUGHPUT = {
    "H100": 6.0, "A100": 3.0, "A6000": 1.0, "A40": 1.1,
    "A40x": 1.1, "A5000": 0.8, "RTX3090": 0.9, "RNGD": 1.5, "TPU-v6e-1": 2.0,
}
_HW_ACTIVE_POWER = {
    "H100": 700, "A100": 400, "A6000": 300, "A40": 300,
    "A40x": 300, "A5000": 230, "RTX3090": 350, "RNGD": 150, "TPU-v6e-1": 200,
}
# KV-cache headroom assumed available per instance, expressed in tokens, for the
# memory feasibility proxy (weights + this many tokens of KV must fit).
_KV_RESERVE_TOKENS = 8192


@dataclass(frozen=True)
class _Template:
    hardware: str
    node_id: str
    tp: int
    role: str | None  # "prefill" | "decode" | None
    mem_gb: float     # per-device memory
    rel_throughput: float
    power_w: float

    def devices_per_instance(self) -> int:
        return self.tp


def _memory_feasible(
    weight_bytes: float, kv_per_tok: float, tp: int, per_device_mem_gb: float
) -> bool:
    """weights (sharded over tp) + KV reserve must fit in the TP group's memory."""
    total_mem_bytes = tp * per_device_mem_gb * (1024 ** 3)
    need = weight_bytes / tp + kv_per_tok * _KV_RESERVE_TOKENS
    return need <= total_mem_bytes


def _enumerate_templates(spec: PlannerSpec, inv: list[Device]) -> list[_Template]:
    cfg = load_model_config(spec.model.name)
    weight_bytes = estimate_weight_bytes(cfg, spec.model.fp)
    kv_per_tok = estimate_kv_bytes_per_token(cfg, spec.model.fp)

    roles: list[str | None] = ["prefill", "decode"] if spec.search_space.pd_disaggregation else [None]
    templates: list[_Template] = []
    for dev in inv:
        for tp in spec.search_space.tp_choices:
            if tp > dev.count:
                continue
            if not _memory_feasible(weight_bytes, kv_per_tok, tp, dev.mem_gb):
                continue
            for role in roles:
                templates.append(
                    _Template(
                        hardware=dev.hardware,
                        node_id=dev.node_id,
                        tp=tp,
                        role=role,
                        mem_gb=dev.mem_gb,
                        rel_throughput=_HW_REL_THROUGHPUT.get(dev.hardware, 1.0),
                        power_w=_HW_ACTIVE_POWER.get(dev.hardware, 300),
                    )
                )
    return templates


def _allocation_from_counts(counts: dict[int, int], templates: list[_Template],
                            spec: PlannerSpec, score: float) -> Allocation:
    instances: list[Instance] = []
    for idx, n in counts.items():
        if n <= 0:
            continue
        t = templates[idx]
        # one Instance object carries all replicas of this template (npu_num = n*tp)
        instances.append(
            Instance(
                node_id=t.node_id,
                hardware=t.hardware,
                model_name=spec.model.name,
                tp=t.tp,
                npu_num=n * t.tp,
                npu_mem_gb=t.mem_gb,
                pd_type=t.role,
            )
        )
    return Allocation(instances=instances, proxy_score=score,
                      meta={"stage": "milp", "counts": dict(counts)})


def solve(spec: PlannerSpec, graph=None, power_lambda: float = 0.001) -> list[Allocation]:
    """Return up to ``spec.solver.top_k`` distinct candidate allocations.

    ``graph`` is accepted for API symmetry (device inventory can be derived from
    it); if None the inventory is built from the spec's topology.
    """
    if graph is not None:
        inv = device_inventory(graph)
    else:
        from .graph_model import build_graph
        inv = device_inventory(build_graph(spec))

    templates = _enumerate_templates(spec, inv)
    if not templates:
        log.warning("no feasible instance templates (memory/tp constraints too tight)")
        return []

    ss = spec.search_space
    allocations: list[Allocation] = []
    forbidden: list[dict[int, int]] = []

    for _ in range(max(1, spec.solver.top_k)):
        model = cp_model.CpModel()
        # count var per template
        n = {}
        for i, t in enumerate(templates):
            # upper bound: how many tp-groups of this hw fit on its node
            hw_total = sum(d.count for d in inv if d.hardware == t.hardware and d.node_id == t.node_id)
            n[i] = model.NewIntVar(0, hw_total // t.tp, f"n_{i}")

        # device-capacity constraint per (node, hardware)
        for dev in inv:
            using = [n[i] * templates[i].tp for i, t in enumerate(templates)
                     if t.hardware == dev.hardware and t.node_id == dev.node_id]
            if using:
                model.Add(sum(using) <= dev.count)

        # at least one instance overall
        model.Add(sum(n.values()) >= 1)

        # P/D balance window (xPyD): total prefill / decode instance counts in range
        if ss.pd_disaggregation:
            prefill_terms = [n[i] for i, t in enumerate(templates) if t.role == "prefill"]
            decode_terms = [n[i] for i, t in enumerate(templates) if t.role == "decode"]
            if prefill_terms:
                model.Add(sum(prefill_terms) >= ss.xpyd_prefill_range[0])
                model.Add(sum(prefill_terms) <= ss.xpyd_prefill_range[1])
            if decode_terms:
                model.Add(sum(decode_terms) >= ss.xpyd_decode_range[0])
                model.Add(sum(decode_terms) <= ss.xpyd_decode_range[1])

        # no-good cuts to force distinct successive solutions
        for fb in forbidden:
            # forbid the exact count vector: sum of |n_i - v_i| >= 1
            diff_bools = []
            for i, v in fb.items():
                b = model.NewBoolVar(f"neq_{i}_{len(diff_bools)}")
                model.Add(n[i] != v).OnlyEnforceIf(b)
                model.Add(n[i] == v).OnlyEnforceIf(b.Not())
                diff_bools.append(b)
            model.AddBoolOr(diff_bools)

        # objective: throughput proxy - lambda * power   (scaled to integers)
        SCALE = 1000
        obj_terms = []
        for i, t in enumerate(templates):
            # per-instance proxy throughput ~ rel_throughput * tp (aggregate compute)
            tput = t.rel_throughput * t.tp
            pwr = t.power_w * t.tp
            coeff = int(round((tput - power_lambda * pwr) * SCALE))
            obj_terms.append(coeff * n[i])
        model.Maximize(sum(obj_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(1, spec.solver.time_limit_sec)
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        counts = {i: int(solver.Value(n[i])) for i in n if solver.Value(n[i]) > 0}
        if not counts:
            break
        score = solver.ObjectiveValue() / SCALE
        allocations.append(_allocation_from_counts(counts, templates, spec, score))
        forbidden.append(counts)

    log.info("Stage-1 produced %d candidate allocation(s)", len(allocations))
    return allocations
