"""End-to-end pipeline: spec -> graph -> MILP(top_k) -> render -> sim -> rank.

Stage-2 simulations of the K candidates run in a thread pool (each is an external
subprocess, so the GIL is not a bottleneck). Setting ``dry_run=True`` stops after
rendering, which is useful for testing Stage-1 + rendering without the simulator.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from . import config_renderer, milp_solver, objective
from .graph_model import build_graph
from .sim_evaluator import evaluate
from .spec_schema import PlannerSpec, load_spec
from .types import Allocation, Infeasible, Metrics
from .utils import get_logger, hash_obj

log = get_logger("planner.orchestrator")


@dataclass
class CandidateResult:
    run_id: str
    allocation: Allocation
    config_path: str
    cli_args: list[str]
    batch_tokens: int
    metrics: Optional[Metrics] = None
    infeasible_reason: Optional[str] = None
    passed: bool = False
    violations: list[str] = field(default_factory=list)
    score: float = float("-inf")


@dataclass
class PlannerResult:
    spec: PlannerSpec
    candidates: list[CandidateResult] = field(default_factory=list)
    pareto: list[str] = field(default_factory=list)  # run_ids on the Pareto front
    best: Optional[CandidateResult] = None
    dry_run: bool = False


def _expand(spec: PlannerSpec, allocations: list[Allocation], out_dir: Path):
    """Cartesian product of allocations x batch_tokens -> rendered candidates."""
    candidates: list[CandidateResult] = []
    for alloc in allocations:
        for bt in spec.search_space.batch_tokens_choices:
            base = hash_obj([alloc.signature(), bt])
            run_id = f"cand_{base}"
            rel_config, cli_args = config_renderer.render(
                alloc, spec, out_dir, run_id, batch_tokens=bt
            )
            candidates.append(
                CandidateResult(
                    run_id=run_id, allocation=alloc, config_path=rel_config,
                    cli_args=cli_args, batch_tokens=bt,
                )
            )
    return candidates


def run(
    spec_path: str | Path,
    out_dir: str | Path = "planner_out",
    jobs: int = 4,
    dry_run: bool = False,
    skip_repo_validation: bool = False,
    timeout_sec: int = 1800,
) -> PlannerResult:
    spec = load_spec(spec_path)
    if not skip_repo_validation:
        problems = spec.validate_against_repo()
        if problems:
            raise ValueError("spec validation failed:\n  - " + "\n  - ".join(problems))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph(spec)
    allocations = milp_solver.solve(spec, graph=graph)
    if not allocations:
        log.warning("Stage-1 found no feasible allocations")
        return PlannerResult(spec=spec, dry_run=dry_run)

    candidates = _expand(spec, allocations, out_dir)
    result = PlannerResult(spec=spec, candidates=candidates, dry_run=dry_run)
    if dry_run:
        log.info("dry run: rendered %d candidate config(s), skipping simulation", len(candidates))
        return result

    # Stage 2: evaluate in parallel
    def _eval(c: CandidateResult) -> CandidateResult:
        res: Union[Metrics, Infeasible] = evaluate(
            c.cli_args, c.run_id, out_dir, timeout_sec=timeout_sec
        )
        if isinstance(res, Infeasible):
            c.infeasible_reason = res.reason
            return c
        c.metrics = res
        c.passed, c.violations = objective.check_constraints(res, spec.requirements)
        c.score = objective.score(res, spec.requirements)
        return c

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(_eval, c): c for c in candidates}
        for fut in as_completed(futures):
            fut.result()  # populated in place

    # rank: passing candidates by score, then Pareto front
    passing = [c for c in candidates if c.passed and c.metrics is not None]
    pf_input = [(c.run_id, c.metrics) for c in passing]
    result.pareto = objective.pareto_front(pf_input, spec.requirements) if pf_input else []
    if passing:
        result.best = max(passing, key=lambda c: c.score)
    log.info(
        "Stage-2 done: %d candidates, %d passed SLO, %d on Pareto front",
        len(candidates), len(passing), len(result.pareto),
    )
    return result
