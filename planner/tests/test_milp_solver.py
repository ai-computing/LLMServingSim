from planner import milp_solver
from planner.spec_schema import PlannerSpec


def _mk_spec(**overrides):
    base = {
        "model": {"name": "meta-llama/Llama-3.1-8B", "fp": 16},
        "workload": {"dataset": "d.jsonl", "num_req": 10},
        "topology": {"nodes": [{"id": "n0", "devices": [{"name": "A6000", "count": 4, "mem_gb": 48}]}]},
        "search_space": {"tp_choices": [1, 2], "batch_tokens_choices": [2048]},
        "solver": {"top_k": 3, "time_limit_sec": 10},
    }
    base.update(overrides)
    return PlannerSpec.model_validate(base)


def test_solve_produces_candidates(spec):
    allocs = milp_solver.solve(spec)
    assert 1 <= len(allocs) <= spec.solver.top_k
    for a in allocs:
        assert a.instances
        # device usage never exceeds inventory
        assert a.total_devices() <= 6  # 2 H100 + 4 A6000


def test_candidates_are_distinct():
    allocs = milp_solver.solve(_mk_spec())
    sigs = {a.signature() for a in allocs}
    assert len(sigs) == len(allocs)  # no-good cuts => all distinct


def test_device_capacity_respected():
    spec = _mk_spec(
        topology={"nodes": [{"id": "n0", "devices": [{"name": "A6000", "count": 4, "mem_gb": 48}]}]},
    )
    for a in milp_solver.solve(spec):
        used = sum(i.npu_num for i in a.instances if i.hardware == "A6000")
        assert used <= 4


def test_tp_is_valid_divisor():
    for a in milp_solver.solve(_mk_spec()):
        for inst in a.instances:
            assert inst.tp <= inst.npu_num
            assert inst.npu_num % inst.tp == 0


def test_infeasible_when_memory_too_small():
    # 1 GB devices cannot hold an 8B model
    spec = _mk_spec(
        topology={"nodes": [{"id": "n0", "devices": [{"name": "A6000", "count": 4, "mem_gb": 1}]}]},
    )
    assert milp_solver.solve(spec) == []
