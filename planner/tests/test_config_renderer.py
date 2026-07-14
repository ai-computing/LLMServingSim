import json

from planner import config_renderer
from planner.types import Allocation, Instance
from planner.utils import REPO_ROOT


def _alloc():
    return Allocation(instances=[
        Instance(node_id="node0", hardware="H100", model_name="meta-llama/Llama-3.1-8B",
                 tp=2, npu_num=2, npu_mem_gb=80, pd_type=None),
        Instance(node_id="node0", hardware="A6000", model_name="meta-llama/Llama-3.1-8B",
                 tp=1, npu_num=4, npu_mem_gb=48, pd_type=None),
    ])


def _load(rel_config: str) -> dict:
    # rel_config is repo-root-relative (staged inside the repo for out-of-repo dirs)
    return json.loads((REPO_ROOT / rel_config).read_text())


def test_render_schema_fields(spec, tmp_path):
    rel, cli = config_renderer.render(_alloc(), spec, tmp_path, "run1", batch_tokens=4096)
    cfg = _load(rel)

    assert cfg["num_nodes"] == 1
    node = cfg["nodes"][0]
    assert node["num_instances"] == 2
    inst = node["instances"][0]
    # TP -> npu_group, device count -> npu_num
    assert inst["npu_group"] == 2
    assert inst["npu_num"] == 2
    assert inst["pd_type"] is None
    assert set(inst.keys()) >= {"model_name", "hardware", "npu_mem", "npu_num", "npu_group", "pd_type"}


def test_cli_args_contain_required_flags(spec, tmp_path):
    rel, cli = config_renderer.render(_alloc(), spec, tmp_path, "run2", batch_tokens=4096)
    assert "--cluster-config" in cli
    assert cli[cli.index("--cluster-config") + 1] == rel  # repo-relative
    assert "--max-num-batched-tokens" in cli
    assert cli[cli.index("--max-num-batched-tokens") + 1] == "4096"
    assert cli[cli.index("--fp") + 1] == "16"


def test_power_block_when_energy_objective(spec, tmp_path):
    # example fixture has a throughput objective only -> no power block
    rel, _ = config_renderer.render(_alloc(), spec, tmp_path, "run3")
    cfg = _load(rel)
    assert "power" not in cfg["nodes"][0]

    spec.requirements.objectives.append(type(spec.requirements.objectives[0])(
        metric="toks_per_wh", direction="max", weight=0.5))
    rel, _ = config_renderer.render(_alloc(), spec, tmp_path, "run4")
    cfg = _load(rel)
    assert "power" in cfg["nodes"][0]
    assert "H100" in cfg["nodes"][0]["power"]["npu"]


def test_tp_group_shape_serialized(spec, tmp_path):
    spec.topology.tp_group_shape = [2, 2]
    rel, _ = config_renderer.render(_alloc(), spec, tmp_path, "run5")
    cfg = _load(rel)
    assert cfg["tp_group_shape"] == [2, 2]
