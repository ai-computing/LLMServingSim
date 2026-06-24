"""Unit tests for webapp.dse.core.interconnect — fabric / catalog resolution."""
from webapp.cluster_builder import ConfigSpec, InstanceSpec, build_cluster_json
from webapp.dse.core.interconnect import (
    _decompose_tp,
    load_fabrics,
    resolve_interconnect,
)

A40_TIERS = [
    {"name": "nvlink", "size": 2, "bw_gbs": 52.8, "latency_ns": 0},
    {"name": "pcie", "size": 2, "bw_gbs": 24.5, "latency_ns": 0},
    {"name": "qpi", "size": 2, "bw_gbs": 21.0, "latency_ns": 0},
]
HW_META = {
    "A40": {"interconnect_bw_gbs": 52.8},
    "A6000": {"interconnect_bw_gbs": 32},
    "H100": {"interconnect_bw_gbs": 900},
}


def _spec(hw, tp, pp=1, dp=1):
    inst = InstanceSpec(hardware=hw, model="m", npu_num=tp * pp,
                        npu_group=pp, pd_type=None)
    return ConfigSpec(label=f"{hw}_tp{tp}", instances=[inst],
                      tp=tp, pp=pp, dp=dp, pd_layout="—")


# ---- _decompose_tp ---------------------------------------------------------

def test_decompose_powers_of_two():
    assert _decompose_tp(2, A40_TIERS) == ([2], [52.8], [0.0])
    assert _decompose_tp(4, A40_TIERS) == ([2, 2], [52.8, 24.5], [0.0, 0.0])
    assert _decompose_tp(8, A40_TIERS) == ([2, 2, 2], [52.8, 24.5, 21.0], [0.0, 0.0, 0.0])


def test_decompose_tp1_is_empty():
    assert _decompose_tp(1, A40_TIERS) == ([], [], [])


def test_decompose_non_factorable_returns_none():
    # TP3 cannot be placed on size-2 tiers.
    assert _decompose_tp(3, A40_TIERS) is None


def test_decompose_tier_larger_than_remaining():
    # Single 8-wide tier; TP4 consumes only the divisible part.
    tiers = [{"size": 8, "bw_gbs": 600.0, "latency_ns": 0}]
    assert _decompose_tp(4, tiers) == ([4], [600.0], [0.0])


# ---- resolve_interconnect: fabric mode ------------------------------------

def test_fabric_tp8_full_hierarchy():
    ic = resolve_interconnect(_spec("A40", 8), {"tiers": A40_TIERS}, HW_META)
    assert ic["tp_group_shape"] == [2, 2, 2]
    assert ic["link_bw"] == [52.8, 24.5, 21.0]


def test_fabric_tp4_partial_hierarchy():
    ic = resolve_interconnect(_spec("A40", 4), {"tiers": A40_TIERS}, HW_META)
    assert ic["tp_group_shape"] == [2, 2]
    assert ic["link_bw"] == [52.8, 24.5]


def test_fabric_tp1_scalar_slowest_tier():
    # No intra-group collective → slowest (outermost) tier as scalar, no shape.
    ic = resolve_interconnect(_spec("A40", 1), {"tiers": A40_TIERS}, HW_META)
    assert ic["tp_group_shape"] is None
    assert ic["link_bw"] == 21.0


def test_fabric_non_factorable_falls_back_to_slowest_scalar():
    ic = resolve_interconnect(_spec("A40", 3), {"tiers": A40_TIERS}, HW_META)
    assert ic["tp_group_shape"] is None
    assert ic["link_bw"] == 21.0  # slowest tier


# ---- resolve_interconnect: catalog scalar fallback ------------------------

def test_catalog_fallback_homogeneous():
    ic = resolve_interconnect(_spec("H100", 2), None, HW_META)
    assert ic == {"link_bw": 900.0, "link_latency": 0, "tp_group_shape": None}


def test_catalog_fallback_heterogeneous_uses_min():
    het = ConfigSpec(
        label="het",
        instances=[InstanceSpec("H100", "m", 2, 1, None),
                   InstanceSpec("A6000", "m", 2, 1, None)],
        tp=2, pp=1, dp=2, pd_layout="—",
    )
    ic = resolve_interconnect(het, None, HW_META)
    assert ic["link_bw"] == 32.0  # min(900, 32)


def test_catalog_fallback_unknown_hw_returns_empty():
    ic = resolve_interconnect(_spec("UNKNOWN", 2), None, {})
    assert ic == {}


# ---- end-to-end with build_cluster_json -----------------------------------

def test_build_cluster_json_embeds_tp_group_shape():
    ic = resolve_interconnect(_spec("A40", 8), {"tiers": A40_TIERS}, HW_META)
    cj = build_cluster_json(
        _spec("A40", 8),
        {"mem_size": 256, "mem_bw": 256, "mem_latency": 0},
        ic["link_bw"], ic["link_latency"],
        tp_group_shape=ic["tp_group_shape"],
    )
    assert cj["tp_group_shape"] == [2, 2, 2]
    assert cj["link_bw"] == [52.8, 24.5, 21.0]


def test_build_cluster_json_no_shape_when_scalar():
    cj = build_cluster_json(
        _spec("H100", 2),
        {"mem_size": 256, "mem_bw": 256, "mem_latency": 0},
        900.0, 0, tp_group_shape=None,
    )
    assert "tp_group_shape" not in cj
    assert cj["link_bw"] == 900.0


# ---- fabrics.yaml presets load --------------------------------------------

def test_load_fabrics_has_a40_preset():
    fabs = load_fabrics()
    assert "a40_8gpu_2socket" in fabs
    tiers = fabs["a40_8gpu_2socket"]["tiers"]
    assert [t["size"] for t in tiers] == [2, 2, 2]
