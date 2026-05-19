"""Configuration constants for the LLMServingSim webapp.

All paths are anchored to the repo root (parent of the webapp/ dir).
SIM_ENV mirrors the env vars used by script/run_a6000_4_sweep.sh so that
the AnalyticalAstra binary can find libprotobuf.so.23 and graph_generator.py
can locate the `python` symlink.
"""
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
LLM_PROFILE_DIR = REPO_ROOT / "llm_profile/perf_models"
CLUSTER_CONFIG_DIR = REPO_ROOT / "cluster_config"
DATASET_DIR = REPO_ROOT / "dataset"
OUTPUT_DIR = REPO_ROOT / "output/web_sweeps"
MAIN_PY = REPO_ROOT / "main.py"

# Simulator env -- required for AnalyticalAstra binary and `python` symlink
# Source: script/run_a6000_4_sweep.sh lines 11-13
SIM_ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": "/tmp/protobuf_prefix/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", ""),
    "PATH": os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", ""),
}

MAX_CONCURRENT = min(10, max(1, (os.cpu_count() or 4) // 2))
CONFIG_TIMEOUT_S = 600  # kill a config run after 10 min
SOFT_CAP = 20           # warn user if sweep exceeds this count

# Default hardware memory specs -- used when building cluster JSONs
# Read from cluster_config/single_node_*.json at startup; these are fallbacks
HW_DEFAULTS = {
    "A6000":     {"mem_size": 40,  "mem_bw": 768,  "mem_latency": 0},
    "H100":      {"mem_size": 80,  "mem_bw": 3000, "mem_latency": 0},
    "RNGD":      {"mem_size": 40,  "mem_bw": 1500, "mem_latency": 0},
    "RTX3090":   {"mem_size": 24,  "mem_bw": 936,  "mem_latency": 0},
    "TPU-v6e-1": {"mem_size": 16,  "mem_bw": 1640, "mem_latency": 0},
}
CPU_MEM_DEFAULT = {"mem_size": 128, "mem_bw": 256, "mem_latency": 0}
LINK_BW_DEFAULT = 112
LINK_LATENCY_DEFAULT = 0
