# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

LLMServingSim is a discrete-event cycle-accurate simulator for LLM inference serving. It models heterogeneous hardware (NPU, CPU, CXL, PIM), multi-instance deployments, Prefill/Decode disaggregation, MoE expert routing, and prefix caching. The simulation backend is [ASTRA-Sim](https://github.com/astra-sim/astra-sim) (network/compute analytical model) with [Chakra](https://github.com/mlcommons/chakra) as the execution graph frontend.

## Build

Requires cloning with submodules (`--recurse-submodules`). Setup is Docker-first:

```bash
./docker.sh       # configure and launch Docker environment
./compile.sh      # install Chakra + build ASTRA-Sim analytical binary
```

`compile.sh` installs Chakra from `astra-sim/extern/graph_frontend/chakra` and compiles the analytical backend at `astra-sim/build/astra_analytical/`. The ns3 backend is present but commented out (work in progress).

## Running a simulation

```bash
python main.py \
    --cluster-config 'cluster_config/single_node_single_instance.json' \
    --fp 16 --block-size 16 \
    --dataset 'dataset/sharegpt_req100_rate10_llama.jsonl' \
    --output 'output/example_run.csv' \
    --num-req 100 --log-interval 1.0
```

See `run.sh` for ready-made examples covering every feature (multi-instance, P/D, MoE, prefix caching, CXL, PIM, power modeling, sub-batch interleaving).

Add `--log-level DEBUG` to print per-layer trace decisions to stderr during a run.

**Important path quirk**: `main.py` calls `os.chdir("astra-sim")` at startup. All relative paths in the ASTRA-Sim-facing code (trace files, workload files, config files) are relative to `astra-sim/`, not the repo root. Dataset and cluster-config paths passed on the CLI are relative to the repo root and are adjusted by `config_builder.py` with `../` prefix.

## Simulation loop architecture

The simulation is a tight Python ↔ ASTRA-Sim IPC loop:

1. **`main.py`** initializes one `Scheduler` per instance, a global `Router`, and a `Controller`, then spawns the ASTRA-Sim binary as a subprocess.
2. Each loop iteration: read ASTRA-Sim stdout → `scheduler.schedule()` → `generate_trace()` → `generate_graph()` → write workload path to ASTRA-Sim stdin.
3. ASTRA-Sim returns a cycle count and NPU/batch IDs; `main.py` maps these back to instance IDs via `npu2inst_mapping`.

Key modules in `inference_serving/`:

| Module | Responsibility |
|---|---|
| `config_builder.py` | Parses `cluster_config/*.json`, writes ASTRA-Sim input YAMLs/JSONs (`network.yml`, `system.json`, `memory_expansion.json`) |
| `scheduler.py` | Per-instance Orca/vLLM-style continuous batching scheduler; owns a `MemoryModel` |
| `router.py` | Distributes incoming requests across instances (RR / RAND / CUSTOM); handles prefill→decode transfer in P/D disaggregation |
| `trace_generator.py` | Builds per-batch `.txt` trace files in `astra-sim/inputs/trace/`; contains performance DB lookups and the MoE expert routing logic |
| `graph_generator.py` | Invokes `chakra.src.converter.converter` to turn `.txt` traces into Chakra execution graphs in `astra-sim/inputs/workload/` |
| `memory_model.py` | Tracks NPU/CPU/CXL memory usage, weight size, KV cache blocks, prefix caching via `RadixCache` |
| `controller.py` | Manages stdin/stdout IPC with the ASTRA-Sim subprocess |
| `radix_tree.py` | `RadixCache` implementing RadixAttention for prefix caching (NPU-tier and second-tier CPU/CXL pool) |
| `power_model.py` | Per-node energy/power modeling across NPU, CPU, DRAM, NIC, storage |
| `pim_model.py` | PIM device latency model; used when `--enable-attn-offloading` is set |
| `attn_utils.py` | Attention latency lookup/prediction utilities (scikit-learn predictor) |
| `gate_function.py` | MoE expert token routing (`GateRouter`) |
| `request.py` | `Request` dataclass: tokens, arrival time, per-request state machine |
| `logger.py` | Coloured ANSI logging helpers; `configure_logger(level)` / `get_logger(name)` |
| `dp_partition_sim.py` | Data-parallel partitioned run helper (used by `run_dp_partition.py`) |

## Cluster configuration

Each run is parameterised by a `cluster_config/*.json` file. Key structure:

- **Top level**: `num_nodes`, `link_bw` (GB/s), `link_latency` (ns), `nodes[]`
- **Per node**: `num_instances`, `cpu_mem` (`mem_size`/`mem_bw`/`mem_latency`), `instances[]`; optional `power`, `cxl_mem`
- **Per instance**: `model_name` (HuggingFace ID), `hardware` (must match a profile in `llm_profile/perf_models/`), `npu_mem`, `npu_num`, `npu_group`, `pd_type` (`"prefill"` / `"decode"` / `null`); optional `placement`, `pim_config`

`npu_num` = total NPUs for the instance; `npu_group` = NPUs forming one tensor-parallel group (the TP degree). Example: `npu_num=4, npu_group=4` → single TP-4 group; `npu_num=4, npu_group=1` → four independent TP-1 replicas. Mismatching these is the most common configuration error.

Model architecture configs live in `model_config/{vendor}/{model}/config.json` (HuggingFace format). `get_config(model_name)` in `utils.py` loads these.

## Adding a new model or hardware

1. **Profile**: run the PyTorch profiler in `llm_profile/` (see `llm_profile/README.md`) to produce per-layer latency CSVs and attention latency models under `llm_profile/perf_models/`.
2. **Memory model** (`inference_serving/memory_model.py`): update `calculate_sizes()` (tensor sizes per layer type) and `get_weight()` if the model architecture differs from Llama.
3. **Trace generator** (`inference_serving/trace_generator.py`): update `synthesize_trace()` to match the layer stack. Invariants to preserve:
   - ATTENTION layer must be separated per request.
   - Output size of layer *i* == input size of layer *i+1*.
   - ALLREDUCE ops must be placed at tensor-parallel synchronization points.

## Dataset format

JSONL, one request per line:
```json
{"input_toks": 128, "output_toks": 512, "arrival_time_ns": 0.0, "input_tok_ids": [1, 2, 3]}
```
`input_tok_ids` is used for prefix cache matching. Generate custom datasets with `dataset/sharegpt_parser.py`.

## Output metrics

- Stdout: throughput (tokens/s), per-instance memory utilisation, prefix hit ratios, power — at `--log-interval` seconds.
- CSV (`--output`): per-request TTFT, TPOT, ITL.

**TTFT definition differs from vLLM**: this simulator measures when computation of the first token *completes*, not when the client receives it, so values will be lower than vLLM's reported TTFT.

Time unit throughout the simulator is nanoseconds (1 GHz clock, `FREQ = 1_000_000_000`).

## Tests

```bash
pytest tests/           # full suite (webapp DSE subsystem)
pytest tests/dse/test_generator.py -v   # single file
```

Tests live in `tests/dse/` and cover `webapp.dse.core.*` (generator, stage-1 filters,
stage-2 predictor, ranker, schemas). The stage-2 tests use real profile data from
`llm_profile/perf_models/`.

## Webapp (DSE UI)

A FastAPI + uvicorn web UI for Design Space Exploration: sweep many hardware/batch
configurations in parallel and inspect Pareto-optimal results.

```bash
./script/serve_webapp.sh   # sets LD_LIBRARY_PATH + PATH, launches on port 8000 with --reload
```

`webapp/` modules:

| Module | Responsibility |
|---|---|
| `app.py` | FastAPI routes: scenario builder, sweep history, SSE event stream, Pareto chart |
| `runner.py` | Asyncio sweep orchestrator; pool of subprocess simulator runs |
| `cluster_builder.py` | Builds cluster-config JSON from `ConfigSpec`/`InstanceSpec` |
| `hardware_catalog.py` | Enumerates available hardware from `llm_profile/perf_models/` |
| `dse/` | Core DSE logic (candidate generator, stage-1 filters, stage-2 latency predictor, ranker) |
| `sim_cache.py` | Deduplicates repeated identical simulator runs |

Sweep outputs land in `output/web_sweeps/{sweep_id}/`. Concurrency is auto-tuned from
available CPU cores and free RAM (`webapp/config.py:compute_max_concurrent`).

## Evaluation / artifact reproduction

```bash
cd evaluation
bash figure_5.sh   # or figure_6.sh … figure_10.sh
bash run_all.sh    # full suite
bash compare.sh 5  # diff against preserved reference outputs
```

Reference outputs and PDFs are under `evaluation/artifacts/`.

## Validation against vLLM

```bash
cd validation
bash run_vllm_tp1.sh   # capture real vLLM metrics
bash run_sim.sh        # run simulator on the same workload
python compare.py      # diff TTFT/TPOT between vLLM and sim
```

Reference stdout captures are `validation/sim_tp*.txt`.
