# A40 / Llama-3.1-8B — LLMServingSim vs vLLM

**Setup:** single A40 (GPU0, TP=1), Llama-3.1-8B, dataset `sharegpt_req300_rate10_llama.jsonl`
(300 reqs, Poisson rate 10/s), fp16, block-size 16, `max-model-len` 4096.

- **Simulator:** `cluster_config/a40_1gpu_validation.json`, A40 perf profile in `llm_profile/perf_models/A40/`.
- **vLLM:** `vllm/vllm-openai:latest` in Docker, served on GPU0. Weights from a local
  Llama-3.1-8B-**Instruct** copy (HF CDN was throttled to ~0.6 MB/s here). Architecture, tokenizer,
  and — with `temperature=0` + fixed `max_tokens` — the compute are identical to base, so
  latency/throughput/power are unaffected by the base-vs-Instruct weight choice.
- **GPU power:** `nvidia-smi` 2 Hz during the vLLM replay; simulator power from its analytical
  power model (NPU component isolated for a fair GPU-to-GPU comparison).

## Results (300 requests)

| Metric | vLLM (measured) | Sim | Δ |
|---|---:|---:|---:|
| **TTFT** p50 | 348.2 ms | 49.3 ms | −85.8% |
| **TTFT** p99 | 3402.6 ms | 67.2 ms | −98.0% |
| **TPOT** p50 | 77.9 ms | 33.4 ms | −57.1% |
| **TPOT** p99 | 128.4 ms | 33.8 ms | −73.7% |
| **Throughput** (generation) | 1373.6 tok/s | 1554.2 tok/s | +13.2% |
| **Throughput** (prompt+gen) | 2484.4 tok/s | 2809.9 tok/s | +13.1% |
| **GPU power** (active mean) | 297.4 W | 299.7 W | **+0.8%** |
| GPU power p95 / max (measured) | 300.3 / 303.0 W | — | — |

Sim makespan 53.6 s vs vLLM wall 60.6 s.  Whole-system sim power (incl. CPU/DRAM/NIC/base) = 510.7 W.

## Reading the numbers

- **GPU power — excellent match (+0.8%).** The A40 ran pinned at ~300 W (its TDP) for the whole
  compute-bound serving window; the simulator's `active_power = 300 W` reproduces this almost exactly.
  (The `nvidia-smi` log only covers the active window, so its "idle" percentile is ~291 W, not true idle ~15 W.)

- **Throughput — close (+13%).** Sim is mildly optimistic on both generation and total token rate,
  consistent with it finishing the workload ~13% faster (53.6 s vs 60.6 s).

- **TPOT — sim underestimates ~57%.** Sim TPOT is essentially flat (~33 ms, p50≈p99), close to a
  *single-request* decode step, whereas real vLLM decode is 78–128 ms because per-token latency grows
  under continuous-batching / memory-bandwidth contention with many concurrent requests. The analytical
  model (per-layer/attention latencies profiled in isolation) does not capture that batch-level slowdown.

- **TTFT — sim much lower (−86%+), low correlation.** Two compounding reasons: (1) *definition* — the
  simulator records when first-token **computation completes**, excluding client/queue/scheduling time
  that vLLM's client-side TTFT includes (documented in `CLAUDE.md`); (2) vLLM shows heavy prefill
  queueing under this arrival rate (TTFT p99 = 3.4 s), which the sim's TTFT does not reflect.

## Takeaway

Power and aggregate throughput validate well (≤13%, power <1%). Per-request latency — especially TPOT
under load and absolute TTFT — is optimistic in the simulator; treat sim TTFT/TPOT as lower bounds and
prefer it for power/throughput-level design exploration rather than tail-latency SLO prediction.

## Reproduce

```bash
# Simulation (CPU, in astrasim container)
docker exec servingsim_docker bash -c 'cd /app/LLMServingSim && python3 main.py \
  --cluster-config cluster_config/a40_1gpu_validation.json --fp 16 --block-size 16 \
  --dataset dataset/sharegpt_req300_rate10_llama.jsonl \
  --output validation/sim_a40_tp1_results.csv --num-req 300 --log-interval 1.0'

# vLLM benchmark on GPU0 (+ GPU power log)
bash validation/run_vllm_a40_bench.sh

# Compare (numpy available in the astrasim container)
docker exec servingsim_docker bash -c 'cd /app/LLMServingSim && python3 validation/compare_a40.py'
```

Artifacts: `sim_a40_tp1_results.csv`, `sim_a40_tp1_stdout.txt`, `vllm_tp1_results.jsonl`,
`vllm_a40_tp1_power.csv`, `compare_a40_summary.csv`.

---

## Matched prefix-caching re-run (RadixAttention ↔ APC)

The original comparison above had **sim prefix caching OFF** while **vLLM APC was ON**
(`enable_prefix_caching=True`, vLLM V1 default). Re-checked with matched settings:

| Sim config | makespan | TTFT p50 | TPOT p50 | gen tput | prefix hit |
|---|---:|---:|---:|---:|---:|
| baseline (no flags) | 53.6 s | 49.3 ms | 33.4 ms | 1554 tok/s | — |
| `--enable-prefix-caching` | 53.6 s | 49.3 ms | 33.4 ms | 1554 tok/s | **4.56%** |
| `--enable-chunked-prefill` | 53.6 s | 49.3 ms | 33.4 ms | 1554 tok/s | — |
| both flags together | **1072 s+ (runaway)** | — | — | — | 98.73% (bogus) |

**Findings**
1. **RadixAttention/APC barely matters on this workload.** The dataset has ~no shared
   prefixes (299/300 unique prompts), so the sim's prefix-cache hit ratio is only **4.56%**
   and every metric is identical to the no-cache baseline. vLLM's APC is likewise inert here.
   ⇒ The sim-vs-vLLM numbers in the section above are unchanged by the prefix-caching mismatch.
2. **Simulator bug — prefix caching + chunked prefill cannot be combined.** Each flag alone
   completes normally (53.6 s makespan). With **both** enabled the run diverges: simulated
   makespan explodes 20×+ (1072 s and climbing) and the prefix-cache hit ratio reads a bogus
   98.73% (343 M / 348 M, far exceeding the 67 K dataset tokens) — i.e. the combined code path
   re-counts/re-processes cached tokens. This blocks a both-flags-on matched run; the matched
   comparison here uses `--enable-prefix-caching` only (the RadixAttention ↔ APC axis the
   question targets). Chunked prefill's effect on the vLLM side was independently minor.

Artifacts: `sim_a40_tp1_nocache_*`, `sim_a40_tp1_apconly_*`, `sim_a40_tp1_chunkonly_*`.

---

## TP=2 (2× A40) — Sim vs vLLM

Same workload (300 reqs), sim `cluster_config/a40_2gpu_tp2_validation.json` (`npu_num=2, npu_group=1`
⇒ npus_per_group=2 = TP-2, loads `perf_models/A40/.../tp2`), `--enable-prefix-caching`.
vLLM `--tensor-parallel-size 2` on GPU0+GPU1 (APC on by default). GPU power = both GPUs summed.

| Metric | vLLM (measured) | Sim | Δ |
|---|---:|---:|---:|
| **TTFT** p50 | 111.0 ms | 18.6 ms | −83.3% |
| **TTFT** p99 | 1713.2 ms | 44.3 ms | −97.4% |
| **TPOT** p50 | 31.8 ms | 10.9 ms | −65.8% |
| **TPOT** p99 | 42.6 ms | 12.1 ms | −71.7% |
| **Throughput** (generation) | 1864.4 tok/s | 2199.4 tok/s | +18.0% |
| **Throughput** (prompt+gen) | 3430.0 tok/s | 3976.4 tok/s | +15.9% |
| **GPU power** (2 GPUs summed, mean) | 577.5 W | 562.7 W | **−2.6%** |
| GPU power p95 / max (measured) | 603.4 W | 611.3 W | — |

Sim req/s 7.92 vs vLLM 6.98. Whole-system sim power = 773.8 W.

### TP=1 vs TP=2 scaling (both engines)

| | vLLM TP1 → TP2 | Sim TP1 → TP2 |
|---|---|---|
| Throughput (gen) | 1374 → 1864 tok/s (**1.36×**) | 1554 → 2199 tok/s (**1.42×**) |
| TPOT p50 | 77.9 → 31.8 ms (2.4× faster) | 33.4 → 10.9 ms (3.1× faster) |
| TTFT p50 | 348 → 111 ms (3.1× faster) | 49.3 → 18.6 ms (2.7× faster) |
| GPU power (total) | 297 → 578 W | 300 → 563 W |

**Findings (consistent with TP=1):**
- **GPU power matches excellently at both scales** (TP1 +0.8%, TP2 −2.6%); A40s run pinned near
  TDP and the sim's `active_power=300 W/NPU` reproduces both single- and dual-GPU totals.
- **Throughput tracks well** (sim +13% at TP1, +18% at TP2; mildly optimistic, slightly more so at TP2)
  and **both engines show similar TP-scaling** (~1.4× generation throughput from 1→2 GPUs).
- **Per-request latency stays optimistic** in the sim (TTFT/TPOT much lower) for the same reasons
  as TP=1 (TTFT definition + no queue/contention modeling; flat decode latency). The gap is even
  larger for TPOT at TP2 (−66%), i.e. the sim under-models cross-GPU/batch decode overhead.

Artifacts: `sim_a40_tp2_*`, `vllm_tp2_results.jsonl`, `vllm_a40_tp2_power.csv`, `compare_a40_tp2_summary.csv`.

---

## TP=2 network bandwidth correction (PCIe guess → measured NVLink)

Initial TP=2 used `link_bw = 32` GB/s (PCIe Gen4 assumption). `nvidia-smi topo -m` shows GPU0–GPU1
are actually **NVLink (NV4 = 4 bonded NVLinks)**, not PCIe. Measured P2P bandwidth (PyTorch
device-to-device copy, 512 MB × 50): **52.8 GB/s unidirectional** (~94% of A40 NVLink's ~56 GB/s/dir
spec). Config updated to `link_bw = 52.8` and TP=2 sim re-run.

**Effect of 32 → 52.8 GB/s on the simulator:**

| | link_bw=32 | link_bw=52.8 |
|---|---:|---:|
| makespan | 37.84 s | 37.80 s |
| TTFT p50 | 18.60 ms | 17.57 ms |
| TTFT p99 | 45.91 ms | 35.35 ms |
| TPOT p50 | 10.88 ms | 10.54 ms |

⇒ **Small effect.** Makespan/throughput essentially unchanged; the clearest change is TTFT p99
(−23%). For TP=2 Llama-3.1-8B the tensor-parallel all-reduce traffic is a small fraction of
compute time, so doubling link bandwidth barely moves aggregate metrics — the run is compute-bound.

**Final TP=2 comparison (sim with measured 52.8 GB/s NVLink):**

| Metric | vLLM | Sim | Δ |
|---|---:|---:|---:|
| TTFT p50 | 111.0 ms | 17.5 ms | −84.2% |
| TPOT p50 | 31.8 ms | 10.5 ms | −66.9% |
| Throughput (gen) | 1864.4 tok/s | 2202.0 tok/s | +18.1% |
| **GPU power** (2 GPUs summed) | 577.5 W | 576.0 W | **−0.3%** |

Interconnect: NVLink (NV4), measured 52.8 GB/s/dir. Artifacts: `sim_a40_tp2_link32_*` (old),
`sim_a40_tp2_*` (NVLink). The original TP=1/TP=2 conclusions are unchanged.

---

## TP=4 (4× A40) — extrapolated profile vs vLLM

**TP=4 profile was NOT measured** — it was extrapolated from measured tp1+tp2 via
`llm_profile/extrapolate_tp_profile.py` (per-layer tp1/tp2 ratio applied once; attention copied
from tp2). Before extrapolating, the measured **tp2 layers.csv was found corrupted** for
`gate_proj`, `up_proj`, `lm_head` (latency 70–190× too small at input≥2; reproducible profiler
bug, confirmed against A6000 reference where tp2≈tp1×0.5). Those 3 layers were repaired
(A40 tp1 × A6000 per-layer tp2/tp1 ratio ≈0.48–0.50) before extrapolation.

Sim `cluster_config/a40_4gpu_tp4_validation.json` (`npu_num=4, npu_group=1` ⇒ TP-4). Interconnect
for TP=4 spans **two NVLink pairs bridged by PCIe** (GPU0-1 NV4, GPU2-3 NV4, GPU0↔GPU2 = PCIe);
measured cross-pair P2P = **24.5 GB/s** (the TP-4 all-reduce bottleneck) → `link_bw = 24.5`.
vLLM `--tensor-parallel-size 4` on GPU0-3.

| Metric | vLLM (measured) | Sim (extrapolated) | Δ |
|---|---:|---:|---:|
| TTFT p50 | 153.9 ms | 21.2 ms | −86.2% |
| TPOT p50 | 41.5 ms | 12.4 ms | −70.0% |
| Throughput (generation) | 1904.0 tok/s | 2220.4 tok/s | +16.6% |
| **GPU power** (4 GPUs summed) | 819.8 W | 1056.3 W | **+28.8%** |

### Scaling across TP (both engines, same workload)

| | vLLM TP1 | vLLM TP2 | vLLM TP4 | Sim TP1 | Sim TP2 | Sim TP4 |
|---|---:|---:|---:|---:|---:|---:|
| Throughput gen (tok/s) | 1374 | 1864 | 1904 | 1554 | 2202 | 2220 |
| TPOT p50 (ms) | 77.9 | 31.8 | 41.5 | 33.4 | 10.9 | 12.4 |
| GPU power total (W) | 297 | 578 | 820 | 300 | 563 | 1056 |
| GPU power per-GPU (W) | 297 | 289 | 205 | 300 | 281 | 264 |

### Findings

1. **TP=4 gives ~no benefit for 8B on this workload — both engines agree.** vLLM throughput
   1864→1904 (TP2→TP4, +2%) and TPOT actually *worsens* (31.8→41.5 ms) from cross-pair PCIe
   all-reduce overhead; the sim likewise plateaus (2202→2220). An 8B model is too small to
   benefit from 4-way TP here. The extrapolated profile reproduces this plateau correctly.

2. **Extrapolated TP=4 behaves consistently** — throughput (+16.6%) and latency optimism match
   the *measured* TP1/TP2 error pattern (+13–18% tput, TTFT/TPOT optimistic). The extrapolation
   introduced no additional aggregate error beyond the simulator's existing biases.

3. **Power model breaks down at TP=4 (sim +29%).** Real A40s draw only ~205 W/GPU at TP4 (vs
   ~290 W at TP1/TP2) because TP-sharded GEMMs are smaller and under-utilize the SMs. The sim's
   power model uses a **flat `active_power = 300 W` regardless of per-kernel utilization**, so it
   overestimates at TP4. At TP1/TP2 the kernels saturated the GPU and the constant held (<3%
   error); at TP4 it does not. To fix, `active_power` should scale with effective compute
   utilization (or be profiled per TP degree).

Artifacts: `sim_a40_tp4_*`, `vllm_tp4_results.jsonl`, `vllm_a40_tp4_power.csv`,
`compare_a40_tp4_summary.csv`. Extrapolation inputs: repaired `tp2/layers.csv`
(`.corrupted.bak` = original), extrapolated `tp4/` (`.fromcorrupted.bak` = pre-repair).
