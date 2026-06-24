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

---

## Re-comparison with corrected (cuda_event) profiles

The original TP1/2/4 comparisons above used profiles measured with the buggy default
`record_function` method, which under-measured **both** layer GEMMs (gate/up/lm_head collapse)
**and** attention — prefill ×1.45, **decode ×6.65**. After re-profiling with `--profile-method
cuda_event` (root-cause fix; see "Profiler bug" section) and re-running the sims:

| Metric (sim vs vLLM Δ) | TP1 buggy → corrected | TP2 buggy → corrected | TP4 buggy → corrected |
|---|---|---|---|
| **TPOT p50** | −57% → **−30%** | −66% → **+38%** | −70% → **−3%** |
| **Throughput (gen)** | +13% → −13% | +18% → −26% | +17% → −22% |
| **TTFT p50** | −86% → −76% | −83% → −39% | −86% → −56% |
| **GPU power** | +0.8% → +0.8% | −0.3% → +1.3% | +29% → +35% |

Corrected absolute numbers (sim / vLLM):

| | TTFT p50 (ms) | TPOT p50 (ms) | gen tput (tok/s) | GPU power (W) |
|---|---|---|---|---|
| TP1 | 84.9 / 348.2 | 54.8 / 77.9 | 1194.8 / 1373.6 | 299.8 / 297.4 |
| TP2 | 67.7 / 111.0 | 43.8 / 31.8 | 1374.4 / 1864.4 | 585.1 / 577.5 |
| TP4 | 67.8 / 153.9 | 40.3 / 41.5 | 1491.0 / 1904.0 | 1103.6 / 819.8 |

### Findings
1. **TPOT optimism was a profiler artifact, not a model limitation.** Fixing decode-attention
   (×6.65 under-measurement) moved TPOT from −57…−70% to within ±3–38% — TP4 is now essentially
   exact (−3%). The simulator's per-token decode model is accurate once fed correct attention data.
2. **Throughput sign flipped** (+13…18% → −13…−26%): heavier corrected compute makes the sim now
   mildly *conservative* rather than optimistic. The sim also shows weaker TP scaling than vLLM
   (sim 1195→1374→1491 vs vLLM 1374→1864→1904) — vLLM's continuous batching extracts more
   throughput than the sim's batching model captures.
3. **Power: TP1/TP2 remain excellent (<1.5%)**; TP4 still over (+35%) due to the flat
   `active_power` power-model limitation (independent of the profiler bug).
4. **TTFT** still lower (definitional: sim = compute-complete, no queue/scheduling) but markedly
   closer than the buggy data (e.g. TP2 −83% → −39%).

Old buggy-data sim results preserved as `sim_a40_tp{1,2,4}_buggy_results.csv`.

---

## Llama-3.1-70B TP=4 — simulation (cross-hardware)

vLLM 70B was **not run here** (no local 70B weights; 141 GB HF download throttled, and serving
needs 4 GPUs loaded). Instead the 70B TP=4 sim is compared across hardware using the repo's
*measured* 70B tp4 profiles (A40 measured this session via cuda_event; A100/H100 from the repo).
Same workload (sharegpt 300 req). link_bw/power: A40 measured (PCIe 24.5 GB/s); A100/H100 from
datasheet (NVLink ~300/450 GB/s, TDP 400/700 W) — estimates, flagged.

| HW | makespan | gen tput (tok/s) | TTFT p50 | TPOT p50 | NPU power (4 GPU) | node energy | efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|
| A40  | 146.7 s | 568  | 423 ms | 194 ms | 1019 W | 187.8 kJ | 444 tok/kJ |
| A100 | 57.6 s  | 1446 | 68 ms  | 44 ms  | 1545 W | 104.6 kJ | 797 tok/kJ |
| H100 | 45.9 s  | 1812 | 45 ms  | 28 ms  | 2717 W | 137.3 kJ | 607 tok/kJ |

Throughput: A100 = 2.5× A40, H100 = 3.2× A40.

### Model scaling on A40 TP=4 (8B vs 70B)

| | makespan | TPOT p50 |
|---|---:|---:|
| 8B  | 55.8 s  | 40.3 ms |
| 70B | 146.7 s | 194.3 ms |

70B is ~2.6× slower makespan / ~4.8× higher per-token latency than 8B at TP=4 (≈8.8× params, but
TP=4 sharding + memory-bound decode compress it to ~5×).

### Findings
1. **A40 runs 70B TP=4 but is throughput/latency-limited**: 568 tok/s, TPOT 194 ms — usable for
   batch/offline but far from interactive SLOs. A100/H100 are 2.5–3.2× faster.
2. **A100 is the most energy-efficient** for 70B here (797 tok/kJ) — H100 is fastest but its 700 W
   TDP makes it less efficient per token than A100 (607 tok/kJ); A40 is least efficient (444).
3. **Caveat**: A40 numbers rest on measured profiles + this-session vLLM-validated 8B accuracy;
   A100/H100 link_bw/power are datasheet estimates, and 70B has no direct vLLM validation here.
   Treat cross-hardware as a *prediction* (relative ordering robust; absolute power approximate).

Artifacts: `sim_{a40,a100,h100}_tp4_70b_*`, `cluster_config/{a40,a100,h100}_4gpu_tp4_70b.json`.

---

## TP=8 extrapolation validity study (8B, 8× A40)

Goal: assess whether `extrapolate_tp_profile.py` produces usable tp8 profiles. Three-way check:
extrapolated tp8 vs directly-measured tp8 (cuda_event, logical TP on 1 GPU) vs real 8-GPU vLLM.
link_bw = 21 GB/s (measured cross-NUMA GPU0↔GPU4, the TP8 all-reduce bottleneck; GPU0-3 and
GPU4-7 sit on different NUMA nodes joined by SYS).

### Level 1 — extrapolation accuracy (extrapolated vs measured)
Per-layer latency MAE = **30%** (median 26%): the geometric tp1→tp2 ratio over-shrinks compute
layers that hit a kernel-launch floor at tp8 (act_fn −88%, down/up/o_proj −48..−54%), while
memory-bound layers are over-kept. Errors partially cancel → **layer-sum error −12.7%**, and the
end-to-end sim impact is **+10% gen / −18% TPOT / −9% makespan** (extrapolated vs measured-profile
sim). So the extrapolation is an acceptable *aggregate* approximation, but per-layer accuracy is
much worse than the large-model case (H100/70B tp4 ≈ 5%) — extrapolation degrades for
small-model × high-TP.

### Level 2 — sim vs reality (the dominant error at TP=8)

| metric | sim (extrap) | sim (measured) | vLLM (real) |
|---|---:|---:|---:|
| gen tput (tok/s) | 1570 | 1422 | **690** |
| TTFT p50 (ms) | 62 | 74 | **18832** |
| TPOT p50 (ms) | 37.7 | 46.0 | **252.5** |
| GPU power (8 GPU, W) | 2128 | 2153 | **892** |

sim(measured) vs vLLM: gen **+106%**, TPOT **−82%**. **Both** sim variants are wildly off — TP=8
on an 8B model *collapses* in reality (690 tok/s, below even TP1's 1374; 18.8 s TTFT) because the
8-way all-reduce over the cross-NUMA SYS link dominates, leaving the GPUs comm-stalled (~111 W
each, vs sim's flat ~265 W → power +140%). The simulator's single-`link_bw` FullyConnected
collective model cannot represent the hierarchical NVLink-pair / PCIe / cross-NUMA topology or
NCCL sync overhead, so it predicts TP=8 is fine.

### Conclusion
- **Extrapolation is valid on the compute side** (~±10–18% end-to-end vs measured profiles); good
  for large models / moderate TP, weaker (per-layer ~30%) for small-model × high-TP.
- **But at TP=8 the limiting error is the simulator's collective-communication model, not the
  profile extrapolation** — neither extrapolated nor measured profiles let the sim reproduce
  vLLM's TP=8 throughput/latency collapse. Improving high-TP fidelity needs a hierarchical
  interconnect + collective-overhead model, not better profiles.
- Practical guidance: trust sim (with measured or extrapolated profiles) up to TP that stays
  on fast intra-node links (here TP≤2 NVLink, TP4 mixed was already +35% power off); treat
  cross-NUMA high-TP predictions as unreliable.

Artifacts: `sim_a40_tp8_{extrap,measured}_*`, `vllm_a40_tp8_*`, `compare_tp8.py`,
`cluster_config/{a40,a40x}_8gpu_tp8.json`; extrapolated profile preserved under perf_models/A40x.

## TP=8 with hierarchical 3-tier interconnect (70B, 8× A40) — calibration (2026-06-24)

Follow-up to the TP=8 study above, which concluded high-TP fidelity needs *a hierarchical
interconnect + collective-overhead model, not better profiles*. The hierarchical model now
exists (`config_builder._create_network_config` + top-level `tp_group_shape`; fabric preset
`a40_8gpu_2socket` = `cluster_config/a40_8gpu_tp8_70b_3tier.json`), with **measured** (not
extrapolated) tp8 70B profiles and the 3-tier topology
`npus_count=[2,2,2]`, `link_bw=[52.8, 24.5, 21.0]` (NVLink pair / intra-NUMA PCIe / cross-socket QPI).
Question: does representing the topology correctly close the sim↔vLLM gap? Model:
Llama-3.1-70B (sim base / vLLM Instruct — architecturally identical), vLLM 0.8.4, TP=8, gpu_util 0.9.

### Saturated serving (ShareGPT, 100 req, arrival rate 10/s)
Identical workload both engines (21,027 input / ~23,339 output tokens).

| metric | sim (3-tier, measured) | vLLM (real) | sim error |
|---|---:|---:|---:|
| total throughput (tok/s) | 619.8 | 305.8 | **+103 %** |
| gen throughput (tok/s) | 326.0 | 160.9 | +103 % |
| request throughput (req/s) | 1.40 | 0.69 | +103 % |
| makespan (s) | 71.6 | 145.1 | −51 % |
| TTFT p50 (ms) | 173.4 | 22 588 | (queueing-dominated — see below) |
| TPOT p50 (ms) | 96.9 | 287.9 | −66 % |

### Unsaturated (15 req, 15 s spacing → concurrency ≈ 1, output capped 32 tok)
Isolates per-request behaviour from queue buildup. TTFT is prefill-only and load-independent.

| metric | sim (3-tier) | vLLM (real) | sim vs vLLM |
|---|---:|---:|---:|
| TTFT p50 (ms) | 92.0 | 186.8 | sim −51 % (≈2× optimistic) |
| TPOT p50 (ms) | 85.5 | 43.4 | sim **+97 %** (2× *pessimistic*) |

### Findings
- **The 3-tier topology fixes representation, not magnitude.** The sim now *expresses* the
  NVLink/PCIe/QPI hierarchy, but on the saturated serving workload it is still **~2.0× optimistic
  on throughput** (consistent across total/gen/req — the makespan is exactly half).
- **TTFT: the 125× saturated gap is almost entirely queueing.** Drop the arrival rate and vLLM
  TTFT collapses 22 588 ms → 187 ms; against sim's 92 ms that is a clean ~2× (prefill +
  scheduling overhead the analytical model underestimates), not a structural error.
- **TPOT is load-dependent and the sim does not model it.** vLLM single-stream decode is
  *faster* than sim (43 vs 85 ms — the analytical per-token cost is conservative), but under
  saturation it balloons to 288 ms (batch contention + per-step all-reduce over PCIe/QPI) while
  sim stays flat at ~90–97 ms. The sim captures neither the low-load speed nor the high-load
  degradation.

### Calibration factor (A40 8-GPU 2-socket, 70B, TP=8)
For serving-throughput estimates on this fabric, scale sim output by **÷2.0** (sim ≈ 2.0× real):

| quantity | sim → real |
|---|---|
| throughput (tok/s, req/s) | × **0.49** |
| makespan / wall time | × **2.0** |
| TPOT under load | not a constant factor — sim ≈ flat; real scales with concurrency (43 ms isolated → 288 ms saturated) |
| TTFT | add queueing (sim omits it); isolated prefill sim × ≈2.0 |

Treat the throughput factor as a first-order correction for *this* fabric/model/TP point only;
the latency behaviour needs a load-dependent collective-overhead model to be predictive.

Artifacts: `cluster_config/a40_8gpu_tp8_70b_3tier.json`, `docs/dse/fabrics.yaml` (a40_8gpu_2socket),
`output/{bench_tp8_70b_3tier,sim_a40_2socket_tp8_70b,sim_lowrate_ttft}.csv`,
`validation/vllm_a40_tp8_lowrate_results.jsonl`, `dataset/sharegpt_lowrate_ttft15.jsonl`;
measured tp8 profiles under `perf_models/A40/meta-llama/Llama-3.1-70B/tp8/`.

### TP=4 control point (70B, 4× A40, intra-socket) — the error is the QPI hop, not TP itself

Same comparison at TP=4 with the 2-tier fabric `[2,2]` = NVLink 52.8 / intra-NUMA PCIe 24.5
(`cluster_config/a40_4gpu_tp4_70b_2tier.json`, GPUs 0–3 on NUMA node 0 — the all-reduce never
crosses QPI). Saturated ShareGPT, 100 req. (vLLM generated ~7% fewer output tokens due to early
EOS; rate-normalised metrics unaffected.)

| metric | sim (2-tier) | vLLM (real) | sim error |
|---|---:|---:|---:|
| total throughput (tok/s) | 420.9 | 498.8 | **−16 % (pessimistic)** |
| gen throughput (tok/s) | 221.4 | 253.4 | −13 % |
| request throughput (req/s) | 0.95 | 1.17 | −19 % |
| makespan (s) | 105.4 | 85.7 | +23 % |
| TPOT p50 (ms) | 145.8 | 152.6 | **−4 % (nearly exact)** |
| TTFT p50 (ms) | 252.2 | 3 553 | queueing (saturated) |

**At TP=4 the simulator is accurate** (TPOT within 4 %, throughput within ~16 % and on the
*conservative* side) — the opposite of TP=8's +103 % optimism. The error is not a function of TP
degree but of whether the collective traverses the slow cross-socket QPI link.

The cleanest evidence is that the sim **inverts the TP4↔TP8 ordering**:

| throughput (tok/s) | TP4 | TP8 | verdict |
|---|---:|---:|---|
| simulator | 420.9 | 619.8 | predicts TP8 **faster** |
| real vLLM | 498.8 | 305.8 | TP4 actually **1.6× faster** |

On real A40 hardware TP=8 is *slower* than TP=4 because the 8-way all-reduce bottlenecks on
cross-socket QPI (21 GB/s); the sim underestimates that sync cost and predicts the wrong winner.
Consequence for calibration: **no single factor** — a correction is needed only for configs whose
collective crosses QPI (≈1.0× for intra-socket TP≤4, ≈0.49× for cross-socket TP8 on this box).

Artifacts: `cluster_config/a40_4gpu_tp4_70b_2tier.json`, `output/sim_tp4_2tier.csv`,
`validation/vllm_a40_tp4_70b_sharegpt100_results.jsonl`.
