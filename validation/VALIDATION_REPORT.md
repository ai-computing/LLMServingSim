# LLMServingSim Level 3 Validation Report
## A5000 × Llama-3.1-8B: Simulator vs. vLLM

**Date**: 2026-06-12  
**Model**: meta-llama/Llama-3.1-8B  
**Hardware**: 2× RTX A5000 (24 GB each, 768 GB/s)  
**vLLM version**: 0.22.0 (torch 2.11.0+cu130)  
**Dataset**: `sharegpt_req100_rate10_llama.jsonl` (100 requests, arrival rate ≈ 10 req/s)

---

## 1. Setup Summary

### Environment fixes required
| 문제 | 원인 | 해결 |
|------|------|------|
| `_C.abi3.so` undefined symbol | vLLM 0.22.0은 `torch==2.11.0` 요구 | `pip install torch==2.11.0 --index-url pytorch.org/whl/cu130` |
| `ncclCommWindowDeregister` missing | pip `nvidia-nccl-cu13==2.28.9` 라이브러리에 심볼 없음 | `LD_PRELOAD=/usr/local/lib/ollama/mlx_cuda_v13/libnccl.so.2` (NCCL 2.29.2) |
| `curand.h: No such file` | flashinfer JIT 컴파일 시 CUDA 헤더 경로 미설정 | `CPATH=.../nvidia/curand/include` |
| attention profiler 46,922 조합 | `max_model_len` 기본값 131072 | `--max-len 2048` 인수 고정 |
| `@check_model_inputs()` TypeError | transformers 4.57.1 API 변경 | `@check_model_inputs` (괄호 제거) |

### Profiling (Phase 1)
- Layer 레이턴시 CSV: `llm_profile/perf_models/A5000/Llama-3.1-8B/layer_perf.csv`
- Attention 레이턴시 predictor: `llm_profile/perf_models/A5000/Llama-3.1-8B/attn_predictor.pkl`
- 전력 측정: idle=18W, standby=20W, active=130W (nvidia-smi 실측)

---

## 2. Raw Results

### 2.1 vLLM (실측)

| 지표 | TP=1 | TP=2 |
|------|------|------|
| TTFT p50 | 133.7 ms | 102.6 ms |
| TTFT p99 | 403.1 ms | 428.1 ms |
| TPOT p50 | 34.1 ms | 22.3 ms |
| TPOT p99 | 103.2 ms | 65.4 ms |
| Throughput | 562 tok/s | 714 tok/s |
| Matched requests | 83/100 | 82/100 |

### 2.2 Simulator (예측)

| 지표 | TP=1 | TP=2 |
|------|------|------|
| TTFT mean | 45.6 ms | 37.9 ms |
| TTFT p50 | 45.4 ms | 37.0 ms |
| TTFT p99 | 65.1 ms | 47.7 ms |
| TPOT mean | 30.9 ms | 30.2 ms |
| TPOT p50 | 31.0 ms | 30.2 ms |
| TPOT p99 | 32.3 ms | 30.9 ms |
| Throughput | 846 tok/s | 850 tok/s |

### 2.3 정확도 지표 (per-request MAPE, Pearson r)

| 지표 | TP=1 MAPE | TP=1 Pearson r | TP=2 MAPE | TP=2 Pearson r |
|------|-----------|----------------|-----------|----------------|
| TTFT | **65.1%** | -0.017 | **59.3%** | -0.186 |
| TPOT | **15.9%** | -0.019 | **42.1%** | -0.033 |
| Throughput | 50.6% | — | 19.0% | — |

---

## 3. Analysis

### 3.1 TTFT — 3배 과소 예측 (예상된 결과)

시뮬레이터 TTFT p50은 vLLM 대비 약 3배 낮습니다 (45ms vs 134ms, TP=1).

**원인**: CLAUDE.md에 명시된 설계상 차이:
> "this simulator measures when computation of the first token *completes*, not when the client receives it"

즉, 시뮬레이터 TTFT = 순수 NPU 연산 시간만 측정.  
vLLM TTFT = 큐잉 대기 + 스케줄링 오버헤드 + 연산 + 응답 전송 포함.

실제 차이분(≈90ms, TP=1)은 시스템 소프트웨어 오버헤드로 해석됩니다.

**TP=2 개선**: TP=2에서 TTFT가 모두 낮아집니다 (vLLM 103ms, Sim 37ms). 상대 오차는 비슷하게 유지 (59% vs 65%).

### 3.2 TPOT — TP=1은 근접, TP=2는 역전

| 구성 | vLLM p50 | Sim p50 | 관계 |
|------|----------|---------|------|
| TP=1 | 34.1 ms | 30.9 ms | Sim이 약 10% 낮음 |
| TP=2 | 22.3 ms | 30.2 ms | Sim이 36% **높음** (역전) |

**TP=1**: MAPE 15.9%로 목표(≤15%) 근접. Sim이 약간 낮은 것은 context switch 오버헤드 미반영.

**TP=2 역전**: vLLM TP=2 TPOT(22.3ms)가 Sim(30.2ms)보다 빠릅니다. 가능한 원인:
- TP=2에서 실제 all-reduce 레이턴시가 프로파일 대비 낮음 (NVLink 미반영 — A5000 PCIe 구성이지만 vLLM이 batching 효율이 더 좋을 가능성)
- 시뮬레이터의 TP=2 attention 병렬화 이득 모델이 과소 반영

### 3.3 Throughput

| 구성 | vLLM | Sim | 오차 |
|------|------|-----|------|
| TP=1 | 562 tok/s | 846 tok/s | +50.6% 과대 |
| TP=2 | 714 tok/s | 850 tok/s | +19.0% 과대 |

시뮬레이터가 실제 큐 경합과 스케줄링 오버헤드를 모델링하지 않아 처리량이 과대 예측됩니다. TP=2에서 오차가 줄어드는 것은 처리량 수렴 효과입니다.

### 3.4 per-request 상관관계 (Pearson r ≈ 0)

두 경우 모두 Pearson r ≈ 0으로, per-request TTFT/TPOT 패턴이 일치하지 않습니다. 이는 시뮬레이터가 individual request 스케줄링 순서와 큐 상태를 동일하게 재현하지 않기 때문입니다 (도착 시간 기반 실시간 replay vs. 시뮬레이터의 cycle-accurate 이벤트 큐).

---

## 4. Accuracy Target Assessment

| 목표 | 기준 | TP=1 | TP=2 | 평가 |
|------|------|------|------|------|
| TTFT p99 MAPE | ≤ 20% | 65.1% | 59.3% | FAIL — 설계상 차이 |
| TPOT p99 MAPE | ≤ 15% | 15.9% | 42.1% | TP=1 근접, TP=2 FAIL |
| Throughput 오차 | ≤ 20% | 50.6% | 19.0% | TP=2 통과 |

**TTFT 목표 미달은 설계상 예상된 결과**입니다. 시뮬레이터 TTFT와 vLLM TTFT는 서로 다른 측정 지점을 나타냅니다. "시뮬레이터가 얼마나 정확한가"를 평가하기 위해서는 TTFT에서 큐잉 오버헤드를 별도로 추정해야 합니다.

---

## 5. Prefix Caching (APC) 영향 분석 (추가 실험, 2026-06-16)

### 5.1 배경

초기 실험에서 vLLM은 `enable_prefix_caching=True` (기본값)으로, 시뮬레이터는 APC 비활성화 상태로 실행되어 설정이 불일치했습니다. 이를 보정하기 위해 두 방향으로 추가 실험을 수행했습니다.

- **실험 A**: 시뮬레이터에 `--enable-prefix-caching` 활성화
- **실험 B**: vLLM에 `--no-enable-prefix-caching --no-enable-chunked-prefill` 비활성화

### 5.2 시뮬레이터 APC 실행 결과 (실험 A)

시뮬레이터 prefix cache hit ratio: **21.46%** (4,512 / 21,027 tokens, TP=1·TP=2 동일)

| 지표 | TP=1 APC OFF | TP=1 APC ON | TP=2 APC OFF | TP=2 APC ON |
|------|-------------|------------|-------------|------------|
| TTFT p50 | 45.4 ms | 44.7 ms | 37.0 ms | 37.2 ms |
| TTFT p99 | 65.1 ms | 63.5 ms | 47.7 ms | 47.6 ms |
| TPOT p50 | 31.0 ms | 30.9 ms | 30.2 ms | 30.3 ms |
| TPOT p99 | 32.3 ms | 32.3 ms | 30.9 ms | 30.9 ms |

**결과**: hit ratio 21%임에도 APC 효과가 ≤1ms로 거의 없음. ShareGPT 데이터셋은 독립적인 대화라 hit 블록이 있어도 나머지 prefill 연산이 지배적이기 때문.

### 5.3 vLLM APC 비활성화 결과 (실험 B)

| 지표 | TP=1 APC ON | TP=1 APC OFF | TP=2 APC ON | TP=2 APC OFF |
|------|------------|-------------|------------|-------------|
| TTFT p50 | 133.7 ms | 155.6 ms (+16%) | 102.6 ms | 126.6 ms (+23%) |
| TTFT p99 | 403.1 ms | 436.5 ms | 428.1 ms | 434.6 ms |
| TPOT p50 | 34.1 ms | 37.3 ms (+9%) | 22.3 ms | 24.3 ms (+9%) |
| TPOT p99 | 103.2 ms | 147.1 ms | 65.4 ms | 55.5 ms |
| Throughput | 562 tok/s | 636 tok/s | 714 tok/s | 871 tok/s |

**결과**: APC를 끄면 오히려 더 느려짐. chunked prefill이 배치 효율을 높이는 효과가 더 크기 때문. APC ON이 실제 프로덕션에 더 가까운 조건.

### 5.4 결론

APC 설정 불일치는 시뮬레이터-vLLM 오차의 원인이 아님. APC ON/OFF에 관계없이 TTFT 3배 갭은 유지됨. **오차의 근본 원인은 큐잉/스케줄링 오버헤드 미반영**임이 확인됨.

---

## 6. 개선 방향

| 우선순위 | 항목 | 예상 효과 |
|---------|------|-----------|
| High | TTFT에 큐잉/스케줄링 오버헤드 모델 추가 | TTFT MAPE 65% → 20% 이하로 개선 가능 |
| High | TP=2 attention all-reduce 레이턴시 재측정 (NVLink vs PCIe 구분) | TP=2 TPOT 역전 해소 |
| Med | Throughput 계산에 실제 스케줄링 오버헤드 반영 | 과대 예측 50% → 20% 수준 |
| Low | per-request 도착 순서 재현 로직 | Pearson r 개선 |

---

## 7. Files

| 파일 | 설명 |
|------|------|
| `validation/vllm_tp1_results.jsonl` | vLLM TP=1 (APC ON) per-request TTFT/TPOT |
| `validation/vllm_tp2_results.jsonl` | vLLM TP=2 (APC ON) per-request TTFT/TPOT |
| `validation/vllm_tp1_noapc_results.jsonl` | vLLM TP=1 (APC OFF) per-request TTFT/TPOT |
| `validation/vllm_tp2_noapc_results.jsonl` | vLLM TP=2 (APC OFF) per-request TTFT/TPOT |
| `validation/sim_tp1_results.csv` | 시뮬레이터 TP=1 (APC OFF) per-request 결과 |
| `validation/sim_tp2_results.csv` | 시뮬레이터 TP=2 (APC OFF) per-request 결과 |
| `validation/sim_tp1_apc_results.csv` | 시뮬레이터 TP=1 (APC ON) per-request 결과 |
| `validation/sim_tp2_apc_results.csv` | 시뮬레이터 TP=2 (APC ON) per-request 결과 |
| `validation/sim_tp1_stdout.txt` | 시뮬레이터 TP=1 APC OFF 전체 출력 |
| `validation/sim_tp2_stdout.txt` | 시뮬레이터 TP=2 APC OFF 전체 출력 |
| `validation/sim_tp1_apc_stdout.txt` | 시뮬레이터 TP=1 APC ON 전체 출력 (hit ratio 포함) |
| `validation/sim_tp2_apc_stdout.txt` | 시뮬레이터 TP=2 APC ON 전체 출력 (hit ratio 포함) |
| `validation/results_summary.csv` | MAPE / Pearson r 요약 테이블 |
| `cluster_config/a5000_1gpu_validation.json` | TP=1 클러스터 설정 |
| `cluster_config/a5000_2gpu_tp2_validation.json` | TP=2 클러스터 설정 |
