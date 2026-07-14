# REPORT — 이종 클러스터 자원 관리 플래너 (MILP · Max-Flow) 구현 및 실험

> 대상 계획서: [`PLAN_MILP_MaxFlow.md`](PLAN_MILP_MaxFlow.md)
> 작성일: 2026-07-14 · 기준 저장소: `ai-computing/LLMServingSim` (main)
> 신규 코드: [`planner/`](planner/) (코어 시뮬레이터 무수정)

---

## 1. 요약

계획서(`PLAN_MILP_MaxFlow.md`)의 2단계 오프라인 플래너를 `planner/` 패키지로 구현하고,
**실제 LLMServingSim 시뮬레이터로 end-to-end 검증**했다.

- **Stage 1 (구조 최적화)**: OR-Tools CP-SAT MILP로 device/memory/xPyD 제약 하에서 Top-K 후보 배치 산출.
- **Stage 2 (시뮬 검증)**: 각 후보를 `cluster_config.json`으로 렌더 → `main.py` 서브프로세스 실행 → `output/*.csv` 파싱 → SLO 제약·파레토 재랭킹.

플래너는 시뮬레이터를 **블랙박스 평가 함수**로만 사용하는 비침습 래퍼로, 코어 코드는 한 줄도 수정하지 않았다.
단위 테스트 23개 + 실제 시뮬 6건으로 동작을 확인했다.

---

## 2. 시스템 구조

```
spec.yaml ──▶ [graph_model] ──▶ [milp_solver]  ─(Top-K)─▶ [config_renderer]
                                  Stage 1: CP-SAT            allocation→cluster_config.json + CLI
                                                                     │
                                                                     ▼
   best_config.json ◀── [report] ◀── [objective] ◀── [sim_evaluator]  Stage 2
   pareto.csv/report.md          SLO·파레토       main.py 서브프로세스 + CSV 파싱
```

| 모듈 | 역할 |
|---|---|
| `spec_schema.py` | pydantic 스펙 + 리포지토리 검증(프로파일/모델 존재 여부) |
| `graph_model.py` | 토폴로지→networkx 그래프, 대역폭/지연 단위 파싱 |
| `milp_solver.py` | **Stage 1**: CP-SAT로 Top-K 후보 (no-good cut으로 상이해 확보) |
| `config_renderer.py` | Allocation→`cluster_config.json` + CLI (tp→`npu_group`, 선택적 power 블록) |
| `sim_evaluator.py` | **Stage 2**: 서브프로세스 실행 + CSV 파싱 + 캐시 + 에너지 파싱 |
| `objective.py` | SLO 제약 + 가중 점수 + 파레토 프론트 |
| `search_orchestrator.py` | 전체 파이프라인 (병렬 Stage 2, dry-run) |
| `report.py` / `cli.py` / `types.py` / `utils.py` | 리포트 / 진입점 / 공유 타입 / 유틸 |

---

## 3. 계획 대비 정합성 수정 (구현 중 확정)

계획서의 가정 일부가 실제 코드와 달랐고, 구현하면서 다음을 확정·수정했다(자세한 내용은 PLAN §0 검증 박스).

| 항목 | 계획 가정 | 실제 | 조치 |
|---|---|---|---|
| TP/PP 필드 | 인스턴스별 `tp`/`pp` 필드 | 필드 없음. `npu_num`+`npu_group`만 존재 | tp→`npu_group`로 매핑, PP는 `[1]` 고정 |
| 경로 처리 | 상대경로 그대로 | `config_builder`가 무조건 `../` 프리픽스 | `stage_path()`로 리포지토리 내부 스테이징 |
| 에너지 출력 | `... J` | `Total energy consumption (kJ)` (kJ) | 파서를 kJ→J 변환으로 수정 |
| 프로파일 경로 | `llm_profile/perf` | `llm_profile/perf_models/` | 경로 수정 |
| ITL | 스칼라 | per-token 배열 문자열 | `ast.literal_eval` 후 p99 집계 |
| 실행 환경 | — | 시뮬에 `LD_LIBRARY_PATH`/`PATH` 필요 | webapp `SIM_ENV`와 동일하게 자동 설정 |

이 중 **경로 버그**와 **에너지 단위 오류**는 실제 시뮬을 돌려서야 드러난 것으로,
계획서의 "Stage 2 실측 검증"이 실효적으로 동작했음을 보여준다.

---

## 4. 검증

- **단위 테스트**: `pytest planner/tests/` → **23 passed** (시뮬 불필요). graph/MILP/renderer/CSV파싱/objective 커버.
- **실제 시뮬 E2E**: 단일 인스턴스 스펙으로 `main.py`를 실제 구동해 metrics 파싱까지 확인.
  생성된 `best_cluster_config.json`은 `main.py`에 그대로 넣어 재현 실행됨.
- **에너지 경로**: `toks_per_wh` 목적함수 포함 시 power 블록 자동 삽입 + stdout 에너지 파싱 검증 (예: `toks_per_wh=2905.77`).

---

## 5. 실험 설정

| 항목 | 값 |
|---|---|
| 모델 | `meta-llama/Llama-3.1-8B`, FP16 |
| 워크로드 | `sharegpt_req100_rate10_llama.jsonl`, **30 requests** |
| 배치 토큰 | `--max-num-batched-tokens 2048` |
| 목적함수 | throughput(0.6) + toks_per_wh(0.4), TTFT ≤ 100 s (완화) |
| 실행 | 플래너 CLI, 6개 스펙 병렬 (20-core 호스트) |

> **주의**: TTFT/TPOT는 시뮬레이터 정의(첫 토큰 *계산 완료* 시점)로 vLLM 대비 낮게 나온다.
> 절대값보다 **구성 간 상대 비교**가 목적이다.

두 가지 실험을 수행했다.

- **실험 A — 하드웨어 비교**: 단일 인스턴스 TP1, A5000 / A6000 / H100.
- **실험 B — TP 스케일링**: A6000에서 TP1 / TP2 / TP4 (각 TP = 해당 수의 GPU).

---

## 6. 실험 A — 하드웨어 비교

![Experiment A](output/planner_experiments/figures/fig1_hardware.png)

| HW | TTFT(ms) | TPOT(ms) | ITL-p99(ms) | Throughput(tok/s) | toks/Wh |
|---|---|---|---|---|---|
| A5000 | 42.34 | 28.40 | 29.70 | 572.97 | 4330.97 |
| A6000 | 72.62 | 28.25 | 69.72 | 587.84 | 3872.45 |
| **H100** | **21.93** | **12.77** | **16.81** | **1141.92** | **4341.15** |

**관찰**
- **H100이 전 지표 지배**: 처리량 ~2×(A6000 대비 1.94×), 지연 최저, 에너지 효율 최고. 이 워크로드에서는 명확한 단일 최적해.
- A5000 vs A6000: 처리량은 A6000이 소폭 높지만, **A5000의 TTFT/ITL-p99가 오히려 우수**. 이는 프로파일 데이터에 내재된 특성으로, 스칼라 프록시만으로는 예측 못 하고 **Stage 2 실측이 있어야 드러나는 비직관적 결과**다(플래너 2단계 설계의 정당성).

---

## 7. 실험 B — TP 스케일링 (A6000)

![Experiment B](output/planner_experiments/figures/fig2_tp_scaling.png)

| TP | TTFT(ms) | TPOT(ms) | Throughput(tok/s) | toks/Wh |
|---|---|---|---|---|
| TP1 | 72.62 | 28.25 | 587.84 | 3872.45 |
| TP2 | 53.78 | 28.16 | 579.41 | 2430.47 |
| TP4 | 47.06 | 30.14 | 549.72 | 1357.21 |

**관찰**
- **TTFT는 TP에 따라 개선**(72.6→47.1 ms, −35%): prefill이 텐서 병렬로 분산되어 첫 토큰 계산이 빨라짐.
- **처리량은 오히려 소폭 감소**(588→550 tok/s): 30-req 소규모·저동시성 워크로드에서는 TP가 처리량을 늘리지 못하고, all-reduce 통신 오버헤드가 TPOT를 되레 증가(28.2→30.1 ms)시킴.
- **에너지 효율은 급락**(3872→1357 toks/Wh, −65%): GPU 수는 TP배로 늘지만 처리량은 정체 → 토큰당 전력이 급증.

**해석**: 이 워크로드에서 TP는 "지연을 사는 대신 처리량·효율을 잃는" 트레이드오프다.
지연 SLO가 빡빡하면 TP2~4가 유효하고, 처리량/효율이 목적이면 TP1이 지배적이다.
플래너의 파레토 랭킹이 이 트레이드오프를 그대로 포착한다.

---

## 8. 결론

- 계획서의 2단계(MILP → 시뮬 검증) 플래너가 **실제 시뮬레이터와 통합되어 동작**함을 확인했다.
- 실험 A/B 모두에서 **구조 프록시만으로는 알 수 없는 실측 특성**(A5000의 낮은 TTFT, TP의 효율 급락)이 Stage 2에서 드러나, 2단계 설계의 필요성이 실증됐다.
- 산출물(`best_cluster_config.json`)은 시뮬레이터에 바로 재투입 가능하다.

---

## 9. 한계 및 향후 과제

| 한계 | 비고 |
|---|---|
| Stage 1 처리량/전력 프록시가 coarse | 하드웨어별 상수 기반. Stage 2가 재랭킹하지만, 프로파일 기반 프록시로 정교화 여지 |
| PP 미지원 | 시뮬 config 노브 부재(`npu_group ≤ npu_num`). `pp_choices=[1]` 고정 |
| ε-제약 파레토 스윕 | 현재 no-good cut 기반 Top-K. 계획 M5의 명시적 ε-스윕은 미구현 |
| 실측 이식(M7) | vLLM/llm-d 인자 변환은 범위 밖 |
| 소규모 워크로드 | 본 실험은 30-req. 고동시성에서 TP 효과는 반전될 수 있음(향후 num_req 스윕 필요) |

---

## 10. 재현 방법

```bash
pip install -r requirements-planner.txt

# 스펙 검증
python -m planner.cli --spec planner/specs/example_hetero_8gpu.yaml --validate-only

# 실험 스펙(6종)은 output/planner_experiments/specs/ 에 있음
export LD_LIBRARY_PATH="/tmp/protobuf_prefix/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
for s in output/planner_experiments/specs/*.yaml; do
  python -m planner.cli --spec "$s" \
    --out-dir "output/planner_experiments/runs/$(basename $s .yaml)" --jobs 1
done

# 그림 재생성
python output/planner_experiments/make_figures.py
```

**아티팩트**
- 실험 스펙: `output/planner_experiments/specs/*.yaml`
- 실행 결과: `output/planner_experiments/runs/*/pareto.csv`
- 그림: `output/planner_experiments/figures/{fig1_hardware,fig2_tp_scaling}.png`
- 단위 테스트: `pytest planner/tests/`
