#!/usr/bin/env bash
#
# models/ 아래 카테고리별 목록에 적힌 모델을 하나씩:
#   vLLM 서버로 띄우기 -> 서버 준비될 때까지 대기 -> tkm_pipeline.py 벤치마크 실행
#   -> 결과 저장 -> 서버 종료 -> 다음 모델
# 순서로 자동 반복하는 스크립트.
#
# 비교 축:
#   models/general.txt              국가별 범용 LLM
#   models/domain/tcm.txt           중국의학(TCM) 도메인 특화 LLM
#   models/domain/western_med.txt   서양의학 도메인 특화 LLM
#
# 사전 준비:
#   pip install vllm litellm --break-system-packages
#   export HF_TOKEN="hf_..."   # 게이트된 모델(Llama, Gemma 계열 등) 접근용
#
# 사용법:
#   chmod +x run_all_models.sh
#   ./run_all_models.sh

set -uo pipefail

# category_label:models_file 쌍의 목록. category_label은 results/, logs/ 아래 하위 경로로 쓰인다.
CATEGORIES=(
  "general:models/general.txt"
  "domain/tcm:models/domain/tcm.txt"
  "domain/western_med:models/domain/western_med.txt"
)

DATA_FILE="KTM_data/2025.json"
STAGE=5
N_TRIALS=7
MAX_WORKERS=7
PORT=8000
API_BASE="http://localhost:${PORT}/v1"
API_KEY="sk-dummy"
READY_TIMEOUT=900          # vLLM 서버 기동 대기 최대 시간(초)
GPU_COOLDOWN=10            # 모델 종료 후 GPU 메모리 정리 대기 시간(초)

SUMMARY_FILE="results/_run_summary.tsv"
mkdir -p results
echo -e "category\tmodel\tstatus\tstarted_at\tfinished_at" > "$SUMMARY_FILE"

wait_for_server() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

run_category() {
  local category="$1" models_file="$2"
  local result_dir="results/${category}"
  local log_dir="logs/${category}"
  mkdir -p "$result_dir" "$log_dir"

  if [[ ! -f "$models_file" ]]; then
    echo "[$category] $models_file 없음 — 건너뜀"
    return
  fi

  while IFS=$'\t' read -r name repo_id extra_args; do
    [[ -z "${name:-}" || "$name" == \#* ]] && continue   # 빈 줄/주석 건너뛰기

    if [[ "$repo_id" == "???" ]]; then
      echo "[$category/$name] repo id가 아직 '???'로 비어있음 — $models_file 을 채운 뒤 다시 실행하세요. 건너뜀."
      echo -e "${category}\t${name}\tSKIPPED_NO_REPO_ID\t-\t-" >> "$SUMMARY_FILE"
      continue
    fi

    echo "=================================================="
    echo "[$category/$name] 시작 ($repo_id)"
    echo "=================================================="
    started_at=$(date +"%Y-%m-%d %H:%M:%S")

    vllm_log="$log_dir/${name}_vllm.log"
    run_log="$log_dir/${name}_run.log"
    output_json="$result_dir/${name}.json"

    # 1) vLLM 서버 기동 (백그라운드)
    # shellcheck disable=SC2086
    vllm serve "$repo_id" \
      --port "$PORT" \
      --served-model-name "$name" \
      $extra_args \
      > "$vllm_log" 2>&1 &
    vllm_pid=$!

    # 2) 서버 준비될 때까지 대기
    if ! wait_for_server; then
      echo "[$category/$name] vLLM 서버가 ${READY_TIMEOUT}초 안에 기동되지 않음 — 건너뜀 (로그: $vllm_log)"
      kill "$vllm_pid" 2>/dev/null
      wait "$vllm_pid" 2>/dev/null
      echo -e "${category}\t${name}\tSERVER_TIMEOUT\t${started_at}\t$(date +"%Y-%m-%d %H:%M:%S")" >> "$SUMMARY_FILE"
      continue
    fi
    echo "[$category/$name] 서버 준비 완료, 벤치마크 시작"

    # 3) 벤치마크 실행
    python3 tkm_pipeline.py \
      --model "openai/${name}" \
      --api-base "$API_BASE" \
      --api-key "$API_KEY" \
      --data "$DATA_FILE" \
      --stage "$STAGE" \
      --n-trials "$N_TRIALS" \
      --max-workers "$MAX_WORKERS" \
      --output "$output_json" \
      > "$run_log" 2>&1
    run_status=$?

    # 4) 서버 종료 + GPU 메모리 정리 대기
    kill "$vllm_pid" 2>/dev/null
    wait "$vllm_pid" 2>/dev/null
    sleep "$GPU_COOLDOWN"

    finished_at=$(date +"%Y-%m-%d %H:%M:%S")
    if [[ $run_status -eq 0 ]]; then
      echo "[$category/$name] 완료 -> $output_json"
      echo -e "${category}\t${name}\tOK\t${started_at}\t${finished_at}" >> "$SUMMARY_FILE"
    else
      echo "[$category/$name] 벤치마크 실행 실패 (로그: $run_log)"
      echo -e "${category}\t${name}\tRUN_FAILED\t${started_at}\t${finished_at}" >> "$SUMMARY_FILE"
    fi

  done < "$models_file"
}

for entry in "${CATEGORIES[@]}"; do
  category="${entry%%:*}"
  models_file="${entry#*:}"
  run_category "$category" "$models_file"
done

echo "=================================================="
echo "전체 완료. 요약: $SUMMARY_FILE"
echo "=================================================="
