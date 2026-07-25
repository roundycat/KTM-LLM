# TKM 국가시험 LLM 벤치마크 파이프라인

Jang et al. (2023), *"GPT-4 can pass the Korean National Licensing Examination
for Korean Medicine Doctors"* (PLOS Digital Health)의 5단계 누적 프롬프트
기법을 재현하는 코드입니다.

**연구 목표**: 한의사 국가시험을 기준으로 (1) 국가별 범용(general-purpose) LLM
성능과 (2) 도메인 특화 LLM(중국의학 vs 서양의학) 성능을 비교합니다.

```
models/
├── general.txt              # 국가별 범용 LLM (중국/미국/한국/프랑스 등)
└── domain/
    ├── tcm.txt               # 중국의학(TCM) 도메인 특화 LLM
    └── western_med.txt       # 서양의학 도메인 특화 LLM
```

## 5단계 기법

| Stage | 추가되는 기법 |
|---|---|
| 0 | 없음 (베이스라인, 한글 원문) |
| 1 | + 한자 병기 (Chinese-term annotation) |
| 2 | + 지시문 영어 번역 |
| 3 | + 문제/보기 영어 번역 |
| 4 | + Exam-optimized instruction (단계적 추론 + 정답 1개 강제) |
| 5 | + Self-consistency (N회 반복 후 다수결) |

## 설치

```bash
pip install litellm --break-system-packages
```

`litellm`은 OpenAI, Anthropic, Google Gemini 등 100개 이상의 모델을
`model="gpt-4o"`, `model="claude-sonnet-4-6"`, `model="gemini/gemini-1.5-pro"`
처럼 문자열만 바꿔서 동일하게 호출할 수 있게 해줍니다.

## API 키

사용할 모델에 맞는 환경변수를 설정하세요.

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
```

## 문항 데이터 형식 (`KTM_data/` 참고)

```json
[
  {
    "id": "Q1",
    "subject": "내과(1)",
    "question_kr": "...",
    "choices_kr": ["...", "...", "...", "..."],
    "correct_answer": 1,
    "tkm_terms": {"간양상항": "肝陽上亢"}
  }
]
```

- `choices_kr`는 4지선다/5지선다 모두 지원 (개수는 자유, 코드가 하드코딩되어 있지 않음)
- `correct_answer`는 1-indexed (4지선다면 1~4, 5지선다면 1~5)
- `tkm_terms`는 한자 병기용 용어 사전 (선택, 없으면 stage 1에서 변화 없음)

## 연도별 데이터 (`KTM_data/`)

| 파일 | 문항 수 | 형식 | 비고 |
|---|---|---|---|
| `KTM_data/2022.json` | 294 | 4지선다 | 실제 기출(제77회) OCR 변환, 이미지 필요 문항 158개는 텍스트만으로 풀 수 없어 제외 |
| `KTM_data/2023.json` | 280 | 4지선다 | 실제 기출, 이미지 필요 문항 제외 |
| `KTM_data/2024.json` | 288 | 4지선다 | 실제 기출, 이미지 필요 문항 제외 |
| `KTM_data/2025.json` | 517 | 5지선다 | 재현 테스트용 예시 문항 |

## 실행 예시

```bash
# GPT-4o, 논문과 동일하게 stage 5 (self-consistency, 7회) 로 전체 재현
python tkm_pipeline.py --model gpt-4o --data KTM_data/2025.json --stage 5 --n-trials 7 --output result_gpt4o.json

# 2022년도 기출로 Claude 테스트 (stage 4까지, self-consistency 없이)
python tkm_pipeline.py --model claude-sonnet-4-6 --data KTM_data/2022.json --stage 4

# 단계별 비교 (Fig 1 재현)
for s in 0 1 2 3 4; do
  python tkm_pipeline.py --model gpt-4o --data KTM_data/2025.json --stage $s
done

# 연도별 전체 비교
for year in 2022 2023 2024 2025; do
  python tkm_pipeline.py --model gpt-4o --data "KTM_data/${year}.json" --stage 5 --n-trials 7 \
    --output "result_gpt4o_${year}.json"
done
```

## 여러 모델 자동 비교 (`run_all_models.sh`)

vLLM으로 로컬/클라우드 GPU에 모델을 하나씩 띄워가며 `models/general.txt`,
`models/domain/tcm.txt`, `models/domain/western_med.txt`에 적힌 모델을 순서대로
전부 벤치마크합니다.

```bash
pip install vllm litellm --break-system-packages
export HF_TOKEN="hf_..."   # 게이트된 모델(Llama, Gemma 계열 등) 접근용
chmod +x run_all_models.sh
./run_all_models.sh
```

결과와 로그는 `models/`와 같은 하위 구조로 쌓입니다.

```
results/
├── general/<모델명>.json
├── domain/tcm/<모델명>.json
├── domain/western_med/<모델명>.json
└── _run_summary.tsv          # 모델별 성공/실패/소요시간 요약
logs/
├── general/<모델명>_{vllm,run}.log
├── domain/tcm/<모델명>_{vllm,run}.log
└── domain/western_med/<모델명>_{vllm,run}.log
```

각 목록 파일에서 repo id가 `???`인 항목은 HuggingFace에서 정확한 repo id를
확인해서 채워야 실행됩니다 (그 전까진 자동으로 건너뜁니다).

## 참고 사항 / 논문과의 차이점

- 논문은 원 실험에서 7회 반복 중 최빈값을 사용했습니다 (본 코드의 `--n-trials 7`과 동일).
- 표(table)·이미지가 포함된 문제는 논문에서 텍스트로만 변환해 입력했는데,
  본 코드는 텍스트 기반 문항만 다루므로 표/이미지가 있는 문항은
  `question_kr`에 표 내용을 텍스트로 미리 풀어서 넣어주세요.
  (`KTM_data/2022~2024.json`은 이미지가 필요한 문항을 자동으로 제외한 상태입니다.)
- 거부 응답("저는 AI라 진단할 수 없습니다" 등) 처리 로직은 논문의
  2.5절 방식을 근사했습니다 (거부 시 최대 3회 재시도).
- `KTM_data/2025.json`은 재현 테스트용 예시 문항이지만, `2022~2024.json`은
  실제 한의사 국가시험 기출문제를 OCR로 변환한 데이터입니다. 저작권/시험 운영 기관
  정책상 공개 배포가 제한될 수 있으니, 이 저장소를 public으로 push하기 전에
  재배포 가능 여부를 꼭 확인하세요.
