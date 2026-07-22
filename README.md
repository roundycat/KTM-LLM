# TKM 국가시험 LLM 벤치마크 파이프라인

Jang et al. (2023), *"GPT-4 can pass the Korean National Licensing Examination
for Korean Medicine Doctors"* (PLOS Digital Health)의 5단계 누적 프롬프트
기법을 재현하는 코드입니다.

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

## 문항 데이터 형식 (`sample_questions.json` 참고)

```json
[
  {
    "id": "Q1",
    "subject": "내과(1)",
    "question_kr": "...",
    "choices_kr": ["...", "...", "...", "...", "..."],
    "correct_answer": 1,
    "tkm_terms": {"간양상항": "肝陽上亢"}
  }
]
```

- `correct_answer`는 1~5 (1-indexed)
- `tkm_terms`는 한자 병기용 용어 사전 (선택, 없으면 stage 1에서 변화 없음)

## 실행 예시

```bash
# GPT-4o, 논문과 동일하게 stage 5 (self-consistency, 7회) 로 전체 재현
python tkm_pipeline.py --model gpt-4o --data sample_questions.json --stage 5 --n-trials 7 --output result_gpt4o.json

# Claude로 stage 4까지만 (self-consistency 없이)
python tkm_pipeline.py --model claude-sonnet-4-6 --data sample_questions.json --stage 4

# 단계별 비교 (Fig 1 재현)
for s in 0 1 2 3 4; do
  python tkm_pipeline.py --model gpt-4o --data sample_questions.json --stage $s
done
```

## 참고 사항 / 논문과의 차이점

- 논문은 원 실험에서 7회 반복 중 최빈값을 사용했습니다 (본 코드의 `--n-trials 7`과 동일).
- 표(table)·이미지가 포함된 문제는 논문에서 텍스트로만 변환해 입력했는데,
  본 코드는 텍스트 기반 문항만 다루므로 표/이미지가 있는 문항은
  `question_kr`에 표 내용을 텍스트로 미리 풀어서 넣어주세요.
- 거부 응답("저는 AI라 진단할 수 없습니다" 등) 처리 로직은 논문의
  2.5절 방식을 근사했습니다 (거부 시 최대 3회 재시도).
- 실제 국시 문항은 저작권/시험 운영 기관 정책상 공개되어 있지 않으므로,
  `sample_questions.json`은 재현 테스트용 예시 문항입니다.
