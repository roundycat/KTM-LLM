# 한의학 국가시험 문제 데이터셋 & 학습 코드

한국보건의료인국가시험원(국시원) 공개 기출문제에서 추출한 **객관식 5지선다 + 정답** 데이터셋과,
**로컬(한국형) 모델 vs 글로벌(범용) 모델**을 같은 시험으로 평가하고,
**KIOM 표준한의학용어집**으로 글로벌 모델에 용어 지식을 주입(RAG)/학습(파인튜닝)하여
향상을 확인하는 코드입니다.

### 실험 순서
1. **로컬(한국형 EXAONE)** 과 **글로벌(범용 Qwen)** 에 한의학 시험을 풀게 함
2. 두 모델의 **평균 정답률** 도출
3. 글로벌 모델에 **한의학 용어(KIOM 표준용어집)** 를 학습 — 두 방식
   - **RAG 주입(기본·무료)**: 문제에 관련 용어 정의를 컨텍스트로 주입 (`rag.py`)
   - **파인튜닝(선택·유료)**: OpenAI SFT (`prepare_terminology.py` → `finetune.py --terminology`)
4. 1~2를 반복하여 **글로벌 모델의 향상(Δ)** 확인 (`report.py`)

- **로컬 모델**: [Ollama](https://ollama.com) 로 띄운 한국형 오픈웨이트 LLM(기본 `exaone3.5:7.8b`).
- **글로벌 모델**: 기본은 무료 로컬 범용 모델(`qwen2.5:7b`). `.env` 로 OpenAI/Gemini 등
  OpenAI 호환 클라우드로 교체 가능. 로컬·클라우드 모두 **동일한 평가 코드**로 채점한다.

> 💸 **완전 무료로 돌리기**: 두 모델 모두 Ollama 로컬 + 용어는 RAG 주입 → API 비용 0.
> OpenAI 파인튜닝은 2026년 신규 사용자에게 닫히는 중이며 무료가 아니므로, 기본 경로는 RAG 입니다.

> ⚠️ **저작권 유의**: 국시원 기출문제와 표준한의학용어집(대한한의학회)은 저작물입니다.
> **개인 학습·연구용**으로만 사용하고 외부 재배포는 주의하세요.

---

## 📊 실험 결과 (전체 517 텍스트 문항)

로컬 **EXAONE 3.5 7.8B**(한국형) vs 글로벌 **Qwen2.5 7B**(범용), 둘 다 Ollama 로컬(GPU).
글로벌 모델에 **KIOM 표준한의학용어집(9,074개)** 을 **RAG로 주입**한 전후 비교.

| 단계 | 모델 | 정답률 | 맞음/전체 |
|---|---|---:|---:|
| ① 학습 전 | 한국형 EXAONE 3.5 7.8B | 43.13% | 223/517 |
| ① 학습 전 | 글로벌 Qwen2.5 7B | 48.55% | 251/517 |
| | **두 모델 평균** | **45.84%** | |
| ② 용어 주입 후 | 글로벌 Qwen2.5 7B **+ 용어 RAG** | **49.52%** | 256/517 |
| | **글로벌 향상(Δ)** | **+0.97%p** | +5문항 |

- 용어 주입 효과는 **용어 의존도가 높은 과목에 집중**된다:
  **본초학 +30.8%p**(7.7→38.5), **침구학 +12.1%p**(30.3→42.4), 부인과학 +6.9%p.
  반대로 임상추론 위주 과목(소아·생리)은 소폭 하락 → 순효과 **+0.97%p**.
- 전체 과목별 표·원자료: `results/report.md`, `results/summary.json`.
- 작은 검증셋(51문항)에서는 ±0으로 노이즈가 크며, 전체 문항으로 봐야 신호가 안정적이다.

> 결과는 GPU·모델 버전·Ollama 양자화에 따라 달라질 수 있다. 재현: `python training/run_all.py --all`.

---

## 목차
1. [데이터셋](#-데이터셋-dataset)
2. [전체 흐름](#-전체-흐름)
3. [빠른 시작](#-빠른-시작)
4. [코드 구성](#-코드-구성-training)
5. [파인튜닝 모드(번호만 vs 해설)](#-파인튜닝-모드)
6. [해설 필드 생성](#-해설-필드-생성)
7. [GitHub Actions 자동 평가](#-github-actions-자동-평가)
8. [비용 가이드](#-비용-가이드)
9. [트러블슈팅](#-트러블슈팅)
10. [데이터 추출 파이프라인](#-데이터-추출-파이프라인-scripts)

---

## 📦 데이터셋 (`dataset/`)

| 파일 | 내용 | 문항 수 |
|---|---|---|
| `한의학_문제.jsonl` | **학습용 메인.** 그림 비의존 순수 텍스트 문항 | 517 |
| `한의학_문제_전체.jsonl` | 그림생략(70) 포함 전체 + 메타데이터 | 587 |
| `한의학_문제_해설.jsonl` | 위에 `해설` 필드를 추가한 버전 *(생성 시)* | — |
| `stats.json` | 회차·교시별 통계 | — |

수집 회차: **제81회 한의사(2026)** 337문항 · **제27회 한약사(2026)** 250문항.

**스키마**
```jsonc
// 한의학_문제.jsonl (메인)
{"question": "...", "options": ["...","...","...","...","..."], "answer": 4, "answer_text": "..."}

// 한의학_문제_전체.jsonl (메타 포함)
{"source":"한의사_81회","교시":1,"과목":"내과학","번호":1,
 "question":"...","options":[...],"answer":4,"answer_text":"...","has_figure":false}

// 한의학_문제_해설.jsonl (add_explanations.py 산출물)
{... 위 필드 ..., "해설":"정답 4번은 ... 때문에 옳다. 2번은 ... 이유로 틀렸다."}
```
- `answer`: 정답 보기 번호(1~5). `answer_text`: 해당 보기 텍스트(100% 일치 검증 완료).
- `has_figure`: 그림/도표 의존 문항 여부(텍스트만으로 풀 수 없는 70문항). 학습·평가에서 자동 제외.

---

## 🔄 전체 흐름

```
  dataset/한의학_문제_전체.jsonl(587문항)     cis.kiom.re.kr 표준한의학용어집
                 │                                     │
                 │                       scripts/fetch_terminology.py
                 │                                     ▼
                 │                         dataset/한의학_용어.jsonl(8.8k 용어)
                 │                                     │
                 ▼                                     ▼
            evaluate.py  ◄──── rag.py(문제에 용어 주입, 무료) ────┐  ③
        (한국형·글로벌 채점)                                       │
                 │                          (선택·유료) 파인튜닝 경로 │
        ①② 로컬 vs 글로벌                  prepare_terminology.py    │
           평균 정답률                       → finetune.py           │
                 │                          → ft:...:hani-term ──────┘
                 ▼  ④
            report.py → results/report.md (전후 비교·향상 Δ)
```

①②③④ 는 위 "실험 순서"에 대응. 기본 경로는 ③에서 **RAG 주입(무료)**.

---

## 🚀 빠른 시작 (무료 RAG 경로)

```bash
# 0) 의존성
pip install -r requirements.txt
cp .env.example .env          # (무료 경로는 OpenAI 키 불필요)

# 1) 로컬 모델 2종 준비 — Ollama (https://ollama.com)
ollama pull exaone3.5:7.8b    # 한국형(LOCAL_MODEL)
ollama pull qwen2.5:7b        # 글로벌 범용(GLOBAL_MODEL)
#   디스크가 빠듯하면 모델 경로를 다른 드라이브로: OLLAMA_MODELS=D:\ollama-models

# 2) KIOM 용어 수집(RAG 지식원)
python scripts/fetch_terminology.py      # → dataset/한의학_용어.jsonl (V2.1 국문 ~8.8천)

# 3) 평가 — 한국형·글로벌 base + 글로벌 RAG(용어 주입)
python training/evaluate.py              # 검증셋(51문항)
python training/evaluate.py --all        # 전체 텍스트 문항(517)

# 4) 종합 리포트(전후 비교·향상 Δ)
python training/report.py                # → results/report.md

# (선택) 1~4 평가+리포트를 한 번에
python training/run_all.py               # --all / --no-rag / --rag-local
```

> **유료 파인튜닝 경로**(OpenAI 권한 필요): `prepare_terminology.py` → `finetune.py --terminology`
> 후 `.env` 의 `GPT4_FINETUNED_MODEL` 에 모델 ID 기입 → `evaluate.py` 재실행 → `report.py`.
> `report.py` 가 RAG/파인튜닝 향상을 모두 표에 포함합니다.

---

## 🧩 코드 구성 (`training/`)

| 파일 | 역할 | 주요 옵션 |
|---|---|---|
| `config.py` | 경로·모델 ID·프롬프트·하이퍼파라미터 + **로컬/글로벌 클라이언트(`make_client`)** 중앙 관리 | — |
| `prepare_data.py` | 기출 데이터셋 → OpenAI chat 포맷 변환 + train/val 분할 | `--with-rationale` |
| `rag.py` | KIOM 용어 인덱스 → 문제에 등장하는 용어 **검색·주입**(한자 우선, 무료) | — |
| `prepare_terminology.py` | KIOM **용어 → Q&A SFT** 변환(유료 파인튜닝용) | `--limit`, `--combine-exam` |
| `add_explanations.py` | 정답 기반 **해설 생성**(재개 가능) → `한의학_문제_해설.jsonl` | `--limit`, `--skip-figure` |
| `finetune.py` | 파일 업로드 → fine‑tuning job 생성 → 진행 모니터링 | `--model {gpt-4,gpt-5,all}`, `--terminology`, `--train-file` |
| `evaluate.py` | **한국형·글로벌** 채점(역할/방식: base·**rag**·ft) + **평균·향상 Δ** | `--all`, `--no-rag`, `--rag-local`, `--models` |
| `report.py` | results/ 종합 → **로컬 vs 글로벌·용어 주입 전후** 리포트(`results/report.md`) | — |
| `run_all.py` | 실험 순서(①②③④) 오케스트레이터 | `--all`, `--no-rag`, `--rag-local` |

> 로컬 모델은 `scripts/fetch_terminology.py`(KIOM 표준한의학용어집 수집)와 함께 동작합니다.
> 용어 수집은 `cis.kiom.re.kr` 의 `search.do?term=%`(전체 반환)를 파싱해 V2.1·국문 ~8.8천 용어를 저장합니다.

모든 모듈은 상단 docstring과 인라인 주석으로 동작/주의점을 설명합니다.

---

## 🎓 파인튜닝 모드

`prepare_data.py` 는 두 가지 학습 타깃을 만들 수 있습니다.

| 모드 | 명령 | assistant 타깃 | 특징 |
|---|---|---|---|
| 번호만(기본) | `prepare_data.py` | `"4"` | 가볍고 빠름. 정답 선택만 학습 |
| 해설 포함(CoT) | `prepare_data.py --with-rationale` | `"{해설}\n\n정답: 4"` | 근거를 먼저 쓰고 답을 내도록 학습. `한의학_문제_해설.jsonl` 필요 |

> 해설 모드는 해설 파일이 없으면 자동으로 번호만 모드로 폴백합니다(경고 출력).

학습/평가 데이터는 **고정 시드(42)** 로 분할되어, `evaluate.py` 의 검증셋과 정확히 동일합니다 →
학습에 쓴 문항으로 평가하는 **데이터 누수(leakage)가 없습니다.**

---

## ✍️ 해설 필드 생성

원본 국시원 자료에는 해설이 없어 LLM으로 생성합니다. 단, **정답을 모델에 알려준 상태**에서
"왜 그 답이 옳은지"를 서술하게 하여(모델이 직접 풀게 하지 않음) 사실 오류 위험을 낮춥니다.

```bash
python training/add_explanations.py --limit 5    # 먼저 5개로 품질 확인
python training/add_explanations.py              # 전체 생성(재개 가능)
```
- **재개 가능**: 중간에 멈춰도 다시 실행하면 이미 만든 해설은 건너뛰고 이어서 생성.
- 결과는 `dataset/한의학_문제_해설.jsonl` 에 저장되며, 이후 `prepare_data.py --with-rationale` 로 학습에 활용.
- ⚠️ 생성 해설은 **참고용**입니다. 학습/배포 전 표본 검수를 권장합니다.

---

## 🤖 GitHub Actions 자동 평가

`.github/workflows/evaluate.yml` — **수동 실행(workflow_dispatch)** 기반(API 비용 때문에 자동 트리거 OFF).

**설정 1회**: 저장소 → Settings → Secrets and variables → Actions → **New repository secret**
→ 이름 `OPENAI_API_KEY`, 값에 API 키 입력.

**실행**: 저장소 **Actions** 탭 → *Evaluate (한의학 문제 정답률)* → **Run workflow**
- `models`: 평가할 모델 ID(공백 구분), 예) `gpt-4o-2024-08-06 ft:gpt-4o-...:hani-exam:...`
- `eval_all`: 전체 문항(true) / 검증셋만(false)

결과는 **Artifacts(`eval-results`)** 로 다운로드되고, 요약은 워크플로 **Summary** 에 표로 표시됩니다.
정기 평가가 필요하면 워크플로의 `schedule:` 주석을 해제하세요(비용 주의).

---

## 💰 비용 가이드

대략적인 호출 수(1회 실행 기준):

| 작업 | API 호출 수 | 비고 |
|---|---|---|
| 검증셋 평가 | 모델당 ~51회 | 짧은 응답(번호) |
| 전체 평가(`--all`) | 모델당 ~517회 | |
| 해설 생성 | ~517~587회 | 응답이 길어 토큰 사용량 ↑ |
| 파인튜닝 | 학습 토큰량 기반 과금 | 모델·에폭에 비례 |

> 실제 비용은 모델 단가에 따라 다릅니다. 먼저 `--limit`/검증셋으로 소규모 확인 후 전체를 돌리세요.
> 추론형 모델(gpt‑5/o‑계열)은 `reasoning_effort=low` 와 넉넉한 출력 토큰으로 평가합니다(빈 응답 방지).

---

## 🛠 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| `openai.AuthenticationError` | `.env` 의 `OPENAI_API_KEY` 누락/오타 |
| 평가에서 정답이 전부 `None` | 추론형 모델인데 출력 토큰이 부족 → `config.EVAL_MAX_TOKENS_REASONING` 상향 |
| 파인튜닝 `model not found / not fine-tunable` | 해당 모델 ID가 파인튜닝 미지원 → `.env` 의 모델 ID를 파인튜닝 가능 스냅샷으로 변경 |
| GPT‑5 관련 파라미터 오류 | `config.REASONING_MODEL_HINTS` 로 추론형을 감지해 `temperature` 등을 자동 생략. 새 모델명은 힌트에 추가 |
| 한글이 깨져 보임(콘솔) | 파일은 UTF‑8 정상. Windows 콘솔 표시 문제 → `chcp 65001` 또는 `PYTHONUTF8=1` |

---

## 🔧 데이터 추출 파이프라인 (`scripts/`)

국시원 PDF는 **텍스트가 전부 벡터(곡선)로 변환**되어(복사 방지) 일반 텍스트 추출이 불가 →
**고해상도 렌더 + 비전 전사**가 유일한 방법. A3 한 페이지 통째로는 작은 한자가 뭉개지므로
**2단 컬럼 타일링**(장변 ≤1980px)으로 가독성 확보. 사진·도표 문항은 국시원이 비공개 처리하여
`[그림 생략]` 표기 후 메인 학습셋에서 제외.

```powershell
python scripts/discover.py --max-pages 30                 # 1) 게시물·첨부 탐색 → manifest.json
python scripts/fetch_render.py --manifest manifest.json   # 2) PDF 다운로드 + 페이지 PNG 렌더
python scripts/tile_render.py --pdf "..." --out "..."     # 3) 2단 컬럼 타일 분할
python scripts/transcribe_api.py questions --slug ...      # 4) 비전 API 전사 (정답표 포함)
python scripts/assemble.py                                # 5) 전사본 + 정답표 조인 → dataset/*.jsonl
```

> 게시판에는 **최신 회차만** 공개됩니다(과년도는 내려감). 새 회차가 올라오면 동일 파이프라인 재실행.
> 원본 PDF/이미지(`raw/`, `img/`, `tiles/`)는 용량·저작권상 레포에서 제외됩니다.

### 품질 메모
- 587문항 전부 정답 매칭, 전부 보기 5개(정답누락 0 / 보기오류 0).
- 비전 전사 특성상 일부 한자·작은 글씨에 국소적 오탈자 가능 → 학습 전 표본 검수 권장.

---

## 📁 디렉터리
```
.
├── .github/workflows/evaluate.yml   # 수동 실행 평가 워크플로
├── dataset/                         # 최종 JSONL 데이터셋
├── training/                        # 파인튜닝 + 평가 + 해설생성 코드
│   ├── config.py
│   ├── prepare_data.py
│   ├── add_explanations.py
│   ├── finetune.py
│   ├── evaluate.py
│   └── run_all.py
├── scripts/                         # PDF→데이터셋 추출 파이프라인
├── requirements.txt
├── .env.example
└── README.md
```
