# -*- coding: utf-8 -*-
"""공통 설정 — 경로, 모델 ID, 프롬프트, 하이퍼파라미터."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
DATA_DIR = ROOT / "data"          # 파인튜닝용 변환 데이터 (gitignore)
RESULTS_DIR = ROOT / "results"    # 평가 결과 (gitignore)

# 587문항 전체(과목/그림여부 메타 포함). 그림 문항은 학습/평가에서 제외한다.
FULL_DATASET = DATASET_DIR / "한의학_문제_전체.jsonl"

# add_explanations.py 가 생성하는, '해설' 필드가 추가된 데이터셋.
#   - 존재하면 prepare_data.py --with-rationale 가 이 파일을 우선 사용한다.
EXPLAINED_DATASET = DATASET_DIR / "한의학_문제_해설.jsonl"

# scripts/fetch_terminology.py 가 수집하는 KIOM 표준한의학용어집(용어↔정의).
#   - prepare_terminology.py 가 이를 Q&A SFT 로 변환해 글로벌 모델 파인튜닝에 사용한다.
TERMINOLOGY_DATASET = DATASET_DIR / "한의학_용어.jsonl"

FINETUNE_TRAIN = DATA_DIR / "finetune_train.jsonl"
FINETUNE_VAL = DATA_DIR / "finetune_val.jsonl"

# 용어 학습용 SFT(기출과 분리 보관). prepare_terminology.py 산출물.
FINETUNE_TERM_TRAIN = DATA_DIR / "finetune_term_train.jsonl"
FINETUNE_TERM_VAL = DATA_DIR / "finetune_term_val.jsonl"

# ---------------------------------------------------------------------------
# 모델 ID
#   - GPT-4 계열은 fine-tuning 가능한 스냅샷을 기본값으로 둔다.
#   - GPT-5 계열은 계정 권한에 따라 다르므로 .env(GPT5_MODEL)로 덮어쓰는 것을 권장.
# ---------------------------------------------------------------------------
GPT4_MODEL = os.getenv("GPT4_MODEL", "gpt-4o-2024-08-06")
GPT5_MODEL = os.getenv("GPT5_MODEL", "gpt-5")

# 파인튜닝 결과 모델 ID(있으면 평가에 포함)
GPT4_FINETUNED_MODEL = os.getenv("GPT4_FINETUNED_MODEL", "").strip()
GPT5_FINETUNED_MODEL = os.getenv("GPT5_FINETUNED_MODEL", "").strip()

# 해설 생성에 사용할 모델(정답이 주어진 상태에서 근거를 서술하므로 일반 chat 모델로 충분).
EXPLANATION_MODEL = os.getenv("EXPLANATION_MODEL", GPT4_MODEL)

# 베이스 모델 매핑(파인튜닝/평가에서 이름으로 참조)
BASE_MODELS = {
    "gpt-4": GPT4_MODEL,
    "gpt-5": GPT5_MODEL,
}

# ---------------------------------------------------------------------------
# 로컬(한국형) 모델 — Ollama 의 OpenAI 호환 엔드포인트로 평가한다.
#   - "글로벌 모델"(OpenAI 클라우드)과 대비되는 "로컬 모델"은 한국형 오픈웨이트 LLM.
#   - EXAONE 3.5(LG) 등을 `ollama pull` 한 뒤 base_url 만 바꿔 OpenAI SDK 그대로 호출한다.
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# Ollama 는 api_key 를 검증하지 않지만, OpenAI SDK 는 빈 키를 거부하므로 더미값을 둔다.
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
# 평가에 쓸 한국형(로컬) 모델 태그 — `ollama list` 에 보이는 이름과 일치해야 한다.
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "exaone3.5:7.8b")

# provider 구분자
PROVIDER_OPENAI = "openai"   # 클라우드(OpenAI 또는 OpenAI 호환: Gemini 등)
PROVIDER_LOCAL = "local"     # 로컬(Ollama) 모델 — OLLAMA_BASE_URL 사용

# ---------------------------------------------------------------------------
# 글로벌(범용) 모델 — '로컬(한국형)' 과 대비되는 일반 모델.
#   무료 실험(RAG)에서는 로컬 범용 오픈모델(Qwen 등)을 글로벌로 둔다(provider=local).
#   클라우드로 바꾸려면 .env 에서 GLOBAL_PROVIDER=openai 로 두고
#   OPENAI_BASE_URL(예: Gemini OpenAI 호환 엔드포인트)+OPENAI_API_KEY 를 설정한다.
# ---------------------------------------------------------------------------
GLOBAL_MODEL = os.getenv("GLOBAL_MODEL", "qwen2.5:7b")
GLOBAL_PROVIDER = os.getenv("GLOBAL_PROVIDER", PROVIDER_LOCAL)

# 모델 역할 라벨(리포트 그룹핑용)
ROLE_KOREAN = "한국형"     # 로컬(EXAONE 등)
ROLE_GLOBAL = "글로벌"     # 범용(Qwen/OpenAI/Gemini 등)


def make_client(provider: str = PROVIDER_OPENAI):
    """provider 에 맞는 OpenAI 호환 클라이언트를 반환한다.

    - local : Ollama 의 OpenAI 호환 엔드포인트(OLLAMA_BASE_URL).
    - openai: 기본은 OpenAI 클라우드. OPENAI_BASE_URL 이 있으면 그 OpenAI 호환
              엔드포인트(Gemini 등)로 보낸다.
    """
    from openai import OpenAI
    if provider == PROVIDER_LOCAL:
        return OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    base = os.getenv("OPENAI_BASE_URL")
    if base:
        return OpenAI(base_url=base, api_key=os.getenv("OPENAI_API_KEY", ""))
    return OpenAI()

# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "당신은 한의사·한약사 국가시험 문제를 푸는 한의학 전문가입니다. "
    "주어진 5지선다 문제를 읽고 정답을 고르세요."
)

# 파인튜닝 정답 형식: 번호만 출력하도록 학습한다.
ANSWER_INSTRUCTION = "정답 번호(1~5) 하나만 숫자로 출력하세요. 설명은 하지 마세요."

# Track A — 추론 개선(CoT). 근거를 먼저 쓰고 마지막 줄에 '정답: N' 형식으로 답하게 한다.
#   parse_answer 가 '정답: N' 을 최우선으로 추출하므로 채점이 안정적이다.
COT_ANSWER_INSTRUCTION = (
    "다음 순서로 답하세요. 먼저 핵심 근거를 2~3문장으로 간단히 쓰고, "
    "마지막 줄에 반드시 '정답: N' (N은 1~5 중 하나) 형식으로 정답 하나만 쓰세요."
)

# 해설 생성용 시스템 프롬프트. 정답을 알려준 상태에서 '왜 그 답인지'를 서술하게 하여
# 모델이 스스로 푸는 것보다 사실 오류 가능성을 낮춘다.
EXPLANATION_SYSTEM_PROMPT = (
    "당신은 한의학 교수입니다. 한의사·한약사 국가시험 5지선다 문제와 '확정된 정답'이 주어집니다. "
    "정답이 왜 옳은지 핵심 근거를 설명하고, 헷갈리기 쉬운 오답이 왜 틀렸는지 간단히 짚어 주세요. "
    "한국어로 3~5문장, 군더더기 없이 작성하세요. 정답 번호를 바꾸려 하지 마세요."
)

# --with-rationale 학습 시 assistant 타깃 형식(해설 후 정답 번호).
RATIONALE_TEMPLATE = "{해설}\n\n정답: {answer}"

# 용어 학습(SFT)용 — KIOM 표준한의학용어집의 '용어→정의'를 Q&A 로 학습시킨다.
TERM_SYSTEM_PROMPT = (
    "당신은 한의학 용어에 정통한 전문가입니다. "
    "한의학 표준 용어의 정의와 의미를 정확하게 설명하세요."
)
# 용어 질문 프롬프트 템플릿(여러 표현으로 다양화해 과적합을 줄인다).
TERM_QUESTION_TEMPLATES = (
    "한의학 용어 '{term}'의 정의를 설명하세요.",
    "'{term}'은(는) 한의학에서 무엇을 뜻합니까?",
    "다음 한의학 용어를 설명하시오: {term}",
)

# ---------------------------------------------------------------------------
# RAG(용어 주입) — 파인튜닝 없이 문제에 등장하는 KIOM 표준 용어 정의를 주입.
#   '글로벌 모델에 용어를 학습' 단계를 무료로 대체(지식 주입 효과 측정).
# ---------------------------------------------------------------------------
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))   # 한 문제에 주입할 최대 용어 수
RAG_MIN_TERM_LEN = 2                            # 1글자 용어는 과매칭 → 제외
RAG_INJECT_HEADER = "[참고] 아래는 문제와 관련될 수 있는 한의학 표준 용어 정의입니다. 필요하면 참고하세요."

# ---------------------------------------------------------------------------
# 하이퍼파라미터 / 실행 옵션
# ---------------------------------------------------------------------------
VAL_RATIO = 0.1          # 학습/검증 분할 비율
RANDOM_SEED = 42
EVAL_CONCURRENCY = 8     # 평가 시 동시 요청 수
REQUEST_TIMEOUT = 60     # 초

# 출력 토큰 예산.
#  - 일반 모델: 번호만 받으면 되므로 작게.
#  - 추론형 모델: max_completion_tokens 에 '추론 토큰'이 포함되므로 넉넉히 줘야
#    실제 답(content)이 잘리지 않는다. 너무 작으면 빈 응답이 나온다.
EVAL_MAX_TOKENS = 16
EVAL_MAX_TOKENS_REASONING = 2048
# 로컬(Ollama) 모델은 '번호만' 지시를 무시하고 설명을 덧붙이는 경우가 있어 약간 넉넉히 준다.
# (parse_answer 가 '정답: N' 또는 마지막 숫자를 견고하게 추출한다.)
EVAL_MAX_TOKENS_LOCAL = 512
# CoT(근거 서술)는 출력이 길어 더 넉넉히.
EVAL_MAX_TOKENS_COT = 640

# Track A — self-consistency: 같은 문제를 N회(temp>0) 샘플해 다수결.
EVAL_SC_SAMPLES = 5
EVAL_SC_TEMPERATURE = 0.7
# 추론형 모델의 추론 강도(비용/지연 절감). 지원하지 않는 모델이면 자동 무시되도록 예외 처리.
EVAL_REASONING_EFFORT = "low"

# 일부 추론형 모델(gpt-5 등)은 temperature 등 샘플링 파라미터를 받지 않는다.
# 이름에 아래 토큰이 들어가면 샘플링 파라미터를 생략한다.
REASONING_MODEL_HINTS = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(model_id: str) -> bool:
    m = model_id.lower()
    return any(h in m for h in REASONING_MODEL_HINTS)
