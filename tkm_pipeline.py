"""
tkm_pipeline.py
================
"GPT-4 can pass the Korean National Licensing Examination for Korean Medicine
Doctors" (Jang et al., 2023, PLOS Digital Health) 논문에서 사용한 5단계 누적
프롬프트 기법을 재현하는 파이프라인.

5 techniques (cumulative, Fig. 1 in the paper):
  1. Chinese-term annotation   : 한의학 전문용어에 한자 병기
  2. English-translated instruction : 시스템 지시문을 영어로 번역
  3. English-translated question    : 문제 지문/보기를 영어로 번역
  4. Exam-optimized instruction     : CoT + "정답 하나만 고르라"는 지시
  5. Self-consistency               : 동일 문제를 N회 반복 후 다수결

다양한 LLM(OpenAI GPT 계열, Anthropic Claude, Google Gemini 등)을 동일한
인터페이스로 호출하기 위해 `litellm`을 사용합니다.

설치:
    pip install litellm --break-system-packages

API 키 설정 (사용할 모델에 맞게 환경변수 설정):
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export GEMINI_API_KEY="..."
    (litellm이 모델 문자열을 보고 자동으로 알맞은 키를 사용합니다)

실행 예:
    python tkm_pipeline.py --model gpt-4o --data sample_questions.json --stage 5 --n-trials 7
    python tkm_pipeline.py --model claude-sonnet-4-6 --data sample_questions.json --stage 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

try:
    from litellm import completion
except ImportError:  # pragma: no cover
    completion = None


# --------------------------------------------------------------------------- #
# 1. Data model
# --------------------------------------------------------------------------- #

@dataclass
class Question:
    id: str
    subject: str
    question_kr: str
    choices_kr: list[str]
    correct_answer: int  # 1-indexed, matches paper's 5-choice format
    # TKM 용어 -> 한자 매핑. 예: {"기허": "氣虛", "어혈": "瘀血"}
    tkm_terms: dict[str, str] = field(default_factory=dict)

    # 캐시 (번역 결과를 매 시행마다 재호출하지 않기 위함)
    _question_en: Optional[str] = None
    _choices_en: Optional[list[str]] = None


# --------------------------------------------------------------------------- #
# 2. LLM call wrapper (다양한 모델 지원)
# --------------------------------------------------------------------------- #

def call_llm(model: str, system: str, user: str, temperature: float = 1.0) -> str:
    """litellm을 통해 어떤 provider의 모델이든 동일하게 호출."""
    if completion is None:
        raise RuntimeError(
            "litellm이 설치되어 있지 않습니다. `pip install litellm` 을 실행하세요."
        )
    resp = completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=1500,
    )
    return resp["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# 3. Technique 1 — Chinese-term annotation
# --------------------------------------------------------------------------- #

def annotate_chinese_terms(text: str, term_map: dict[str, str]) -> str:
    """한의학 용어 뒤에 (한자)를 병기. 예: '기허' -> '기허(氣虛)'"""
    annotated = text
    # 긴 용어부터 치환해야 부분 문자열 충돌을 피할 수 있음
    for term in sorted(term_map, key=len, reverse=True):
        hanja = term_map[term]
        annotated = annotated.replace(term, f"{term}({hanja})")
    return annotated


# --------------------------------------------------------------------------- #
# 4. Technique 2 & 3 — English translation (instruction / question)
# --------------------------------------------------------------------------- #

TRANSLATE_SYSTEM = (
    "You are a professional medical translator specializing in Traditional "
    "Korean Medicine (TKM). Translate the given Korean text into natural, "
    "precise English. Preserve any parenthetical Chinese-character "
    "annotations exactly as given. Output ONLY the translation, nothing else."
)


def translate_to_english(text: str, model: str) -> str:
    return call_llm(model=model, system=TRANSLATE_SYSTEM, user=text, temperature=0.0)


def get_translated_question(q: Question, model: str) -> tuple[str, list[str]]:
    """문제/보기를 번역하고 캐시. 이미 번역돼 있으면 재사용."""
    if q._question_en is None:
        q._question_en = translate_to_english(q.question_kr, model)
    if q._choices_en is None:
        # 보기 5개를 한 번에 번역 (번호 유지)
        joined = "\n".join(f"{i+1}. {c}" for i, c in enumerate(q.choices_kr))
        translated_joined = translate_to_english(joined, model)
        # 파싱: "1. ..." 형태 라인을 추출
        lines = [l.strip() for l in translated_joined.splitlines() if l.strip()]
        parsed = []
        for l in lines:
            m = re.match(r"^\d+\.\s*(.*)$", l)
            parsed.append(m.group(1) if m else l)
        # 혹시 파싱이 어긋나면 원본 개수만큼 안전하게 자르기/채우기
        if len(parsed) != len(q.choices_kr):
            parsed = (parsed + q.choices_kr)[: len(q.choices_kr)]
        q._choices_en = parsed
    return q._question_en, q._choices_en


# --------------------------------------------------------------------------- #
# 5. Prompt builder — stages 0~4 (누적 적용), stage 5 = self-consistency wrapper
# --------------------------------------------------------------------------- #

BASE_INSTRUCTION_KR = (
    "다음은 한의사 국가시험 문제입니다. 보기 중 가장 적절한 답 하나를 고르세요."
)

EXAM_OPTIMIZED_INSTRUCTION_EN = (
    "You are taking the Korean National Licensing Examination for Korean "
    "Medicine Doctors. Read the question and the five answer choices "
    "carefully. Reason step by step, considering the relevant Traditional "
    "Korean Medicine (TKM) knowledge needed. After your reasoning, you MUST "
    "select exactly ONE choice as your final answer. "
    "End your response with a final line in exactly this format:\n"
    "FINAL ANSWER: <choice number>"
)


def build_prompt(q: Question, stage: int, model_for_translation: str) -> tuple[str, str]:
    """
    stage 0: 원문 그대로, 한글 지시문
    stage 1: + 한자 병기
    stage 2: + 지시문 영어 번역
    stage 3: + 문제/보기 영어 번역
    stage 4: + exam-optimized instruction (CoT + 정답 하나 강제)
    (stage 5 = self-consistency는 이 프롬프트를 N번 호출하는 것으로 별도 처리)
    반환값: (system_prompt, user_prompt)
    """
    question_text = q.question_kr
    choices = list(q.choices_kr)
    instruction = BASE_INSTRUCTION_KR

    if stage >= 1:
        question_text = annotate_chinese_terms(question_text, q.tkm_terms)
        choices = [annotate_chinese_terms(c, q.tkm_terms) for c in choices]

    if stage >= 2:
        instruction = (
            "The following is a question from the Korean National Licensing "
            "Examination for Korean Medicine Doctors. Choose the single best "
            "answer among the choices."
        )

    if stage >= 3:
        en_q, en_choices = get_translated_question(q, model_for_translation)
        # 영어 번역본을 쓰되, 한자 병기(stage1)는 이미 원문에 반영되어 있으므로
        # 번역 함수가 괄호 안 한자를 보존하도록 프롬프트에 명시되어 있음.
        question_text, choices = en_q, en_choices

    system_prompt = EXAM_OPTIMIZED_INSTRUCTION_EN if stage >= 4 else instruction

    choices_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
    user_prompt = f"{question_text}\n\n{choices_block}"

    return system_prompt, user_prompt


# --------------------------------------------------------------------------- #
# 6. Answer extraction (논문 2.5절 Encoding of answers 규칙 반영)
# --------------------------------------------------------------------------- #

REFUSAL_PATTERNS = [
    "i am an ai", "i'm an ai", "cannot provide medical", "not authorized",
    "consult a medical professional", "academic integrity", "academic ethics",
]


def is_refusal(response: str) -> bool:
    low = response.lower()
    return any(p in low for p in REFUSAL_PATTERNS)


def extract_answer(response: str) -> Optional[int]:
    """
    'FINAL ANSWER: N' 형식을 우선 탐색하고, 없으면 응답 내 마지막에 등장하는
    1~5 숫자를 정답으로 간주 (논문의 관대한 채점 방식 근사).
    "정답이 여러 개" 또는 "정답 없음"이라고 답하면 오답 처리.
    """
    low = response.lower()
    if "more than one" in low or "no correct answer" in low or "정답이 없" in low:
        return None

    m = re.search(r"final answer\s*:\s*([1-5])", response, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # fallback: 응답 끝부분에서 1~5 숫자 탐색
    tail = response[-200:]
    nums = re.findall(r"\b([1-5])\b", tail)
    if nums:
        return int(nums[-1])
    return None


# --------------------------------------------------------------------------- #
# 7. Single-trial & self-consistency runners
# --------------------------------------------------------------------------- #

def run_single_trial(
    q: Question, model: str, stage: int, translation_model: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[int]:
    """한 번의 시행. 거부 응답이면 재시도(논문 방식)."""
    translation_model = translation_model or model
    system_prompt, user_prompt = build_prompt(q, stage, translation_model)

    for _ in range(max_retries):
        response = call_llm(model=model, system=system_prompt, user=user_prompt)
        if is_refusal(response):
            continue  # 논문처럼 거부 응답은 버리고 재시도
        return extract_answer(response)
    return None  # 재시도 초과 시 오답 처리


def run_self_consistency(
    q: Question, model: str, n_trials: int = 7, translation_model: Optional[str] = None,
) -> tuple[Optional[int], list[Optional[int]]]:
    """stage 4 프롬프트로 N회 독립 시행 후 다수결(최빈값)로 최종 답 결정."""
    answers = [
        run_single_trial(q, model, stage=4, translation_model=translation_model)
        for _ in range(n_trials)
    ]
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, answers
    final = Counter(valid).most_common(1)[0][0]
    return final, answers


# --------------------------------------------------------------------------- #
# 8. Dataset evaluation
# --------------------------------------------------------------------------- #

def evaluate(
    questions: list[Question],
    model: str,
    stage: int,
    n_trials: int = 1,
    translation_model: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    stage 0~4: 문항당 1회 채점.
    stage 5   : self-consistency(N trials, 다수결) 적용.
    """
    results = []
    correct = 0

    for q in questions:
        if stage >= 5:
            answer, trials = run_self_consistency(
                q, model, n_trials=n_trials, translation_model=translation_model
            )
        else:
            answer = run_single_trial(q, model, stage=stage, translation_model=translation_model)
            trials = [answer]

        is_correct = answer == q.correct_answer
        correct += int(is_correct)
        results.append(
            {
                "id": q.id,
                "subject": q.subject,
                "predicted": answer,
                "correct_answer": q.correct_answer,
                "is_correct": is_correct,
                "trials": trials,
            }
        )
        if verbose:
            mark = "O" if is_correct else "X"
            print(f"[{mark}] {q.id} ({q.subject}): predicted={answer}, answer={q.correct_answer}")

    accuracy = correct / len(questions) if questions else 0.0
    summary = {
        "model": model,
        "stage": stage,
        "n_trials": n_trials if stage >= 5 else 1,
        "n_questions": len(questions),
        "n_correct": correct,
        "accuracy": accuracy,
        "results": results,
    }
    return summary


# --------------------------------------------------------------------------- #
# 9. CLI
# --------------------------------------------------------------------------- #

def load_questions(path: str) -> list[Question]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        Question(
            id=item["id"],
            subject=item.get("subject", ""),
            question_kr=item["question_kr"],
            choices_kr=item["choices_kr"],
            correct_answer=item["correct_answer"],
            tkm_terms=item.get("tkm_terms", {}),
        )
        for item in raw
    ]


def main():
    parser = argparse.ArgumentParser(description="TKM licensing exam LLM benchmark pipeline")
    parser.add_argument("--model", required=True, help="예: gpt-4o, claude-sonnet-4-6, gemini/gemini-1.5-pro")
    parser.add_argument("--data", required=True, help="문항 JSON 파일 경로")
    parser.add_argument(
        "--stage", type=int, default=5, choices=[0, 1, 2, 3, 4, 5],
        help="0=베이스라인 ... 4=exam-optimized instruction까지, 5=+self-consistency",
    )
    parser.add_argument("--n-trials", type=int, default=7, help="stage 5일 때 반복 횟수")
    parser.add_argument(
        "--translation-model", default=None,
        help="번역에 쓸 모델(기본값: --model과 동일). 번역만 저렴한 모델로 하고 싶을 때 사용",
    )
    parser.add_argument("--output", default=None, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    questions = load_questions(args.data)
    summary = evaluate(
        questions,
        model=args.model,
        stage=args.stage,
        n_trials=args.n_trials,
        translation_model=args.translation_model,
    )

    print("\n=== SUMMARY ===")
    print(f"Model       : {summary['model']}")
    print(f"Stage       : {summary['stage']}")
    print(f"N questions : {summary['n_questions']}")
    print(f"Accuracy    : {summary['accuracy']*100:.2f}%")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장됨: {args.output}")


if __name__ == "__main__":
    main()
