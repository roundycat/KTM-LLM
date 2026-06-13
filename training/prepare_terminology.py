# -*- coding: utf-8 -*-
"""KIOM 표준한의학용어집 → OpenAI Fine-tuning chat 포맷(SFT)으로 변환.

목적
----
글로벌(OpenAI) 모델에 '한의학 용어'를 학습시키기 위한 SFT 데이터를 만든다.
각 용어를 "용어 설명을 요구하는 질문 → 표준 정의" 형태의 1턴 대화로 변환한다.

  system    : config.TERM_SYSTEM_PROMPT (한의학 용어 전문가)
  user      : "한의학 용어 '간(肝)'의 정의를 설명하세요." (템플릿 다양화)
  assistant : 표준 정의 (+ 동의어)

생성물
------
- data/finetune_term_train.jsonl, data/finetune_term_val.jsonl

옵션
----
- --limit N        : 용어 수 제한(비용 조절). 0=전체.
- --min-def N      : 정의 길이 N자 미만 용어 제외(기본 6).
- --combine-exam   : 기출 학습셋(train 분할)도 함께 섞는다.
                     → 용어 지식 + MCQ 응답 형식을 동시에 학습(평가 점수에 유리,
                       단 '용어 효과'만 분리하려면 끄는 게 깨끗하다). val 은 항상 제외.

사용법
------
  python training/prepare_terminology.py                 # 용어만, 전체
  python training/prepare_terminology.py --limit 2000    # 비용 절감
  python training/prepare_terminology.py --combine-exam  # 용어+기출(train)
"""
from __future__ import annotations
import argparse
import json
import random

import prepare_data  # 기출 train 분할 재사용(--combine-exam)
from config import (
    TERMINOLOGY_DATASET, FULL_DATASET, DATA_DIR,
    FINETUNE_TERM_TRAIN, FINETUNE_TERM_VAL,
    TERM_SYSTEM_PROMPT, TERM_QUESTION_TEMPLATES,
    VAL_RATIO, RANDOM_SEED,
)


def load_terms() -> list[dict]:
    if not TERMINOLOGY_DATASET.exists():
        raise SystemExit(
            f"{TERMINOLOGY_DATASET} 없음 → 먼저 scripts/fetch_terminology.py 실행")
    with open(TERMINOLOGY_DATASET, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def term_to_chat(row: dict, idx: int) -> dict:
    """용어 1개 → chat SFT 예시. 질문 템플릿은 idx 로 순환해 다양화."""
    headword = row.get("headword") or row["term"]
    q = TERM_QUESTION_TEMPLATES[idx % len(TERM_QUESTION_TEMPLATES)].format(term=headword)
    answer = row["definition"].strip()
    syn = (row.get("synonyms") or "").strip()
    if syn:
        answer += f"\n동의어: {syn}"
    return {
        "messages": [
            {"role": "system", "content": TERM_SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": answer},
        ]
    }


def exam_train_examples() -> list[dict]:
    """기출 데이터의 'train 분할'만 chat 예시로 반환(val 은 제외 → 누수 방지)."""
    rows = prepare_data.load_rows(with_rationale=False)
    usable = [r for r in rows if not r.get("has_figure", False)]
    random.seed(RANDOM_SEED)
    random.shuffle(usable)
    n_val = max(1, int(len(usable) * VAL_RATIO))
    train_rows = usable[n_val:]  # val(앞 n_val) 제외
    return [prepare_data.to_chat_example(r, with_rationale=False) for r in train_rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="용어 수 제한(0=전체)")
    ap.add_argument("--min-def", type=int, default=6, help="정의 최소 길이(자)")
    ap.add_argument("--combine-exam", action="store_true",
                    help="기출 train 분할도 함께 학습")
    args = ap.parse_args()
    run(limit=args.limit, min_def=args.min_def, combine_exam=args.combine_exam)


def run(limit: int = 0, min_def: int = 6, combine_exam: bool = False) -> None:
    terms = load_terms()
    terms = [t for t in terms if len(t.get("definition", "").strip()) >= min_def]

    # 고정 시드로 셔플(재현 가능) 후 제한
    random.seed(RANDOM_SEED)
    random.shuffle(terms)
    if limit:
        terms = terms[:limit]

    term_examples = [term_to_chat(t, i) for i, t in enumerate(terms)]

    # train/val 분할(용어 기준)
    n_val = max(1, int(len(term_examples) * VAL_RATIO))
    val_examples = term_examples[:n_val]
    train_examples = term_examples[n_val:]

    n_exam = 0
    if combine_exam:
        exam = exam_train_examples()
        n_exam = len(exam)
        train_examples += exam
        random.shuffle(train_examples)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path, subset in [(FINETUNE_TERM_TRAIN, train_examples),
                         (FINETUNE_TERM_VAL, val_examples)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in subset:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # 대략적 학습 토큰 추정(문자수/2 ≈ 토큰, 한국어 보수적 가정)
    approx_tokens = sum(len(m["content"]) for ex in train_examples
                        for m in ex["messages"]) // 2

    print(f"용어 {len(terms)}개 → SFT 예시 {len(term_examples)}개")
    if combine_exam:
        print(f"  + 기출 train {n_exam}개 결합")
    print(f"  학습: {len(train_examples)}개 → {FINETUNE_TERM_TRAIN}")
    print(f"  검증: {len(val_examples)}개 → {FINETUNE_TERM_VAL}")
    print(f"  대략 학습 토큰 ≈ {approx_tokens:,} (에폭당). 비용은 모델 단가 × 에폭 수에 비례.")


if __name__ == "__main__":
    main()
