# -*- coding: utf-8 -*-
"""데이터셋(JSONL) → OpenAI Fine-tuning chat 포맷으로 변환.

생성물
------
- data/finetune_train.jsonl, data/finetune_val.jsonl
- 각 줄은 {"messages": [system, user, assistant]} (OpenAI SFT 표준 포맷).

두 가지 학습 타깃 모드
----------------------
1) 기본(번호만):      assistant = "4"
   → 정답 선택만 빠르게 학습. 가볍고 평가 파싱도 단순.
2) --with-rationale:  assistant = "{해설}\n\n정답: 4"
   → 해설(근거)을 먼저 쓰고 정답을 내는 Chain-of-Thought 식 학습.
     dataset/한의학_문제_해설.jsonl(= add_explanations.py 산출물)이 있어야 하며,
     해설이 빈 문항은 자동으로 번호만 모드로 대체된다.

기타
----
- 그림 의존 문항(has_figure=True)은 텍스트만으로 풀 수 없으므로 항상 제외.
- 고정 시드(config.RANDOM_SEED)로 train/val 분할 → 재현 가능.
  evaluate.py 도 동일 시드로 검증셋을 재현하므로 학습/평가가 겹치지 않는다.

사용법
------
    python training/prepare_data.py                  # 번호만 모드
    python training/prepare_data.py --with-rationale # 해설 포함(CoT) 모드
"""
from __future__ import annotations
import argparse
import json
import random

from config import (
    FULL_DATASET, EXPLAINED_DATASET, DATA_DIR, FINETUNE_TRAIN, FINETUNE_VAL,
    SYSTEM_PROMPT, ANSWER_INSTRUCTION, RATIONALE_TEMPLATE, VAL_RATIO, RANDOM_SEED,
)


def load_rows(with_rationale: bool) -> list[dict]:
    """학습 소스 로드. 해설 모드이면 해설 데이터셋을 우선 사용한다."""
    path = EXPLAINED_DATASET if (with_rationale and EXPLAINED_DATASET.exists()) else FULL_DATASET
    if with_rationale and not EXPLAINED_DATASET.exists():
        print(f"[경고] {EXPLAINED_DATASET} 없음 → 해설 없이 번호만 모드로 진행합니다. "
              f"(먼저 add_explanations.py 실행)")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_user_prompt(row: dict) -> str:
    """문제 + 번호 매긴 보기 + 출력 지시를 사용자 메시지로 구성."""
    lines = [row["question"].strip(), ""]
    for i, opt in enumerate(row["options"], start=1):
        lines.append(f"{i}. {opt.strip()}")
    lines.append("")
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines)


def format_assistant_target(row: dict, with_rationale: bool) -> str:
    """학습 타깃(정답) 텍스트. 해설이 있으면 CoT 형식, 없으면 번호만."""
    rationale = (row.get("해설") or "").strip()
    if with_rationale and rationale:
        return RATIONALE_TEMPLATE.format(해설=rationale, answer=row["answer"])
    return str(row["answer"])


def to_chat_example(row: dict, with_rationale: bool) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_prompt(row)},
            {"role": "assistant", "content": format_assistant_target(row, with_rationale)},
        ]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-rationale", action="store_true",
                    help="해설을 포함한 CoT 형식으로 학습 데이터 생성")
    args = ap.parse_args()
    run(args.with_rationale)


def run(with_rationale: bool = False) -> None:
    rows = load_rows(with_rationale)
    usable = [r for r in rows if not r.get("has_figure", False)]
    skipped = len(rows) - len(usable)

    # 고정 시드로 셔플 후 분할(재현 가능)
    random.seed(RANDOM_SEED)
    random.shuffle(usable)
    n_val = max(1, int(len(usable) * VAL_RATIO))
    val_rows, train_rows = usable[:n_val], usable[n_val:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path, subset in [(FINETUNE_TRAIN, train_rows), (FINETUNE_VAL, val_rows)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(to_chat_example(r, with_rationale),
                                   ensure_ascii=False) + "\n")

    mode = "해설 포함(CoT)" if with_rationale else "번호만"
    n_rat = sum(1 for r in usable if with_rationale and (r.get("해설") or "").strip())
    print(f"모드: {mode}")
    print(f"전체 {len(rows)}문항 중 그림문항 {skipped}개 제외 → 사용 {len(usable)}개")
    if with_rationale:
        print(f"  해설 포함 타깃: {n_rat}개 (나머지는 번호만)")
    print(f"  학습: {len(train_rows)}개 → {FINETUNE_TRAIN}")
    print(f"  검증: {len(val_rows)}개 → {FINETUNE_VAL}")


if __name__ == "__main__":
    main()
