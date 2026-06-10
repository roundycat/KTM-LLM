# -*- coding: utf-8 -*-
"""각 문항에 '해설' 필드를 생성해 새 데이터셋(dataset/한의학_문제_해설.jsonl)을 만든다.

핵심 아이디어
-------------
원본 국시원 자료에는 해설이 없다. 그래서 LLM으로 해설을 생성하되,
**정답을 모델에게 알려준 상태**에서 "왜 이 답이 옳은지"를 서술하게 한다.
모델이 스스로 정답을 고르게 하는 것보다 사실 오류 위험이 낮다.

특징
----
- 재개 가능(resumable): 출력 파일에 이미 해설이 있는 (source,교시,번호) 는 건너뛴다.
  중간에 멈춰도 다시 실행하면 이어서 생성한다.
- 동시 요청(ThreadPoolExecutor)으로 속도 확보, 실패 시 재시도.
- 그림 문항(has_figure=True)도 텍스트가 있으면 해설을 생성하되, 그림 의존이 크면
  부정확할 수 있으므로 기본적으로 --skip-figure 로 제외 가능.

⚠️ 비용/정확도
- 문항 수만큼 API 호출이 일어난다(약 517~587회). 비용이 발생한다.
- 생성된 해설은 참고용이다. 학습/배포 전 표본 검수를 권장한다.

사용법
------
    python training/add_explanations.py                 # 전체 생성(재개 가능)
    python training/add_explanations.py --limit 5       # 5개만(테스트)
    python training/add_explanations.py --skip-figure   # 그림 문항 제외
"""
from __future__ import annotations
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from config import (
    FULL_DATASET, EXPLAINED_DATASET, EXPLANATION_MODEL,
    EXPLANATION_SYSTEM_PROMPT, EVAL_CONCURRENCY, REQUEST_TIMEOUT,
    is_reasoning_model,
)

client = OpenAI()


def row_key(row: dict) -> tuple:
    """문항을 유일하게 식별하는 키(재개 시 중복 생성 방지)."""
    return (row.get("source"), row.get("교시"), row.get("번호"), row["question"][:30])


def load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_user_prompt(row: dict) -> str:
    """문제 + 보기 + 확정 정답을 모델에 전달."""
    lines = [f"[문제] {row['question'].strip()}", ""]
    for i, opt in enumerate(row["options"], start=1):
        lines.append(f"{i}. {opt.strip()}")
    lines += ["", f"[확정 정답] {row['answer']}번: {row['answer_text'].strip()}",
              "", "위 정답에 대한 해설을 작성하세요."]
    return "\n".join(lines)


def make_explanation(row: dict) -> str:
    kwargs = dict(
        model=EXPLANATION_MODEL,
        messages=[
            {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
        ],
        max_completion_tokens=400,
        timeout=REQUEST_TIMEOUT,
    )
    # 추론형 모델은 temperature 미지원 → 일반 모델에만 약간의 다양성 부여
    if not is_reasoning_model(EXPLANATION_MODEL):
        kwargs["temperature"] = 0.3
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="생성 개수 제한(0=전체)")
    ap.add_argument("--skip-figure", action="store_true", help="그림 문항 제외")
    args = ap.parse_args()

    rows = load_jsonl(FULL_DATASET)
    if args.skip_figure:
        rows = [r for r in rows if not r.get("has_figure", False)]

    # 이미 생성된 해설 로드(재개)
    done = {row_key(r): r for r in load_jsonl(EXPLAINED_DATASET) if r.get("해설")}
    todo = [r for r in rows if row_key(r) not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"대상 {len(rows)}문항 | 이미 완료 {len(done)} | 이번에 생성 {len(todo)} "
          f"| 모델 {EXPLANATION_MODEL}")

    results = dict(done)  # 키 -> 해설 포함 row

    def worker(row):
        for attempt in range(3):
            try:
                exp = make_explanation(row)
                return row, exp, None
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return row, None, str(e)
                time.sleep(2 * (attempt + 1))

    failed = 0
    with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as ex:
        futures = [ex.submit(worker, r) for r in todo]
        for i, fut in enumerate(as_completed(futures), start=1):
            row, exp, err = fut.result()
            if err:
                failed += 1
                print(f"  [실패] {row_key(row)} : {err}")
            else:
                results[row_key(row)] = {**row, "해설": exp}
            if i % 20 == 0 or i == len(todo):
                print(f"  진행 {i}/{len(todo)} (실패 {failed})")

    # 원본 순서를 유지하며 저장(해설 없는 항목은 해설="" 로 둠)
    EXPLAINED_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPLAINED_DATASET, "w", encoding="utf-8") as f:
        for r in rows:
            merged = results.get(row_key(r), {**r, "해설": ""})
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")

    n_with = sum(1 for r in rows if results.get(row_key(r), {}).get("해설"))
    print(f"\n저장 완료: {EXPLAINED_DATASET}")
    print(f"  해설 포함 {n_with}/{len(rows)}문항 (실패 {failed})")


if __name__ == "__main__":
    main()
