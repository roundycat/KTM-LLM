# -*- coding: utf-8 -*-
"""전체 실험 오케스트레이터 — 사용자 '순서' 자동화 (무료 RAG 경로).

  1) 한국형(로컬 EXAONE)·글로벌(범용 Qwen)에 한의학 시험을 풀게 함
  2) 두 모델의 평균 정답률 도출
  3) 글로벌 모델에 한의학 용어를 '주입'(RAG) — 파인튜닝 없이 무료
  4) 1~2 반복(=글로벌 RAG 평가)하여 글로벌 향상(Δ) 확인 + 리포트

전제
----
- Ollama 가 떠 있고 LOCAL_MODEL/GLOBAL_MODEL 이 pull 되어 있어야 한다.
- dataset/한의학_용어.jsonl 이 있어야 RAG 가 동작(없으면 scripts/fetch_terminology.py).

사용법:
    python training/run_all.py                # 한국형+글로벌 base + 글로벌 RAG
    python training/run_all.py --all          # 전체 문항
    python training/run_all.py --no-rag       # base 비교만
    python training/run_all.py --rag-local    # 한국형에도 RAG

(유료) 파인튜닝 경로를 쓰려면: prepare_terminology.py → finetune.py --terminology
이후 .env 의 GPT4_FINETUNED_MODEL 을 채우고 evaluate.py 를 다시 실행하면
report.py 가 파인튜닝 향상까지 표에 포함한다.
"""
from __future__ import annotations
import argparse

import evaluate
import report
from rag import TermIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="전체 텍스트 문항으로 평가")
    ap.add_argument("--no-rag", dest="rag", action="store_false", help="RAG 변형 제외")
    ap.add_argument("--rag-local", action="store_true", help="한국형에도 RAG 적용")
    ap.add_argument("--no-local", action="store_true", help="한국형 평가 생략")
    ap.add_argument("--no-global", action="store_true", help="글로벌 평가 생략")
    ap.add_argument("--models", nargs="*", help="글로벌 자리 모델 ID(OpenAI/파인튜닝)")
    ap.add_argument("--local-models", nargs="*", help="한국형 자리 로컬 태그")
    ap.set_defaults(rag=True)
    args = ap.parse_args()

    rows = evaluate.select_eval_rows(args.all)
    index = TermIndex() if args.rag else None
    if args.rag and (index is None or len(index) == 0):
        print("[경고] 용어 인덱스 비어있음 → RAG 비활성화 (scripts/fetch_terminology.py 먼저)")
        args.rag = False
        index = None

    targets = evaluate.resolve_targets(args)
    print(f"\n########## 1~4. 평가 ({len(rows)}문항, RAG={'on' if args.rag else 'off'}) ##########")
    for t in targets:
        print(f"  - {t['role']}/{'rag' if t['rag'] else 'base'}: {t['id']} [{t['provider']}]")

    summary = []
    for t in targets:
        try:
            summary.append(evaluate.evaluate_model(t, rows, index))
        except Exception as e:  # noqa: BLE001
            print(f"[평가 오류] {t['id']}[{t['provider']}]: {e}")
    evaluate.save_results(summary, args.all, len(rows))

    print("\n########## 종합 리포트 ##########")
    report.main()


if __name__ == "__main__":
    main()
