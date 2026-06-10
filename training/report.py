# -*- coding: utf-8 -*-
"""평가 결과 종합 리포트 — 로컬(한국형) vs 글로벌, 용어 주입(RAG)/파인튜닝 전후 비교.

기본적으로 results/summary.json(evaluate.py 가 한 번의 실행에서 쓴 일관된 묶음)을
읽는다. 없으면 results/eval_*.json 을 모은다(과거 파일 혼입 주의).

  1) 한국형(로컬)·글로벌(범용) 모델 정답률
  2) 두 모델 평균 정답률
  3) (글로벌 용어 주입/학습 후) 향상된 글로벌 정답률 — RAG·파인튜닝 각각
  4) 글로벌 향상(Δ) = (RAG 또는 파인튜닝) − 베이스

각 결과는 role(한국형/글로벌)·variant(base/rag/ft) 필드로 분류한다.

사용법:
  python training/report.py
"""
from __future__ import annotations
import json

from config import RESULTS_DIR, ROLE_KOREAN, ROLE_GLOBAL


def load_results() -> tuple[list[dict], int]:
    """summary.json(최신 실행) 우선, 없으면 eval_*.json 글롭."""
    sj = RESULTS_DIR / "summary.json"
    if sj.exists():
        data = json.loads(sj.read_text(encoding="utf-8"))
        return data.get("models", []), data.get("n", 0)
    out = []
    for p in sorted(RESULTS_DIR.glob("eval_*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return out, (out[0]["n"] if out else 0)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return f"{x*100:.2f}%" if x is not None else "—"


def signed(x):
    return ("+" if x >= 0 else "") + f"{x*100:.2f}%p" if x is not None else "—"


def build_report(results: list[dict], n: int) -> str:
    def grp(role, variant):
        return [r for r in results if r.get("role") == role and r.get("variant") == variant]

    def avg(role, variant):
        return mean([r["accuracy"] for r in grp(role, variant)])

    korean_base = avg(ROLE_KOREAN, "base")
    global_base = avg(ROLE_GLOBAL, "base")
    global_rag = avg(ROLE_GLOBAL, "rag")
    global_ft = avg(ROLE_GLOBAL, "ft")

    pair_avg = mean([v for v in (korean_base, global_base) if v is not None]) \
        if (korean_base is not None and global_base is not None) else None

    L = ["# 한의학 시험 — 로컬(한국형) vs 글로벌 · 용어 주입 전후 비교\n",
         f"평가 문항 수: **{n}** (그림 문항 제외)\n",
         "## 모델별 정답률\n",
         "| 역할 | 방식 | 모델 | 정답률 | 맞음/전체 |",
         "|---|---|---|---:|---:|"]
    vlabel = {"base": "베이스", "rag": "용어 RAG", "ft": "파인튜닝"}
    for role in (ROLE_KOREAN, ROLE_GLOBAL):
        for variant in ("base", "rag", "ft"):
            for r in grp(role, variant):
                L.append(f"| {role} | {vlabel.get(variant, variant)} | `{r['model']}` "
                         f"| {pct(r['accuracy'])} | {r['correct']}/{r['n']} |")
    L.append("")

    L.append("## ① 로컬·글로벌 평균 (학습 전)\n")
    L.append(f"- 한국형(로컬) 평균: **{pct(korean_base)}**")
    L.append(f"- 글로벌(베이스) 평균: **{pct(global_base)}**")
    L.append(f"- **두 모델 평균: {pct(pair_avg)}**\n")

    L.append("## ② 글로벌 용어 주입/학습 후\n")
    shown = False
    for label, val in [("용어 RAG", global_rag), ("파인튜닝", global_ft)]:
        if val is None:
            continue
        shown = True
        imp = (val - global_base) if global_base is not None else None
        pair_after = mean([v for v in (korean_base, val) if v is not None]) \
            if korean_base is not None else None
        L.append(f"- 글로벌({label}) 평균: **{pct(val)}**  "
                 f"(향상 Δ {signed(imp)}, 한국형·글로벌 평균 {pct(pair_after)})")
    if not shown:
        L.append("- (아직 글로벌 향상 결과 없음 — RAG/파인튜닝 평가 후 재실행)")
    L.append("")

    # 과목별 — 모델별로 한 열씩(역할/방식 표기)
    L.append("## 과목별 정답률 (참고)\n")
    cols = [r for role in (ROLE_KOREAN, ROLE_GLOBAL) for variant in ("base", "rag", "ft")
            for r in grp(role, variant)]
    subjects = sorted({s for r in cols for s in r.get("by_subject", {})})
    if subjects and cols:
        def head(r):
            return f"{r['role']}/{vlabel.get(r['variant'], r['variant'])}"
        L.append("| 과목 | " + " | ".join(head(r) for r in cols) + " |")
        L.append("|---|" + "---:|" * len(cols))
        for s in subjects:
            cells = [pct(r.get("by_subject", {}).get(s)) for r in cols]
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        L.append("")

    return "\n".join(L)


def main() -> None:
    results, n = load_results()
    if not results:
        raise SystemExit(f"{RESULTS_DIR} 에 결과가 없습니다. 먼저 evaluate.py 실행.")
    report = build_report(results, n)
    out = RESULTS_DIR / "report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n리포트 저장: {out}")


if __name__ == "__main__":
    main()
