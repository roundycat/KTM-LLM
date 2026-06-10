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


VLABEL_ATOM = {"base": "베이스", "rag": "용어 RAG", "ft": "파인튜닝", "cot": "CoT"}


def vlabel(variant: str) -> str:
    """variant 문자열을 사람이 읽는 라벨로. 예: 'cot+sc5' → 'CoT+자기일관성×5'."""
    parts = []
    for a in variant.split("+"):
        if a in VLABEL_ATOM:
            parts.append(VLABEL_ATOM[a])
        elif a.startswith("sc"):
            parts.append(f"자기일관성×{a[2:]}")
        else:
            parts.append(a)
    return "+".join(parts)


def build_report(results: list[dict], n: int) -> str:
    roles = [ROLE_KOREAN, ROLE_GLOBAL] + sorted(
        {r.get("role") for r in results} - {ROLE_KOREAN, ROLE_GLOBAL})

    def of(role):
        return [r for r in results if r.get("role") == role]

    def base_of(role):
        b = mean([r["accuracy"] for r in of(role) if r.get("variant") == "base"])
        return b if b is not None else mean([r["accuracy"] for r in of(role)
                                             if r.get("variant") == "ft"])

    # 표시 순서: base 먼저, 그 외 variant는 정답률 순
    def ordered(role):
        rs = of(role)
        return sorted(rs, key=lambda r: (r.get("variant") != "base", -r["accuracy"]))

    L = ["# 한의학 시험 — 로컬(한국형) vs 글로벌 · 방식별 향상 비교\n",
         f"평가 문항 수: **{n}** (그림 문항 제외)\n",
         "## 모델별 정답률\n",
         "| 역할 | 방식 | 모델 | 정답률 | 맞음/전체 |",
         "|---|---|---|---:|---:|"]
    for role in roles:
        for r in ordered(role):
            L.append(f"| {role} | {vlabel(r['variant'])} | `{r['model']}` "
                     f"| {pct(r['accuracy'])} | {r['correct']}/{r['n']} |")
    L.append("")

    kb, gb = base_of(ROLE_KOREAN), base_of(ROLE_GLOBAL)
    pair = mean([v for v in (kb, gb) if v is not None]) \
        if (kb is not None and gb is not None) else None
    L.append("## ① 로컬·글로벌 평균 (베이스)\n")
    L.append(f"- 한국형(로컬) 평균: **{pct(kb)}**")
    L.append(f"- 글로벌(범용) 평균: **{pct(gb)}**")
    L.append(f"- **두 모델 평균: {pct(pair)}**\n")

    L.append("## ② 방식별 향상 (base 대비 Δ)\n")
    any_var = False
    for role in roles:
        base = base_of(role)
        extras = [r for r in ordered(role) if r.get("variant") != "base"]
        if not extras:
            continue
        any_var = True
        L.append(f"**[{role}]** base {pct(base)}")
        for r in extras:
            imp = (r["accuracy"] - base) if base is not None else None
            L.append(f"- {vlabel(r['variant'])}: **{pct(r['accuracy'])}** (Δ {signed(imp)})")
        L.append("")
    if not any_var:
        L.append("- (base 외 방식 결과 없음 — --cot/--sc/--rag 로 평가)\n")

    # 과목별 — 모델·방식별로 한 열씩
    L.append("## 과목별 정답률 (참고)\n")
    cols = [r for role in roles for r in ordered(role)]
    subjects = sorted({s for r in cols for s in r.get("by_subject", {})})
    if subjects and cols:
        L.append("| 과목 | " + " | ".join(f"{r['role']}/{vlabel(r['variant'])}" for r in cols) + " |")
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
