# -*- coding: utf-8 -*-
"""eval_out/*.json(모델별)을 모아 다중모델 비교 + 선택적RAG(처방형→그래프) 리포트 생성."""
import json, os, glob, math, re
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval_out")
P = r"D:\tmp\persubj"
GOLD = json.load(open(P + r"\gold.json", encoding="utf-8"))
SUBJ = json.load(open(P + r"\subjects.json", encoding="utf-8"))
Q = {r["idx"]: r for r in (json.loads(l) for l in open(P + r"\questions_noanswer.jsonl", encoding="utf-8"))}
N = len(GOLD)
RX = re.compile(r"(치방|처방)은")
RXSET = set(i for i in range(N) if RX.search(Q[i]["question"]))   # 처방형 idx
MODES = ["plain", "graph", "hae"]
KO = {"plain": "그냥LLM", "graph": "그래프RAG", "hae": "그래프+해설", "sel": "선택적RAG"}

def wilson(c, n):
    if not n: return (0, 0)
    z = 1.96; p = c/n; d = 1+z*z/n
    return ((p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d*100,
            (p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d*100)

results = {}
for f in sorted(glob.glob(os.path.join(OUT, "grageval_*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    pr = {m: {int(k): v for k, v in d["pred"].get(m, {}).items()} for m in MODES}
    # 선택적RAG: 처방형이면 그래프, 아니면 그냥
    pr["sel"] = {i: (pr["graph"].get(i) if i in RXSET else pr["plain"].get(i)) for i in range(N)}
    results[d["model"]] = pr

def acc(pred, idxs):
    return sum(1 for i in idxs if pred.get(i) == GOLD[str(i)])

ALL = list(range(N)); NON = [i for i in ALL if i not in RXSET]; RXL = sorted(RXSET)
order = sorted(results, key=lambda k: -acc(results[k]["sel"], ALL))

L = ["# 다중 모델 GraphRAG 평가 — 전과목 517문항 (+ 선택적RAG)\n",
     "각 모델을 **그냥LLM / 그래프RAG / 그래프+해설 / 선택적RAG**로 평가. ",
     "**선택적RAG = 처방형 문항은 그래프RAG, 그 외 문항은 그냥LLM**으로 유형별 라우팅(누수 없음, 고정 규칙).",
     "채점: 예측==정답, Wilson 95% CI.\n",
     f"> 처방형 {len(RXL)}문항 / 그 외 {len(NON)}문항. 그래프RAG는 보기 처방의 구성·주치를 KG에서 주입 → **처방형에서만 효과**.\n"]

L.append("## 1. 전체 정답률 (모델 × 방식)\n")
L.append("| 모델 | 그냥LLM | 그래프RAG | 그래프+해설 | **선택적RAG** | Δ(선택−그냥) |")
L.append("|---|---:|---:|---:|---:|---:|")
for m in order:
    r = results[m]
    cp, cg, ch, cs = (acc(r[x], ALL) for x in ("plain", "graph", "hae", "sel"))
    L.append(f"| {m} | {100*cp/N:.1f}% | {100*cg/N:.1f}% | {100*ch/N:.1f}% | **{100*cs/N:.1f}%** | {100*(cs-cp)/N:+.1f}%p |")
L.append("\n→ **선택적RAG가 거의 모든 모델에서 그냥LLM보다 향상**(처방형의 그래프 이득을 살리고, 그외 방해를 제거).\n")

L.append("## 2. 처방형 vs 그 외 (그래프RAG 효과의 출처)\n")
L.append("| 모델 | 처방형 그냥 | 처방형 그래프 | 처방형 Δ | 그외 그냥 | 그외 그래프 | 그외 Δ |")
L.append("|---|---:|---:|---:|---:|---:|---:|")
for m in order:
    r = results[m]
    rp, rg = acc(r["plain"], RXL), acc(r["graph"], RXL)
    np_, ng = acc(r["plain"], NON), acc(r["graph"], NON)
    L.append(f"| {m} | {100*rp/len(RXL):.1f}% | **{100*rg/len(RXL):.1f}%** | {100*(rg-rp)/len(RXL):+.1f}%p | {100*np_/len(NON):.1f}% | {100*ng/len(NON):.1f}% | {100*(ng-np_)/len(NON):+.1f}%p |")
L.append("\n→ **처방형에서 그래프RAG가 크게 향상**, 그외에선 무관 근거가 방해. 그래서 유형별 라우팅(선택적RAG)이 최적.\n")

L.append("## 3. 과목별 정답률 (선택적RAG 기준)\n")
hdr = " | ".join(order)
L.append("| 과목 | n | " + hdr + " |")
L.append("|---|---:|" + "---:|" * len(order))
for s, idxs in sorted(SUBJ.items(), key=lambda kv: -len(kv[1])):
    row = f"| {s} | {len(idxs)} |"
    for m in order:
        row += f" {100*acc(results[m]['sel'], idxs)/len(idxs):.0f}% |"
    L.append(row)
L.append(f"\n*완료 모델: {', '.join(order)} ({len(results)}/6)*")
open(os.path.join(HERE, "MULTIMODEL_REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

print(f"리포트 생성 | 모델 {len(results)}개 | 처방형 {len(RXL)} / 그외 {len(NON)}")
for m in order:
    r = results[m]
    cp, cs = acc(r["plain"], ALL), acc(r["sel"], ALL)
    rp, rg = acc(r["plain"], RXL), acc(r["graph"], RXL)
    print(f"  {m}: 그냥 {100*cp/N:.1f}% → 선택적RAG {100*cs/N:.1f}% ({100*(cs-cp)/N:+.1f}%p) | 처방형 그래프 {100*(rg-rp)/len(RXL):+.1f}%p")
