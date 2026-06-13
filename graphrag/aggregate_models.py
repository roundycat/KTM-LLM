# -*- coding: utf-8 -*-
"""eval_out/*.json(모델별 GraphRAG 결과)를 모아 다중모델 비교 리포트(MD) 생성."""
import json, os, glob, math
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval_out")
P = r"D:\tmp\persubj"
GOLD = json.load(open(P + r"\gold.json", encoding="utf-8"))
SUBJ = json.load(open(P + r"\subjects.json", encoding="utf-8"))
N = len(GOLD)
MODES = ["plain", "graph", "hae"]
MODE_KO = {"plain": "그냥LLM", "graph": "그래프RAG", "hae": "그래프+해설"}

def wilson(c, n):
    if not n: return (0, 0)
    z = 1.96; p = c/n; d = 1+z*z/n
    return ((p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d*100,
            (p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d*100)

results = {}
for f in sorted(glob.glob(os.path.join(OUT, "grageval_*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    results[d["model"]] = {m: {int(k): v for k, v in d["pred"].get(m, {}).items()} for m in MODES}

def acc(pred, idxs):
    return sum(1 for i in idxs if pred.get(i) == GOLD[str(i)])

L = ["# 다중 모델 GraphRAG 평가 — 전과목 517문항\n",
     "로컬 모델별로 **그냥LLM / 그래프RAG / 그래프+해설RAG** 3방식 정답률(누수 없는 해설 LOO). 채점: 예측==정답, Wilson 95% CI.\n",
     f"> 원본 bigse0u1(gpt-4o-mini): 그냥 59.6% / 벡터 59.6% / 그래프 59.2% / 보기Graph 60.0%. 아래는 본 작업의 로컬모델 재현·확장.\n"]
L.append("## 1. 전체 정답률 (모델 × 방식)\n")
L.append("| 모델 | 그냥LLM | 그래프RAG | 그래프+해설 | 최고 Δ(해설-그냥) |")
L.append("|---|---:|---:|---:|---:|")
for mdl, res in sorted(results.items(), key=lambda kv: -acc(kv[1].get("hae", {}), range(N))):
    cells = []
    for m in MODES:
        c = acc(res.get(m, {}), range(N)); cells.append(f"{100*c/N:.1f}%")
    dp = (acc(res.get("hae", {}), range(N)) - acc(res.get("plain", {}), range(N))) / N * 100
    L.append(f"| {mdl} | {cells[0]} | {cells[1]} | **{cells[2]}** | {dp:+.1f}%p |")
L.append("")
# 과목별 (각 모델 최고 방식 = hae 기준)
L.append("## 2. 과목별 정답률 (그래프+해설RAG 기준)\n")
hdr = " | ".join(sorted(results, key=lambda k: -acc(results[k].get("hae", {}), range(N))))
L.append("| 과목 | n | " + hdr + " |")
L.append("|---|---:|" + "---:|" * len(results))
for s, idxs in sorted(SUBJ.items(), key=lambda kv: -len(kv[1])):
    row = f"| {s} | {len(idxs)} |"
    for mdl in sorted(results, key=lambda k: -acc(results[k].get("hae", {}), range(N))):
        c = acc(results[mdl].get("hae", {}), idxs)
        row += f" {100*c/len(idxs):.0f}% |"
    L.append(row)
L.append(f"\n*완료 모델: {', '.join(results.keys())} ({len(results)}/6)*")
open(os.path.join(HERE, "MULTIMODEL_REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("리포트 생성:", os.path.join(HERE, "MULTIMODEL_REPORT.md"), f"| 모델 {len(results)}개")
for mdl, res in results.items():
    print(f"  {mdl}: " + " ".join(f"{m}={100*acc(res.get(m,{}),range(N))/N:.1f}%" for m in MODES))
