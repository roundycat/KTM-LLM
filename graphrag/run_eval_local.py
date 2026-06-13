# -*- coding: utf-8 -*-
"""로컬(Ollama) GraphRAG 평가 — 레이트리밋/에이전트 없이 확실히 완주.
 모드 3종을 한 번에: 그냥LLM / 그래프RAG / 그래프+해설RAG. 과목별·전체 정답률 + Wilson CI.
사용: python run_eval_local.py --model qwen2.5:7b
"""
import json, re, math, argparse, requests, sys, time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

OLLAMA = "http://localhost:11434/v1/chat/completions"
P = r"D:\tmp\persubj"
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
GOLD = json.load(open(P + r"\gold.json", encoding="utf-8"))
SUBJ = json.load(open(P + r"\subjects.json", encoding="utf-8"))
GRAPH = {r["idx"]: r for r in load(P + r"\eval_input.jsonl")}       # evidence=그래프
HAE   = {r["idx"]: r for r in load(P + r"\eval_input_hae.jsonl")}   # evidence=그래프+해설
N = len(GOLD)

def fmt(o): return "\n".join(f"{i+1}. {x}" for i, x in enumerate(o))
PLAIN = "다음 한의사·한약사 국가고시 5지선다 문제의 정답을 고르세요. 정답 번호 하나만 숫자로 답하세요.\n\n문제: {q}\n보기:\n{opts}\n\n정답 번호(1-5):"
RAG = ("아래 '근거'(지식그래프: 보기 처방의 구성·주치·계통, 증상부합 처방, 유사문항 풀이)를 참고해 "
       "한의사·한약사 국가고시 5지선다 정답을 고르세요. 근거가 무관하면 무시하고 네 한의학 지식으로 답하라. "
       "정답 번호 하나만 숫자로 답하세요.\n\n[근거]\n{ctx}\n\n문제: {q}\n보기:\n{opts}\n\n정답 번호(1-5):")

def parse(t):
    t = t or ""
    m = re.search(r"정답[^0-9]{0,4}([1-5])", t) or re.search(r"([1-5])", t)
    return int(m.group(1)) if m else -1

def ask(model, prompt):
    for _ in range(3):
        try:
            r = requests.post(OLLAMA, json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                            "temperature": 0, "max_tokens": 1536}, timeout=180)
            return parse(r.json()["choices"][0]["message"]["content"])
        except Exception:
            time.sleep(2)
    return -1

def run_mode(model, mode, idxs):
    def one(i):
        q = GRAPH[i]
        if mode == "plain":
            pr = PLAIN.format(q=q["question"], opts=fmt(q["options"]))
        else:
            ctx = (HAE if mode == "hae" else GRAPH)[i]["evidence"] or "(관련 근거 없음)"
            pr = RAG.format(ctx=ctx[:3500], q=q["question"], opts=fmt(q["options"]))
        return i, ask(model, pr)
    pred = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for n, (i, p) in enumerate(ex.map(one, idxs), 1):
            pred[i] = p
            if n % 20 == 0:
                print(f"  [{mode}] {n}/{len(idxs)}", flush=True)
    return pred

def wilson(c, n):
    if not n: return (0, 0)
    z = 1.96; p = c/n; d = 1+z*z/n
    return ((p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d*100,
            (p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d*100)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--rx-only", action="store_true", help="처방형(치방/처방은?)만")
    ap.add_argument("--modes", default="plain,graph,hae")
    ap.add_argument("--out", default=P + r"\local_eval_result.json"); a = ap.parse_args()
    RX = re.compile(r"(치방|처방)은\?")
    IDXS = [i for i in range(N) if (not a.rx_only or RX.search(GRAPH[i]["question"]))]
    modes = [m for m in a.modes.split(",") if m]
    print(f"모델 {a.model} | {'처방형' if a.rx_only else '전체'} {len(IDXS)}문항 | 모드 {','.join(modes)}", flush=True)
    res = {}
    for mode in modes:
        t0 = time.time(); res[mode] = run_mode(a.model, mode, IDXS)
        c = sum(1 for i in IDXS if res[mode][i] == GOLD[str(i)])
        lo, hi = wilson(c, len(IDXS))
        print(f"== {mode}: {c}/{len(IDXS)} = {100*c/len(IDXS):.2f}% (CI {lo:.1f}-{hi:.1f}) [{int(time.time()-t0)}s]", flush=True)
    json.dump({"model": a.model, "rx_only": a.rx_only, "idxs": IDXS,
               "pred": {m: {str(k): v for k, v in res[m].items()} for m in res}},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    # 과목별 비교표
    hdr = "".join(f"{m:>9}" for m in modes)
    print(f"\n{'과목':16}{'n':>4}{hdr}")
    for s, sidx in sorted(SUBJ.items(), key=lambda kv: -len(kv[1])):
        idxs = [i for i in sidx if i in IDXS]
        if not idxs: continue
        row = "".join(f"{100*sum(1 for i in idxs if res[m][i]==GOLD[str(i)])/len(idxs):8.1f}%" for m in modes)
        print(f"{s:16}{len(idxs):>4}{row}")
    print(f"\n저장: {a.out}", flush=True)

if __name__ == "__main__":
    main()
