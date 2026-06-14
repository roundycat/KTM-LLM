# -*- coding: utf-8 -*-
"""bigse0u1(최신) GraphRAG 방식 그대로 복제 — 6 로컬모델 평가.
 step3(extract_seeds_vector + graph_retrieve + vector_retrieve)·step4를 동일 로직으로 재현.
 BGE-m3 임베딩·검색은 그대로, 그래프 쿼리만 Python으로(동일 결과), Neo4j/Chroma 불요.
사용: python bigse_replicate.py [--rx-only] [--k 20]
"""
import os, json, re, argparse, requests, time
import numpy as np
from sentence_transformers import SentenceTransformer

SRC = r"D:\tmp\bigse_latest"           # 최신 bigse0u1 데이터/문항
DATA = os.path.join(SRC, "data")
EMB_MODEL = "BAAI/bge-m3"
EMBCACHE = r"D:\tmp\bige_emb"
os.makedirs(EMBCACHE, exist_ok=True)
MODELS = ["qwen2.5:7b", "gemma2:9b", "exaone3.5:7.8b", "llama3.1:8b", "mistral:7b", "solar:10.7b"]
RX = re.compile(r'(치방|처방)은')

BASE_PROMPT = "다음은 한의사 국가고시 5지선다 문제입니다. 정답 번호 하나만 숫자로 답하세요.\n\n문제: {q}\n보기:\n{opts}\n\n정답 번호(1-5)만 출력:"
RAG_PROMPT = "아래 '근거'를 참고하여 한의사 국가고시 5지선다 문제의 정답을 고르세요.\n근거가 문제와 무관하면 무시하고 네 지식으로 답하라.\n정답 번호 하나만 숫자로 답하세요.\n\n[근거]\n{ctx}\n\n문제: {q}\n보기:\n{opts}\n\n정답 번호(1-5)만 출력:"

def L(p): return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
def fmt(o): return "\n".join(f"{i+1}. {x}" for i, x in enumerate(o))
def parse_choice(t):
    m = re.search(r'[1-5]', t or ""); return int(m.group()) if m else -1

print("BGE-m3 로딩…", flush=True)
EMB = SentenceTransformer(EMB_MODEL)
def embed(texts, tag):
    cache = os.path.join(EMBCACHE, tag + ".npy")
    if os.path.exists(cache): return np.load(cache)
    v = EMB.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True)
    v = np.asarray(v, dtype=np.float32); np.save(cache, v); return v

# ---- KG 로드 ----
nodes = L(os.path.join(DATA, "kg_all_nodes.jsonl"))
edges = L(os.path.join(DATA, "kg_all_edges.jsonl"))
node = {n["id"]: n for n in nodes}
# 증상/변증 노드 (seed 추출용)
sym = [n for n in nodes if n.get("type") in ("증상", "변증") and len(n.get("name_ko", "")) >= 2]
sym_text = [n["name_ko"] + (" " + n.get("일반인설명", "") if n.get("일반인설명") else "") for n in sym]
sym_emb = embed(sym_text, "sym")
# 처방 -주치-> 증상,  처방 -구성-> 약재
rx_juchi = {}   # 처방id -> set(증상id)
rx_gu = {}      # 처방id -> [약재명]
for e in edges:
    if e["type"] == "주치":
        rx_juchi.setdefault(e["src"], set()).add(e["dst"])
    elif e["type"] == "구성":
        rx_gu.setdefault(e["src"], []).append(node.get(e["dst"], {}).get("name_ko", ""))

def extract_seeds_vector(question, top_k=10, threshold=0.25):
    q = EMB.encode([question], normalize_embeddings=True)
    sims = (sym_emb @ q.T).squeeze()
    out = []
    for i in sims.argsort()[::-1][:top_k]:
        if float(sims[i]) < threshold: break
        out.append(sym[i]["id"])
    return out

def graph_retrieve(seeds):
    if not seeds: return []
    sset = set(seeds)
    scored = []
    for rxid, juset in rx_juchi.items():
        inter = juset & sset
        if inter: scored.append((len(inter), rxid, inter))
    scored.sort(key=lambda t: -t[0])
    rows = []
    for score, rxid, inter in scored[:6]:
        p = node.get(rxid, {})
        rows.append({"처방": p.get("name_ko", ""), "한자": p.get("name_hanja", ""),
                     "계통": p.get("계통", ""),
                     "matched": [node.get(s, {}).get("name_ko", "") for s in inter],
                     "약재": [a for a in rx_gu.get(rxid, []) if a][:8]})
    return rows

# ---- 벡터 청크(원본/임상) ----
def load_chunks(fname):
    rows = L(os.path.join(DATA, fname)); return [r["id"] for r in rows], [r["text"] for r in rows]
rx_ids, rx_docs = load_chunks("처방_rag_chunks.jsonl")
cl_ids, cl_docs = load_chunks("처방_rag_chunks_clinical.jsonl")
rx_emb = embed(rx_docs, "rx"); cl_emb = embed(cl_docs, "clinical")

def vector_retrieve(question, k, clinical):
    ids, docs, emb = (cl_ids, cl_docs, cl_emb) if clinical else (rx_ids, rx_docs, rx_emb)
    q = EMB.encode([question], normalize_embeddings=True)
    sims = (emb @ q.T).squeeze()
    top = sims.argsort()[::-1][:k]
    return [(ids[i], docs[i]) for i in top]

def build_context(graph_rows, chunks):
    lines = []
    if graph_rows:
        lines.append("[그래프 근거: 증상에 부합하는 처방]")
        for g in graph_rows:
            lines.append(f"- {g['처방']}({g['한자']}) [{g['계통']}내과] | 부합 증상: {', '.join(g['matched'])} | 구성: {', '.join(g['약재'][:8])}")
    if chunks:
        lines.append("\n[본문 근거]")
        for cid, doc in chunks:
            lines.append(f"- {doc}")
    return "\n".join(lines)

def call_llm(model, prompt):
    for _ in range(3):
        try:
            r = requests.post("http://localhost:11434/api/generate",
                              json={"model": model, "prompt": prompt, "stream": False,
                                    "options": {"temperature": 0, "num_predict": 1536}}, timeout=180)
            return r.json().get("response", "")
        except Exception:
            time.sleep(2)
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rx-only", action="store_true"); ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", default=r"D:\tmp\bige_replicate_result.json"); a = ap.parse_args()
    Q = L(os.path.join(SRC, "eval", "한의학_문제.jsonl"))
    src = {re.sub(r"\s+", "", r["question"])[:80]: r for r in L(os.path.join(SRC, "eval", "한의학_문제_원본.jsonl"))}
    if a.rx_only: Q = [q for q in Q if RX.search(q["question"])]
    print(f"문항 {len(Q)} | k={a.k} | 모델 {len(MODELS)}", flush=True)
    allres = {}
    for model in MODELS:
        print(f"\n===== {model} =====", flush=True)
        st = {"base": 0, "vec": 0, "rag": 0, "n": 0}; subj = {}
        for i, q in enumerate(Q, 1):
            opts = fmt(q["options"]); gold = q["answer"]
            is_rx = bool(RX.search(q["question"]))
            meta = src.get(re.sub(r"\s+", "", q["question"])[:80], {"과목": "미상"})
            sb = meta.get("과목", "미상")
            b = parse_choice(call_llm(model, BASE_PROMPT.format(q=q["question"], opts=opts)))
            v = parse_choice(call_llm(model, RAG_PROMPT.format(ctx=build_context([], vector_retrieve(q["question"], a.k, True)), q=q["question"], opts=opts)))
            seeds = extract_seeds_vector(q["question"]); g = graph_retrieve(seeds)
            clinical = is_rx
            ctx = build_context(g, vector_retrieve(q["question"], a.k, clinical))
            r = parse_choice(call_llm(model, RAG_PROMPT.format(ctx=ctx, q=q["question"], opts=opts)))
            st["n"] += 1; st["base"] += (b == gold); st["vec"] += (v == gold); st["rag"] += (r == gold)
            d = subj.setdefault(sb, {"base": 0, "vec": 0, "rag": 0, "n": 0})
            d["n"] += 1; d["base"] += (b == gold); d["vec"] += (v == gold); d["rag"] += (r == gold)
            if i % 20 == 0: print(f"  {i}/{len(Q)}", flush=True)
        p = lambda x: f"{100*x/st['n']:.1f}%"
        print(f"== {model}: 그냥 {p(st['base'])} | 벡터RAG {p(st['vec'])} | GraphRAG {p(st['rag'])}", flush=True)
        allres[model] = {"overall": st, "subject": subj}
        json.dump(allres, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\n저장: {a.out}")
    print(f"\n{'모델':16}{'그냥':>9}{'벡터RAG':>9}{'GraphRAG':>10}")
    for m, r in allres.items():
        s = r["overall"]; print(f"{m:16}{100*s['base']/s['n']:8.1f}%{100*s['vec']/s['n']:8.1f}%{100*s['rag']/s['n']:9.1f}%")

if __name__ == "__main__":
    main()
