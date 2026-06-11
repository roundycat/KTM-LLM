"""
step3_graphrag_query.py — 그래프 RAG 핵심.
질문 → (1)증상 추출 → (2)Neo4j 경로검색 + (3)Chroma 본문검색 → (4)LLM 답변(근거·출처).

사용:
  export NEO4J_URI=... NEO4J_USER=... NEO4J_PW=...
  # 답변 LLM 선택 (둘 중 하나)
  export LLM_PROVIDER=ollama   LLM_MODEL=qwen2.5
  export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=...   # (LLM_MODEL 기본 claude-sonnet-4-6)
  python step3_graphrag_query.py "가슴이 두근거리고 쉽게 피로해요"
"""
import os, sys, json, functools
from neo4j import GraphDatabase
import chromadb
from sentence_transformers import SentenceTransformer

NEO4J = (os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
         os.environ.get("NEO4J_USER", "neo4j"),
         os.environ.get("NEO4J_PW", "neo4j"))
EMB_MODEL = "BAAI/bge-m3"
DB_PATH, COLL = "./chroma_db", "hani"

# ---------- 공통 리소스 ----------
@functools.lru_cache(maxsize=1)
def emb_model(): return SentenceTransformer(EMB_MODEL)

@functools.lru_cache(maxsize=1)
def chroma():
    return chromadb.PersistentClient(path=DB_PATH).get_collection(COLL)

@functools.lru_cache(maxsize=1)
def driver(): return GraphDatabase.driver(NEO4J[0], auth=(NEO4J[1], NEO4J[2]))

@functools.lru_cache(maxsize=1)
def name_index():
    """질문에서 증상/변증명을 찾기 위한 이름→id 사전."""
    idx = []
    for l in open("kg_all_nodes.jsonl", encoding="utf-8"):
        n = json.loads(l)
        if n["type"] in ("증상", "변증") and len(n.get("name_ko", "")) >= 2:
            idx.append((n["name_ko"], n["id"]))
    return idx

# ---------- (1) 증상/변증 추출 ----------
def extract_seeds(question):
    """질문 텍스트에서 그래프 노드 이름과 직접 매칭 (한의학 시험 문어체에 적합)."""
    seeds, names = [], []
    for nm, nid in name_index():
        if nm in question:
            seeds.append(nid); names.append(nm)
    return seeds, names

EXTRACT_PROMPT = """다음 질문에서 한의학 증상·병증·변증에 해당하는 용어를 추출하세요.
한의학 교과서 표현으로, 쉼표로만 구분해 출력하세요. 없으면 빈 문자열만 출력.
예) 심계항진, 기허, 음허화동

질문: {q}
한의학 용어:"""

def extract_seeds_llm(question):
    """LLM으로 구어체 → 한의학 용어 변환 후 그래프 노드 매칭 (구어체 질문에 적합)."""
    raw = call_llm(EXTRACT_PROMPT.format(q=question)).strip()
    if not raw:
        return [], []
    terms = [t.strip() for t in raw.replace("，", ",").split(",") if t.strip()]
    idx = name_index()
    seeds, names = [], []
    seen = set()
    for term in terms:
        for nm, nid in idx:
            if (term in nm or nm in term) and nid not in seen:
                seeds.append(nid); names.append(nm); seen.add(nid)
    # LLM 추출 실패 시 원본 텍스트 직접 매칭으로 fallback
    if not seeds:
        return extract_seeds(question)
    return seeds, names

# ---------- (2) 그래프 검색 ----------
GRAPH_Q = """
MATCH (p:Node {label:'처방'})-[:REL {type:'주치'}]->(s:Node)
WHERE s.id IN $seeds
WITH p, collect(DISTINCT s.name_ko) AS matched, count(DISTINCT s) AS score
ORDER BY score DESC LIMIT 6
MATCH (p)-[:REL {type:'구성'}]->(h:Node)
RETURN p.name_ko AS 처방, p.name_hanja AS 한자, p.계통 AS 계통,
       matched, collect(h.name_ko) AS 약재, score
"""
def graph_retrieve(seeds):
    if not seeds: return []
    with driver().session() as s:
        return [r.data() for r in s.run(GRAPH_Q, seeds=seeds)]

# ---------- (3) 본문(벡터) 검색 ----------
def vector_retrieve(question, k=5):
    q = emb_model().encode([question], normalize_embeddings=True).tolist()
    res = chroma().query(query_embeddings=q, n_results=k)
    return list(zip(res["ids"][0], res["documents"][0]))

# ---------- (4) LLM 호출 ----------
def call_llm(prompt):
    prov = os.environ.get("LLM_PROVIDER", "ollama")
    if prov == "anthropic":
        import anthropic
        m = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
        r = anthropic.Anthropic().messages.create(
            model=m, max_tokens=900, messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    if prov == "openai":
        from openai import OpenAI
        m = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        r = OpenAI().chat.completions.create(
            model=m, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
    # 기본: Ollama (로컬·무료)
    import requests
    m = os.environ.get("LLM_MODEL", "qwen2.5")
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": m, "prompt": prompt, "stream": False}, timeout=180)
    return r.json().get("response", "")

# ---------- 근거 조립 + 답변 ----------
def build_context(graph_rows, chunks):
    lines = []
    if graph_rows:
        lines.append("[그래프 근거: 증상에 부합하는 처방]")
        for g in graph_rows:
            lines.append(f"- {g['처방']}({g['한자']}) [{g['계통']}내과] "
                         f"| 부합 증상: {', '.join(g['matched'])} "
                         f"| 구성: {', '.join(g['약재'][:8])}")
    if chunks:
        lines.append("\n[본문 근거]")
        for cid, doc in chunks:
            lines.append(f"- {doc}")
    return "\n".join(lines)

PROMPT = """당신은 한의학 지식베이스를 활용하는 보조 도우미입니다.
아래 '근거'에 있는 정보만으로 한국어로 답하세요. 
근거가 문제와 무관하면 무시하고 네 지식으로 답하라
가능한 처방 후보와 그 근거(부합 증상·구성)를 제시하고, 마지막에
"※ 본 답변은 참고용이며 최종 진단·처방은 면허 한의사가 판단해야 합니다."를 덧붙이세요.

[환자 질문]
{q}

[근거]
{ctx}
"""
def answer(question):
    if os.environ.get("USE_LLM_EXTRACT"):
        seeds, names = extract_seeds_llm(question)
    else:
        seeds, names = extract_seeds(question)
    g = graph_retrieve(seeds)
    c = vector_retrieve(question, k=5)
    ctx = build_context(g, c)
    out = call_llm(PROMPT.format(q=question, ctx=ctx))
    return out, names, g

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "가슴이 두근거리고 쉽게 피로해요"
    ans, names, g = answer(q)
    print("● 추출된 증상/변증:", names)
    print("● 그래프 처방 후보:", [r["처방"] for r in g])
    print("\n● 답변:\n" + ans)
