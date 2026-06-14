"""
step2_build_vectordb.py — 처방/용어 청크를 임베딩해 로컬 Chroma에 적재.
임베딩: BAAI/bge-m3 (로컬·무료). 저장 위치: ./chroma_db
컬렉션: hani_rx (처방), hani_term (용어) — 분리 적재
"""
import json, os
import chromadb
from sentence_transformers import SentenceTransformer

COLLECTIONS = {
    "hani_rx":          ["data/처방_rag_chunks.jsonl"],
    "hani_rx_clinical": ["data/처방_rag_chunks_clinical.jsonl"],
    "hani_term":        ["data/한의학용어_rag_chunks.jsonl"],
}
MODEL   = "BAAI/bge-m3"
DB_PATH = "./chroma_db"

def load_chunks(files):
    rows = []
    for fn in files:
        if not os.path.exists(fn):
            print(f"(건너뜀) 없음: {fn}"); continue
        for l in open(fn, encoding="utf-8"):
            rows.append(json.loads(l))
    return rows

def flatten_meta(m):
    out = {"source_file": ""}
    for k, v in (m or {}).items():
        out[k] = ", ".join(map(str, v)) if isinstance(v, list) else (v if v is not None else "")
    return out

def main():
    model  = SentenceTransformer(MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)

    for coll_name, files in COLLECTIONS.items():
        chunks = load_chunks(files)
        if not chunks:
            continue
        print(f"\n[{coll_name}] {len(chunks)}개 임베딩 중…")
        texts = [c["text"] for c in chunks]
        embs  = model.encode(texts, batch_size=64, show_progress_bar=True,
                             normalize_embeddings=True).tolist()
        try:
            client.delete_collection(coll_name)
        except Exception:
            pass
        col = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})
        B = 2000
        for i in range(0, len(chunks), B):
            sl = slice(i, i + B)
            col.add(
                ids=[c["id"] for c in chunks[sl]],
                embeddings=embs[sl],
                documents=[c["text"] for c in chunks[sl]],
                metadatas=[flatten_meta(c.get("metadata")) for c in chunks[sl]],
            )
        print(f"  → {DB_PATH} (collection='{coll_name}', {len(chunks)}건)")

if __name__ == "__main__":
    main()
