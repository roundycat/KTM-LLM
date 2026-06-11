"""
step2_build_vectordb.py — 처방/용어 청크를 임베딩해 로컬 Chroma에 적재.
임베딩: BAAI/bge-m3 (로컬·무료). 저장 위치: ./chroma_db
"""
import json, os
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_FILES = ["처방_rag_chunks.jsonl", "한의학용어_rag_chunks.jsonl"]
MODEL = "BAAI/bge-m3"
DB_PATH = "./chroma_db"
COLL = "hani"

def load_chunks():
    rows = []
    for fn in CHUNK_FILES:
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
    chunks = load_chunks()
    print(f"청크 {len(chunks)}개 임베딩 시작…")
    model = SentenceTransformer(MODEL)
    texts = [c["text"] for c in chunks]
    embs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True).tolist()
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLL)
    except Exception:
        pass
    col = client.create_collection(COLL, metadata={"hnsw:space": "cosine"})
    B = 2000
    for i in range(0, len(chunks), B):
        sl = slice(i, i+B)
        col.add(
            ids=[c["id"] for c in chunks[sl]],
            embeddings=embs[sl],
            documents=[c["text"] for c in chunks[sl]],
            metadatas=[flatten_meta(c.get("metadata")) for c in chunks[sl]],
        )
    print(f"Chroma 적재 완료 → {DB_PATH} (collection='{COLL}', {len(chunks)}건)")

if __name__ == "__main__":
    main()
