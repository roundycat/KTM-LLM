"""
enrich_주치.py — 처방_rag_chunks의 주치증상을 파싱해
kg_all_edges.jsonl에 누락된 주치 엣지를 추가한다.
실행: python enrich_주치.py [--dry-run]
"""
import json, argparse
from collections import defaultdict

NODES_FILE  = "data/kg_all_nodes.jsonl"
EDGES_FILE  = "data/kg_all_edges.jsonl"
CHUNKS_FILE = "data/처방_rag_chunks.jsonl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="추가하지 않고 통계만 출력")
    args = ap.parse_args()

    # 증상/변증 노드 name_ko → id
    name_to_id = {}
    for l in open(NODES_FILE, encoding="utf-8"):
        n = json.loads(l)
        if n["type"] in ("증상", "변증") and n.get("name_ko"):
            name_to_id[n["name_ko"]] = n["id"]

    # 기존 주치 엣지 (중복 방지)
    existing = set()
    for l in open(EDGES_FILE, encoding="utf-8"):
        e = json.loads(l)
        if e["type"] == "주치":
            existing.add((e["src"], e["dst"]))

    # 처방 청크에서 새 주치 엣지 생성
    new_edges = []
    seen = set(existing)
    per_rx = defaultdict(list)

    for l in open(CHUNKS_FILE, encoding="utf-8"):
        chunk = json.loads(l)
        rx_id = chunk["id"]
        rx_nm = chunk.get("metadata", {}).get("처방명", rx_id)
        for sym in chunk.get("metadata", {}).get("주치증상", []):
            if sym in name_to_id:
                dst = name_to_id[sym]
                if (rx_id, dst) not in seen:
                    edge = {"src": rx_id, "dst": dst, "type": "주치"}
                    new_edges.append(edge)
                    per_rx[rx_nm].append(sym)
                    seen.add((rx_id, dst))

    print(f"기존 주치 엣지: {len(existing):,}개")
    print(f"추가될 주치 엣지: {len(new_edges):,}개")
    print(f"영향 받는 처방 수: {len(per_rx):,}개")

    # 주요 처방 확인
    targets = ["육군자탕", "윤마환", "대시호탕", "온포음", "수비전",
               "단치소요산", "생철락음", "열다한소탕"]
    print("\n[진단 케이스 처방 추가 엣지]")
    for nm in targets:
        syms = per_rx.get(nm, [])
        print(f"  {nm}: {len(syms)}개 추가 → {syms[:5]}")

    if args.dry_run:
        print("\n--dry-run 모드: 실제 파일은 수정하지 않습니다.")
        return

    # kg_all_edges.jsonl에 추가
    with open(EDGES_FILE, "a", encoding="utf-8") as f:
        for e in new_edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n{EDGES_FILE}에 {len(new_edges)}개 엣지 추가 완료.")
    print("이제 step1을 다시 실행해 Neo4j에 반영하세요:")
    print("  python step1_load_neo4j.py")

if __name__ == "__main__":
    main()
