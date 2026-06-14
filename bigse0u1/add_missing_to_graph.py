"""
add_missing_to_graph.py — add_missing_rx.py로 추가한 24개 처방을
kg_all_nodes.jsonl / kg_all_edges.jsonl에도 추가.

실행:
  python add_missing_to_graph.py --dry-run
  python add_missing_to_graph.py
  → 이후 python step1_load_neo4j.py 로 Neo4j에 반영
"""
import json, argparse

NODES_FILE  = "data/kg_all_nodes.jsonl"
EDGES_FILE  = "data/kg_all_edges.jsonl"
CHUNKS_FILE = "data/처방_rag_chunks.jsonl"


def load_lookup(nodes_file):
    herb_idx = {}   # name_ko → id
    sym_idx  = {}   # name_ko → id
    existing = set()
    for l in open(nodes_file, encoding="utf-8"):
        n = json.loads(l)
        existing.add(n["id"])
        if n.get("type") == "약재":
            herb_idx[n["name_ko"]] = n["id"]
        elif n.get("type") in ("증상", "변증"):
            sym_idx[n["name_ko"]] = n["id"]
    return herb_idx, sym_idx, existing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    herb_idx, sym_idx, existing_ids = load_lookup(NODES_FILE)
    print(f"기존 노드: {len(existing_ids)}개  약재: {len(herb_idx)}개  증상: {len(sym_idx)}개")

    new_nodes, new_edges = [], []

    for l in open(CHUNKS_FILE, encoding="utf-8"):
        c = json.loads(l)
        if not c["id"].startswith("MISS_"):
            continue
        if c["id"] in existing_ids:
            continue

        m = c["metadata"]
        rx_id   = c["id"]
        name_ko = m.get("처방명", "")
        hanja   = m.get("처방한자", "")
        system  = m.get("계통", "")
        source  = m.get("출전", "")

        # 처방 노드
        new_nodes.append({
            "id": rx_id, "type": "처방",
            "name_ko": name_ko, "name_hanja": hanja,
            "출전": source, "계통": system, "페이지": "",
        })

        # 계통 엣지
        if system:
            new_edges.append({"src": rx_id, "dst": f"SYS-{system}", "type": "계통"})

        # 구성 엣지 (약재 노드가 있는 것만)
        matched_herbs = 0
        for herb in m.get("구성약재", []):
            if herb in herb_idx:
                new_edges.append({"src": rx_id, "dst": herb_idx[herb], "type": "구성",
                                  "용량": "", "단위": "", "수치법": ""})
                matched_herbs += 1

        # 주치 엣지 (증상 노드가 있는 것만)
        matched_sym = 0
        for sym in m.get("주치증상", []):
            if sym in sym_idx:
                new_edges.append({"src": rx_id, "dst": sym_idx[sym], "type": "주치"})
                matched_sym += 1

        print(f"  {name_ko}({hanja}) [{system}]"
              f"  약재 {matched_herbs}/{len(m.get('구성약재',[]))}매칭"
              f"  주치 {matched_sym}/{len(m.get('주치증상',[]))}매칭")

    print(f"\n추가할 노드: {len(new_nodes)}개  엣지: {len(new_edges)}개")

    if args.dry_run:
        print("--dry-run: 저장 안 함")
        return

    with open(NODES_FILE, "a", encoding="utf-8") as f:
        for n in new_nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")

    with open(EDGES_FILE, "a", encoding="utf-8") as f:
        for e in new_edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n✅ {NODES_FILE}  (+{len(new_nodes)}개)")
    print(f"✅ {EDGES_FILE}  (+{len(new_edges)}개)")
    print("\n다음 단계: python step1_load_neo4j.py")


if __name__ == "__main__":
    main()
