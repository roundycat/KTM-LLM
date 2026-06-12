"""
step1_load_neo4j.py — 통합 그래프(kg_all_*.jsonl)를 Neo4j에 적재.
환경변수: NEO4J_URI, NEO4J_USER, NEO4J_PW
"""
import json, os
from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PW = os.environ.get("NEO4J_PW", "neo4j")

def read(fn):
    with open(fn, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def scalarize(d):
    # Neo4j 속성은 스칼라/스칼라배열만 허용 → 리스트는 문자열로
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = ", ".join(map(str, v)) if v else ""
        else:
            out[k] = v
    return out

def main():
    nodes = [scalarize(n) for n in read("data/kg_all_nodes.jsonl")]
    edges = [scalarize(e) for e in read("data/kg_all_edges.jsonl")]
    driver = GraphDatabase.driver(URI, auth=(USER, PW))
    with driver.session() as s:
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
        for i in range(0, len(nodes), 1000):
            s.run("""
                UNWIND $rows AS row
                MERGE (n:Node {id: row.id})
                SET n += row, n.label = row.type
            """, rows=nodes[i:i+1000])
        for i in range(0, len(edges), 1000):
            s.run("""
                UNWIND $rows AS row
                MATCH (a:Node {id: row.src}), (b:Node {id: row.dst})
                MERGE (a)-[r:REL {type: row.type}]->(b)
                SET r += row
            """, rows=edges[i:i+1000])
    driver.close()
    print(f"적재 완료: 노드 {len(nodes)} / 엣지 {len(edges)}")

if __name__ == "__main__":
    main()
