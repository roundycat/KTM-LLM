"""
한의학 지식그래프 → Neo4j 적재 스크립트
사용법:
  pip install neo4j
  # Neo4j(로컬 Desktop 또는 Aura)에서 DB를 띄운 뒤, 접속정보를 환경변수로:
  export NEO4J_URI="bolt://localhost:7687"
  export NEO4J_USER="neo4j"
  export NEO4J_PW="your-password"
  python load_to_neo4j.py  # kg_all_nodes.jsonl, kg_all_edges.jsonl 와 같은 폴더에서 실행
"""
import json, os
from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PW = os.environ.get("NEO4J_PW", "neo4j")

def read(fn):
    with open(fn, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def main():
    nodes = read("kg_all_nodes.jsonl")
    edges = read("kg_all_edges.jsonl")
    driver = GraphDatabase.driver(URI, auth=(USER, PW))
    with driver.session() as s:
        # 제약(인덱스) — id 유일성
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
        # 노드: type을 라벨로, 나머지는 속성으로 (배치 처리)
        for i in range(0, len(nodes), 1000):
            batch = nodes[i:i+1000]
            s.run(
                """
                UNWIND $rows AS row
                MERGE (n:Node {id: row.id})
                SET n += row, n.label = row.type
                """, rows=batch)
        # 엣지: type을 관계 타입처럼 property로 저장(동적 타입 대신 단일 REL+type)
        for i in range(0, len(edges), 1000):
            batch = edges[i:i+1000]
            s.run(
                """
                UNWIND $rows AS row
                MATCH (a:Node {id: row.src}), (b:Node {id: row.dst})
                MERGE (a)-[r:REL {type: row.type}]->(b)
                SET r += row
                """, rows=batch)
    driver.close()
    print(f"적재 완료: 노드 {len(nodes)} / 엣지 {len(edges)}")

if __name__ == "__main__":
    main()
