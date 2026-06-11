# 한의학 그래프 RAG — 실행 키트

질문을 넣으면 그래프(Neo4j)와 본문(Chroma)을 검색해 LLM이 근거와 함께
답하는 최소 작동 예제입니다. 모두 **로컬·무료**로 시작할 수 있어요.
(유료가 될 수 있는 건 답변 LLM에 외부 API를 쓸 때뿐입니다.)

## 0. 준비
- Python 3.10+ 설치
- 같은 폴더에 데이터 파일을 둡니다:
  `kg_all_nodes.jsonl`, `kg_all_edges.jsonl`,
  `처방_rag_chunks.jsonl`, `한의학용어_rag_chunks.jsonl`,
  그리고 채점용 `한의학_문제.jsonl`
- 패키지 설치:  `pip install -r requirements.txt`

## 1. Neo4j 설치 & 그래프 적재  (무료·로컬)
1) Neo4j Desktop을 공식 사이트에서 받아 설치 → 새 DBMS 생성 → Start.
   (비밀번호를 정하고 기억해두세요.)
2) 접속정보를 환경변수로 넣고 적재:
```
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PW="정한_비밀번호"
python step1_load_neo4j.py
```

## 2. 벡터DB(Chroma) 구축  (무료·로컬, 임베딩=BGE-m3)
```
python step2_build_vectordb.py
```
처음 실행 시 임베딩 모델(BGE-m3)을 자동 내려받습니다(수 GB, 1회).

## 3. 질문해보기 (그래프 RAG)
답변 LLM을 고릅니다. 둘 중 하나:
- (A) 완전 무료·로컬: Ollama 설치 후 `ollama pull qwen2.5` →
  `export LLM_PROVIDER=ollama LLM_MODEL=qwen2.5`
- (B) 유료 API(품질 ↑): `export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=...`
  (또는 `LLM_PROVIDER=openai OPENAI_API_KEY=...`)
```
python step3_graphrag_query.py "가슴이 두근거리고 쉽게 피로해요"
```

## 4. 문제집으로 채점 (그냥 LLM vs 그래프 RAG)
```
python step4_eval.py --n 50
```
517문제 중 n개를 뽑아 두 방식의 정답률을 비교합니다.

## 비용 요약
- Neo4j(로컬), Chroma, BGE-m3 임베딩 → **0원**
- 답변 LLM → Ollama면 0원 / 외부 API면 호출당 소액

## 주의
- 이 키트는 의료 의사결정 보조·교육용입니다. 답변에는 근거 출처가 붙지만,
  최종 판단은 면허 한의사가 해야 합니다.
- `팔강귀속` 등 자동 분류 관계는 검수가 필요합니다.
