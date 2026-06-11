# 한의학 GraphRAG

환자의 증상 설명을 입력하면 지식 그래프(Neo4j)와 벡터 검색(Chroma)을 함께 활용해
LLM이 처방 후보와 근거를 제시하는 시스템입니다.

---

## 데이터 구성

### 지식 그래프 (`kg_all_nodes.jsonl`, `kg_all_edges.jsonl`)

| 노드 | 설명 | 개수 |
|------|------|------|
| 처방 | 한약 처방 (구성·출전·계통) | 3,094 |
| 약재 | 구성 약재 (귀경·성·미·분류) | 645 |
| 증상 | 처방의 주치 적응증 | 3,873 |
| 변증 | 팔강 8증 (음·양·표·리·한·열·허·실) | 8 |
| 증상지표 | 환자 말투에 가까운 팔강변증 설문 문항 | 54 |
| 장부 | 귀경 대상 장기 | 13 |
| 계통 | 오장 내과 계통 | 5 |

| 엣지 | 의미 | 개수 |
|------|------|------|
| 구성 | 처방 → 약재 | 25,565 |
| 주치 | 처방 → 증상/변증 | 8,509 |
| 지표 | 증상지표 → 변증 | 54 |
| 귀경 | 약재 → 장부 | 972 |
| 계통 | 처방 → 오장 계통 | 3,094 |

**추론 경로:**
```
환자 증상 → 변증 → 적응 변증 → 처방 → 약재 → 장부
```
예: "쉽게 피로해진다" → 허증 → 신허(腎虛) → 가미사육탕 → 숙지황·산약·산수유

### 벡터 검색용 본문
- `처방_rag_chunks.jsonl` — 처방별 자기완결 본문 3,094개
- `한의학용어_rag_chunks.jsonl` — 용어 정의 본문 5,878개

### 평가 데이터
- `한의학_문제.jsonl` — 한의사 국가고시 5지선다 517문제
- `처방_문제.jsonl` — 그 중 처방형 문제 86개만 추출

---

## 작동 방식

질문이 들어오면 다음 4단계로 처리됩니다.

```
질문
 ├─ 1. 증상 추출 (문자열 매칭 or LLM)
 ├─ 2. 그래프 검색 (Neo4j — 증상에 맞는 처방 탐색)
 ├─ 3. 벡터 검색 (Chroma — 유사 본문 5개)
 └─ 4. LLM 답변 (그래프 + 벡터 근거 기반)
```

---

## 설치 및 실행

### 패키지 설치
```bash
pip install -r requirements.txt
```

### Step 1 — 그래프 적재 (Neo4j)
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PW="비밀번호"
python step1_load_neo4j.py
```

### Step 2 — 벡터DB 구축 (Chroma)
```bash
python step2_build_vectordb.py
# 첫 실행 시 BGE-m3 모델 자동 다운로드 (수 GB, 1회)
```

### Step 3 — 질문
```bash
# LLM 선택 (셋 중 하나)
export LLM_PROVIDER=ollama    # 로컬·무료 (ollama pull qwen2.5 필요)
export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-...
export LLM_PROVIDER=openai    OPENAI_API_KEY=sk-...

python step3_graphrag_query.py "가슴이 두근거리고 쉽게 피로해요"
```

### Step 4 — 정답률 비교 평가
```bash
python step4_eval.py --n 50       # 전체 문제 중 50개 샘플
python step4_eval.py --rx-only    # 처방형 문제 86개 전부
```

세 가지 방식을 비교합니다:
- **그냥 LLM** — 검색 없이 LLM만 사용
- **벡터 RAG** — Chroma 벡터 검색만 사용
- **GraphRAG** — 그래프 + 벡터 검색 모두 사용

---

## 비용

| 항목 | 비용 |
|------|------|
| Neo4j (로컬), Chroma, BGE-m3 임베딩 | 무료 |
| 답변 LLM (Ollama) | 무료 |
| 답변 LLM (외부 API) | 호출당 소액 |

---

## 출처
- 표준한의학용어집 (2006)
- 한의대 내과학 교과서 처방 데이터 (2024)
- 팔강변증 한의표준임상진료지침 (2023)
- 한의사 국가고시 문제집

본 시스템은 교육·연구 보조용이며, 최종 진단과 처방은 면허 한의사가 판단해야 합니다.
