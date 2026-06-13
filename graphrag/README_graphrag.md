# 한의학 GraphRAG

환자 증상을 입력하면 지식 그래프(Neo4j)와 벡터 검색(Chroma)을 결합해
LLM이 처방 후보와 근거를 제시하는 시스템입니다.
한의사 국가고시 517문제로 그냥 LLM / 벡터 RAG / GraphRAG / 보기Graph 정답률을 비교 평가합니다.

---

## 파일 구성

```
k_llm/
├── data/
│   ├── kg_all_nodes.jsonl              # 지식 그래프 노드 (7,692개)
│   ├── kg_all_edges.jsonl              # 지식 그래프 엣지 (~40,000개)
│   ├── 처방_rag_chunks.jsonl            # 처방 벡터 청크 원본 (3,094개)
│   ├── 처방_rag_chunks_clinical.jsonl   # 처방 임상 표현 청크 (2,213개)
│   └── 한의학용어_rag_chunks.jsonl       # 한의학 용어 벡터 청크 (9,074개)
├── eval/
│   ├── 한의학_문제.jsonl                 # 국가고시 5지선다 517문제
│   ├── 한의학_문제_원본.jsonl             # 과목·그림 라벨 포함 원본
│   ├── 처방_문제.jsonl                   # 처방형 문제 86개 (별도 추출)
│   ├── diag_plain_ok_rag_fail.jsonl     # 진단 파일 (LLM정답·GraphRAG오답)
│   └── experiment.txt                   # 실험 결과 기록
├── step1_load_neo4j.py         # 그래프 → Neo4j 적재
├── step2_build_vectordb.py     # 청크 → Chroma 임베딩
├── step3_graphrag_query.py     # GraphRAG 쿼리 엔진
├── step4_eval.py               # 정답률 비교 평가
├── enrich_주치.py              # 주치 엣지 보강 (청크 메타 → 그래프)
├── rewrite_chunks.py           # 처방 청크 임상 표현 재작성
└── requirements.txt
```

---

## 데이터 구성

### 지식 그래프 노드 (총 7,692개)

| 타입 | 설명 | 개수 |
|------|------|------|
| 처방 | 한약 처방 (구성·출전·계통·한자명) | 3,094 |
| 증상 | 처방의 주치 적응증·병증 | 3,873 |
| 약재 | 처방 구성 약재 (귀경·성·미·분류) | 645 |
| 증상지표 | 팔강변증 설문 문항 (구어체) | 54 |
| 장부 | 귀경 대상 장기 | 13 |
| 변증 | 팔강 8증 (음·양·표·리·한·열·허·실) | 8 |
| 계통 | 오장 내과 계통 | 5 |

**처방 노드 주요 필드:** `id`, `name_ko`, `name_hanja`, `계통`, `출전`

**약재 노드 주요 필드:** `id`, `name_ko`, `name_hanja`, `성(性)`, `미(味)`, `귀경`, `분류`

**변증 노드 주요 필드:** `id`, `name_ko`, `팔강축`, `정의`

### 지식 그래프 엣지 (총 39,996개)

| 타입 | 의미 | 수 |
|------|------|----|
| 구성 | 처방 → 약재 (처방 구성 약재) | 25,565 |
| 주치 | 처방 → 증상/변증 (적응 증상) | 8,858 |
| 계통 | 처방 → 계통 (오장 분류) | 3,094 |
| 팔강귀속 | 증상 → 변증 (팔강 분류) | 1,447 |
| 귀경 | 약재 → 장부 (약재 작용 부위) | 972 |
| 지표 | 증상지표 → 변증 (설문→변증 매핑) | 54 |
| 포함 | 변증 → 변증 (상위 개념 포함) | 6 |

### 그래프 연결 구조

```
[증상지표] ──지표──▶ [변증]
                        │
[증상] ──팔강귀속──▶ [변증]
  ▲
  │ 주치
[처방] ──구성──▶ [약재] ──귀경──▶ [장부]
  │
  └──계통──▶ [계통]
```

**검색 경로 (GraphRAG):**
```
질문 텍스트 → LLM이 한의학 증상 용어 추출(seeds)
           → Neo4j: 처방 -[주치]→ seeds 역방향 검색
           → 부합 증상 수 기준 처방 후보 상위 6개
           → 각 처방의 구성 약재 함께 조회
```

### 벡터 검색용 청크

| 컬렉션 | 파일 | 청크 수 | 내용 | 용도 |
|--------|------|---------|------|------|
| `hani_rx` | 처방_rag_chunks.jsonl | 3,094 | 처방별 주치·구성·출전 본문 | GraphRAG 벡터 보조 |
| `hani_rx_clinical` | 처방_rag_chunks_clinical.jsonl | 2,213 | LLM이 재작성한 임상 표현 본문 | 벡터 RAG |
| `hani_term` | 한의학용어_rag_chunks.jsonl | 9,074 | 한의학 용어 정의 본문 | 용어 질문 |

- **GraphRAG**는 `hani_rx`(원본)를 사용 — 그래프 근거가 주도하고 벡터가 보조
- **벡터 RAG**는 `hani_rx_clinical`(임상 표현)을 사용 — 시험 문제 문체와 유사도가 높음
- 컬렉션을 분리한 이유: 임상 표현 청크를 GraphRAG에 섞으면 설득력 있는 틀린 근거가 포함돼 정답률이 떨어짐

**처방 청크 메타데이터:** `처방명`, `처방한자`, `계통`, `주치증상[]`, `구성약재[]`, `출전`, `출처`, `페이지`

**용어 청크 메타데이터:** `term`, `hanja`, `category`, `synonyms`, `source`

### 평가 데이터

| 파일 | 설명 | 수 |
|------|------|----|
| 한의학_문제.jsonl | 한의사 국가고시 5지선다 | 517 |
| 한의학_문제_원본.jsonl | 과목·그림 라벨 포함 원본 | 517 |
| 처방_문제.jsonl | 처방 선택형만 추출 | 86 |

---

## 사용 기술

### 그래프 DB — Neo4j

[Neo4j](https://neo4j.com/) Community Edition을 로컬에서 실행합니다.
Bolt 프로토콜(`bolt://localhost:7687`)로 Python `neo4j` 드라이버가 접속합니다.

**Neo4j 스키마:**

```
(:Node {id, label, name_ko, name_hanja, 계통, 출전, ...})
  -[:REL {type: "주치" | "구성" | "귀경" | "계통" | "팔강귀속" | "지표" | "포함"}]->
(:Node {...})
```

모든 노드를 단일 레이블 `:Node`로 저장하고 `label` 속성으로 타입을 구분합니다.

**GraphRAG에서 사용하는 Cypher 쿼리:**

```cypher
MATCH (p:Node {label:'처방'})-[:REL {type:'주치'}]->(s:Node)
WHERE s.id IN $seeds
WITH p, collect(DISTINCT s.name_ko) AS matched, count(DISTINCT s) AS score
ORDER BY score DESC LIMIT 6
MATCH (p)-[:REL {type:'구성'}]->(h:Node)
RETURN p.name_ko AS 처방, p.name_hanja AS 한자, p.계통 AS 계통,
       matched, collect(h.name_ko) AS 약재, score
```

질문에서 추출된 증상 ID(`$seeds`)와 `주치` 엣지로 연결된 처방을 역방향으로 탐색해
부합 증상 수(`score`) 기준 상위 6개 처방과 구성 약재를 한 번에 가져옵니다.

### 임베딩 모델 — BGE-m3

`BAAI/bge-m3`를 사용합니다. Beijing Academy of AI(BAAI)에서 공개한 다국어 임베딩 모델로,
한국어·중국어·영어가 혼재된 한의학 텍스트에 적합합니다.

| 항목 | 내용 |
|------|------|
| 모델 | `BAAI/bge-m3` |
| 벡터 차원 | 1,024 |
| 언어 | 100+ 다국어 (한국어·한자 포함) |
| 유사도 | Cosine similarity |
| 실행 환경 | 로컬 (sentence-transformers, 첫 실행 시 ~2GB 자동 다운로드) |
| 비용 | 무료 |

Chroma 컬렉션 생성 시 `{"hnsw:space": "cosine"}`으로 코사인 공간을 설정해
정규화된 벡터의 내적이 코사인 유사도와 동일하도록 합니다.

### 벡터 DB — Chroma

`chromadb.PersistentClient`로 `./chroma_db/` 디렉토리에 영구 저장합니다.
컬렉션은 `hani_rx`(처방 원본), `hani_rx_clinical`(처방 임상), `hani_term`(용어) 세 개로 분리해
방식에 따라 다른 컬렉션을 조회합니다.

---

## 4단계 파이프라인

### Step 1 — 그래프 적재 (`step1_load_neo4j.py`)

`kg_all_nodes.jsonl`과 `kg_all_edges.jsonl`을 Neo4j에 적재합니다.

- 노드: `:Node {id, label, name_ko, name_hanja, ...}`
- 엣지: `:REL {type}` (src → dst)
- 인덱스: `Node.id`, `Node.label`
- 데이터(특히 `kg_all_edges.jsonl`)를 수정한 뒤에는 반드시 재실행해야 Neo4j에 반영됩니다.

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PW="비밀번호"
python step1_load_neo4j.py
```

### Step 2 — 벡터DB 구축 (`step2_build_vectordb.py`)

처방/용어 청크를 BGE-m3로 임베딩해 Chroma에 저장합니다.

- 임베딩 모델: `BAAI/bge-m3` (첫 실행 시 자동 다운로드 ~2GB)
- 저장 경로: `./chroma_db/`
- 컬렉션: `hani_rx` (3,094건), `hani_rx_clinical` (2,213건), `hani_term` (9,074건)
- 청크 데이터를 수정하면 이 단계를 다시 실행해야 합니다 (수십 분 소요).

```bash
python step2_build_vectordb.py
```

### Step 3 — 질문 답변 (`step3_graphrag_query.py`)

질문 하나를 받아 4단계로 처리합니다.

```
질문 텍스트
  ① extract_seeds_llm — LLM이 임상 표현을 한의학 용어로 번역 후 그래프 노드 매칭
  ② graph_retrieve    — Neo4j Cypher: 처방 -[주치]→ seeds 역방향 탐색
  ③ vector_retrieve   — BGE-m3 임베딩 + Chroma 유사도 상위 k개
  ④ call_llm          — 그래프+벡터 근거를 합쳐 LLM 답변 생성
```

**① seed 추출 방식 — LLM 번역**

시험 문제는 임상 표현("소화가 안 되고 구토")을 쓰지만 그래프 노드는 한의학 진단 용어("비기허약, 위기불화")로 등록되어 있어 직접 문자열 매칭이 불가능합니다. LLM이 먼저 임상 표현을 한의학 용어로 번역한 뒤 그래프 노드와 매칭합니다.

```
질문: "소화가 안 되고 가슴이 답답하며 구토"
  → LLM 추출: "비기허약, 위기불화, 담음"
  → 그래프 매칭: SY-비기허약 → 육군자탕 발견
```

**LLM 선택:**

```bash
# 로컬 (Ollama, 무료)
export LLM_PROVIDER=ollama LLM_MODEL=qwen2.5

# OpenAI
export LLM_PROVIDER=openai OPENAI_API_KEY=sk-...

# Anthropic
export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-...
```

```bash
python step3_graphrag_query.py "가슴이 두근거리고 쉽게 피로해요"
```

**주요 설계 결정:**
- `STOP_NAMES = {"한다"}`: 종결어미 "~한다"가 증상 노드 "汗多(한다)"와 충돌하는 버그 방지
- `temperature=0`: 재현 가능한 결과를 위해 모든 LLM 호출에 고정

### Step 4 — 정답률 비교 평가 (`step4_eval.py`)

4가지 방식을 동시에 실행해 정답률을 비교합니다.

| 방식 | 설명 | 컬렉션 |
|------|------|--------|
| 그냥 LLM | 근거 없이 LLM만 사용 | — |
| 벡터 RAG | Chroma 벡터 검색 근거만 사용 | `hani_rx_clinical` |
| GraphRAG | LLM seed 추출 + 그래프 + 벡터 검색 | `hani_rx` |
| 보기Graph | 보기 처방명 5개를 Neo4j에 직접 조회해 구성·주치·계통 제공 | Neo4j 직접 |

**보기Graph 동작 방식:**
```
보기(1~5번) 처방명 추출 → 합방(" 합 " 기준 분리) → Neo4j 직접 조회
→ 각 처방의 구성약재·주치·계통 LLM에 전달 → 정답 선택
```
보기에 정답이 반드시 포함되어 있다는 점을 활용해 그래프 조회 대상을 처음부터 특정합니다.
Neo4j 매칭이 없으면 plain LLM으로 fallback합니다.

**주요 옵션:**

```bash
python step4_eval.py                       # 전체 517문제
python step4_eval.py --n 50 --seed 42      # 50개 샘플, 시드 고정
python step4_eval.py --rx-only             # 처방형 86문제만
python step4_eval.py --rx-only --k 10      # 벡터 청크 수 10개
python step4_eval.py --skip-figure         # 그림 문제 제외
```

**출력 지표:**
- 과목별 / 처방형 vs 그외 / 그림 유무별 정답률
- `recall@ctx`: 처방형 문제에서 정답 처방명이 검색 근거에 포함된 비율
- `eval/diag_plain_ok_rag_fail.jsonl`: 그냥 LLM은 맞고 GraphRAG는 틀린 문제 진단 파일

---

## 주치 엣지 보강 (`enrich_주치.py`)

처방_rag_chunks의 `주치증상` 메타데이터를 파싱해 그래프에 누락된 주치 엣지를 추가합니다.

```bash
python enrich_주치.py --dry-run   # 통계만 출력 (파일 수정 없음)
python enrich_주치.py             # 실제 추가
```

추가 후 Step 1을 다시 실행해 Neo4j에 반영해야 합니다.

---

## 주요 이슈 및 수정사항

### "한다" 종결어미 오매칭 (SY-0410)

**현상**

증상 노드 중 `汗多(한다)` — 땀이 많은 증상 — 의 `name_ko` 값이 `"한다"`로 등록되어 있어,
한국어 문장의 종결어미 `"~한다"`와 완전 일치하는 오매칭이 발생했다.

예를 들어 `"월 1~2회 발작이 오르고 한다"` 같은 문장에서 `한다`가 seed로 추출되어
汗多와 연결된 엉뚱한 처방들이 그래프 근거로 올라왔다.
거의 모든 질문에 `SY-0410(한다)`가 seed에 포함되어 그래프 검색 결과 전체가 오염되는 심각한 버그였다.

**수정**

`step3_graphrag_query.py`에 `STOP_NAMES` 집합을 추가해 매칭에서 제외했다.

```python
STOP_NAMES = {"한다"}   # SY-0410 汗多: '한다'가 종결어미 '~한다'와 충돌

def extract_seeds(question):
    seeds, names = [], []
    seen = set()
    for nm, nid in name_index():
        if nm in STOP_NAMES:
            continue
        ...
```

---

## 전체 실행 순서

```bash
# 1. 패키지 설치 (최초 1회)
pip install -r requirements.txt

# 2. 그래프 → Neo4j 적재 (데이터 변경 시 재실행)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PW="비밀번호"
python step1_load_neo4j.py

# 3. 벡터DB 구축 (데이터 변경 시 재실행, 수십 분 소요)
python step2_build_vectordb.py

# 4. LLM API 키 설정
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# 5. 질문 테스트
python step3_graphrag_query.py "소화가 안되고 입맛이 없으며 기운이 없다"

# 6. 정답률 평가 (처방형 86문제)
python step4_eval.py --rx-only --k 5 --seed 42
```

---

## 평가 결과

### 전체 517문제

| 방식 | 정답률 |
|------|-------|
| 그냥 LLM | 59.6% |
| 벡터 RAG | 59.6% |
| GraphRAG | 59.2% |
| **보기Graph** | **60.0%** |

**과목별:**

| 과목 | n | 그냥 LLM | 벡터 RAG | GraphRAG | 보기Graph |
|------|---|---------|---------|---------|---------|
| 한의학 기초 | 107 | 73.8% | 74.8% | 76.6% | 72.9% |
| 한약학 응용 | 104 | 66.3% | 66.3% | 63.5% | 67.3% |
| 내과학1 | 66 | 54.5% | 51.5% | 51.5% | 54.5% |
| 침구학 | 33 | 54.5% | 54.5% | 51.5% | 51.5% |
| 내과학2 | 30 | 30.0% | 30.0% | 33.3% | 23.3% |
| 보건의약 관계 법규 | 30 | 50.0% | 56.7% | 43.3% | 50.0% |
| 부인과학 | 29 | 51.7% | 51.7% | 51.7% | 62.1% |
| 예방의학 | 22 | 68.2% | 59.1% | 68.2% | 68.2% |
| 소아과학 | 21 | 71.4% | 61.9% | 61.9% | 71.4% |
| 한방생리학 | 16 | 56.2% | 75.0% | 56.2% | 62.5% |
| 신경정신과학 | 15 | 66.7% | 66.7% | 80.0% | 73.3% |
| 본초학 | 13 | 30.8% | 38.5% | 38.5% | 30.8% |
| 외과학 | 6 | 33.3% | 16.7% | 33.3% | 33.3% |
| 안이비인후과학 | 5 | 40.0% | 40.0% | 40.0% | 40.0% |

**유형별:**

| 유형 | n | 그냥 LLM | 벡터 RAG | GraphRAG | 보기Graph |
|------|---|---------|---------|---------|---------|
| 그외 | 431 | 65.4% | 65.7% | 64.5% | 65.4% |
| 처방형 | 86 | 30.2% | 29.1% | 32.6% | 32.6% |

전체 문제에서는 방식 간 차이가 거의 없어요(59~60%). LLM이 이미 강한 지식을 갖고 있어서 RAG의 추가 효과가 희석되기 때문입니다. 처방형 문제에서만 방식 간 차이가 두드러집니다.

---

### 처방형 문제 86개 기준 (`--rx-only --k 5 --seed 42`)

| 방식 | 정답률 |
|------|-------|
| 그냥 LLM | 30.2% |
| 벡터 RAG | 29.1% |
| GraphRAG | 31.4% |
| **보기Graph** | **34.9%** |

**과목별:**

| 과목 | n | 그냥 LLM | 벡터 RAG | GraphRAG | 보기Graph |
|------|---|---------|---------|---------|---------|
| 내과학1 | 36 | 27.8% | 38.9% | 36.1% | 33.3% |
| 부인과학 | 16 | 31.2% | 25.0% | 31.2% | **50.0%** |
| 내과학2 | 15 | 33.3% | 33.3% | 33.3% | 20.0% |
| 소아과학 | 8 | 25.0% | 12.5% | 25.0% | 37.5% |
| 신경정신과학 | 5 | 40.0% | 20.0% | 20.0% | 40.0% |
| 안이비인후과학 | 3 | 66.7% | 0.0% | 0.0% | 33.3% |
| 외과학 | 2 | 0.0% | 0.0% | 50.0% | 0.0% |
| 한약학 응용 | 1 | 0.0% | 0.0% | 0.0% | 100.0% |

- **벡터 RAG**: 임상 표현 청크(`hani_rx_clinical`) 사용
- **GraphRAG**: LLM seed 추출 + Neo4j 그래프 탐색 + `hani_rx` 원본 청크
- **보기Graph**: 보기 처방명 직접 Neo4j 조회 — 정답이 반드시 보기에 있다는 점 활용

`recall@ctx` (GraphRAG 처방형 정답이 검색 근거에 포함된 비율): 약 4.7%

### 그냥 LLM → 벡터 RAG → GraphRAG 순으로 성능이 올라가지 않는 이유

이론적으로는 정보가 추가될수록 성능이 올라야 하지만, 실제로는 그렇지 않습니다. 세 가지 복합적인 원인이 있습니다.

**① 데이터 부족 (가장 큰 원인)**

recall@ctx가 4.7%입니다. 86문제 중 4개만 검색 근거에 정답 처방이 포함되어 있고, 나머지 82개는 LLM에게 엉뚱한 처방 정보를 제공하고 있습니다. 근거에 정답이 없는 RAG는 효과를 낼 수 없습니다.

**② LLM 기본 지식이 이미 강함**

gpt-4o-mini는 한의학 시험 문제를 학습 데이터로 접했을 가능성이 높습니다. 아무 근거 없이도 30%를 맞추는 것이 그 증거입니다. RAG가 추가 이득을 주려면 LLM이 모르는 정보를 제공해야 하는데, 이미 알고 있는 정보라면 RAG가 의미가 없습니다.

**③ 나쁜 근거 = 방해**

정답이 없는 근거가 들어오면 LLM이 그 정보에 이끌려 오히려 틀리는 경우가 발생합니다. 프롬프트에 "근거가 무관하면 무시하라"고 명시했지만 완벽하게 작동하지 않습니다.

**보기Graph(34.9%)가 유일하게 LLM을 이긴 이유**가 이를 증명합니다. 정답이 반드시 보기 안에 있고 그 정보를 그래프에서 정확히 가져올 때만 RAG가 실질적인 효과를 냅니다. 단, 보기 처방이 Neo4j에 없는 경우(86문제 중 약 60%)는 plain LLM으로 fallback합니다.

---

## 비용

| 항목 | 비용 |
|------|------|
| Neo4j (로컬 Community), Chroma, BGE-m3 임베딩 | 무료 |
| Ollama 로컬 LLM | 무료 |
| OpenAI / Anthropic API | 호출당 소액 (GraphRAG는 질문당 LLM 2회 호출) |

---

## 출처

- 표준한의학용어집 (2006)
- 한의대 내과학 교과서 처방 데이터 (2024)
- 팔강변증 한의표준임상진료지침 (2023)
- 한의사 국가고시 문제집

본 시스템은 교육·연구 보조용이며, 최종 진단과 처방은 면허 한의사가 판단해야 합니다.
