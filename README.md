# 한의학 국가시험 RAG · GraphRAG 연구 (KTM-LLM)

국시원 공개 기출 **5지선다 517문항**으로, 한의학 지식을 **검색증강(RAG)** 으로 주입했을 때
로컬 오픈 LLM의 정답률이 오르는지를 검증한다. 두 갈래의 RAG만 다룬다.

1. **용어 RAG** — 문제에 등장하는 KIOM 표준용어 정의를 컨텍스트로 주입(`training/`).
2. **GraphRAG** — 한의학 처방 지식그래프(Neo4j)와 벡터검색(Chroma·BGE-m3)을 결합해 처방 근거를
   주입. 원본 시스템(`bigse0u1/`)과 그 확장·선택적RAG(`graphrag/`).

> 모든 평가는 **로컬 추론(Ollama, GPU), 비용 $0**(원본 일부만 gpt-4o-mini). 모델 가중치 학습(파인튜닝)은
> 하지 않고 **사전학습 모델 그대로 추론**한다. 정답은 채점에만 쓰고 모델에 주지 않는다(누수 없음).

> 📚 상세: [graphrag/PAPER_한의학_GraphRAG.md](graphrag/PAPER_한의학_GraphRAG.md) ·
> [graphrag/MULTIMODEL_REPORT.md](graphrag/MULTIMODEL_REPORT.md) · 원본 GraphRAG [bigse0u1/](bigse0u1/) ·
> 원자료 [results/](results/)

---

## 📊 전체 결과

### A. GraphRAG

처방 선택형 86문항이 핵심 — 전체 517 평균으론 LLM 기본 지식에 희석되어 방식 간 차이가 작지만(59~60%),
처방형으로 좁히면 검색 근거의 효과가 드러난다. 본 프로젝트는 두 GraphRAG 접근을 모두 수록한다.

#### A-1. 원본 시스템(`bigse0u1/`) — seed 벡터화 GraphRAG (처방형 86)

질문 임베딩을 증상 노드 임베딩과 직접 비교(seed 벡터화)해 처방을 역검색한다. **단계적 개선**으로
처방형 정답률과 검색 recall을 크게 끌어올렸다(LLM=gpt-4o-mini, `--rx-only --k 20`).

| 단계 | 주요 변경 | 그냥 LLM | 벡터 RAG | GraphRAG | recall@ctx |
|---|---|---:|---:|---:|---:|
| 초기 (k=5) | — | 30.2% | 29.1% | 31.4% | 4.7% |
| 임상청크 재작성 + k=20 | symptom-first 청크, k 확대 | 30.2% | 37.2% | 30.2% | 7.0% |
| **누락처방 추가 + seed 벡터화** | 24개 처방 보완, LLM번역→벡터추출 | 30.2% | **38.4%** | **43.0%** | **12.8%** |

- **seed 벡터화가 핵심**: LLM이 임상표현→한문용어로 번역하던 단계를 제거하고 질문·증상 임베딩을 직접
  비교 → GraphRAG **31.4%→43.0%(+11.6%p)**, recall@ctx **4.7%→12.8%**.
- 로컬 모델 비교(처방형 86, k=20):

| 모델 | 그냥 LLM | 벡터 RAG | GraphRAG |
|---|---:|---:|---:|
| qwen2.5 | 32.6% | 31.4% | **34.9%** |
| exaone3.5 | 25.6% | 27.9% | **32.6%** |
| llama3.1 | 23.3% | 27.9% | **31.4%** |
| mistral | 27.9% | 26.7% | 24.4% |

→ qwen·exaone·llama는 **GraphRAG ≥ 벡터RAG ≥ 그냥LLM** 순. **mistral만 역효과**(한문 용어 근거를 못 살림).

#### A-2. 확장(`graphrag/`) — 보기Graph 기반 선택적RAG (전체 517, 6모델)

보기 5개 처방을 그래프에서 직접 조회해 **구성·주치·계통**을 주입하고, **처방형이면 그래프RAG / 그 외면
그냥LLM**으로 유형 라우팅(정답 미사용). Neo4j 없이 순수 Python 재현.

| 모델 | 그냥LLM | 그래프RAG | **선택적RAG** | Δ(선택−그냥) | 처방형 그래프 Δ |
|---|---:|---:|---:|---:|---:|
| **qwen2.5 7B** | 48.7% | 48.0% | **51.3%** | **+2.5%p** | **+15.1%p** ⭐ |
| gemma2 9B | 47.0% | 47.0% | **47.6%** | +0.6%p | +3.5%p |
| exaone3.5 7.8B | 42.6% | 43.5% | **42.7%** | +0.2%p | +1.2%p |
| llama3.1 8B | 34.6% | 29.2% | **34.8%** | +0.2%p | +1.2%p |
| solar 10.7B | 32.3% | 27.5% | **32.5%** | +0.2%p | +1.2%p |
| mistral 7B | 31.7% | 28.6% | **30.4%** | −1.4%p | −8.1%p |

- 효과는 **강하게 모델 의존적**: 한자·중국어에 강한 qwen 압도(처방형 +15.1%p), 영어권 mistral 역효과.
- 전체×과목별 표 → [graphrag/MULTIMODEL_REPORT.md](graphrag/MULTIMODEL_REPORT.md).

> 공통 결론: **그래프RAG는 처방형에서만 효과**, **효과는 모델의 한자/한의학 이해력에 비례**, 무관 근거는
> 오히려 방해(→ 유형 선택적 적용·검색 recall 개선이 관건).

### B. 용어 RAG — KIOM 표준용어 정의 주입 (전체 517)

문제+보기에 **등장하는 용어**의 정의를 한자 우선 매칭으로 상위 6개 주입(임베딩·API 불필요·결정론적).

| 모델 | base | **+용어RAG** | Δ |
|---|---:|---:|---:|
| 한국형 EXAONE 3.5 7.8B | 43.71% | **46.62%** | **+2.91%p** |
| 한국형 SOLAR 10.7B | 33.08% | 34.62% | +1.54%p |
| 글로벌 Qwen2.5 7B | 48.55% | 49.52% | +0.97%p |
| 글로벌 Llama3.1 8B | 34.24% | 33.46% | −0.78%p |

- **RAG 효과 = 모델의 한국어 능력에 비례**(EXAONE +2.91%p > … > Llama −0.78%p).
- 과목별로는 **용어가 곧 답인 과목에 집중**(글로벌 Qwen 기준): 본초학 **+30.8%p**(7.69→38.46),
  침구학 **+12.1%p**(30.30→42.42), 부인과 +6.9%p / 임상추론 과목은 역효과(소아 −9.5, 생리 −6.3%p).

> 참고: temperature 0이라도 배치·런타임 차로 런 간 ±1문항 변동이 있다. 표 수치는 각 행 내 일관된 런 기준.

---

## 📦 데이터

### 평가 문항 (`dataset/`, `graphrag/eval/`)

| 자원 | 내용 | 규모 |
|---|---|---:|
| `한의학_문제.jsonl` | 그림 비의존 텍스트 5지선다(평가 메인) | 517 |
| `한의학_문제_전체.jsonl` | 그림생략(70) 포함 + 과목·정답 메타 | 587 |
| `처방_문제.jsonl` | 처방 선택형(문항 끝 "치방은?/처방은?")만 추출 | 86 |
| `한의학_용어.jsonl` | KIOM 표준한의학용어집(V2.1·국문) — 용어RAG 지식원 | 9,074 |

수집 회차: **제81회 한의사(2026)** · **제27회 한약사(2026)**. 정답은 국시원 공식 정답표 기준(100% 매칭 검증).
스키마: `{source, 교시, 과목, 번호, question, options[5], answer(1~5), answer_text, has_figure}`.

### 한의학 처방 지식그래프 (`graphrag/data/`, `bigse0u1/data/`)

노드 **7,692** / 엣지 **39,996**.

| 노드 타입 | 설명 | 수 | | 엣지 타입 | 의미 | 수 |
|---|---|---:|---|---|---|---:|
| 처방 | 구성·출전·계통·한자 | 3,094 | | 구성 | 처방→약재 | 25,565 |
| 증상 | 주치 적응증·병증 | 3,873 | | 주치 | 처방→증상/변증 | 8,858 |
| 약재 | 귀경·성·미·분류 | 645 | | 계통 | 처방→계통 | 3,094 |
| 증상지표 | 팔강변증 설문 | 54 | | 팔강귀속 | 증상→변증 | 1,447 |
| 장부 | 귀경 대상 장기 | 13 | | 귀경 | 약재→장부 | 972 |
| 변증 | 팔강 8증 | 8 | | 지표 | 증상지표→변증 | 54 |
| 계통 | 오장 내과 계통 | 5 | | 포함 | 변증→변증 | 6 |

**연결 구조**
```
[증상지표] ──지표──▶ [변증] ◀──팔강귀속── [증상]
                                              ▲ 주치
[처방] ──구성──▶ [약재] ──귀경──▶ [장부]      │
   └──계통──▶ [계통]                          [처방]
```

**벡터 청크(Chroma)**: `hani_rx`(처방 원본 3,118) · `hani_rx_clinical`(임상 표현 2,237) · `hani_term`(용어 9,074).
임상 표현 청크는 `rewrite_chunks.py`가 주치증상(한문)을 시험 문체의 증상 묘사로 변환하고 처방명을 마지막
줄로 옮겨 임베딩이 증상에 anchoring되게 만든 것. 누락 처방 24개는 `add_missing_rx.py`로 보완.

> ⚠️ **저작권**: 국시원 기출·표준한의학용어집·교과서 처방 데이터는 저작물입니다. 개인 학습·연구용으로만.

---

## 🔬 시스템 · 방법

### GraphRAG 아키텍처

| 구성 | 기술 | 비고 |
|---|---|---|
| 그래프 DB | **Neo4j** Community (`bolt://localhost:7687`) | 단일 레이블 `:Node`, `label` 속성으로 타입 구분 |
| 임베딩 | **BAAI/bge-m3** (1,024차원, cosine) | 한·중·영 혼재 텍스트용 다국어, 로컬·무료 |
| 벡터 DB | **Chroma** PersistentClient | `{"hnsw:space":"cosine"}`, 컬렉션 3종 분리 |
| LLM | Ollama(로컬) / OpenAI / Anthropic | `temperature=0` 고정(재현성) |

**핵심 Cypher** — 추출된 증상(`$seeds`)과 `주치` 엣지로 연결된 처방을 역탐색, 부합 증상 수 상위 6개 + 구성약재:
```cypher
MATCH (p:Node {label:'처방'})-[:REL {type:'주치'}]->(s:Node)
WHERE s.id IN $seeds
WITH p, collect(DISTINCT s.name_ko) AS matched, count(DISTINCT s) AS score
ORDER BY score DESC LIMIT 6
MATCH (p)-[:REL {type:'구성'}]->(h:Node)
RETURN p.name_ko AS 처방, p.name_hanja AS 한자, p.계통 AS 계통, matched,
       collect(h.name_ko) AS 약재, score
```

**seed 추출 — LLM 번역 → 벡터화로 진화**: 시험 문제는 임상표현("소화가 안 되고 구토")을, 그래프 노드는
진단용어("비기허약, 위기불화")를 쓴다. 초기엔 LLM이 번역 후 매칭했으나, **질문·증상 노드 임베딩을 직접
비교(seed 벡터화)** 하도록 바꿔 번역 오류를 제거하고 처방형 GraphRAG를 43.0%까지 끌어올렸다.

### 4단계 파이프라인 (`bigse0u1/`, `graphrag/step1~4`)
```
step1_load_neo4j      그래프(노드·엣지) → Neo4j 적재
step2_build_vectordb  청크 → BGE-m3 임베딩 → Chroma (수십 분)
step3_graphrag_query  질문 → seed 추출 → Neo4j 역검색 + Chroma 검색 → LLM 답변
step4_eval            그냥LLM / 벡터RAG / GraphRAG 정답률 비교 (+ recall@ctx)
```

### 용어 RAG (`training/rag.py`, `training/evaluate.py`)
```
문제+보기 ─▶ 등장 용어 검색(한자 우선 → 한국어 한글경계 보충) ─▶ 정의 상위6 주입 ─▶ LLM 풀이 ─▶ 정답 파싱
```
한자 매칭은 일반어와 거의 안 겹쳐 고정밀, 한국어는 단어 경계 정규식으로 오매칭 차단(`대장균`에 `대장` 등).
1글자 용어·일반어 불용어 제외. 임베딩/API 불필요 → 무료·결정론적.

### 누수 방지

| 구성 | 시험 문항/정답 사용 | 판정 |
|---|---|---|
| 지식그래프·용어·청크 | 교과서·용어집으로 구축, 문항 무관 | ✅ |
| 그냥LLM / 그래프RAG / 용어RAG / 선택적RAG | 테스트 시 문항만(정답 분리·미사용) | ✅ |

---

## 🐛 주요 이슈 — "한다" 종결어미 오매칭 (SY-0410)

증상 노드 `汗多(한다)`의 `name_ko`가 `"한다"`라, 한국어 종결어미 `~한다`와 완전 일치하는 오매칭 발생.
거의 모든 질문에 `SY-0410`이 seed로 끼어 그래프 검색 전체가 오염되는 심각한 버그였다.
→ `step3_graphrag_query.py`에 `STOP_NAMES = {"한다"}`를 추가해 매칭에서 제외.

---

## 🚀 빠른 시작 / 재현

```bash
# ── 용어 RAG (training/) ── 용어 인덱스는 dataset/한의학_용어.jsonl 에 포함됨
pip install -r requirements.txt
ollama pull qwen2.5:7b ; ollama pull exaone3.5:7.8b   # (D: 권장: OLLAMA_MODELS=D:\ollama-models)
python training/evaluate.py --all            # base + 글로벌 +용어RAG (전체 517)
python training/evaluate.py --all --rag-local
python training/report.py                    # → results/report.md

# ── GraphRAG (graphrag/, Neo4j 불요 확장) ──
pip install -r graphrag/requirements_graphrag.txt
python graphrag/build_evidence.py            # 보기 처방 그래프 근거 생성
python graphrag/run_all_models.py            # 6모델 그냥/그래프/선택적RAG
python graphrag/aggregate_models.py          # → graphrag/MULTIMODEL_REPORT.md

# ── 원본 GraphRAG (bigse0u1/, Neo4j+Chroma) ──
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PW=비밀번호
python bigse0u1/step1_load_neo4j.py          # 그래프 적재
python bigse0u1/step2_build_vectordb.py      # 벡터DB 구축
python bigse0u1/step3_graphrag_query.py "소화가 안되고 입맛이 없으며 기운이 없다"
python bigse0u1/step4_eval.py --rx-only --k 20   # 처방형 86 평가
```

---

## 📁 프로젝트 구조

```bash
KTM-LLM/ (main_1)
├── training/          # 용어 RAG 실험
│   ├── rag.py · evaluate.py · config.py · report.py
├── graphrag/          # GraphRAG 확장(정본): 보기Graph 선택적RAG, 6모델
│   ├── build_evidence.py · run_all_models.py · aggregate_models.py
│   ├── step1~4_*.py · data/  · PAPER_한의학_GraphRAG.md · MULTIMODEL_REPORT.md
├── bigse0u1/          # 원본 GraphRAG(github 브랜치 bigse0u1): seed 벡터화 + 단계적 개선
│   ├── step1~4_*.py · enrich_주치.py · rewrite_chunks.py
│   ├── add_missing_rx.py · check_clinical_recall.py · data/ · eval/
├── dataset/           # 평가 문항 + KIOM 용어
├── results/           # 모델별 평가 결과 JSON(원자료)
└── README.md
```

---

## ⚠️ 한계

- **그래프RAG는 처방형(n=86)에서만 효과** — 전체 평균으론 희석. 처방형 CI가 넓어 점추정은 지시적.
- **증상→처방 검색 recall이 낮음**(초기 4.7% → seed 벡터화·누락처방 보완 후 12.8%) — 여전히 개선 여지.
- 지식그래프는 **내과 처방 중심** — 본초·침구·법규 등은 직접 근거가 없어 그냥LLM 사용(선택적RAG).
- **용어RAG 효과는 모델 한국어 능력·과목 의존**(순효과 작음, 일부 과목 역효과).
- 영어권 모델(mistral)은 한문 그래프 근거를 못 살려 GraphRAG가 오히려 역효과.
- 단일 회차(한의사81·한약사27)·비전 전사 데이터의 드문 오탈자 가능. 긴 근거에서 형식 이탈은 보수적 오답 처리.

## 📑 출처

표준한의학용어집(2006) · 한의대 내과학 교과서 처방 데이터(2024) · 팔강변증 한의표준임상진료지침(2023) ·
국시원 공개 기출(한의사 제81회·한약사 제27회). 교육·연구 보조용이며 최종 진단·처방은 면허 한의사가 판단한다.
