# -*- coding: utf-8 -*-
"""해설 데이터를 누수 없는 RAG(Leave-One-Out)로 추가.
 - 해설_rag_chunks.jsonl: 과목별 해설 지식원(출처 포함, 레포 저장)
 - 각 문항 X에 대해 '같은 과목 + 질문 유사도' 상위 k개 '다른 문항'의 해설을 근거로 부착(자기 X 제외 → 누수 없음)
 - 기존 그래프 근거(per_question_evidence.jsonl)에 [유사 문항 풀이 예시] 블록을 합쳐 eval_input_hae.jsonl 생성
"""
import json, re, os
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
HAE = r"D:\정하민\한의학 문제 데이터\dataset\한의학_문제_해설.jsonl"
QN = r"D:\tmp\persubj\questions_noanswer.jsonl"
GEV = os.path.join(DATA, "per_question_evidence.jsonl")
def L(p): return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]

# 해설: (source,교시,번호) → 해설
hae = {(r["source"], r["교시"], r["번호"]): r.get("해설", "") for r in L(HAE)}
qn = {r["idx"]: r for r in L(QN)}
gev = {r["idx"]: r for r in L(GEV)}

# 해설 RAG 청크(출처 포함) 저장
RXSRC = "Claude Opus 생성·2단계 적대검증(정확도 96.9%) 해설 — dataset/한의학_문제_해설.jsonl"
with open(os.path.join(DATA, "해설_rag_chunks.jsonl"), "w", encoding="utf-8") as f:
    for i, q in qn.items():
        h = hae.get((q["source"], q["교시"], q["번호"]), "")
        if h:
            f.write(json.dumps({"idx": i, "과목": q["과목"], "source": q["source"], "교시": q["교시"],
                                "번호": q["번호"], "해설": h, "출처": RXSRC}, ensure_ascii=False) + "\n")

# 토큰화(2자 이상 한글/한자 어절) — 질문 유사도용
STOP = set("문제 환자 다음 경우 무엇 것은 옳은 가장 모두 이때 한다 그것 어느 위해 통해 대한 따라 설명".split())
def toks(s):
    return [t for t in re.findall(r"[가-힣]{2,}|[一-鿿]{2,}", s or "") if t not in STOP]

# 각 문항의 토큰셋(질문+보기)
tset = {i: set(toks(q["question"] + " " + " ".join(q["options"]))) for i, q in qn.items()}
idxs = sorted(qn)

def loo_examples(x, k=3):
    """x와 같은 과목 우선 + 질문 토큰 겹침 상위 k '다른 문항' 해설."""
    sx, subx = tset[x], qn[x]["과목"]
    scored = []
    for y in idxs:
        if y == x:  # ← 누수 방지: 자기 자신 제외
            continue
        hy = hae.get((qn[y]["source"], qn[y]["교시"], qn[y]["번호"]), "")
        if not hy:
            continue
        overlap = len(sx & tset[y])
        if overlap == 0:
            continue
        score = overlap + (3 if qn[y]["과목"] == subx else 0)  # 같은 과목 가중
        scored.append((score, overlap, y, hy))
    scored.sort(key=lambda t: -t[0])
    return scored[:k]

n_with = 0
with open(r"D:\tmp\persubj\eval_input_hae.jsonl", "w", encoding="utf-8") as f:
    for i, q in qn.items():
        base_ev = gev.get(i, {}).get("evidence", "")
        ex = loo_examples(i, k=3)
        lines = []
        if ex:
            n_with += 1
            lines.append("[유사한 다른 문항의 풀이 예시 — 참고용(이 문항의 정답이 아님)]")
            for sc, ov, y, hy in ex:
                qy = qn[y]
                lines.append(f"· ({qy['과목']}) {hy[:280]}")
        hae_block = "\n".join(lines)
        full_ev = (base_ev + ("\n\n" + hae_block if hae_block else "")).strip()
        f.write(json.dumps({"idx": i, "과목": q["과목"], "question": q["question"],
                            "options": q["options"], "evidence": full_ev}, ensure_ascii=False) + "\n")

print(f"해설 RAG 청크 저장: {DATA}\\해설_rag_chunks.jsonl ({sum(1 for i in qn if hae.get((qn[i]['source'],qn[i]['교시'],qn[i]['번호'])))}개)")
print(f"eval_input_hae.jsonl 생성: {len(qn)}문항 (해설 예시 부착 {n_with})")
