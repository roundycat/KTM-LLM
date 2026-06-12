# -*- coding: utf-8 -*-
"""Neo4j 없이 순수 Python으로 GraphRAG 근거 생성(보기Graph + 증상→처방) + 출처 저장.
입력: graphrag/data/*(KG 노드·처방 청크), D:\\tmp\\persubj\\questions_noanswer.jsonl(idx·과목·문항)
출력: graphrag/data/per_question_evidence.jsonl  — idx별 그래프 근거 + 출처(레포 저장)
"""
import json, re, os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
QFILE = r"D:\tmp\persubj\questions_noanswer.jsonl"
OUT = os.path.join(DATA, "per_question_evidence.jsonl")

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

# ── 처방 청크: 처방명 → 메타(구성·주치·계통·출전·출처·source) ──
rx_by_name = {}
sym_to_rx = {}   # 주치증상명 → set(처방명)
for c in load_jsonl(os.path.join(DATA, "처방_rag_chunks.jsonl")):
    m = c["metadata"]; nm = m["처방명"]
    rx_by_name[nm] = m
    for s in m.get("주치증상", []):
        sym_to_rx.setdefault(s, set()).add(nm)

# ── 노드: 증상/변증 이름(시드 추출용), 처방 보조 ──
sym_names = []   # (name, id)
for n in load_jsonl(os.path.join(DATA, "kg_all_nodes.jsonl")):
    if n.get("type") in ("증상", "변증") and len(n.get("name_ko", "")) >= 2:
        sym_names.append(n["name_ko"])
sym_names = sorted(set(sym_names), key=len, reverse=True)  # 긴 이름 우선
STOP = {"한다"}

def parse_rx_names(options):
    """보기에서 처방명 추출(합방 ' 합 ' 분리, 한자괄호 제거)."""
    names = []
    for o in options:
        nm = o.split("(")[0].strip()
        names.extend([x.strip() for x in nm.split(" 합 ")] if " 합 " in nm else [nm])
    return names

def opt_graph(options):
    """보기Graph: 각 보기 처방을 KG에서 조회."""
    rows, srcs = [], set()
    for nm in parse_rx_names(options):
        m = rx_by_name.get(nm)
        if not m:
            continue
        rows.append({
            "처방": nm, "한자": m.get("처방한자", ""), "계통": m.get("계통", ""),
            "주치": m.get("주치증상", [])[:8], "구성": m.get("구성약재", [])[:12],
            "출전": m.get("출전", ""),
        })
        if m.get("출처"): srcs.add(m["출처"])
        if m.get("source"): srcs.add(m["source"])
    return rows, srcs

def symptom_graph(question, topn=6):
    """증상→처방: 질문에 등장하는 증상명으로 주치 처방 후보 랭킹."""
    seeds = []
    seen = set()
    for nm in sym_names:
        if nm in STOP: continue
        if nm in question and nm not in seen:
            seeds.append(nm); seen.add(nm)
    if not seeds:
        return [], seeds
    score = {}
    for s in seeds:
        for rx in sym_to_rx.get(s, ()):
            score[rx] = score.get(rx, 0) + 1
    top = sorted(score.items(), key=lambda kv: -kv[1])[:topn]
    rows = []
    for rx, sc in top:
        m = rx_by_name[rx]
        rows.append({"처방": rx, "한자": m.get("처방한자", ""), "계통": m.get("계통", ""),
                     "부합증상수": sc, "구성": m.get("구성약재", [])[:10], "출전": m.get("출전", "")})
    return rows, seeds

def evidence_text(opt_rows, sym_rows, seeds):
    L = []
    if opt_rows:
        L.append("[보기 처방의 지식그래프 정보]")
        for g in opt_rows:
            L.append(f"- {g['처방']}({g['한자']}) [{g['계통']}] | 주치: {', '.join(g['주치'])} | 구성: {', '.join(g['구성'])} | 출전: {g['출전']}")
    if sym_rows:
        L.append(f"[질문 증상({', '.join(seeds[:8])})에 부합하는 처방 후보]")
        for g in sym_rows:
            L.append(f"- {g['처방']}({g['한자']}) [{g['계통']}] | 부합증상 {g['부합증상수']} | 구성: {', '.join(g['구성'])}")
    return "\n".join(L)

qs = load_jsonl(QFILE)
n_opt = n_sym = 0
with open(OUT, "w", encoding="utf-8") as f:
    for q in qs:
        opt_rows, srcs = opt_graph(q["options"])
        sym_rows, seeds = symptom_graph(q["question"])
        if opt_rows: n_opt += 1
        if sym_rows: n_sym += 1
        srcs |= {rx_by_name[r["처방"]].get("출처","") for r in sym_rows if r["처방"] in rx_by_name}
        f.write(json.dumps({
            "idx": q["idx"], "과목": q["과목"],
            "evidence": evidence_text(opt_rows, sym_rows, seeds),
            "보기처방_매칭": [r["처방"] for r in opt_rows],
            "증상시드": seeds[:10],
            "sources": sorted(x for x in srcs if x),
        }, ensure_ascii=False) + "\n")

print(f"근거 생성 {len(qs)}문항 → {OUT}")
print(f"보기Graph 매칭 있음: {n_opt}/{len(qs)}  | 증상→처방 후보 있음: {n_sym}/{len(qs)}")
print(f"처방 청크 {len(rx_by_name)}개, 증상명 인덱스 {len(sym_names)}개")
