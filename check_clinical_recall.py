"""
임상 청크 컬렉션(hani_rx_clinical) recall 진단.
python check_clinical_recall.py
"""
import json, re
from step3_graphrag_query import vector_retrieve, COLL_RX, COLL_RX_CLINICAL, chroma

QFILE = "eval/한의학_문제.jsonl"
RX_PRESCRIPTION = re.compile(r'(치방|처방)은\?')

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

Q = [q for q in load_jsonl(QFILE) if RX_PRESCRIPTION.search(q["question"])]
print(f"처방형 문제: {len(Q)}개")

# DB 커버리지 확인
coll_orig  = chroma(COLL_RX)
coll_clin  = chroma(COLL_RX_CLINICAL)
print(f"COLL_RX 청크 수: {coll_orig.count()}")
print(f"COLL_RX_CLINICAL 청크 수: {coll_clin.count()}\n")

# k별 recall 측정
for k in [5, 10, 20, 50]:
    orig_hit = clin_hit = 0
    for q in Q:
        gold = q["answer"]
        answer_text = q["options"][gold - 1].split("(")[0].strip() if gold >= 1 else ""
        orig_docs = " ".join(doc for _, doc in vector_retrieve(q["question"], k=k, coll=COLL_RX))
        clin_docs = " ".join(doc for _, doc in vector_retrieve(q["question"], k=k, coll=COLL_RX_CLINICAL))
        orig_hit += int(answer_text in orig_docs)
        clin_hit += int(answer_text in clin_docs)
    n = len(Q)
    print(f"k={k:2d} | 원본 {orig_hit}/{n} ({orig_hit/n*100:.1f}%)  임상 {clin_hit}/{n} ({clin_hit/n*100:.1f}%)")

# 정답 처방이 DB에 있는지 커버리지 확인
print("\n=== 정답 처방 DB 커버리지 ===")
no_cover = []
for q in Q:
    gold = q["answer"]
    answer_text = q["options"][gold - 1].split("(")[0].strip() if gold >= 1 else ""
    # 전체 컬렉션에서 처방명 검색
    results = coll_orig.get(where={"처방명": {"$eq": answer_text}}, limit=1)
    if not results["ids"]:
        no_cover.append(answer_text)

print(f"DB에 없는 정답 처방: {len(no_cover)}/{len(Q)}개")
for nm in sorted(set(no_cover))[:20]:
    print(f"  - {nm}")
