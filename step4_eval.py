"""
step4_eval.py — 국가고시 문제집으로 세 가지 방식 정답률 비교.
  - 그냥 LLM  : 전부 근거 없이
  - 항상 RAG  : 전부 그래프+벡터 근거 검색
  - 라우팅    : 처방 선택형만 RAG, 나머지는 그냥 LLM
사용:  python step4_eval.py --n 50
"""
import os, sys, json, re, random, argparse
from step3_graphrag_query import call_llm, extract_seeds, graph_retrieve, vector_retrieve, build_context

QFILE = "한의학_문제.jsonl"
RX_PRESCRIPTION = re.compile(r'(치방|처방)은\?\s*$')

BASE_PROMPT = """다음은 한의사 국가고시 5지선다 문제입니다. 정답 번호 하나만 숫자로 답하세요.

문제: {q}
보기:
{opts}

정답 번호(1-5)만 출력:"""

RAG_PROMPT = """아래 '근거'를 참고하여 한의사 국가고시 5지선다 문제의 정답을 고르세요.
근거가 문제와 무관하면 무시하고 네 지식으로 답하라
정답 번호 하나만 숫자로 답하세요.

[근거]
{ctx}

문제: {q}
보기:
{opts}

정답 번호(1-5)만 출력:"""

def fmt_opts(opts):
    return "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))

def parse_choice(text):
    m = re.search(r'[1-5]', text or "")
    return int(m.group()) if m else -1

def rag_answer(question, opts):
    seeds, _ = extract_seeds(question)
    ctx = build_context(graph_retrieve(seeds), vector_retrieve(question, k=5))
    return parse_choice(call_llm(RAG_PROMPT.format(ctx=ctx, q=question, opts=opts)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Q = [json.loads(l) for l in open(QFILE, encoding="utf-8")]
    random.seed(args.seed); random.shuffle(Q)
    Q = Q[:args.n]

    base_ok = rag_ok = route_ok = 0
    rx_base_ok = rx_rag_ok = rx_total = 0

    for i, q in enumerate(Q, 1):
        opts = fmt_opts(q["options"])
        gold = q["answer"]
        is_rx = bool(RX_PRESCRIPTION.search(q["question"]))

        b = parse_choice(call_llm(BASE_PROMPT.format(q=q["question"], opts=opts)))
        r = rag_answer(q["question"], opts)
        route = r if is_rx else b

        base_ok  += (b     == gold)
        rag_ok   += (r     == gold)
        route_ok += (route == gold)

        if is_rx:
            rx_total += 1
            rx_base_ok += (b == gold)
            rx_rag_ok  += (r == gold)

        tag = "[처방형]" if is_rx else "        "
        print(f"[{i:3}/{len(Q)}] {tag} 정답 {gold} | 그냥={b} RAG={r} 라우팅={route}")

    n = len(Q)
    print("\n===== 전체 결과 =====")
    print(f"그냥 LLM   : {base_ok}/{n}  ({base_ok/n*100:.1f}%)")
    print(f"항상 RAG   : {rag_ok}/{n}  ({rag_ok/n*100:.1f}%)")
    print(f"라우팅     : {route_ok}/{n}  ({route_ok/n*100:.1f}%)")

    if rx_total:
        print(f"\n===== 처방 선택형 부분집합 ({rx_total}문제) =====")
        print(f"그냥 LLM   : {rx_base_ok}/{rx_total}  ({rx_base_ok/rx_total*100:.1f}%)")
        print(f"항상 RAG   : {rx_rag_ok}/{rx_total}  ({rx_rag_ok/rx_total*100:.1f}%)")
    else:
        print("\n(처방 선택형 문제 없음 — 더 많은 문제로 실행하세요)")

if __name__ == "__main__":
    main()
