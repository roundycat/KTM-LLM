"""
step4_eval.py — 국가고시 문제집으로 방식별 정답률 비교.
  - 그냥 LLM   : 근거 없이
  - 벡터 RAG   : Chroma 벡터 검색만
  - GraphRAG   : BGE-m3 seed 벡터화 → 그래프+벡터 근거 검색
사용:
  python step4_eval.py --n 50
  python step4_eval.py            # 전체 517
  python step4_eval.py --rx-only --k 10 --seed 42
"""
import os, json, re, random, argparse
from collections import defaultdict
from step3_graphrag_query import (call_llm, extract_seeds_vector,
    graph_retrieve, vector_retrieve,
    build_context, COLL_RX, COLL_RX_CLINICAL)

QFILE  = "eval/한의학_문제.jsonl"
SOURCE = "eval/한의학_문제_원본.jsonl"
RX_PRESCRIPTION = re.compile(r'(치방|처방)은\?')

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

def norm(s):
    return re.sub(r"\s+", "", s or "")[:80]

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def build_subject_index(rows):
    idx = {}
    for r in rows:
        idx[norm(r["question"])] = {"과목": r.get("과목", "미상"),
                                    "has_figure": bool(r.get("has_figure", False))}
    return idx

def vector_rag_answer(question, opts, k=5, coll=COLL_RX_CLINICAL):
    ctx = build_context([], vector_retrieve(question, k=k, coll=coll))
    return parse_choice(call_llm(RAG_PROMPT.format(ctx=ctx, q=question, opts=opts)))

def rag_answer(question, opts, k=5, coll=COLL_RX, return_ctx=False):
    seeds, _ = extract_seeds_vector(question)
    g = graph_retrieve(seeds)
    ctx = build_context(g, vector_retrieve(question, k=k, coll=coll))
    choice = parse_choice(call_llm(RAG_PROMPT.format(ctx=ctx, q=question, opts=opts)))
    if return_ctx:
        return choice, seeds, ctx
    return choice

def pct(a, b): return f"{a/b*100:.1f}%" if b else "  -  "

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="문제 수 제한 (0=전체)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rx-only", action="store_true", help="처방 선택형만")
    ap.add_argument("--k", type=int, default=5, help="벡터 청크 수")
    ap.add_argument("--source", default=SOURCE, help="과목라벨 원본 파일")
    ap.add_argument("--skip-figure", action="store_true", help="그림 문제 제외")
    ap.add_argument("--diag", default="eval/diag_plain_ok_rag_fail.jsonl")
    args = ap.parse_args()

    Q = load_jsonl(QFILE)
    src_idx = build_subject_index(load_jsonl(args.source)) if os.path.exists(args.source) else {}
    matched = sum(1 for q in Q if norm(q["question"]) in src_idx)
    print(f"문제 {len(Q)}개 | 과목라벨 매칭 {matched}개 ({pct(matched, len(Q))})")

    if args.rx_only:
        Q = [q for q in Q if RX_PRESCRIPTION.search(q["question"])]
        print(f"처방 선택형: {len(Q)}개")
    random.seed(args.seed); random.shuffle(Q)
    if args.n:
        Q = Q[:args.n]

    by = lambda: {"base": 0, "vec": 0, "rag": 0, "n": 0}
    overall   = by()
    stat_subj = defaultdict(by)
    stat_type = defaultdict(by)
    stat_fig  = defaultdict(by)
    diag = []
    rx_ctx_n = rx_ctx_hit = 0

    for i, q in enumerate(Q, 1):
        meta = src_idx.get(norm(q["question"]), {"과목": "미상", "has_figure": False})
        subject, has_fig = meta["과목"], meta["has_figure"]
        if args.skip_figure and has_fig:
            continue
        opts = fmt_opts(q["options"]); gold = q["answer"]
        is_rx = bool(RX_PRESCRIPTION.search(q["question"]))

        b = parse_choice(call_llm(BASE_PROMPT.format(q=q["question"], opts=opts)))
        v = vector_rag_answer(q["question"], opts, k=args.k)

        if is_rx:
            r, seeds_r, ctx_r = rag_answer(q["question"], opts, k=args.k, coll=COLL_RX_CLINICAL, return_ctx=True)
            answer_text = q["options"][gold - 1] if gold >= 1 else ""
            ctx_has_ans = answer_text.split("(")[0] in ctx_r
            rx_ctx_n += 1; rx_ctx_hit += int(ctx_has_ans)
        else:
            r = rag_answer(q["question"], opts, k=args.k, coll=COLL_RX)

        rb, rv, rr = (b==gold), (v==gold), (r==gold)
        for d in (overall, stat_subj[subject],
                  stat_type["처방형" if is_rx else "그외"],
                  stat_fig["그림" if has_fig else "텍스트"]):
            d["n"]+=1; d["base"]+=rb; d["vec"]+=rv; d["rag"]+=rr

        if rb and not rr:
            if is_rx:
                diag.append({
                    "q": q["question"][:80], "과목": subject, "rx": is_rx,
                    "gold": gold, "answer_text": answer_text,
                    "그냥": b, "벡터": v, "Graph": r,
                    "정답이_근거에_있었나": ctx_has_ans,
                    "seeds": seeds_r, "ctx_head": ctx_r[:500],
                })
            else:
                _, seeds_r, ctx_r = rag_answer(q["question"], opts, k=args.k, coll=COLL_RX, return_ctx=True)
                at = q["options"][gold - 1] if gold >= 1 else ""
                diag.append({
                    "q": q["question"][:80], "과목": subject, "rx": is_rx,
                    "gold": gold, "answer_text": at,
                    "그냥": b, "벡터": v, "Graph": r,
                    "정답이_근거에_있었나": at.split("(")[0] in ctx_r,
                    "seeds": seeds_r, "ctx_head": ctx_r[:500],
                })

        tag = "[처방]" if is_rx else "     "
        print(f"[{i:3}/{len(Q)}] {tag}<{subject[:6]:6}> 정답{gold} "
              f"| 그냥={b} 벡터={v} Graph={r}")

    def table(title, d):
        print(f"\n## {title}")
        print(f"{'키':<14}{'n':>4}{'그냥':>8}{'벡터':>8}{'Graph':>8}")
        for key, v in sorted(d.items(), key=lambda x: -x[1]['n']):
            print(f"{key:<14}{v['n']:>4}{pct(v['base'],v['n']):>8}{pct(v['vec'],v['n']):>8}"
                  f"{pct(v['rag'],v['n']):>8}")

    o = overall
    print(f"\n===== 전체 n={o['n']} =====")
    print(f"그냥 LLM : {pct(o['base'],o['n'])}   벡터 RAG : {pct(o['vec'],o['n'])}   GraphRAG : {pct(o['rag'],o['n'])}")
    table("과목별", stat_subj)
    table("유형별", stat_type)
    table("그림 유무별", stat_fig)

    if rx_ctx_n:
        print(f"\n검색 recall (처방형 정답이 근거에 포함): "
              f"{rx_ctx_hit}/{rx_ctx_n} ({rx_ctx_hit/rx_ctx_n*100:.1f}%)")

    with open(args.diag, "w", encoding="utf-8") as f:
        for d in diag:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"그냥정답·Graph오답 {len(diag)}건 → {args.diag}")

if __name__ == "__main__":
    main()
