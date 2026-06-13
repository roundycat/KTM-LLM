"""
rewrite_chunks.py — 처방 청크를 임상 한국어 표현으로 재구성.
주치증상(한의학 한문 용어) → 환자가 호소하는 임상 증상 설명으로 LLM 변환.

실행:
  python rewrite_chunks.py --dry-run        # 처음 5개만 미리보기
  python rewrite_chunks.py --n 50           # 50개만 처리 (테스트)
  python rewrite_chunks.py                  # 전체 처리 (2,213개, ~$0.33)

출력: 처방_rag_chunks_clinical.jsonl
  → step2_build_vectordb.py 재실행 시 hani_rx 컬렉션에 병합됨
"""

import json, os, argparse
from step3_graphrag_query import call_llm

IN_FILE  = "data/처방_rag_chunks.jsonl"
OUT_FILE = "data/처방_rag_chunks_clinical.jsonl"

PROMPT = """한의사 국가고시 5지선다 문제에서 이 처방이 정답인 상황을 2~3문장으로 묘사하세요.
규칙:
1. 반드시 환자 증상 묘사로 시작하세요 (처방명·한자명 언급 금지)
2. 국가고시 문체 사용 ("~하며", "~을 호소하고", "~한 경우")
3. 마지막 줄에만 "처방: 처방명(한자) | 계통: 계통명" 형식으로 처방 정보 기재

처방명: {name}({hanja})
주치증상: {juchi}
구성약재: {herbs}
계통: {system}

증상 묘사 (처방명 언급 없이):"""


def load_unique_chunks(path):
    """(처방명, 주치 튜플) 기준 중복 제거 후 반환."""
    seen = set()
    result = []
    for l in open(path, encoding="utf-8"):
        c = json.loads(l)
        m = c.get("metadata", {})
        juchi = tuple(sorted(m.get("주치증상", [])))
        if not juchi:
            continue
        key = (m.get("처방명", ""), juchi)
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


def already_done(path):
    """이미 처리된 id 집합 (재시작 지원)."""
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="처리 수 제한 (0=전체)")
    ap.add_argument("--dry-run", action="store_true", help="처음 5개만 출력, 저장 안 함")
    ap.add_argument("--overwrite", action="store_true", help="기존 출력 파일 삭제 후 처음부터 재생성")
    args = ap.parse_args()

    if args.overwrite and os.path.exists(OUT_FILE):
        os.remove(OUT_FILE)
        print(f"기존 {OUT_FILE} 삭제 완료 → 처음부터 재생성")

    chunks = load_unique_chunks(IN_FILE)
    if args.dry_run:
        chunks = chunks[:5]
    elif args.n:
        chunks = chunks[:args.n]

    done = set() if args.dry_run else already_done(OUT_FILE)
    pending = [c for c in chunks if c["id"] + "_clin" not in done]
    print(f"총 {len(chunks)}개 중 {len(pending)}개 미처리 → 처리 시작")

    out_f = None if args.dry_run else open(OUT_FILE, "a", encoding="utf-8")

    try:
        for i, c in enumerate(pending, 1):
            m = c.get("metadata", {})
            name   = m.get("처방명", "")
            hanja  = m.get("처방한자", "")
            juchi  = ", ".join(m.get("주치증상", []))
            herbs  = ", ".join(m.get("구성약재", [])[:10])
            system = m.get("계통", "")

            prompt = PROMPT.format(name=name, hanja=hanja, juchi=juchi,
                                   herbs=herbs, system=system)
            clinical = call_llm(prompt).strip()

            new_text = (
                f"{clinical}\n"
                f"처방: {name}({hanja}) | 계통: {system} | 구성: {herbs}"
            )
            new_chunk = {
                "id": c["id"] + "_clin",
                "text": new_text,
                "metadata": m,
            }

            if args.dry_run:
                print(f"\n[{i}] {name}")
                print(f"  원본 주치: {juchi}")
                print(f"  → 임상 설명: {clinical}")
            else:
                out_f.write(json.dumps(new_chunk, ensure_ascii=False) + "\n")
                out_f.flush()

            if i % 50 == 0:
                print(f"  [{i}/{len(pending)}] {name} 완료")

    finally:
        if out_f:
            out_f.close()

    if not args.dry_run:
        done_total = sum(1 for _ in open(OUT_FILE, encoding="utf-8"))
        print(f"\n완료: {OUT_FILE} ({done_total}개)")
        print("다음 단계: python step2_build_vectordb.py")



if __name__ == "__main__":
    main()
