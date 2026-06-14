"""
add_missing_rx.py — 누락된 24개 처방을 처방_rag_chunks.jsonl에 추가.

합방(3개): 기존 DB에서 두 처방을 찾아 합치기
단방(21개): LLM으로 처방 데이터 생성 후 임상 청크도 자동 생성

실행:
  python add_missing_rx.py --dry-run   # 처음 5개만 미리보기
  python add_missing_rx.py             # 전체 추가
"""
import json, os, argparse
from step3_graphrag_query import call_llm

CHUNKS_FILE   = "data/처방_rag_chunks.jsonl"
CLINICAL_FILE = "data/처방_rag_chunks_clinical.jsonl"

MISSING_SINGLE = [
    "강활계지탕", "곡맥지출환", "공제환", "교애탕", "기국지황탕",
    "도화탕", "마황세신부자탕", "마황행인감초석고탕", "뭉석곤담환",
    "미후등식장탕", "보궁환", "보음전", "보익양위탕", "비원전",
    "생철락음", "성유탕", "온포음", "윤마환", "이모영수탕",
    "태화음", "한다열소탕",
]

MISSING_COMPOUND = [
    ("우귀음 합 이중탕",       ["우귀음",    "이중탕"]),
    ("육미지황탕 합 생맥산",   ["육미지황탕", "생맥산"]),
    ("팔물탕 합 익위승양탕",   ["팔물탕",   "익위승양탕"]),
]

DATA_PROMPT = """한의학 처방 {name}에 대한 정보를 정확하게 JSON으로만 출력하세요.
다른 텍스트는 절대 포함하지 마세요.

{{
  "처방한자": "한자 이름",
  "계통": "간계/심계/비계/폐계/신계/기타계 중 가장 적합한 것",
  "주치증상": ["주치 키워드1", "주치 키워드2"],
  "구성약재": ["약재1", "약재2", "약재3"],
  "출전": "동의보감/방약합편/동의수세보원 등 원전명"
}}"""

CLINICAL_PROMPT = """한의사 국가고시 5지선다 문제에서 이 처방이 정답인 상황을 2~3문장으로 묘사하세요.
규칙:
1. 반드시 환자 증상 묘사로 시작하세요 (처방명·한자명 언급 금지)
2. 국가고시 문체 사용 ("~하며", "~을 호소하고", "~한 경우")
3. 마지막 줄에만 "처방: 처방명(한자) | 계통: 계통명" 형식으로 처방 정보 기재

처방명: {name}({hanja})
주치증상: {juchi}
구성약재: {herbs}
계통: {system}

증상 묘사 (처방명 언급 없이):"""


def load_db():
    db = {}
    for l in open(CHUNKS_FILE, encoding="utf-8"):
        c = json.loads(l)
        nm = c["metadata"].get("처방명", "")
        if nm:
            db[nm] = c
    return db


def already_added():
    done = set()
    if os.path.exists(CHUNKS_FILE):
        for l in open(CHUNKS_FILE, encoding="utf-8"):
            c = json.loads(l)
            if c["id"].startswith("MISS_"):
                done.add(c["metadata"].get("처방명", ""))
    return done


def make_chunk(name, hanja, system, juchi_list, herbs_list, source, idx):
    juchi_str = ", ".join(juchi_list)
    herbs_str = ", ".join(herbs_list[:10])
    text = (f"{name}({hanja}) [{system}내과]\n"
            f"주치: {juchi_str}\n"
            f"구성: {herbs_str}")
    return {
        "id": f"MISS_{idx:04d}",
        "text": text,
        "metadata": {
            "처방명": name,
            "처방한자": hanja,
            "계통": system,
            "주치증상": juchi_list,
            "구성약재": herbs_list,
            "출전": source,
            "출처": "국가고시_누락보완",
            "source": "추가분",
        }
    }


def make_clinical(chunk):
    m = chunk["metadata"]
    prompt = CLINICAL_PROMPT.format(
        name=m["처방명"], hanja=m["처방한자"],
        juchi=", ".join(m["주치증상"]),
        herbs=", ".join(m["구성약재"][:10]),
        system=m["계통"],
    )
    clinical = call_llm(prompt).strip()
    return {
        "id": chunk["id"] + "_clin",
        "text": f"{clinical}\n처방: {m['처방명']}({m['처방한자']}) | 계통: {m['계통']} | 구성: {', '.join(m['구성약재'][:10])}",
        "metadata": m,
    }


def llm_rx_data(name, idx):
    raw = call_llm(DATA_PROMPT.format(name=name)).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        d = json.loads(raw)
        return make_chunk(
            name=name,
            hanja=d.get("처방한자", ""),
            system=d.get("계통", "기타계"),
            juchi_list=d.get("주치증상", []),
            herbs_list=d.get("구성약재", []),
            source=d.get("출전", ""),
            idx=idx,
        )
    except Exception as e:
        print(f"  ⚠️  JSON 파싱 실패 ({name}): {e}\n  raw: {raw[:300]}")
        return None


def compound_chunk(compound_name, components, db, idx):
    parts = [db[c] for c in components if c in db]
    missing = [c for c in components if c not in db]
    if missing:
        print(f"  ⚠️  합방 구성 처방 없음: {missing}")
    if not parts:
        return None

    def dedup(lst):
        seen, out = set(), []
        for x in lst:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    hanja  = " 合 ".join(p["metadata"].get("처방한자", "") for p in parts)
    system = parts[0]["metadata"].get("계통", "")
    juchi  = dedup(j for p in parts for j in p["metadata"].get("주치증상", []))
    herbs  = dedup(h for p in parts for h in p["metadata"].get("구성약재", []))
    return make_chunk(compound_name, hanja, system, juchi, herbs, "합방", idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db   = load_db()
    done = already_added()
    print(f"기존 DB: {len(db)}개  이미 추가됨: {len(done)}개\n")

    new_chunks   = []
    new_clinical = []
    idx = 1

    # 합방
    print("=== 합방 처리 (DB 병합) ===")
    for compound_name, components in MISSING_COMPOUND:
        if compound_name in done:
            print(f"  skip (이미 있음): {compound_name}"); idx += 1; continue
        chunk = compound_chunk(compound_name, components, db, idx)
        if chunk:
            clinical = make_clinical(chunk)
            new_chunks.append(chunk)
            new_clinical.append(clinical)
            print(f"  ✅ {compound_name}  주치: {', '.join(chunk['metadata']['주치증상'][:3])}")
        idx += 1

    # 단방
    print("\n=== 단방 처리 (LLM) ===")
    for name in MISSING_SINGLE:
        if name in done:
            print(f"  skip (이미 있음): {name}"); idx += 1; continue
        if args.dry_run and len(new_chunks) >= 5:
            break
        print(f"  처리중: {name}...", end=" ", flush=True)
        chunk = llm_rx_data(name, idx)
        if chunk:
            clinical = make_clinical(chunk)
            new_chunks.append(chunk)
            new_clinical.append(clinical)
            print(f"✅ ({chunk['metadata']['처방한자']}) [{chunk['metadata']['계통']}]")
        idx += 1

    print(f"\n생성: {len(new_chunks)}개")

    if args.dry_run:
        print("\n[dry-run] 저장 안 함. 샘플:")
        for c, cl in zip(new_chunks[:2], new_clinical[:2]):
            print(f"\n  처방: {c['metadata']['처방명']}({c['metadata']['처방한자']})")
            print(f"  주치: {c['metadata']['주치증상']}")
            print(f"  구성: {c['metadata']['구성약재'][:5]}")
            print(f"  임상: {cl['text'][:120]}")
        return

    with open(CHUNKS_FILE,   "a", encoding="utf-8") as f:
        for c in new_chunks:   f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(CLINICAL_FILE, "a", encoding="utf-8") as f:
        for c in new_clinical: f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n✅ {CHUNKS_FILE}  (+{len(new_chunks)}개)")
    print(f"✅ {CLINICAL_FILE} (+{len(new_clinical)}개)")
    print("\n다음 단계: python step2_build_vectordb.py")


if __name__ == "__main__":
    main()
