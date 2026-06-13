# -*- coding: utf-8 -*-
"""
Anthropic Claude 비전 API로 타일 이미지를 전사하여 transcribe/ JSON을 생성한다.
(서브에이전트 수작업을 자동화하는 모듈 — 새 회차도 사람 개입 없이 처리)

준비:
  pip install anthropic
  set ANTHROPIC_API_KEY=...        (PowerShell: $env:ANTHROPIC_API_KEY="...")

문항 전사:
  python scripts/transcribe_api.py questions --slug 한의사_81회 --gyo 1교시 --gnum 1 --nums 1-80
정답표 전사:
  python scripts/transcribe_api.py answers   --slug 한의사_81회

설계 메모:
- 타일은 한 페이지당 c1s1,c1s2,c2s1,c2s2(좌상,좌하,우상,우하) 순서로 읽기.
- 한 번의 API 호출에 너무 많은 이미지를 넣으면 "장변 2000px" 제약/토큰 폭증이 생기므로
  PAGES_PER_CALL 페이지씩 끊어 호출하고, 번호 기준으로 병합/중복제거한다.
- system 프롬프트는 prompt caching 으로 비용 절감.
"""
import os, re, sys, json, glob, base64, argparse, time
from collections import defaultdict

try:
    from anthropic import Anthropic
except ImportError:
    sys.exit("pip install anthropic 필요")

MODEL = "claude-opus-4-8"          # 정확도 우선. 비용을 줄이려면 claude-sonnet-4-6
PAGES_PER_CALL = 3                 # 호출당 페이지 수 (이미지 = 4 * 이 값)
TRANS = "transcribe"
TILES = "tiles"

client = Anthropic()

Q_SYSTEM = """너는 한국 한의사/한약사 국가시험(객관식 5지선다) 문제지를 고해상도로 잘라낸
타일 이미지를 정확히 전사하는 전문 전사기다. 규칙:
- 각 문항: {"교시":정수,"번호":정수,"question":문두 전체,"options":[보기5개]}
- question 에는 환자 사례/보기/표를 모두 포함(표는 마크다운). 한자는 보이는 그대로.
  페이지 footer(예 "2/16")와 교시/과목 머리글은 제외.
- options 는 ①②③④⑤ 순서의 텍스트 5개. 동그라미 번호는 제거. 항상 정확히 5개.
- 그림이 <자료(비공개)>로 빠진 자리에는 "[그림 생략]" 을 적는다.
- 멀티미디어 비공개로 문두/보기가 아예 없는 번호는 만들지 말고 생략.
- 타일은 겹쳐 잘렸으니 같은 문항이 중복되면 더 완전한 쪽으로 한 번만.
출력: JSON 배열만. 설명/코드펜스 금지."""

A_SYSTEM = """너는 한국 보건의료인 국가시험 정답표(이미지)를 전사한다. 표 컬럼은
[교시 | 과목 | 문제번호 | 최종답안]. 모든 행을 전사:
{"교시":정수,"과목":문자열,"번호":정수,"answer":정수(1~5)}
복수정답/전항정답/비공개 등 특수표기는 answer 에 문자열 그대로. 출력은 JSON 배열만."""


def img_block(path):
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode()
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data}}


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\[.*\]", text, re.S)
    return json.loads(m.group(0) if m else text)


def call(system, content, max_tokens=16000, retries=3):
    for i in range(retries):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=max_tokens,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
            )
            return msg.content[0].text
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))


def page_num(path):
    m = re.search(r"p(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else 0


def transcribe_questions(slug, gyo, gnum, nums):
    folder = os.path.join(TILES, slug, gyo)
    tiles = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not tiles:
        sys.exit(f"타일 없음: {folder}")
    by_page = defaultdict(list)
    for t in tiles:
        by_page[page_num(t)].append(t)
    pages = sorted(by_page)

    merged = {}
    for i in range(0, len(pages), PAGES_PER_CALL):
        chunk = pages[i:i + PAGES_PER_CALL]
        content = [{"type": "text",
                    "text": f"{gyo} 페이지 {chunk} 타일. 교시={gnum}. "
                            f"문제번호 범위 {nums}. 보이는 모든 문항을 전사하라."}]
        for p in chunk:
            for t in sorted(by_page[p]):  # c1s1,c1s2,c2s1,c2s2 순
                content.append(img_block(t))
        out = extract_json(call(Q_SYSTEM, content))
        for q in out:
            n = q.get("번호")
            if n is None:
                continue
            q["교시"] = gnum
            if n not in merged or len(q.get("question", "")) > len(merged[n].get("question", "")):
                merged[n] = q
        print(f"  pages {chunk}: +{len(out)} (누적 {len(merged)})")

    result = [merged[n] for n in sorted(merged)]
    os.makedirs(TRANS, exist_ok=True)
    out_path = os.path.join(TRANS, f"{slug}__{gyo}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[*] {out_path}: {len(result)} 문항")


def transcribe_answers(slug):
    folder = os.path.join(TILES, slug, "답안")
    # 정답표는 단일 컬럼 → img/ 의 통 페이지를 써도 됨
    tiles = sorted(glob.glob(os.path.join(folder, "*.png"))) or \
            sorted(glob.glob(os.path.join("img", slug, "답안_*.png")))
    if not tiles:
        sys.exit(f"답안 이미지 없음: {folder} 또는 img/{slug}/답안_*.png")
    rows = {}
    for i in range(0, len(tiles), 4):
        content = [{"type": "text", "text": "정답표 전사. 모든 행."}]
        for t in tiles[i:i + 4]:
            content.append(img_block(t))
        for a in extract_json(call(A_SYSTEM, content)):
            rows[(a["교시"], a["번호"])] = a
    result = sorted(rows.values(), key=lambda x: (x["교시"], x["번호"]))
    os.makedirs(TRANS, exist_ok=True)
    out_path = os.path.join(TRANS, f"{slug}__답안.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[*] {out_path}: {len(result)} 행")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("questions")
    q.add_argument("--slug", required=True)
    q.add_argument("--gyo", required=True)
    q.add_argument("--gnum", type=int, required=True)
    q.add_argument("--nums", default="1-100")
    a = sub.add_parser("answers")
    a.add_argument("--slug", required=True)
    args = ap.parse_args()

    if args.cmd == "questions":
        transcribe_questions(args.slug, args.gyo, args.gnum, args.nums)
    else:
        transcribe_answers(args.slug)


if __name__ == "__main__":
    main()
