# -*- coding: utf-8 -*-
"""KIOM 표준한의학용어집 수집 → dataset/한의학_용어.jsonl

cis.kiom.re.kr/terminology/search.do 는 검색어로 term=%(와일드카드)를 주면
전 버전·언어의 모든 용어(약 9천 레코드)를 페이지네이션 없이 한 번에 반환한다.
각 레코드의 표제어/메타/정의를 파싱하고, 기본적으로 V2.1·국문만 추려 저장한다.

레코드 HTML 구조(요약):
  <span class='style1'>간(肝)</span>
  <span class='term_filterType'>[V2.1][생리][국문]</span><br>
  <span class='content_bold'>① 오장의 하나 ... [동] 간장(肝臟) ...</span>

산출 스키마(한 줄당 1용어):
  {"term":"간","hanja":"肝","headword":"간(肝)","version":"V2.1",
   "category":"생리","lang":"국문","definition":"① 오장의 하나 ...","synonyms":"간장(肝臟), ..."}

⚠️ 저작권: 표준한의학용어집(대한한의학회)은 저작물이다. 개인 학습/연구용으로만 쓰고
   재배포에 주의한다. (대용량 원본 HTML 캐시는 raw/ 에 저장되며 gitignore 됨)

사용법:
  python scripts/fetch_terminology.py                 # V2.1 · 국문 전부(기본)
  python scripts/fetch_terminology.py --version all   # 전 버전
  python scripts/fetch_terminology.py --lang all      # 국문+영문
  python scripts/fetch_terminology.py --no-cache      # 캐시 무시하고 재다운로드
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset" / "한의학_용어.jsonl"
RAW_CACHE = ROOT / "raw" / "terminology_all.html"   # raw/ 는 gitignore

BASE = "https://cis.kiom.re.kr/terminology/search.do"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; hani-exam terminology fetch)"}

# [버전][분류][언어] 형태의 메타 추출
FILTER_RE = re.compile(r"\[([^\]]*)\]\[([^\]]*)\]\[([^\]]*)\]")


def fetch_html(use_cache: bool) -> str:
    """term=%(전체) 검색 결과 HTML 을 가져온다(캐시 가능)."""
    if use_cache and RAW_CACHE.exists():
        print(f"[캐시 사용] {RAW_CACHE} ({RAW_CACHE.stat().st_size:,} bytes)")
        return RAW_CACHE.read_text(encoding="utf-8")
    print(f"[다운로드] {BASE}?term=% (대용량 ~20MB, 수십 초 소요)")
    # params 로 넘기면 requests 가 % → %25 로 인코딩한다.
    r = requests.get(BASE, params={"term": "%"}, headers=HEADERS, timeout=180)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    html = r.text
    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    RAW_CACHE.write_text(html, encoding="utf-8")
    print(f"  {len(html):,} chars 저장 → {RAW_CACHE}")
    return html


def parse_records(html: str) -> list[dict]:
    """style1(표제어) 단위로 레코드 창을 잘라 BeautifulSoup 으로 필드 추출.

    중첩 테이블(시맨틱/네이버 메뉴)에 영향받지 않도록, 다음 표제어 직전까지를
    한 레코드 창으로 보고 그 안의 '첫' 메타/정의 span 만 사용한다.
    """
    starts = [m.start() for m in re.finditer(r"<span class='style1'", html)]
    starts.append(len(html))
    out = []
    for i in range(len(starts) - 1):
        window = html[starts[i]:starts[i + 1]]
        soup = BeautifulSoup(window, "html.parser")
        head_el = soup.select_one(".style1")
        meta_el = soup.select_one(".term_filterType")
        def_el = soup.select_one(".content_bold") or soup.select_one(".content")
        if not (head_el and meta_el):
            continue
        headword = head_el.get_text(" ", strip=True)
        fm = FILTER_RE.search(meta_el.get_text(" ", strip=True))
        if not fm:
            continue
        version, category, lang = (g.strip() for g in fm.groups())
        definition = def_el.get_text(" ", strip=True) if def_el else ""

        # 표제어/한자 분리: "간(肝)" → ("간", "肝")
        term, hanja = headword, ""
        hm = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", headword)
        if hm:
            term, hanja = hm.group(1).strip(), hm.group(2).strip()

        # 정의 뒤에 붙는 마커([동]동의어, [대]대응어, [참]참고 등)를 본문에서 분리.
        #   본문 = 첫 마커 이전, notes = 마커 이후 전체, synonyms = [동] 내용.
        notes, synonyms = "", ""
        mk = re.search(r"\[[가-힣]{1,3}\]", definition)
        if mk:
            notes = definition[mk.start():].strip()
            definition = definition[:mk.start()].strip()
            sm = re.search(r"\[\s*동\s*\]\s*([^\[]*)", notes)
            if sm:
                synonyms = sm.group(1).strip(" .")

        out.append({
            "term": term, "hanja": hanja, "headword": headword,
            "version": version, "category": category, "lang": lang,
            "definition": definition, "synonyms": synonyms, "notes": notes,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="V2.1", help="버전 필터(V2.1/V2.0/V1.0/all)")
    ap.add_argument("--lang", default="국문", help="언어 필터(국문/영문/all)")
    ap.add_argument("--no-cache", action="store_true", help="캐시 무시하고 재다운로드")
    args = ap.parse_args()

    html = fetch_html(use_cache=not args.no_cache)
    recs = parse_records(html)
    print(f"파싱된 전체 레코드: {len(recs)}")

    def keep(r: dict) -> bool:
        if args.version != "all" and r["version"].upper() != args.version.upper():
            return False
        if args.lang != "all" and r["lang"] != args.lang:
            return False
        return bool(r["term"] and r["definition"])

    recs = [r for r in recs if keep(r)]

    # 중복 제거: 같은 표제어라도 '정의가 다르면' 동형이의어이므로 보존한다.
    #   (term, hanja) 만으로 묶으면 동형이의 199여 건이 정의 손실로 삭제됨.
    seen, uniq = set(), []
    for r in recs:
        k = (r["term"], r["hanja"], r["definition"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cats = Counter(r["category"] for r in uniq)
    print(f"필터({args.version}/{args.lang}) 후 저장: {len(uniq)}개 → {OUT}")
    print("분류 상위:", dict(cats.most_common(12)))


if __name__ == "__main__":
    main()
