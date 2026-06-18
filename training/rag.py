# -*- coding: utf-8 -*-
"""용어 RAG — 문제에 등장하는 KIOM 표준 용어 정의를 검색해 프롬프트에 주입.

파인튜닝 없이 '글로벌 모델에 용어 학습' 단계를 무료로 대체한다(추론 시 지식 주입).
검색은 가벼운 어휘 매칭이다: 용어(국문) 또는 한자가 '문제+보기' 텍스트에 부분
문자열로 등장하면 후보로 보고, 더 구체적인(긴) 용어를 우선해 상위 K개를 주입한다.

  - 임베딩/외부 API 불필요 → 완전 무료, 의존성 없음, 결정론적(재현 가능).
  - 1글자 용어는 과매칭을 유발하므로 제외(RAG_MIN_TERM_LEN).
"""
from __future__ import annotations
import json
import re

from config import (
    TERMINOLOGY_DATASET, RAG_TOP_K, RAG_MIN_TERM_LEN, RAG_INJECT_HEADER,
)


def load_terms() -> list[dict]:
    if not TERMINOLOGY_DATASET.exists():
        return []
    with open(TERMINOLOGY_DATASET, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# 한국어 표제어 중 일반어와 겹쳐 오매칭을 유발하는 흔한 단어(부분문자열 노이즈).
#   예: "조사하고자"→고자, "행정구역"→구역, "표본추출"→표본, "다음에서"→다음
KOREAN_STOPWORDS = {
    "다음", "증상", "약물", "조사", "표본", "구역", "사하", "고자", "경우", "방법",
    "사용", "문제", "설명", "부위", "정도", "이용", "대조", "포함", "구성", "발생",
    "관찰", "측정", "선택", "결과", "기능", "작용", "상태", "부분", "전체", "이상",
}


class TermIndex:
    """용어 매칭 인덱스.

    정밀도를 위해 '한자 매칭'을 우선한다(의학 문항은 핵심 용어를 한자로 포함하는
    경우가 많고, 한자는 일반어와 거의 겹치지 않아 오매칭이 적다). 부족하면 한국어
    표제어로 보충하되, 흔한 일반어(KOREAN_STOPWORDS)는 제외한다.
    """

    def __init__(self, terms: list[dict] | None = None, min_len: int = RAG_MIN_TERM_LEN):
        if terms is None:
            terms = load_terms()
        self.min_len = min_len
        # 한자: 일반어와 거의 안 겹쳐 단순 부분문자열로 충분.
        self.hanja_entries: list[tuple[str, int, dict]] = []
        # 한국어: 더 긴 단어의 일부에 오매칭되지 않도록 한글 경계 정규식으로 매칭.
        #   예) '대장균'에 '대장', '간기능검사'에 '간기', '심하다'에 '심하' 매치 방지.
        self.korean_entries: list[tuple[object, int, dict]] = []
        for r in terms:
            if not (r.get("definition") or "").strip():
                continue
            hanja = (r.get("hanja") or "").strip()
            term = (r.get("term") or "").strip()
            if len(hanja) >= min_len:
                self.hanja_entries.append((hanja, len(hanja), r))
            if len(term) >= min_len and term not in KOREAN_STOPWORDS:
                pat = re.compile(r"(?<![가-힣])" + re.escape(term) + r"(?![가-힣])")
                self.korean_entries.append((pat, len(term), r))
        # 더 긴(구체적인) 키를 먼저
        self.hanja_entries.sort(key=lambda e: e[1], reverse=True)
        self.korean_entries.sort(key=lambda e: e[1], reverse=True)

    def __len__(self) -> int:
        return len(self.hanja_entries) + len(self.korean_entries)

    def retrieve(self, text: str, k: int = RAG_TOP_K) -> list[dict]:
        out, picked = [], set()

        def add(r):
            rid = (r.get("term"), r.get("hanja"))
            if rid in picked:
                return
            picked.add(rid)
            out.append(r)

        for key, _, r in self.hanja_entries:    # 1) 한자 우선(고정밀, 부분문자열)
            if len(out) >= k:
                break
            if key in text:
                add(r)
        for pat, _, r in self.korean_entries:   # 2) 한국어 보충(한글 경계 매칭)
            if len(out) >= k:
                break
            if pat.search(text):
                add(r)
        return out


def format_injection(terms: list[dict]) -> str:
    """검색된 용어들을 프롬프트에 넣을 참고 블록 문자열로 만든다."""
    if not terms:
        return ""
    lines = [RAG_INJECT_HEADER]
    for r in terms:
        head = r.get("headword") or r.get("term")
        df = (r.get("definition") or "").strip()
        syn = (r.get("synonyms") or "").strip()
        line = f"- {head}: {df}"
        if syn:
            line += f" (동의어: {syn})"
        lines.append(line)
    return "\n".join(lines)


def question_text(row: dict) -> str:
    """매칭 대상 텍스트(문제 + 보기)."""
    return row.get("question", "") + " " + " ".join(row.get("options", []))
