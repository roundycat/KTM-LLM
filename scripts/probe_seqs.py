# -*- coding: utf-8 -*-
"""국시원 출제자료 게시판 seq 범위를 훑어 한의사/한약사 등 기출 게시물이
과년도까지 남아있는지 탐침한다."""
import re, sys, time, requests
from bs4 import BeautifulSoup

VIEW = "https://www.kuksiwon.or.kr/CollectOfQuestions/brd/m_116/view.do"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": VIEW}
KW = ["한의사", "한약사"]

lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (120, 195)
hits = []
for seq in range(lo, hi + 1):
    try:
        r = requests.get(VIEW, params={"seq": seq, "itm_seq_1": 0, "itm_seq_2": 0,
                                       "multi_itm_seq": 0}, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "lxml")
        txt = soup.get_text(" ", strip=True)
        # 제목 추정: 'NNNN년도 제NN회 ... 기출문제'
        m = re.search(r"(\d{4}년도\s*제?\s*\d+회[^\n]{0,30}?(?:한의사|한약사|의사|약사|간호사|치과의사)[^\n]{0,20}기출문제)", txt)
        title = m.group(1) if m else ""
        files = len(soup.select("a[href*='down.do']"))
        mark = "  <<<" if any(k in title for k in KW) else ""
        if title or files:
            print(f"seq={seq:>4} files={files} {title[:50]}{mark}")
        if any(k in title for k in KW):
            hits.append((seq, title, files))
    except Exception as e:
        print(f"seq={seq} ERR {e}")
    time.sleep(0.25)

print("\n한의/한약 매칭:", len(hits))
for h in hits:
    print(h)
