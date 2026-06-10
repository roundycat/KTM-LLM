# -*- coding: utf-8 -*-
"""
국시원(한국보건의료인국가시험원) 출제자료 게시판에서
한의사/한약사 등 한의학 관련 기출문제 게시물과 첨부파일을 자동 탐색한다.

게시판: CollectOfQuestions  (m_116)
목록 URL : https://www.kuksiwon.or.kr/CollectOfQuestions/brd/m_116/list.do?page=N
상세 URL : .../view.do?seq=SEQ
다운로드 : .../down.do?brd_id=CollectOfQuestions&seq=SEQ&data_tp=A&file_seq=K
"""
import re, json, sys, time, argparse
import requests
from bs4 import BeautifulSoup

BASE = "https://www.kuksiwon.or.kr"
LIST = BASE + "/CollectOfQuestions/brd/m_116/list.do"
VIEW = BASE + "/CollectOfQuestions/brd/m_116/view.do"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": LIST,
}

# 관심 직종 키워드 (한의학 관련). 필요시 추가.
DEFAULT_KEYWORDS = ["한의사", "한약사"]


def get(url, **kw):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30, **kw)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2)


def parse_list_page(html):
    """목록 페이지에서 (seq, title, date) 추출."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for a in soup.select("a[href*='view.do']"):
        href = a.get("href", "")
        m = re.search(r"seq=(\d+)", href)
        if not m:
            # onclick 방식일 수 있음
            continue
        seq = int(m.group(1))
        title = a.get_text(" ", strip=True)
        if title:
            rows.append((seq, title))
    # onclick="fn_view('123')" 패턴도 처리
    for a in soup.find_all(attrs={"onclick": re.compile(r"\d{2,}")}):
        oc = a.get("onclick", "")
        m = re.search(r"(\d{2,})", oc)
        if m:
            seq = int(m.group(1))
            title = a.get_text(" ", strip=True)
            if title:
                rows.append((seq, title))
    # 중복 제거 (seq 기준)
    seen, out = set(), []
    for seq, title in rows:
        if seq not in seen:
            seen.add(seq)
            out.append((seq, title))
    return out


def get_total_pages(html):
    soup = BeautifulSoup(html, "lxml")
    txt = soup.get_text(" ", strip=True)
    m = re.search(r"Total\s*:?\s*([\d,]+)", txt)
    total = int(m.group(1).replace(",", "")) if m else None
    return total


def scan_listing(max_pages=20, keywords=None):
    """게시판 전체를 페이지별로 훑어 키워드 매칭 게시물 수집."""
    keywords = keywords or DEFAULT_KEYWORDS
    found = []
    for page in range(1, max_pages + 1):
        r = get(LIST, params={"page": page})
        rows = parse_list_page(r.text)
        if not rows:
            break
        for seq, title in rows:
            if any(k in title for k in keywords):
                found.append({"seq": seq, "title": title, "page": page})
        time.sleep(0.5)
    # seq 중복 제거
    uniq = {}
    for f in found:
        uniq[f["seq"]] = f
    return sorted(uniq.values(), key=lambda x: -x["seq"])


def parse_attachments(seq):
    """상세 페이지에서 첨부파일 (file_seq, filename) 목록 추출."""
    r = get(VIEW, params={"seq": seq, "itm_seq_1": 0, "itm_seq_2": 0, "multi_itm_seq": 0})
    soup = BeautifulSoup(r.text, "lxml")
    files = []
    for a in soup.select("a[href*='down.do']"):
        href = a.get("href", "")
        m = re.search(r"file_seq=(\d+)", href)
        if not m:
            continue
        fseq = int(m.group(1))
        fname = a.get_text(" ", strip=True) or f"file_{fseq}"
        dl = href if href.startswith("http") else BASE + "/CollectOfQuestions/brd/m_116/" + href.lstrip("./")
        files.append({"file_seq": fseq, "filename": fname, "url": dl})
    # 제목
    title = ""
    h = soup.find(["h3", "h4", "td", "th"], string=re.compile("한의사|한약사"))
    if h:
        title = h.get_text(" ", strip=True)
    return {"seq": seq, "title": title, "files": files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=20)
    ap.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS)
    ap.add_argument("--out", default="manifest.json")
    args = ap.parse_args()

    print(f"[*] 게시판 스캔 (키워드={args.keywords}, 최대 {args.max_pages}페이지)")
    posts = scan_listing(args.max_pages, args.keywords)
    print(f"[*] 매칭 게시물 {len(posts)}건")

    manifest = []
    for p in posts:
        info = parse_attachments(p["seq"])
        info["title"] = info["title"] or p["title"]
        manifest.append(info)
        print(f"    seq={p['seq']:>4}  files={len(info['files'])}  {info['title']}")
        time.sleep(0.4)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[*] 저장: {args.out}")


if __name__ == "__main__":
    main()
