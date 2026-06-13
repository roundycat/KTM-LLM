# -*- coding: utf-8 -*-
"""
manifest.json 을 읽어 각 게시물의 PDF를 내려받고(raw/) 페이지를 PNG로 렌더(img/)한다.
파일명에서 회차/교시/구분을 추론해 일관된 폴더 구조로 정리.

사용:
  python scripts/fetch_render.py --manifest manifest.json --scale 2.2
"""
import os, re, json, argparse, time
import requests
import pypdfium2 as pdfium

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://www.kuksiwon.or.kr/CollectOfQuestions/brd/m_116/list.do",
}

RAW = "raw"
IMG = "img"


def slugify_title(title):
    """게시물 제목에서 'NN회 직종' 형태의 짧은 슬러그 추출."""
    m_round = re.search(r"제?\s*(\d+)\s*회", title)
    m_job = re.search(r"(한의사|한약사|의사|치과의사|약사|간호사)", title)
    rnd = m_round.group(1) if m_round else "x"
    job = m_job.group(1) if m_job else "직종"
    return f"{job}_{rnd}회"


def classify_file(fname):
    """첨부 파일명에서 교시/구분 추론."""
    if re.search(r"답안|정답", fname):
        return "답안"
    m = re.search(r"(\d+)\s*교시", fname)
    if m:
        return f"{m.group(1)}교시"
    return re.sub(r"[\\/:*?\"<>|.]+", "_", fname)[:30]


def download(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return os.path.getsize(out)
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            with open(out, "wb") as f:
                f.write(r.content)
            return len(r.content)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def render_pdf(pdf_path, out_dir, prefix, scale):
    os.makedirs(out_dir, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    n = len(pdf)
    paths = []
    for i in range(n):
        out = os.path.join(out_dir, f"{prefix}_p{i+1:02d}.png")
        if not os.path.exists(out):
            pdf[i].render(scale=scale).to_pil().save(out)
        paths.append(out)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--scale", type=float, default=2.2)
    ap.add_argument("--only", help="슬러그 필터 (예: 한의사_81회)")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    summary = []
    for post in manifest:
        slug = slugify_title(post["title"])
        if args.only and args.only not in slug:
            continue
        raw_dir = os.path.join(RAW, slug)
        img_dir = os.path.join(IMG, slug)
        os.makedirs(raw_dir, exist_ok=True)
        print(f"=== {slug} (seq={post['seq']}) ===")
        for fobj in post["files"]:
            kind = classify_file(fobj["filename"])
            pdf_path = os.path.join(raw_dir, f"{kind}.pdf")
            size = download(fobj["url"], pdf_path)
            pages = render_pdf(pdf_path, img_dir, kind, args.scale)
            print(f"    {kind:8s} {size:>9d}B  ->  {len(pages)} pages")
            summary.append({"slug": slug, "kind": kind, "pdf": pdf_path,
                            "pages": pages})
        time.sleep(0.3)

    with open("render_index.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[*] 저장: render_index.json")


if __name__ == "__main__":
    main()
