# -*- coding: utf-8 -*-
"""
PDF 페이지를 고해상도로 렌더한 뒤 세로 방향으로 N개 스트립(겹침 포함)으로 잘라
비전 모델 다운스케일(~1568px) 한계를 우회한다. 작은 한자/사례문 가독성을 크게 높임.

사용:
  python scripts/tile_render.py --pdf "raw/한의사_81회/2교시.pdf" --out tiles/한의사_81회/2교시 \
      --scale 4.0 --strips 3 --overlap 0.08 --first 2
"""
import os, argparse
import pypdfium2 as pdfium
from PIL import Image

# 비전 입력 긴 변 권장 상한(가독성 유지). 폭은 이 값으로 맞춤.
TARGET_W = 1500


def _vsplit(img, strips, overlap, tag, w):
    """이미지를 세로 strips개로 (겹침 포함) 자른다."""
    h = img.size[1]
    strip_h = h // strips
    ov = int(strip_h * overlap)
    out = []
    for i in range(strips):
        top = max(0, i * strip_h - ov)
        bot = min(h, (i + 1) * strip_h + ov)
        out.append((f"{tag}s{i+1}", img.crop((0, top, w, bot))))
    return out


def tile_page(pil, strips, overlap, columns):
    """columns=2면 좌/우 컬럼으로 먼저 분할 후 각 컬럼을 세로 strips개로 자른다."""
    w, h = pil.size
    if columns >= 2:
        # 컬럼 분할: 가운데 약간 겹치게
        gut = int(w * 0.50)
        cov = int(w * 0.03)
        cols = [("c1", pil.crop((0, 0, gut + cov, h))),
                ("c2", pil.crop((gut - cov, 0, w, h)))]
    else:
        cols = [("", pil)]

    out = []
    for ctag, cimg in cols:
        cw = cimg.size[0]
        if cw > TARGET_W:
            sc = TARGET_W / cw
            cimg = cimg.resize((TARGET_W, int(cimg.size[1] * sc)), Image.LANCZOS)
            cw = TARGET_W
        out.extend(_vsplit(cimg, strips, overlap, ctag, cw))

    # 비전 API 제약: 다중 이미지 요청 시 장변 2000px 초과 금지 -> 1980px로 캡
    capped = []
    for tag, im in out:
        longest = max(im.size)
        if longest > 1980:
            sc = 1980 / longest
            im = im.resize((int(im.size[0] * sc), int(im.size[1] * sc)), Image.LANCZOS)
        capped.append((tag, im))
    return capped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=4.5)
    ap.add_argument("--strips", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.12)
    ap.add_argument("--columns", type=int, default=2)
    ap.add_argument("--first", type=int, default=2,
                    help="이 페이지 번호(1-base)부터 처리 (표지 건너뛰기)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pdf = pdfium.PdfDocument(args.pdf)
    n = len(pdf)
    total = 0
    for pi in range(args.first - 1, n):
        pil = pdf[pi].render(scale=args.scale).to_pil()
        for tag, strip in tile_page(pil, args.strips, args.overlap, args.columns):
            out = os.path.join(args.out, f"p{pi+1:02d}_{tag}.png")
            strip.save(out)
            total += 1
    print(f"[*] {args.pdf}: {n}p -> {total} strips in {args.out}")


if __name__ == "__main__":
    main()
