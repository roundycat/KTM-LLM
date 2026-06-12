# -*- coding: utf-8 -*-
"""해설 검증용 배치 생성: 텍스트 517문항을 문제+정답+해설과 함께 35배치로."""
import json, os, glob
ROOT = r"D:\정하민\한의학 문제 데이터"
SRC = os.path.join(ROOT, r"dataset\한의학_문제_해설.jsonl")
INDIR = r"D:\tmp\verify_in"
OUTDIR = r"D:\tmp\verify_out"
os.makedirs(INDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)
for f in glob.glob(os.path.join(INDIR, "*.jsonl")): os.remove(f)
for f in glob.glob(os.path.join(OUTDIR, "*.jsonl")): os.remove(f)

rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
items = [r for r in rows if not r.get("has_figure", False) and r.get("해설")]
BATCH = 15
n = 0
for i in range(0, len(items), BATCH):
    chunk = items[i:i+BATCH]
    idx = i // BATCH
    with open(os.path.join(INDIR, f"vbatch_{idx:03d}.jsonl"), "w", encoding="utf-8") as f:
        for r in chunk:
            f.write(json.dumps({
                "source": r["source"], "교시": r["교시"], "과목": r.get("과목"),
                "번호": r["번호"], "question": r["question"], "options": r["options"],
                "answer": r["answer"], "answer_text": r.get("answer_text"), "해설": r["해설"],
            }, ensure_ascii=False) + "\n")
    n += 1
print(f"BATCHES={n} ITEMS={len(items)} INDIR={INDIR} OUTDIR={OUTDIR}")
