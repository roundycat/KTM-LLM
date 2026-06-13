# -*- coding: utf-8 -*-
"""전 로컬 모델 × (plain/graph/hae) × 517문항 순차 평가 큐. 재개 가능(완료 모델 건너뜀)."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval_out"); os.makedirs(OUT, exist_ok=True)
MODELS = ["qwen2.5:7b", "gemma2:9b", "exaone3.5:7.8b", "llama3.1:8b", "mistral:7b", "solar:10.7b"]
for m in MODELS:
    o = os.path.join(OUT, "grageval_" + m.replace(":", "_").replace(".", "_") + ".json")
    if os.path.exists(o):
        print("[skip 완료됨]", m, flush=True); continue
    print(f"\n===================== MODEL {m} =====================", flush=True)
    r = subprocess.run([sys.executable, "-u", "run_eval_local.py", "--model", m, "--out", o])
    print(f"[done] {m} exit={r.returncode} -> {o}", flush=True)
print("\n========== ALL MODELS DONE ==========", flush=True)
