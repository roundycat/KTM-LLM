# -*- coding: utf-8 -*-
"""전 14과목 용어RAG Δ 마크다운 표 생성(법규 2라벨 병합)."""
import json, os
from collections import defaultdict
RES = r"D:\정하민\한의학 문제 데이터\results"
M = [("Qwen2.5","eval_글로벌_base_qwen2.5_7b.json","eval_글로벌_rag_qwen2.5_7b.json"),
     ("EXAONE 3.5","eval_한국형_base_exaone3.5_7.8b.json","eval_한국형_rag_exaone3.5_7.8b.json"),
     ("SOLAR 10.7B","eval_한국형_base_solar_10.7b.json","eval_한국형_rag_solar_10.7b.json"),
     ("Llama3.1","eval_글로벌_base_llama3.1_8b.json","eval_글로벌_rag_llama3.1_8b.json")]
def norm(s): return "보건의약관계법규" if "법규" in s else s
def ps(path):
    d=json.load(open(os.path.join(RES,path),encoding="utf-8")); c=defaultdict(int); n=defaultdict(int)
    for r in d.get("details",[]):
        s=norm(r.get("과목","미상")); n[s]+=1; c[s]+=1 if r.get("correct") else 0
    return c,n
data={}; nmap={}
for name,bf,rf in M:
    bc,bn=ps(bf); rc,rn=ps(rf); data[name]=(bc,bn,rc,rn)
    for s in bn: nmap[s]=bn[s]
names=[m[0] for m in M]
subs=sorted(nmap, key=lambda x:-nmap[x])
print("| 과목 | n | " + " | ".join(names) + " |")
print("|---|---:|" + "---:|"*len(names))
for s in subs:
    row=f"| {s} | {nmap[s]} |"
    for nm in names:
        bc,bn,rc,rn=data[nm]
        d=100*rc[s]/rn[s]-100*bc[s]/bn[s] if bn.get(s) and rn.get(s) else None
        cell = f"{d:+.1f}" if d is not None else "-"
        if d is not None and abs(d)>=10: cell=f"**{cell}**"
        row+=f" {cell} |"
    print(row)
