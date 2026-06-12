# -*- coding: utf-8 -*-
"""2차 강화 검증 확정 9건 교정 적용(공식정답 절대 불변) + 통합 검증 리포트 재생성."""
import json, os
ROOT = r"D:\정하민\한의학 문제 데이터"
SRC = os.path.join(ROOT, r"dataset\한의학_문제_해설.jsonl")
REPORT = os.path.join(ROOT, r"dataset\해설_검증_리포트.md")
P1 = r"D:\tmp\claude\D----------------\46815c06-5503-4947-9834-2ab3326e4316\tasks\wjygid9dm.output"
P2 = r"D:\tmp\claude\D----------------\46815c06-5503-4947-9834-2ab3326e4316\tasks\w7adg6lcl.output"
c1 = json.load(open(P1, encoding="utf-8"))["result"]
c2 = json.load(open(P2, encoding="utf-8"))["result"]

rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
index = {(r["source"], r["교시"], r["번호"]): r for r in rows}

# pass-2 교정문(JSON에서 가져옴): 3번, 13번은 제공된 final_correction 사용
p2map = {(c["source"], c["period"], c["qnum"]): c for c in c2["confirmed"]}
def fc(src, per, q):
    return p2map[(src, per, q)]["final_correction"].strip()

# 전체 교체(공식정답 유지 확인 완료)
FULL = {
    ("한의사_81회", 2, 3): fc("한의사_81회", 2, 3),    # 축혈=3 유지, 열입혈실(5) 감별
    ("한의사_81회", 4, 13): fc("한의사_81회", 4, 13),  # 귀비탕=2 지지(중대)
    ("한약사_27회", 2, 51): (  # 정답 4(읽기 상오) 유지 + 相畏/相惡 정의 교정 + 소스표기 플래그
        "이 문항은 '병용투여로 약물의 효능이 감소하는 것'을 묻는다. 표준 중약학(中藥學)의 칠정(七情) "
        "정의상 한 약물이 다른 약물의 치료 효능을 감소·약화시키는 관계는 본래 相惡(상오)이며, 확정 정답은 "
        "4번(읽기 '상오')이다. 相畏(상외)는 효능 감소가 아니라 한 약물의 독성·부작용이 다른 약물에 의해 "
        "경감·억제되는 관계(예: 半夏畏生薑)이고, 相殺(상쇄)는 한 약이 다른 약의 독성을 없애는 관계, 相使(상사)는 "
        "주약의 효능을 보조해 효과를 증강하는 관계다. 따라서 '효능 감소'는 相惡 개념이며 相畏(독성 경감)와 "
        "혼동하지 않아야 한다. ※ 보기 3·4의 한자 표기(相惡/相畏)가 혼재되어 있어 원본 표기 확인이 필요하다."
    ),
    ("한의사_81회", 2, 100): (  # 정답 4 유지, 자기모순 제거
        "연명의료결정법령상 말기환자 진단 시 고려해야 할 기준에는 임상적 증상, 종전의 진료 경과, 다른 진료 "
        "방법의 가능 여부 등이 포함된다. 본 문항은 '기준이 아닌 것'을 묻고 있으며, 확정 정답은 4번 "
        "'약물 투여에 따른 개선 정도'로, 이는 해당 법령이 말기환자 진단 고려기준으로 열거한 항목에 포함되지 "
        "않는다. 말기환자 진단은 환자의 주관적 요소가 아니라 의학적·객관적 근거에 따라 이루어진다."
    ),
}

# 부분 수정(정답 불변, 사실만 교정)
MINOR = {
    ("한의사_81회", 2, 87): (
        "1·2·4·5번 중 부당 금품수수, 진료기록부 거짓작성, 무자격자 고용 의료행위 등은 자격정지",
        "1·2·5번의 부당 금품수수, 진료기록부 거짓작성, 무자격자 고용 의료행위 등은 자격정지",
    ),
    ("한의사_81회", 3, 20): (
        "부교감 항진과 심독성을 일으키며",
        "전압의존성 나트륨 채널을 지속적으로 활성화시켜 신경·심근의 흥분과 심독성을 일으키며",
    ),
    ("한의사_81회", 4, 53): (
        "'인사이존위지의(因思而遠慕謂之慮)'",
        "'인사이원모위지려(因思而遠慕謂之慮)'",
    ),
    ("한약사_27회", 1, 27): (
        "결정형 수산화칼슘 단정(單晶)을 포함하는 조직 특징도 감초에 부합한다.",
        "유세포에 결정형 단정(식물 조직의 특징적 결정은 일반적으로 수산칼슘=옥살산칼슘이며, 지문의 '수산화칼슘'은 원본 표기 확인이 필요)과 전분립을 갖는 조직 특징도 감초에 부합한다.",
    ),
    ("한약사_27회", 2, 8): (
        "patulin과 ochratoxin A, citreoviridin은 각각 신장·간·신경 독성을 주로 나타내고",
        "patulin은 위장관·세포(면역) 독성, ochratoxin A는 신장 독성, citreoviridin은 신경 독성을 주로 나타내고",
    ),
}

applied, misses = [], []
for k, new in FULL.items():
    r = index.get(k)
    if r and new:
        r["해설"] = new; applied.append((k, "전체교체"))
    else:
        misses.append((k, "FULL 실패"))
for k, (old, new) in MINOR.items():
    r = index.get(k)
    if r and old in r["해설"]:
        r["해설"] = r["해설"].replace(old, new); applied.append((k, "부분수정"))
    else:
        misses.append((k, "MINOR old 미일치"))

with open(SRC, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"2차 교정 적용 {len(applied)}/9")
for k, h in applied: print("  OK", k, h)
for k, w in misses: print("  MISS", k, w)

# ── 통합 리포트 재생성(1차 + 2차) ──
def sev_rank(s): return {"중대":0,"경미":1}.get(s,9)
c1c = sorted(c1["confirmed"], key=lambda c:(sev_rank(c["final_severity"]), c["source"], c["period"], c["qnum"]))
c2c = sorted(c2["confirmed"], key=lambda c:(sev_rank(c["final_severity"]), c["source"], c["period"], c["qnum"]))
# distinct 오류 문항 수(51은 두 패스에 걸쳐 서로 다른 오류)
keys = set((c["source"],c["period"],c["qnum"]) for c in c1["confirmed"]) | set((c["source"],c["period"],c["qnum"]) for c in c2["confirmed"])
n_distinct = len(keys)
acc = 100*(517-n_distinct)/517

L=[]
L.append("# 답안 해설 — 의학 정확성 검증 리포트\n")
L.append("> Claude Opus 4.8 다중 에이전트가 **517개 해설 전수**를 적대적으로 검증했습니다. **2단계 검증**을 거쳤습니다: "
         "1차(배치당 1리뷰어) → 2차 강화(배치당 **독립 리뷰어 2명 union**으로 recall 상향). 작성 기준일 2026-06-12.\n")
L.append("## 1. 결과 요약 (정직)\n")
L.append("| 패스 | 방식 | 검증 | 1차 플래그 | 확정 오류 | 비고 |")
L.append("|---|---|---:|---:|---:|---|")
L.append(f"| 1차 | 배치당 리뷰어 1명 + 독립 재검증 | 517 | {c1['flagged']} | **{len(c1['confirmed'])}** (중대 {sum(1 for c in c1['confirmed'] if c['final_severity']=='중대')}) | 47 에이전트 |")
L.append(f"| 2차 강화 | 배치당 **리뷰어 2명 union** + 독립 확정 | 517 | {c2['flagged']} | **{len(c2['confirmed'])}** (중대 {sum(1 for c in c2['confirmed'] if c['final_severity']=='중대')}) | 92 에이전트, **1차가 놓친 신규** |")
L.append("")
L.append(f"- **두 패스 합산 = 서로 다른 오류 문항 {n_distinct}건**(51번은 1·2차에서 각기 다른 오류로 2회 교정). **모두 교정 완료.**")
L.append(f"- **실측 해설 정확도 ≈ {acc:.1f}% ({517-n_distinct}/517).** ⚠️ 솔직히: 1차 단일검토만으론 98.5%로 보였으나, **2차 강화(리뷰어 2명)에서 {len(c2['confirmed'])}건이 추가 발견**되어 실제는 더 낮았다. **강한 검토일수록 더 찾는다** — 잔존 오류 가능성 배제 불가, 전문가 최종감수 권장.")
L.append("- 정답(answer)은 국시원 공식이라 **불변**. 모든 교정은 공식 정답을 그대로 두고 해설 문장만 바로잡았다.\n")
L.append("## 2. 방법\n")
L.append("- **적대 검증**: 공식 정답을 불변 진실로 두고 **해설의 의학적 사실오류**(본초 효능·귀경, 경혈, 방제 구성, 법규 조문, 생리·병리 기전, 수치, 오답 분석 논리)만 점검.")
L.append("- **2차 강화**: 배치마다 중점이 다른 독립 리뷰어 2명(A: 본초·방제·경혈 / B: 수치·법규·생리병리·논리)을 돌려 플래그 **합집합(union)** → recall 향상. 이후 **독립 최종 판정자**가 거짓양성 제거.")
L.append("- 하니스: `scripts/verify_explanations_workflow.js`(1차)·동 2차 강화 워크플로.\n")

def block(title, items, off):
    L.append(f"## {title}\n")
    for i,c in enumerate(items, 1):
        cur = index.get((c["source"],c["period"],c["qnum"]))
        L.append(f"### {off+i}. {c['source']} {c['period']}교시 {c['qnum']}번 · {c.get('subject','')} — **{c['final_severity']}**")
        L.append(f"- **오류**: {c.get('first_issue', c.get('union_issue',''))}")
        L.append(f"- **판정**: {c['reason']}")
        L.append("")
block("3. 1차 확정 오류 (8건, 교정 완료)", c1c, 0)
block("4. 2차 강화 신규 확정 오류 (9건, 교정 완료)", c2c, 8)
L.append("## 5. 소스(전사) 데이터 주의 — 별도 확인 필요\n")
L.append("- **한약사 27회 2교시 51번**: '효능 감소'를 묻는데 보기 3=상오(相惡)·보기 4=상오(相畏)로 한자 표기가 혼재(공식정답 4). "
         "전사 단계의 한자 혼입 가능성 → 원본 보기 표기 확인 필요. 해설은 공식정답(4)을 유지하되 표준 정의(효능감소=相惡)를 명시.")
L.append("- **한약사 27회 1교시 27번**: 지문이 '수산화칼슘의 단정'이라 하나 식물 유세포의 특징적 결정은 수산칼슘(옥살산칼슘)이다. 지문 표기 확인 필요(정답 감초는 불변).")
L.append("")
L.append("## 6. 주의\n")
L.append("- 해설은 LLM 생성물이다. 2단계 다중에이전트 검증으로 17건(문항 기준 "+str(n_distinct)+"건)을 교정했으나 **잔존 오류 가능성은 배제할 수 없으며**, 배포 전 한의학 전문가 최종 감수를 권장한다.")
L.append("- 원본 해설·교정 이력은 git에 보존된다.")
open(REPORT,"w",encoding="utf-8").write("\n".join(L)+"\n")
print("리포트 재생성:", REPORT, f"| distinct {n_distinct} | acc {acc:.1f}%")
