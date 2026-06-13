# -*- coding: utf-8 -*-
"""모델 평가 — 한의학 문제를 풀게 해서 정답률(전체/과목별)을 비교.

대상 모델은 '역할(role)'과 '방식(variant)'으로 구분한다.
  - role    : 한국형(로컬 EXAONE 등) / 글로벌(범용 Qwen·OpenAI·Gemini 등)
  - variant : base(그대로) / rag(용어 주입) / ft(파인튜닝 모델)
  - provider: local(Ollama) / openai(OpenAI 또는 OpenAI 호환 클라우드)

무료 실험(RAG) 기본 동작(인자 없이 실행):
  - 한국형(LOCAL_MODEL) base
  - 글로벌(GLOBAL_MODEL) base + rag(용어 주입)
  - 평가 후 '한국형 vs 글로벌' 평균과 '글로벌 향상(rag-base)' 출력

사용법:
    python training/evaluate.py                 # 검증셋, 한국형+글로벌(+RAG)
    python training/evaluate.py --all           # 전체 텍스트 문항
    python training/evaluate.py --no-rag        # RAG 변형 제외(순수 base 비교)
    python training/evaluate.py --rag-local     # 한국형에도 RAG 적용
    python training/evaluate.py --models ft:gpt-4o-...:hani-term   # 글로벌에 파인튜닝 모델 추가
    python training/evaluate.py --vote 3 --rag-local    # 환각 완화: 3회 생성·2회 합의 시만 인정
"""
from __future__ import annotations
import argparse
import json
import re
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    FULL_DATASET, RESULTS_DIR, SYSTEM_PROMPT, ANSWER_INSTRUCTION, COT_ANSWER_INSTRUCTION,
    LOCAL_MODEL, GLOBAL_MODEL, GLOBAL_PROVIDER, ROLE_KOREAN, ROLE_GLOBAL,
    GPT4_FINETUNED_MODEL, GPT5_FINETUNED_MODEL,
    VAL_RATIO, RANDOM_SEED, EVAL_CONCURRENCY, EVAL_MAX_TOKENS,
    EVAL_MAX_TOKENS_REASONING, EVAL_MAX_TOKENS_LOCAL, EVAL_MAX_TOKENS_COT,
    EVAL_REASONING_EFFORT, EVAL_SC_SAMPLES, EVAL_SC_TEMPERATURE,
    REQUEST_TIMEOUT, RAG_TOP_K, is_reasoning_model, make_client,
    PROVIDER_OPENAI, PROVIDER_LOCAL,
)
from collections import Counter
from rag import TermIndex, format_injection, question_text

CIRCLED = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}

_CLIENTS: dict[str, object] = {}


def get_client(provider: str):
    if provider not in _CLIENTS:
        _CLIENTS[provider] = make_client(provider)
    return _CLIENTS[provider]


def load_rows() -> list[dict]:
    rows = []
    with open(FULL_DATASET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if not r.get("has_figure", False)]


def select_eval_rows(use_all: bool) -> list[dict]:
    rows = load_rows()
    if use_all:
        return rows
    random.seed(RANDOM_SEED)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_RATIO))
    return rows[:n_val]


def build_prompt(row: dict, rag: bool = False, index: TermIndex | None = None,
                 cot: bool = False) -> str:
    lines = [row["question"].strip(), ""]
    for i, opt in enumerate(row["options"], start=1):
        lines.append(f"{i}. {opt.strip()}")
    lines.append("")
    lines.append(COT_ANSWER_INSTRUCTION if cot else ANSWER_INSTRUCTION)
    prompt = "\n".join(lines)
    if rag and index is not None and len(index):
        inj = format_injection(index.retrieve(question_text(row), k=RAG_TOP_K))
        if inj:
            prompt = inj + "\n\n" + prompt
    return prompt


def _to_num(ch: str) -> int:
    return CIRCLED[ch] if ch in CIRCLED else int(ch)


def parse_answer(text: str) -> int | None:
    """모델 출력에서 정답 번호(1~5)를 견고하게 추출.

    우선순위(설명형 로컬 모델 대응):
      1) "정답/답 ... N" 결론 표기(숫자 또는 ①~⑤)        — 가장 신뢰
      2) "N번" 형태(마지막 것 = 결론부 우선)
      3) 원문자 ①~⑤ (텍스트 내 '마지막 등장' 우선)
      4) 독립 숫자 1~5 (나이·용량 등 다자리 숫자의 일부는 경계로 배제, 마지막 우선)
    """
    if not text:
        return None
    # 1) "정답/답 : N" (조사 은/는/이, 콜론 허용; 숫자 또는 원문자)
    m = re.search(r"(?:정답|답)\s*[은는이]?\s*[:：]?\s*([1-5①-⑤])", text)
    if m:
        return _to_num(m.group(1))
    # 2) "N번"(마지막)
    bm = re.findall(r"([1-5])\s*번", text)
    if bm:
        return int(bm[-1])
    # 3) 원문자(마지막 등장 우선 — dict 순서 편향 제거)
    circ = [(text.rfind(ch), num) for ch, num in CIRCLED.items() if ch in text]
    if circ:
        return max(circ)[1]
    # 4) 독립 숫자(마지막). 다자리 숫자의 일부(나이·용량)와 단위 카운터(3회·5세 등)는 배제.
    nums = re.findall(r"(?<![0-9])([1-5])(?![0-9회세개년월일명시분초차번째도])", text)
    return int(nums[-1]) if nums else None


def _chat_once(client, model_id: str, provider: str, messages: list,
               max_out: int, temperature: float) -> str:
    """1회 호출 → 응답 텍스트. provider 별 파라미터 차이를 흡수."""
    if provider == PROVIDER_LOCAL:
        resp = client.chat.completions.create(
            model=model_id, messages=messages,
            max_tokens=max_out, temperature=temperature, timeout=REQUEST_TIMEOUT)
        return resp.choices[0].message.content or ""

    # --- 클라우드(openai 호환) ---
    reasoning = is_reasoning_model(model_id)
    kwargs = dict(
        model=model_id, messages=messages,
        max_completion_tokens=EVAL_MAX_TOKENS_REASONING if reasoning else max_out,
        timeout=REQUEST_TIMEOUT,
    )
    if reasoning:
        kwargs["reasoning_effort"] = EVAL_REASONING_EFFORT
    else:
        kwargs["temperature"] = temperature
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        # reasoning_effort 미지원 모델일 때만 빼고 1회 재시도. 그 외는 상위에 위임.
        msg = str(e).lower()
        if "reasoning_effort" in kwargs and (
                "reasoning_effort" in msg or "unsupported" in msg or "parameter" in msg):
            kwargs.pop("reasoning_effort", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return resp.choices[0].message.content or ""


def ask(model_id: str, provider: str, row: dict, rag: bool = False,
        index: TermIndex | None = None, cot: bool = False, sc_n: int = 1,
        sc_min: int = 0) -> int | None:
    """한 문항을 물어 정답 번호(1~5)를 받아 파싱.

    - cot   : 근거 서술 후 '정답: N' (출력 토큰 예산 ↑)
    - sc_n  : >1 이면 temp>0 으로 sc_n 회 샘플 후 다수결(self-consistency)
    - sc_min: >0 이면 최다 득표가 sc_min 표 미만일 때 기권(None) — 환각 완화.
              예) sc_n=3, sc_min=2 → 3회 생성 중 2회 이상 일치해야 정답 인정.
    """
    client = get_client(provider)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(row, rag, index, cot)},
    ]
    if cot:
        max_out = EVAL_MAX_TOKENS_COT
    elif provider == PROVIDER_LOCAL:
        max_out = EVAL_MAX_TOKENS_LOCAL
    else:
        max_out = EVAL_MAX_TOKENS

    if sc_n and sc_n > 1:
        votes = []
        for _ in range(sc_n):
            a = parse_answer(_chat_once(client, model_id, provider, messages,
                                        max_out, EVAL_SC_TEMPERATURE))
            if a is not None:
                votes.append(a)
        if not votes:
            return None
        top, cnt = Counter(votes).most_common(1)[0]
        # 환각 완화: 합의(sc_min 표) 미달이면 기권 — 불확실할 때 찍지 않는다.
        if sc_min and cnt < sc_min:
            return None
        return top
    return parse_answer(_chat_once(client, model_id, provider, messages, max_out, 0))


def evaluate_model(target: dict, rows: list[dict], index: TermIndex | None = None) -> dict:
    mid, provider = target["id"], target["provider"]
    role, rag = target["role"], target.get("rag", False)
    cot, sc_n = target.get("cot", False), target.get("sc", 1)
    sc_min = target.get("sc_min", 0)
    variant = target.get("variant") or variant_of(mid, rag, cot, sc_n, sc_min)
    tag = f"{role}/{variant}"
    print(f"\n=== 평가: {mid} [{provider}] ({tag}, {len(rows)}문항) ===")
    correct = 0
    by_subject = defaultdict(lambda: [0, 0])
    details = []

    def worker(idx_row):
        idx, row = idx_row
        for attempt in range(3):
            try:
                pred = ask(mid, provider, row, rag, index, cot, sc_n, sc_min)
                return idx, pred, None
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return idx, None, str(e)
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as ex:
        futures = [ex.submit(worker, (i, r)) for i, r in enumerate(rows)]
        done = 0
        for fut in as_completed(futures):
            idx, pred, err = fut.result()
            row = rows[idx]
            gold = row["answer"]
            ok = (pred == gold)
            correct += int(ok)
            subj = row.get("과목", "미상")
            by_subject[subj][0] += int(ok)
            by_subject[subj][1] += 1
            details.append({
                "source": row.get("source"), "과목": subj, "번호": row.get("번호"),
                "gold": gold, "pred": pred, "correct": ok, "error": err,
            })
            done += 1
            if done % 25 == 0 or done == len(rows):
                print(f"  진행 {done}/{len(rows)} | 누적 정답 {correct}")

    acc = correct / len(rows) if rows else 0.0
    subj_acc = {s: round(v[0] / v[1], 4) for s, v in sorted(by_subject.items())}
    # 미응답(빈/파싱불가 출력): 오답과 구분해 기록(채점엔 오답으로 반영되지만 진단용).
    no_answer = sum(1 for d in details if d["pred"] is None and not d["error"])
    errored = sum(1 for d in details if d["error"])
    if no_answer or errored:
        label = "기권(합의미달)/미응답" if sc_min else "미응답/파싱실패"
        print(f"  ⚠ {label} {no_answer} · 호출오류 {errored} (오답으로 집계됨)")
    return {
        "model": mid, "provider": provider, "role": role, "rag": rag,
        "variant": variant, "cot": cot, "sc": sc_n, "sc_min": sc_min,
        "n": len(rows), "correct": correct, "accuracy": round(acc, 4),
        "no_answer": no_answer, "errored": errored,
        "by_subject": subj_acc, "details": details,
    }


def variant_of(model_id: str, rag: bool = False, cot: bool = False, sc: int = 1,
               sc_min: int = 0) -> str:
    """방식 라벨. 예: base / rag / cot / cot+sc5 / vote3 / rag+vote3 / ft.

    vote{N} = N회 생성 후 과반(sc_min표) 이상 일치 시만 인정(미달 시 기권) — 환각 완화.
    sc{N}   = N회 생성 후 단순 다수결(기권 없음).
    """
    parts = []
    if rag:
        parts.append("rag")
    if cot:
        parts.append("cot")
    if sc and sc > 1:
        parts.append(f"vote{sc}" if sc_min else f"sc{sc}")
    if not parts:
        return "ft" if str(model_id).startswith("ft:") else "base"
    return "+".join(parts)


def _spec(m, prov, role, rag=False, cot=False, sc=1, sc_min=0):
    return {"id": m, "provider": prov, "role": role, "rag": rag, "cot": cot, "sc": sc,
            "sc_min": sc_min, "variant": variant_of(m, rag, cot, sc, sc_min)}


def resolve_targets(args) -> list[dict]:
    """평가 대상 스펙(dict) 목록을 만든다.

    글로벌 모델엔 base 외에 RAG / CoT / CoT+SC 등 '방식' 변형을 더한다.
    """
    targets: list[dict] = []
    sc_n = getattr(args, "sc", 1) or 1
    use_cot = getattr(args, "cot", False)
    local_cot = getattr(args, "local_cot", False)
    rag_cot = getattr(args, "rag_cot", False)
    # 환각 완화: N회 생성 → 과반 이상 일치 시만 인정(미달 시 기권). 예) 3회 → 2표 이상.
    vote_n = getattr(args, "vote", 0) or 0
    vote_min = (vote_n // 2 + 1) if vote_n > 1 else 0

    # 한국형(로컬) — base(+옵션 변형)
    if not args.no_local:
        for m in (args.local_models or [LOCAL_MODEL]):
            if not m:
                continue
            targets.append(_spec(m, PROVIDER_LOCAL, ROLE_KOREAN))
            if args.rag and args.rag_local:
                targets.append(_spec(m, PROVIDER_LOCAL, ROLE_KOREAN, rag=True))
            if vote_n > 1:
                targets.append(_spec(m, PROVIDER_LOCAL, ROLE_KOREAN,
                                     sc=vote_n, sc_min=vote_min))
                if args.rag and args.rag_local:
                    targets.append(_spec(m, PROVIDER_LOCAL, ROLE_KOREAN, rag=True,
                                         sc=vote_n, sc_min=vote_min))
            if local_cot:
                if use_cot:
                    targets.append(_spec(m, PROVIDER_LOCAL, ROLE_KOREAN, cot=True))
                if sc_n > 1:
                    targets.append(_spec(m, PROVIDER_LOCAL, ROLE_KOREAN, cot=True, sc=sc_n))

    # 글로벌(범용/클라우드)
    if not args.no_global:
        if args.models:
            g_specs = [(m, PROVIDER_OPENAI) for m in args.models if m]
        else:
            g_specs = [(GLOBAL_MODEL, GLOBAL_PROVIDER)]
            for ft in (GPT4_FINETUNED_MODEL, GPT5_FINETUNED_MODEL):
                if ft:
                    g_specs.append((ft, PROVIDER_OPENAI))
        for m, prov in g_specs:
            if not m:
                continue
            is_ft = str(m).startswith("ft:")
            targets.append(_spec(m, prov, ROLE_GLOBAL))  # base 또는 ft
            if is_ft:
                continue  # 파인튜닝 모델엔 추가 방식 변형을 만들지 않음
            if args.rag:
                targets.append(_spec(m, prov, ROLE_GLOBAL, rag=True))
            if vote_n > 1:
                targets.append(_spec(m, prov, ROLE_GLOBAL, sc=vote_n, sc_min=vote_min))
                if args.rag:
                    targets.append(_spec(m, prov, ROLE_GLOBAL, rag=True,
                                         sc=vote_n, sc_min=vote_min))
            if use_cot:
                targets.append(_spec(m, prov, ROLE_GLOBAL, cot=True))
            if sc_n > 1:
                targets.append(_spec(m, prov, ROLE_GLOBAL, cot=True, sc=sc_n))
            if rag_cot and args.rag and sc_n > 1:
                targets.append(_spec(m, prov, ROLE_GLOBAL, rag=True, cot=True, sc=sc_n))

    seen, out = set(), []
    for t in targets:
        key = (t["id"], t["provider"], t["rag"], t["cot"], t["sc"], t.get("sc_min", 0))
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def _group_avgs(summary: list[dict]) -> dict:
    """역할별 base 정답률과, 각 방식(variant)의 정답률·향상(Δ)을 일반적으로 계산."""
    out = {"roles": {}}
    role_base = {}
    for role in [ROLE_KOREAN, ROLE_GLOBAL] + sorted(
            {r["role"] for r in summary} - {ROLE_KOREAN, ROLE_GLOBAL}):
        rs = [r for r in summary if r.get("role") == role]
        if not rs:
            continue
        base = _mean([r["accuracy"] for r in rs if r.get("variant") == "base"])
        if base is None:  # base 가 없으면 ft 를 기준으로
            base = _mean([r["accuracy"] for r in rs if r.get("variant") == "ft"])
        role_base[role] = base
        variants = {}
        for r in rs:
            v = r["variant"]
            variants[v] = {
                "acc": r["accuracy"],
                "delta": (round(r["accuracy"] - base, 4) if base is not None else None),
            }
        out["roles"][role] = {"base": base, "variants": variants}

    kb, gb = role_base.get(ROLE_KOREAN), role_base.get(ROLE_GLOBAL)
    if kb is not None and gb is not None:
        out["pair_avg"] = round((kb + gb) / 2, 4)
    return out


def save_results(summary: list[dict], eval_all: bool, n: int) -> dict:
    """모델별 eval_*.json 저장 + 평균 정답률 출력 + summary.json 저장."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for r in summary:
        safe = f"{r['role']}_{r['variant']}_{r['model']}".replace("/", "_").replace(":", "_")
        with open(RESULTS_DIR / f"eval_{safe}.json", "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    print("\n================ 정답률 요약 ================")
    print(f"{'역할/방식':14s} {'모델':28s} {'정답률':>8s}  {'맞음/전체'}")
    for r in summary:
        print(f"{r['role']+'/'+r['variant']:14s} {r['model']:28s} "
              f"{r['accuracy']*100:7.2f}%  {r['correct']}/{r['n']}")

    avgs = _group_avgs(summary)
    print("\n---------------- 평균 정답률 / 향상 ----------------")
    for role, d in avgs["roles"].items():
        print(f"  [{role}] base {(d['base'] or 0)*100:6.2f}%")
        for v, info in d["variants"].items():
            if v == "base":
                continue
            dl = info["delta"] or 0
            s = "+" if dl >= 0 else ""
            print(f"      {v:14s}: {info['acc']*100:6.2f}%  (Δ {s}{dl*100:.2f}%p)")
    if avgs.get("pair_avg") is not None:
        print(f"  ▶ 두 모델 평균(base): {avgs['pair_avg']*100:6.2f}%")

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "models": [{k: v for k, v in r.items() if k != "details"} for r in summary],
            "averages": avgs, "eval_all": eval_all, "n": n,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {RESULTS_DIR}")
    return avgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="전체 텍스트 문항 평가")
    ap.add_argument("--models", nargs="*", help="글로벌 자리에 넣을 모델 ID(OpenAI/파인튜닝)")
    ap.add_argument("--local-models", nargs="*", help="한국형 자리에 넣을 로컬(Ollama) 태그")
    ap.add_argument("--no-local", action="store_true", help="한국형(로컬) 평가 생략")
    ap.add_argument("--no-global", action="store_true", help="글로벌 평가 생략")
    ap.add_argument("--no-rag", dest="rag", action="store_false", help="RAG(용어 주입) 변형 제외")
    ap.add_argument("--rag-local", action="store_true", help="한국형에도 RAG 적용")
    ap.add_argument("--cot", action="store_true", help="CoT(근거 후 정답) 변형 추가")
    ap.add_argument("--sc", type=int, default=1, help="self-consistency 샘플 수(>1이면 CoT+SC 변형)")
    ap.add_argument("--rag-cot", action="store_true", help="RAG+CoT+SC 결합 변형 추가")
    ap.add_argument("--local-cot", action="store_true", help="한국형에도 CoT/SC 적용")
    ap.add_argument("--vote", type=int, default=0,
                    help="환각 완화: 문항당 N회 생성, 과반 이상 일치 시만 인정(미달 시 기권). "
                         "한국형·글로벌 모두 적용. 예) --vote 3 → 2표 이상")
    ap.add_argument("--limit", type=int, default=0, help="평가 문항 수 제한(스모크 테스트용)")
    ap.set_defaults(rag=True)
    args = ap.parse_args()

    rows = select_eval_rows(args.all)
    if args.limit:
        rows = rows[: args.limit]
    targets = resolve_targets(args)

    # RAG 변형이 하나라도 있으면 용어 인덱스를 빌드. 비어있으면 RAG 변형 제외.
    need_rag = any(t["rag"] for t in targets)
    index = TermIndex() if need_rag else None
    if need_rag and (index is None or len(index) == 0):
        print("[경고] 용어 인덱스 비어있음 → RAG 변형 제외 (scripts/fetch_terminology.py 먼저)")
        targets = [t for t in targets if not t["rag"]]
        index = None

    print(f"평가셋 {len(rows)}문항 | 용어 {len(index) if index else 0} | 대상:")
    for t in targets:
        print(f"  - {t['role']}/{t['variant']}: {t['id']} [{t['provider']}]")

    summary = []
    for t in targets:
        try:
            summary.append(evaluate_model(t, rows, index))
        except Exception as e:  # noqa: BLE001
            print(f"[오류] {t['id']}[{t['provider']}]: {e}")
            continue
    save_results(summary, args.all, len(rows))


if __name__ == "__main__":
    main()
