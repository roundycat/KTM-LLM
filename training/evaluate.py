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
    FULL_DATASET, RESULTS_DIR, SYSTEM_PROMPT, ANSWER_INSTRUCTION,
    LOCAL_MODEL, GLOBAL_MODEL, GLOBAL_PROVIDER, ROLE_KOREAN, ROLE_GLOBAL,
    GPT4_FINETUNED_MODEL, GPT5_FINETUNED_MODEL,
    VAL_RATIO, RANDOM_SEED, EVAL_CONCURRENCY, EVAL_MAX_TOKENS,
    EVAL_MAX_TOKENS_REASONING, EVAL_MAX_TOKENS_LOCAL, EVAL_REASONING_EFFORT,
    REQUEST_TIMEOUT, RAG_TOP_K, is_reasoning_model, make_client,
    PROVIDER_OPENAI, PROVIDER_LOCAL,
)
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


def build_prompt(row: dict, rag: bool = False, index: TermIndex | None = None) -> str:
    lines = [row["question"].strip(), ""]
    for i, opt in enumerate(row["options"], start=1):
        lines.append(f"{i}. {opt.strip()}")
    lines.append("")
    lines.append(ANSWER_INSTRUCTION)
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


def ask(model_id: str, provider: str, row: dict,
        rag: bool = False, index: TermIndex | None = None) -> int | None:
    """모델에게 한 문항을 물어 정답 번호(1~5)를 받아 파싱."""
    client = get_client(provider)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(row, rag, index)},
    ]

    if provider == PROVIDER_LOCAL:
        resp = client.chat.completions.create(
            model=model_id, messages=messages,
            max_tokens=EVAL_MAX_TOKENS_LOCAL, temperature=0, timeout=REQUEST_TIMEOUT)
        return parse_answer(resp.choices[0].message.content or "")

    # --- 클라우드(openai 호환) ---
    reasoning = is_reasoning_model(model_id)
    kwargs = dict(
        model=model_id, messages=messages,
        max_completion_tokens=EVAL_MAX_TOKENS_REASONING if reasoning else EVAL_MAX_TOKENS,
        timeout=REQUEST_TIMEOUT,
    )
    if reasoning:
        kwargs["reasoning_effort"] = EVAL_REASONING_EFFORT
    else:
        kwargs["temperature"] = 0
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        # reasoning_effort 미지원 모델일 때만 그 파라미터를 빼고 1회 재시도.
        # 인증/레이트리밋 등 그 외 오류는 상위(worker)의 백오프 재시도에 맡긴다.
        msg = str(e).lower()
        if "reasoning_effort" in kwargs and (
                "reasoning_effort" in msg or "unsupported" in msg or "parameter" in msg):
            kwargs.pop("reasoning_effort", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return parse_answer(resp.choices[0].message.content or "")


def evaluate_model(target: dict, rows: list[dict], index: TermIndex | None = None) -> dict:
    mid, provider = target["id"], target["provider"]
    role, rag = target["role"], target.get("rag", False)
    variant = target.get("variant") or variant_of(mid, rag)
    tag = f"{role}/{variant}"
    print(f"\n=== 평가: {mid} [{provider}] ({tag}, {len(rows)}문항) ===")
    correct = 0
    by_subject = defaultdict(lambda: [0, 0])
    details = []

    def worker(idx_row):
        idx, row = idx_row
        for attempt in range(3):
            try:
                pred = ask(mid, provider, row, rag, index)
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
        print(f"  ⚠ 미응답/파싱실패 {no_answer} · 호출오류 {errored} (오답으로 집계됨)")
    return {
        "model": mid, "provider": provider, "role": role, "rag": rag,
        "variant": variant,
        "n": len(rows), "correct": correct, "accuracy": round(acc, 4),
        "no_answer": no_answer, "errored": errored,
        "by_subject": subj_acc, "details": details,
    }


def variant_of(model_id: str, rag: bool) -> str:
    if rag:
        return "rag"
    if str(model_id).startswith("ft:"):
        return "ft"
    return "base"


def resolve_targets(args) -> list[dict]:
    """평가 대상 스펙(dict) 목록을 만든다."""
    targets: list[dict] = []

    # 한국형(로컬)
    if not args.no_local:
        for m in (args.local_models or [LOCAL_MODEL]):
            if m:
                targets.append({"id": m, "provider": PROVIDER_LOCAL, "role": ROLE_KOREAN,
                                "rag": False, "variant": "base"})

    # 글로벌(범용/클라우드) — (모델ID, provider) 스펙으로 구성
    if not args.no_global:
        if args.models:
            # 사용자가 글로벌 자리에 직접 넣은 모델은 클라우드(OpenAI/호환)로 취급
            g_specs = [(m, PROVIDER_OPENAI) for m in args.models if m]
        else:
            g_specs = [(GLOBAL_MODEL, GLOBAL_PROVIDER)]
            for ft in (GPT4_FINETUNED_MODEL, GPT5_FINETUNED_MODEL):
                if ft:
                    g_specs.append((ft, PROVIDER_OPENAI))  # 파인튜닝 모델은 OpenAI
        for m, prov in g_specs:
            if not m:
                continue
            is_ft = str(m).startswith("ft:")
            targets.append({"id": m, "provider": prov, "role": ROLE_GLOBAL,
                            "rag": False, "variant": "ft" if is_ft else "base"})
            # 파인튜닝 모델엔 RAG 변형을 만들지 않는다(향상 Δ 의미를 분리).
            if args.rag and not is_ft:
                targets.append({"id": m, "provider": prov, "role": ROLE_GLOBAL,
                                "rag": True, "variant": "rag"})

    # 한국형 RAG(옵션)
    if args.rag and args.rag_local and not args.no_local:
        for m in (args.local_models or [LOCAL_MODEL]):
            if m:
                targets.append({"id": m, "provider": PROVIDER_LOCAL, "role": ROLE_KOREAN,
                                "rag": True, "variant": "rag"})

    seen, out = set(), []
    for t in targets:
        key = (t["id"], t["provider"], t["rag"])
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _group_avgs(summary: list[dict]) -> dict:
    """역할·variant 별 평균 정답률을 계산한다."""
    def avg(role, variant):
        xs = [r["accuracy"] for r in summary
              if r.get("role") == role and r.get("variant") == variant]
        return round(sum(xs) / len(xs), 4) if xs else None

    out = {
        "korean_base": avg(ROLE_KOREAN, "base"),
        "korean_rag": avg(ROLE_KOREAN, "rag"),
        "global_base": avg(ROLE_GLOBAL, "base"),
        "global_rag": avg(ROLE_GLOBAL, "rag"),
        "global_ft": avg(ROLE_GLOBAL, "ft"),
    }
    if out["korean_base"] is not None and out["global_base"] is not None:
        out["pair_avg"] = round((out["korean_base"] + out["global_base"]) / 2, 4)
    if out["global_rag"] is not None and out["global_base"] is not None:
        out["global_improvement_rag"] = round(out["global_rag"] - out["global_base"], 4)
    if out["global_ft"] is not None and out["global_base"] is not None:
        out["global_improvement_ft"] = round(out["global_ft"] - out["global_base"], 4)
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
    print("\n---------------- 평균 정답률 ----------------")
    if avgs["korean_base"] is not None:
        print(f"  한국형(로컬) 평균        : {avgs['korean_base']*100:6.2f}%")
    if avgs["global_base"] is not None:
        print(f"  글로벌(베이스) 평균      : {avgs['global_base']*100:6.2f}%")
    if avgs.get("pair_avg") is not None:
        print(f"  ▶ 한국형+글로벌 두 모델 평균: {avgs['pair_avg']*100:6.2f}%")
    if avgs["global_rag"] is not None:
        print(f"  글로벌(용어 RAG) 평균    : {avgs['global_rag']*100:6.2f}%")
    if avgs.get("global_improvement_rag") is not None:
        s = "+" if avgs["global_improvement_rag"] >= 0 else ""
        print(f"  ▶ 글로벌 향상(RAG-베이스): {s}{avgs['global_improvement_rag']*100:.2f}%p")
    if avgs.get("global_improvement_ft") is not None:
        s = "+" if avgs["global_improvement_ft"] >= 0 else ""
        print(f"  ▶ 글로벌 향상(파인튜닝-베이스): {s}{avgs['global_improvement_ft']*100:.2f}%p")

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
    ap.set_defaults(rag=True)
    args = ap.parse_args()

    rows = select_eval_rows(args.all)
    index = TermIndex() if args.rag else None
    if args.rag and (index is None or len(index) == 0):
        print("[경고] 용어 인덱스가 비어있음 → RAG 비활성화 (먼저 scripts/fetch_terminology.py 실행)")
        args.rag = False
        index = None

    targets = resolve_targets(args)
    print(f"평가셋 {len(rows)}문항 | RAG={'on' if args.rag else 'off'}"
          f"{f'({len(index)} 용어)' if index else ''} | 대상:")
    for t in targets:
        print(f"  - {t['role']}/{'rag' if t['rag'] else 'base'}: {t['id']} [{t['provider']}]")

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
