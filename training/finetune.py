# -*- coding: utf-8 -*-
"""OpenAI Fine-tuning 실행 — 파일 업로드 → job 생성 → 진행 모니터링.

사용법:
    # 한 모델만
    python training/finetune.py --model gpt-4
    python training/finetune.py --model gpt-5

    # 두 모델 모두 (config.BASE_MODELS 의 gpt-4, gpt-5)
    python training/finetune.py --model all

    # 임의의 베이스 모델 ID 직접 지정
    python training/finetune.py --model-id gpt-4o-2024-08-06

주의:
    - GPT-5 계열의 파인튜닝 가능 여부/모델 ID는 계정 권한에 따라 다릅니다.
      .env 의 GPT5_MODEL 을 실제 파인튜닝 가능한 ID로 설정하세요.
      파인튜닝을 지원하지 않으면 OpenAI 측에서 오류를 반환합니다.
"""
from __future__ import annotations
import argparse
import time

from openai import OpenAI

from config import (
    FINETUNE_TRAIN, FINETUNE_VAL, FINETUNE_TERM_TRAIN, FINETUNE_TERM_VAL,
    BASE_MODELS, RANDOM_SEED,
)

client = OpenAI()


def upload_file(path) -> str:
    print(f"[업로드] {path}")
    with open(path, "rb") as f:
        obj = client.files.create(file=f, purpose="fine-tune")
    print(f"  file_id = {obj.id}")
    return obj.id


def run_finetune(base_model_id: str, train_path=FINETUNE_TRAIN,
                 val_path=FINETUNE_VAL, suffix="hani-exam") -> str | None:
    """베이스 모델 하나에 대해 파인튜닝 job을 만들고 완료까지 폴링.

    train_path/val_path/suffix 를 바꾸면 용어 SFT 등 다른 데이터로도 학습할 수 있다.
    """
    print(f"\n=== 파인튜닝 시작: {base_model_id} (suffix={suffix}) ===")
    print(f"  학습 파일: {train_path}")
    train_id = upload_file(train_path)
    val_id = upload_file(val_path) if val_path and val_path.exists() else None

    job = client.fine_tuning.jobs.create(
        training_file=train_id,
        validation_file=val_id,
        model=base_model_id,
        seed=RANDOM_SEED,
        suffix=suffix,
    )
    print(f"[job 생성] id={job.id} status={job.status}")

    seen = set()
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        # 새 이벤트 출력
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=20)
        for ev in reversed(events.data):
            if ev.id not in seen:
                seen.add(ev.id)
                print(f"  [{ev.created_at}] {ev.message}")
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(15)

    print(f"[완료] status={job.status}")
    if job.status == "succeeded":
        print(f"  ★ 파인튜닝 모델 ID: {job.fine_tuned_model}")
        print(f"    → .env 의 해당 *_FINETUNED_MODEL 에 넣고 evaluate.py 로 평가하세요.")
        return job.fine_tuned_model
    else:
        print("  파인튜닝 실패/취소. 위 이벤트 로그를 확인하세요.")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", choices=["gpt-4", "gpt-5", "all"],
                   help="config.BASE_MODELS 의 별칭")
    g.add_argument("--model-id", help="베이스 모델 ID 직접 지정")
    # 학습 데이터/접미사 — 비우면 기출 SFT(FINETUNE_TRAIN/VAL) 사용.
    ap.add_argument("--terminology", action="store_true",
                    help="용어 SFT(FINETUNE_TERM_TRAIN/VAL)로 학습. suffix=hani-term")
    ap.add_argument("--train-file", help="학습 JSONL 직접 지정")
    ap.add_argument("--val-file", help="검증 JSONL 직접 지정")
    ap.add_argument("--suffix", help="파인튜닝 모델 접미사")
    args = ap.parse_args()

    if args.model_id:
        targets = [args.model_id]
    elif args.model == "all":
        targets = [BASE_MODELS["gpt-4"], BASE_MODELS["gpt-5"]]
    else:
        targets = [BASE_MODELS[args.model]]

    # 학습 데이터 결정: --train-file > --terminology > 기본(기출)
    if args.train_file:
        from pathlib import Path
        train_path = Path(args.train_file)
        val_path = Path(args.val_file) if args.val_file else None
        suffix = args.suffix or "hani-custom"
    elif args.terminology:
        train_path, val_path = FINETUNE_TERM_TRAIN, FINETUNE_TERM_VAL
        suffix = args.suffix or "hani-term"
    else:
        train_path, val_path = FINETUNE_TRAIN, FINETUNE_VAL
        suffix = args.suffix or "hani-exam"

    results = {}
    for mid in targets:
        try:
            results[mid] = run_finetune(mid, train_path, val_path, suffix)
        except Exception as e:  # noqa: BLE001
            print(f"[오류] {mid}: {e}")
            results[mid] = None

    print("\n=== 요약 ===")
    for mid, ft in results.items():
        print(f"  {mid} → {ft or '실패'}")


if __name__ == "__main__":
    main()
