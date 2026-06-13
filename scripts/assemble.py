# -*- coding: utf-8 -*-
"""
전사된 문항 JSON + 정답표 JSON을 조인하여 학습용 JSONL을 생성한다.

입력:
  transcribe/<slug>__<교시>.json        : 문항 [{교시,번호,question,options[5]}]
  transcribe/parts/<slug>__<교시>_*.json : 분할 전사본(있으면 병합 우선)
  transcribe/<slug>__답안.json          : 정답표 [{교시,과목,번호,answer}]

출력(dataset/):
  한의학_문제.jsonl        : 그림 없는 순수 텍스트 문항 (학습용 메인). {question, options, answer, answer_text}
  한의학_문제_전체.jsonl   : 그림생략 포함 전체 + 메타데이터
  stats.json               : 통계
"""
import os, re, json, glob
from collections import defaultdict

TRANS = "transcribe"
PARTS = os.path.join(TRANS, "parts")
OUT = "dataset"

# 처리할 회차(slug)와 교시 목록
EXAMS = {
    "한의사_81회": ["1교시", "2교시", "3교시", "4교시"],
    "한약사_27회": ["1교시", "2교시"],
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_parts(slug, gyo):
    """parts/ 에 분할본이 있으면 그것을 병합, 없으면 단일 파일 사용."""
    part_paths = sorted(glob.glob(os.path.join(PARTS, f"{slug}__{gyo}_*.json")))
    items = {}
    if part_paths:
        for p in part_paths:
            for q in load_json(p):
                n = q["번호"]
                # 중복 시 question 길이가 더 긴(더 완전한) 쪽 채택
                if n not in items or len(q.get("question", "")) > len(items[n].get("question", "")):
                    items[n] = q
        src = f"parts:{len(part_paths)}"
    else:
        single = os.path.join(TRANS, f"{slug}__{gyo}.json")
        for q in load_json(single):
            items[q["번호"]] = q
        src = "single"
    return items, src


def build_answer_index(slug):
    """(교시,번호) -> {answer, 과목}"""
    path = os.path.join(TRANS, f"{slug}__답안.json")
    idx = {}
    for a in load_json(path):
        idx[(int(a["교시"]), int(a["번호"]))] = {
            "answer": a["answer"],
            "과목": a.get("과목", ""),
        }
    return idx


def gyo_to_int(gyo):
    return int(re.match(r"(\d+)", gyo).group(1))


def main():
    os.makedirs(OUT, exist_ok=True)
    clean, full = [], []
    stats = {"exams": {}, "total_questions": 0, "clean": 0, "with_figure": 0,
             "no_answer": 0, "bad_options": 0}

    for slug, gyos in EXAMS.items():
        ans_idx = build_answer_index(slug)
        ex_stat = {"questions": 0, "clean": 0, "with_figure": 0,
                   "no_answer": 0, "by_gyo": {}}
        for gyo in gyos:
            items, src = merge_parts(slug, gyo)
            gnum = gyo_to_int(gyo)
            gcount = 0
            for num in sorted(items):
                q = items[num]
                question = (q.get("question") or "").strip()
                options = q.get("options") or []
                key = (gnum, num)
                ainfo = ans_idx.get(key)

                # 옵션 5개 검증
                if len(options) != 5:
                    stats["bad_options"] += 1

                # 정답 처리
                answer = None
                answer_text = None
                subject = ""
                if ainfo:
                    subject = ainfo["과목"]
                    a = ainfo["answer"]
                    if isinstance(a, int) or (isinstance(a, str) and a.strip().isdigit()):
                        answer = int(a)
                        if 1 <= answer <= len(options):
                            answer_text = options[answer - 1]
                if answer is None:
                    stats["no_answer"] += 1
                    ex_stat["no_answer"] += 1

                has_fig = "[그림 생략]" in question
                rec_full = {
                    "source": slug,
                    "교시": gnum,
                    "과목": subject,
                    "번호": num,
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "answer_text": answer_text,
                    "has_figure": has_fig,
                }
                full.append(rec_full)

                # 메인 학습셋: 정답 있고, 옵션 5개, 그림 의존 아님
                if answer is not None and len(options) == 5 and not has_fig:
                    clean.append({
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "answer_text": answer_text,
                    })
                    ex_stat["clean"] += 1
                    stats["clean"] += 1
                if has_fig:
                    ex_stat["with_figure"] += 1
                    stats["with_figure"] += 1

                gcount += 1
                ex_stat["questions"] += 1
                stats["total_questions"] += 1
            ex_stat["by_gyo"][gyo] = {"count": gcount, "source": src}
        stats["exams"][slug] = ex_stat

    # 쓰기
    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(OUT, "한의학_문제.jsonl"), clean)
    write_jsonl(os.path.join(OUT, "한의학_문제_전체.jsonl"), full)
    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n[*] dataset/한의학_문제.jsonl       : {len(clean)} 문항 (학습용 메인)")
    print(f"[*] dataset/한의학_문제_전체.jsonl  : {len(full)} 문항 (그림포함+메타)")


if __name__ == "__main__":
    main()
