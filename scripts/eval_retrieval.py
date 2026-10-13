"""Đánh giá retrieval: Recall@k (hit rate), MRR@k, tách theo loại câu hỏi, khoảng tin cậy 95%.

  python scripts/eval_retrieval.py --qa data/eval/qa.jsonl --k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from legal_qa.retriever import Retriever  # noqa: E402


def bootstrap_ci(values, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", type=Path, default=ROOT / "data" / "eval" / "qa.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    args = ap.parse_args()

    qa = [json.loads(line) for line in open(args.qa, encoding="utf-8") if line.strip()]
    retriever = Retriever()
    known = {c["aid"] for c in retriever.chunks}

    rows, skipped = [], 0
    for it in qa:
        gold = it["aid"] if isinstance(it["aid"], list) else [it["aid"]]
        if not any(g in known for g in gold):
            skipped += 1  # nhãn trỏ tới Điều không có trong index
            continue
        res = retriever.search(it["question"], k=args.k)
        rank = next((i + 1 for i, h in enumerate(res) if h["aid"] in gold), None)
        rows.append({"question": it["question"], "type": it.get("type", "unknown"), "gold": gold,
                     "rank": rank, "got": [h["aid"] for h in res]})

    def stats(subset):
        hits = [r["rank"] is not None for r in subset]
        rr = [1 / r["rank"] if r["rank"] else 0.0 for r in subset]
        lo, hi = bootstrap_ci(hits)
        return len(subset), np.mean(hits), lo, hi, np.mean(rr)

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    lines = [f"| Nhóm | Số câu | Recall@{args.k} | CI 95% | MRR@{args.k} |", "|---|---|---|---|---|"]
    for name, subset in [("TẤT CẢ", rows), *sorted(by_type.items())]:
        n, rec, lo, hi, mrr = stats(subset)
        lines.append(f"| {name} | {n} | {rec:.3f} | [{lo:.3f}, {hi:.3f}] | {mrr:.3f} |")
    table = "\n".join(lines)
    print(table)
    if skipped:
        print(f"\n⚠️ Bỏ qua {skipped} câu vì nhãn không có trong index (có thể do số Điều parse sai).")

    args.out.mkdir(exist_ok=True)
    (args.out / "retrieval_latest.md").write_text(table + "\n", encoding="utf-8")
    fails = [r for r in rows if r["rank"] is None]
    (args.out / "retrieval_failures.json").write_text(json.dumps(fails, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(f"\n{len(fails)} câu thất bại → {args.out / 'retrieval_failures.json'}")


if __name__ == "__main__":
    main()
