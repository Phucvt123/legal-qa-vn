"""docx → làm sạch → tách Điều → kiểm tra → chunk → embedding → artifacts/.

Ví dụ:
  python scripts/build_index.py --skip-embed   # chỉ kiểm tra parse, rất nhanh
  python scripts/build_index.py --strict       # build thật, dừng nếu số Điều sai
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from legal_qa.config import settings  # noqa: E402
from legal_qa.ingest.chunker import chunk_article  # noqa: E402
from legal_qa.ingest.cleaning import clean_text, load_phrase_fixes  # noqa: E402
from legal_qa.ingest.loaders import load_text  # noqa: E402
from legal_qa.ingest.parser import parse_articles, validate  # noqa: E402

REQUIRED_COLUMNS = {"file", "so_hieu", "title", "status"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="Dừng nếu số Điều thiếu/trùng/sai số lượng")
    ap.add_argument("--skip-embed", action="store_true", help="Chỉ kiểm tra parse, không tính embedding")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    with open(settings.data_dir / "metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not REQUIRED_COLUMNS <= set(rows[0]):
        sys.exit(f"metadata.csv phải có các cột: {sorted(REQUIRED_COLUMNS)}")

    fixes = load_phrase_fixes(settings.data_dir / "ocr_fixes.csv")
    debug_dir = settings.artifacts_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    chunks, doc_reports, failed = [], [], False
    for row in rows:
        path = settings.data_dir / "raw" / row["file"]
        if not path.exists():
            sys.exit(f"❌ Thiếu file {path}")

        raw = load_text(path)
        cleaned, clean_report = clean_text(raw, fixes)
        (debug_dir / f"{path.stem}.raw.txt").write_text(raw, encoding="utf-8")
        (debug_dir / f"{path.stem}.clean.txt").write_text(cleaned, encoding="utf-8")

        articles, warnings = parse_articles(cleaned)
        expected = int(row["expected_articles"]) if row.get("expected_articles") else None
        val = validate(articles, expected)
        failed |= not val.ok

        print(f"\n{'✅' if val.ok else '❌'} {row['title']} ({row['so_hieu']}): {val.count} điều"
              + (f" / kỳ vọng {expected}" if expected else ""))
        if val.missing:
            print(f"   Thiếu: {val.missing[:30]}")
        if val.duplicates:
            print(f"   Trùng: {val.duplicates[:30]}")
        print("   Làm sạch:", dict(clean_report.counts))
        if clean_report.removed_samples:
            print("   Dòng đã xóa (mẫu):", clean_report.removed_samples[:5])
        for w in warnings[:15]:
            print("   ⚠️", w)

        doc_chunks = [c for a in articles for c in chunk_article(a, row, settings.max_chunk_chars)]
        chunks.extend(doc_chunks)
        doc_reports.append({"file": row["file"], "so_hieu": row["so_hieu"], "articles": val.count,
                            "expected": expected, "missing": val.missing, "duplicates": val.duplicates,
                            "chunks": len(doc_chunks), "warnings": warnings, "cleaning": clean_report.as_dict()})

    lens = sorted(len(c["text"]) for c in chunks)
    print(f"\nTổng {len(chunks)} chunks | độ dài min {lens[0]}, median {lens[len(lens)//2]}, max {lens[-1]}")
    print(f"Văn bản sau làm sạch để soát lỗi: {debug_dir}")

    if failed and args.strict:
        sys.exit("Parse chưa đạt (--strict). Sửa file gốc hoặc data/ocr_fixes.csv rồi chạy lại.")
    if args.skip_embed:
        return

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(settings.embed_model, device=device)
    model.max_seq_length = 512
    t0 = time.time()
    vecs = model.encode([c["text"] for c in chunks], batch_size=args.batch_size,
                        normalize_embeddings=True, show_progress_bar=True)
    print(f"Embedding xong trên {device} trong {time.time() - t0:.0f}s")

    out = settings.artifacts_dir
    np.save(out / "embeddings.npy", vecs.astype(np.float16))
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "embed_model": settings.embed_model,
        "dim": int(vecs.shape[1]),
        "num_chunks": len(chunks),
        "max_chunk_chars": settings.max_chunk_chars,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "documents": doc_reports,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu index vào {out}")


if __name__ == "__main__":
    main()
