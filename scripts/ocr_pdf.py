"""OCR file PDF scan thành .docx để đưa vào pipeline (chỉ dùng khi không có bản Word/text).

Cài Tesseract kèm gói tiếng Việt:
  Ubuntu/Colab: apt-get install -y tesseract-ocr tesseract-ocr-vie
  Windows:      cài từ https://github.com/UB-Mannheim/tesseract/wiki, tick "Vietnamese"
  macOS:        brew install tesseract tesseract-lang

Ví dụ:
  python scripts/ocr_pdf.py duong_dan/blld_2019_scan.pdf
  → data/raw/blld_2019_scan_ocr_raw.docx + file báo cáo độ tin cậy từng trang
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ocr_page(page, dpi: int, lang: str, psm: int) -> tuple[str, float]:
    import pytesseract
    from PIL import Image, ImageOps

    pix = page.get_pixmap(dpi=dpi)
    img = ImageOps.grayscale(Image.open(io.BytesIO(pix.tobytes("png"))))
    data = pytesseract.image_to_data(img, lang=lang, config=f"--oem 1 --psm {psm}",
                                     output_type=pytesseract.Output.DICT)
    lines: dict[tuple, list[str]] = {}
    confs: list[float] = []
    for j, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][j], data["par_num"][j], data["line_num"][j])
        lines.setdefault(key, []).append(word)
        conf = float(data["conf"][j])
        if conf >= 0:
            confs.append(conf)
    text = "\n".join(" ".join(words) for words in lines.values())
    return text, round(sum(confs) / len(confs), 1) if confs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, help="Mặc định: data/raw/<tên>_ocr_raw.docx")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--lang", default="vie")
    ap.add_argument("--psm", type=int, default=3, help="Chế độ phân trang của Tesseract (3 = tự động)")
    ap.add_argument("--force-ocr", action="store_true", help="OCR cả những trang đã có lớp chữ")
    ap.add_argument("--min-conf", type=float, default=80.0, help="Trang dưới ngưỡng này sẽ được cảnh báo")
    args = ap.parse_args()

    import docx
    import fitz
    import pytesseract

    try:
        langs = pytesseract.get_languages(config="")
    except pytesseract.TesseractNotFoundError:
        sys.exit("❌ Chưa cài Tesseract (xem hướng dẫn ở đầu file).")
    if args.lang not in langs:
        sys.exit(f"❌ Thiếu gói ngôn ngữ '{args.lang}'. Hiện có: {langs}")

    out = args.out or ROOT / "data" / "raw" / f"{args.pdf.stem}_ocr_raw.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    document = docx.Document()
    report = []

    with fitz.open(str(args.pdf)) as pdf:
        for i, page in enumerate(pdf, 1):
            layer = page.get_text().strip()
            if len(layer) > 50 and not args.force_ocr:
                text, method, conf = layer, "text-layer", None   # PDF đã có chữ: không cần OCR
            else:
                text, conf = ocr_page(page, args.dpi, args.lang, args.psm)
                method = "ocr"
            for line in text.splitlines():
                if line.strip():
                    document.add_paragraph(line.strip())
            report.append({"page": i, "method": method, "confidence": conf, "chars": len(text)})
            print(f"Trang {i}/{len(pdf)}: {method}" + (f", độ tin cậy {conf}" if conf is not None else ""))

    document.save(str(out))
    report_path = out.with_name(f"{out.stem}_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "method", "confidence", "chars"])
        writer.writeheader()
        writer.writerows(report)

    low = [r["page"] for r in report if r["confidence"] is not None and r["confidence"] < args.min_conf]
    print(f"\n✅ {out}\n📄 Báo cáo: {report_path}")
    if low:
        print(f"⚠️ Các trang có độ tin cậy < {args.min_conf}: {low}")
    print("⚠️ Hãy dò lại CHỮ SỐ (số ngày, số tháng, tỷ lệ %, số Điều) trên các trang này so với bản gốc. "
          "Pipeline không tự sửa chữ số.")


if __name__ == "__main__":
    main()
