"""Đọc văn bản từ .docx / .pdf / .txt, giữ đúng thứ tự đoạn văn và bảng."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator


def _iter_docx_lines(path: Path) -> Iterator[str]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    # Duyệt theo thứ tự xuất hiện: file OCR hay đặt một phần nội dung trong bảng
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield Paragraph(child, document).text
        elif tag == "tbl":
            for row in Table(child, document).rows:
                previous = None
                for cell in row.cells:
                    text = cell.text
                    if text != previous:  # ô gộp (merged) bị lặp lại
                        yield text
                    previous = text


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return "\n".join(_iter_docx_lines(path))
    if suffix == ".pdf":
        import fitz
        with fitz.open(str(path)) as pdf:
            return "\n".join(page.get_text() for page in pdf)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".doc":
        raise ValueError(f"{path.name}: định dạng .doc cũ không hỗ trợ, hãy lưu lại thành .docx")
    raise ValueError(f"{path.name}: không hỗ trợ định dạng {suffix}")
