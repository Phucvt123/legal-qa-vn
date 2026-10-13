"""Làm sạch văn bản luật, đặc biệt là văn bản OCR.

Nguyên tắc an toàn:
- Chỉ sửa ký tự, tiêu đề Điều/Chương, dấu khoản, ngắt dòng và cụm từ KHÔNG chứa chữ số.
- KHÔNG BAO GIỜ tự sửa chữ số: "6 tháng" bị OCR thành "8 tháng" thì LLM cũng không phát hiện được.
  Chữ số sai phải sửa tay trong file gốc.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

CHAR_MAP = str.maketrans({
    "\u00d0": "Đ",   # Ð (Eth) hay bị OCR nhầm với Đ
    "\u00f0": "đ",
    "\u00a0": " ",
    "\u200b": "",
    "\ufeff": "",
    "\t": " ",
})

JUNK_LINES = {"Đang theo dõi"}
REPEAT_THRESHOLD = 8  # dòng ngắn lặp >= 8 lần: thường là header/footer trang

RE_PAGE_NUMBER = re.compile(r"^(?:Trang\s*)?\d{1,3}(?:\s*/\s*\d{1,3})?$", re.IGNORECASE)
RE_DIEU_OCR = re.compile(
    r"^[ĐD]\s?[iíìỉĩịl1]\s?[êềếểễệeé]\s?[uưúù]\s*(?P<num>\d{1,3}[a-z]?)\s*(?P<punct>[.:,;])?\s*"
)
RE_CHUONG_OCR = re.compile(r"^C\s?h\s?[uư]\s?[oơ]\s?n\s?g\s+(?P<num>[IVXLCl1]+)\b\s*[.:]?\s*")
RE_INLINE_KHOAN1 = re.compile(r"\s+[il1]\.\s+")
RE_KHOAN_OCR = re.compile(r"^[il|]\.\s+")
RE_KHOAN = re.compile(r"^\d{1,2}\.\s")
RE_DIEM = re.compile(r"^[a-zđ]\)\s")
TERMINALS = (".", ":", ";", "!", "?")

# Lỗi đã quan sát trong POC. Bổ sung thêm qua data/ocr_fixes.csv
DEFAULT_PHRASE_FIXES = {
    "tối thiếu": "tối thiểu",
    "tổng sổ người": "tổng số người",
    "ngưòi": "người",
    "nguời": "người",
    "nguòi": "người",
    "lao dộng": "lao động",
    "hơp đồng": "hợp đồng",
}


@dataclass
class CleanReport:
    counts: Counter = field(default_factory=Counter)
    removed_samples: list[str] = field(default_factory=list)

    def remove(self, line: str) -> None:
        self.counts["xóa dòng rác / header / số trang"] += 1
        if len(self.removed_samples) < 15 and line not in self.removed_samples:
            self.removed_samples.append(line)

    def as_dict(self) -> dict:
        return {"counts": dict(self.counts), "removed_samples": self.removed_samples}


def load_phrase_fixes(path: Path | None) -> dict[str, str]:
    fixes = dict(DEFAULT_PHRASE_FIXES)
    if path is not None and path.exists():
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                wrong, right = (row.get("wrong") or "").strip(), (row.get("right") or "").strip()
                if wrong:
                    fixes[wrong] = right
    for wrong, right in fixes.items():
        if re.search(r"\d", wrong + right):
            raise ValueError(f"Cụm sửa '{wrong}' -> '{right}' chứa chữ số. Chữ số phải sửa tay trong file gốc.")
    return fixes


def _is_structural(line: str) -> bool:
    return bool(line.startswith(("Điều ", "Chương ", "Mục ")) or RE_KHOAN.match(line) or RE_DIEM.match(line))


def _is_block_start(line: str) -> bool:
    return _is_structural(line) or (line.isupper() and len(line) > 3)


def _normalize_heading(line: str, report: CleanReport) -> list[str]:
    m = RE_DIEU_OCR.match(line)
    if m:
        rest = line[m.end():].strip()
        # Có dấu câu sau số, hoặc tiêu đề viết hoa: là tiêu đề Điều.
        # "Điều 35 của Bộ luật này..." (chữ thường) là dẫn chiếu, giữ nguyên.
        if m.group("punct") or rest[:1].isupper():
            head = f"Điều {m.group('num')}."
            for mm in RE_INLINE_KHOAN1.finditer(rest):
                after = rest[mm.end():]
                if after[:1].isupper():  # "…doanh nghiệp i. Tổ chức…" → tách khoản 1
                    report.counts["tách khoản 1 dính vào tiêu đề Điều"] += 1
                    return [f"{head} {rest[:mm.start()].strip()}".strip(), f"1. {after}"]
            new = f"{head} {rest}".strip()
            if new != line:
                report.counts["chuẩn hóa tiêu đề Điều"] += 1
            return [new]

    m = RE_CHUONG_OCR.match(line)
    if m:
        roman = m.group("num").replace("l", "I").replace("1", "I").upper()
        rest = line[m.end():].strip()
        new = f"Chương {roman}" + (f" {rest}" if rest else "")
        if new != line:
            report.counts["chuẩn hóa tiêu đề Chương"] += 1
        return [new]
    return [line]


def _fix_khoan_marker(line: str, report: CleanReport) -> str:
    m = RE_KHOAN_OCR.match(line)
    if m and line[m.end():][:1].isupper():
        report.counts["sửa dấu khoản 'i.' → '1.'"] += 1
        return "1. " + line[m.end():]
    return line


def _join_broken_lines(lines: list[str], report: CleanReport) -> list[str]:
    out: list[str] = []
    for line in lines:
        if out and not _is_block_start(line) and line[:1].islower() and not out[-1].endswith(TERMINALS):
            out[-1] = f"{out[-1]} {line}"
            report.counts["nối dòng bị ngắt"] += 1
        else:
            out.append(line)
    return out


def clean_text(text: str, phrase_fixes: dict[str, str] | None = None) -> tuple[str, CleanReport]:
    report = CleanReport()
    fixes = phrase_fixes if phrase_fixes is not None else load_phrase_fixes(None)
    patterns = [(w, re.compile(rf"(?<!\w){re.escape(w)}(?!\w)"), r) for w, r in fixes.items()]

    text = unicodedata.normalize("NFC", text).translate(CHAR_MAP)
    lines = [re.sub(r" {2,}", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    freq = Counter(lines)
    repeated = {ln for ln, c in freq.items()
                if c >= REPEAT_THRESHOLD and len(ln) < 60 and not _is_structural(ln)}

    fixed: list[str] = []
    for line in lines:
        if line in JUNK_LINES or line in repeated or RE_PAGE_NUMBER.match(line):
            report.remove(line)
            continue
        for part in _normalize_heading(line, report):
            part = _fix_khoan_marker(part, report)
            for wrong, pat, right in patterns:
                part, n = pat.subn(right, part)
                if n:
                    report.counts[f"sửa cụm '{wrong}'"] += n
            fixed.append(part)

    return "\n".join(_join_broken_lines(fixed, report)), report
