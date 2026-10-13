"""Tách văn bản đã làm sạch thành danh sách Điều.

Chống lỗi OCR bằng ràng buộc "số Điều tăng dần": một dòng 'Điều N.' chỉ được nhận là
tiêu đề nếu N lớn hơn Điều trước đó tối đa max_gap. Dòng không thỏa mãn được coi là nội dung.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

RE_DIEU = re.compile(r"^Điều\s+(?P<num>\d{1,3})(?P<suffix>[a-z]?)\.\s*(?P<title>.*)$")
RE_CHUONG = re.compile(r"^Chương\s+(?P<num>[IVXLC]+)\b\s*(?P<rest>.*)$")
RE_MUC = re.compile(r"^Mục\s+\d+")


@dataclass
class Article:
    number: int
    suffix: str
    title: str
    chapter: str | None
    lines: list[str] = field(default_factory=list)

    @property
    def article(self) -> str:
        return f"{self.number}{self.suffix}"

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class ValidationResult:
    count: int
    expected: int | None
    missing: list[int]
    duplicates: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.duplicates and (self.expected is None or self.count == self.expected)


def parse_articles(text: str, max_gap: int = 3) -> tuple[list[Article], list[str]]:
    articles: list[Article] = []
    warnings: list[str] = []
    chapter: str | None = None
    current: Article | None = None
    expect_title: str | None = None  # "chapter" | "section" | None
    last = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if m := RE_CHUONG.match(line):
            chapter = f"Chương {m['num']}"
            rest = m["rest"].strip(" .:-")
            if rest:
                chapter += f" - {rest.title()}"
                expect_title = None
            else:
                expect_title = "chapter"
            continue
        if RE_MUC.match(line):
            expect_title = "section"
            continue
        if expect_title and line.isupper():
            if expect_title == "chapter" and chapter:
                chapter += f" - {line.title()}"
            expect_title = None
            continue
        expect_title = None

        if m := RE_DIEU.match(line):
            num, suffix = int(m["num"]), m["suffix"]
            if (last < num <= last + max_gap) or (suffix and num == last):
                if num > last + 1:
                    warnings.append(f"Nhảy từ Điều {last} lên Điều {num}: có thể mất tiêu đề do OCR")
                current = Article(num, suffix, m["title"].strip(), chapter)
                articles.append(current)
                last = num
                continue
            warnings.append(f"Bỏ qua tiêu đề 'Điều {num}{suffix}' sau Điều {last} (không tăng dần): {line[:80]}")

        if current is not None:
            current.lines.append(line)

    return articles, warnings


def validate(articles: list[Article], expected: int | None = None) -> ValidationResult:
    nums = [a.number for a in articles if not a.suffix]
    upper = max([*nums, expected or 0])
    missing = sorted(set(range(1, upper + 1)) - set(nums))
    duplicates = [k for k, c in Counter(a.article for a in articles).items() if c > 1]
    return ValidationResult(len(articles), expected, missing, duplicates)
