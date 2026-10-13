"""1 Điều = 1 chunk; Điều dài thì tách theo ranh giới Khoản (kết luận từ POC E1)."""
from __future__ import annotations

import re

from .parser import Article

RE_KHOAN = re.compile(r"^(\d{1,2})\.\s+")


def _group_by_khoan(lines: list[str]) -> list[list[str]]:
    groups, cur = [], []
    for line in lines:
        if RE_KHOAN.match(line) and cur:
            groups.append(cur)
            cur = []
        cur.append(line)
    if cur:
        groups.append(cur)
    return groups


def split_body(lines: list[str], max_chars: int) -> list[tuple[list[str], str]]:
    body = "\n".join(lines)
    if len(body) <= max_chars:
        return [([], body)]
    parts, buf, khoan, size = [], [], [], 0
    for group in _group_by_khoan(lines):
        text = "\n".join(group)
        if buf and size + len(text) > max_chars:
            parts.append((khoan, "\n".join(buf)))
            buf, khoan, size = [], [], 0
        buf.append(text)
        size += len(text) + 1
        if m := RE_KHOAN.match(group[0]):
            khoan.append(m.group(1))
    if buf:
        parts.append((khoan, "\n".join(buf)))
    return parts


def chunk_article(art: Article, doc: dict, max_chars: int) -> list[dict]:
    aid = f"{doc['so_hieu']}|{art.article}"
    header = f"{doc['title']} ({doc['so_hieu']}) - Điều {art.article}. {art.title}"  # giữ format POC
    return [
        {
            "id": f"{aid}#{i}",
            "aid": aid,
            "so_hieu": doc["so_hieu"],
            "doc_title": doc["title"],
            "status": doc["status"],
            "replaced_by": doc.get("replaced_by", ""),
            "effective_date": doc.get("effective_date", ""),
            "chapter": art.chapter,
            "article": art.article,
            "article_title": art.title,
            "khoan": khoan,
            "text": f"{header}\n{body}".strip(),
        }
        for i, (khoan, body) in enumerate(split_body(art.lines, max_chars))
    ]
