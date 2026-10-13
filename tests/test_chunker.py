from legal_qa.ingest.chunker import chunk_article
from legal_qa.ingest.parser import Article

DOC = {"so_hieu": "45/2019/QH14", "title": "Bộ luật Lao động 2019", "status": "con_hieu_luc"}


def test_short_article_one_chunk():
    art = Article(139, "", "Nghỉ thai sản", None, ["1. Lao động nữ được nghỉ."])
    chunks = chunk_article(art, DOC, max_chars=1800)
    assert len(chunks) == 1
    assert chunks[0]["aid"] == "45/2019/QH14|139"
    assert chunks[0]["text"].startswith("Bộ luật Lao động 2019 (45/2019/QH14) - Điều 139. Nghỉ thai sản")


def test_long_article_split_by_khoan():
    lines = [f"{i}. " + "nội dung " * 40 for i in range(1, 6)]
    art = Article(35, "", "Quyền đơn phương", None, lines)
    chunks = chunk_article(art, DOC, max_chars=800)
    assert len(chunks) > 1
    assert all(c["aid"] == "45/2019/QH14|35" for c in chunks)
    assert chunks[0]["khoan"][0] == "1"
