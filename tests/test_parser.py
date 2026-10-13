from legal_qa.ingest.parser import parse_articles, validate


def test_parse_with_chapter_title():
    text = "BỘ LUẬT\nChương I\nNHỮNG QUY ĐỊNH CHUNG\nĐiều 1. Phạm vi\nNội dung 1.\nĐiều 2. Đối tượng\nNội dung 2."
    arts, warnings = parse_articles(text)
    assert [a.article for a in arts] == ["1", "2"]
    assert arts[0].chapter == "Chương I - Những Quy Định Chung"
    assert arts[1].text == "Nội dung 2."
    assert not warnings


def test_non_monotonic_heading_is_content():
    text = "Điều 1. A\nx\nĐiều 90. Sai\ny\nĐiều 2. B\nz"
    arts, warnings = parse_articles(text)
    assert [a.article for a in arts] == ["1", "2"]
    assert "Điều 90. Sai" in arts[0].lines
    assert warnings


def test_validate_missing():
    arts, _ = parse_articles("Điều 1. A\nx\nĐiều 3. C\nz")
    result = validate(arts, expected=3)
    assert result.missing == [2]
    assert not result.ok
