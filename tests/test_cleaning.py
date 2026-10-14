import pytest

from legal_qa.ingest.cleaning import clean_text, load_phrase_fixes


def test_eth_and_ocr_heading():
    out, _ = clean_text("Ðiểu 5, Phạm vi điều chỉnh\nNội dung.")
    assert out.splitlines()[0] == "Điều 5. Phạm vi điều chỉnh"


def test_split_khoan_merged_into_title():
    out, _ = clean_text("Điều 68. Quyền thương lượng tập thể i. Tổ chức đại diện có quyền.")
    assert out.splitlines() == ["Điều 68. Quyền thương lượng tập thể", "1. Tổ chức đại diện có quyền."]


def test_khoan_marker_fix():
    out, _ = clean_text("Điều 1. A\ni. Người lao động có quyền.")
    assert out.splitlines()[1] == "1. Người lao động có quyền."


def test_join_broken_line():
    out, _ = clean_text("1. Người lao động có quyền\nnghỉ hằng năm.")
    assert out == "1. Người lao động có quyền nghỉ hằng năm."


def test_cross_reference_not_heading():
    out, _ = clean_text("Điều 35 của Bộ luật này")
    assert out == "Điều 35 của Bộ luật này"


def test_remove_junk_and_page_numbers():
    out, report = clean_text("Điều 1. A\nĐang theo dõi\n12\n1. Nội dung.")
    assert out.splitlines() == ["Điều 1. A", "1. Nội dung."]
    assert report.counts["xóa dòng rác / header / số trang"] == 2


def test_phrase_fix_applied_and_numbers_kept():
    out, _ = clean_text("1. Tỷ lệ tối thiếu là 06 tháng.")
    assert out == "1. Tỷ lệ tối thiểu là 06 tháng."


def test_phrase_fix_with_digits_rejected(tmp_path):
    p = tmp_path / "fixes.csv"
    p.write_text("wrong,right\n8 tháng,6 tháng\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_phrase_fixes(p)
def test_ocr_dien_heading():
    out, _ = clean_text("Điền 194. Thời hiệu yêu cầu giải quyết\n1. Thời hiệu là 06 tháng.")
    assert out.splitlines()[0] == "Điều 194. Thời hiệu yêu cầu giải quyết"
    assert "06 tháng" in out  # chữ số giữ nguyên


def test_reference_not_treated_as_heading():
    line = "Điều 35 của Bộ luật này quy định quyền đơn phương chấm dứt."
    out, _ = clean_text(line)
    assert out == line
