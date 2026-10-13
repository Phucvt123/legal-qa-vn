"""Prompt (giữ tinh thần prompt POC E5: correct 0.95, faithful 1.0)."""
from __future__ import annotations

from .config import ACTIVE_STATUS

REFUSAL = "Không tìm thấy căn cứ pháp lý trong dữ liệu hiện có."
DISCLAIMER = "Thông tin chỉ mang tính tham khảo, không thay thế tư vấn pháp lý."

SYSTEM_PROMPT = f"""Bạn là trợ lý tra cứu pháp luật lao động Việt Nam. Quy tắc:
1. CHỈ dùng thông tin trong phần Ngữ cảnh. Không dùng kiến thức bên ngoài.
2. Mỗi ý phải kèm trích dẫn dạng (Điều X, <số hiệu văn bản>).
3. Nếu Ngữ cảnh không đủ để trả lời hoặc câu hỏi nằm ngoài lĩnh vực lao động, chỉ trả lời đúng câu: "{REFUSAL}"
4. Nếu căn cứ thuộc văn bản HẾT HIỆU LỰC, phải nói rõ và nêu văn bản thay thế.
5. Trả lời ngắn gọn, dễ hiểu, giữ nguyên chính xác các con số trong điều luật.
6. Nếu có trả lời, kết thúc bằng câu: "{DISCLAIMER}" """


def format_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        if h["status"] == ACTIVE_STATUS:
            status = "còn hiệu lực"
        else:
            status = f"HẾT HIỆU LỰC, thay bởi {h.get('replaced_by') or 'văn bản khác'}"
        blocks.append(f"[{h['so_hieu']} - {status}]\n{h['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Ngữ cảnh:\n{format_context(hits)}\n\nCâu hỏi: {question}"},
    ]
