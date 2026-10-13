"""Ghép retrieval + LLM, stream câu trả lời và hiển thị nguồn đã trích dẫn."""
from __future__ import annotations

import logging
import re
import time
from typing import Iterator

from .config import settings
from .llm import LLMClient, LLMUnavailable
from .prompts import REFUSAL, build_messages
from .retriever import Retriever

logger = logging.getLogger(__name__)
RE_CITED = re.compile(r"Điều\s+(\d{1,3}[a-z]?)")


class RAGPipeline:
    def __init__(self, retriever: Retriever | None = None, llm: LLMClient | None = None):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()

    @staticmethod
    def _sources_block(answer: str, hits: list[dict]) -> str:
        cited = set(RE_CITED.findall(answer))
        seen, shown = set(), []
        for h in hits:
            if h["article"] in cited and h["aid"] not in seen:
                seen.add(h["aid"])
                shown.append(h)
        title = "Nguồn trích dẫn"
        if not shown:  # câu trả lời không ghi số Điều: hiện các điều đã tra cứu
            title, shown = "Các điều luật đã tra cứu", hits[:3]
        lines = [f"- **Điều {h['article']}. {h['article_title']}** ({h['so_hieu']})" for h in shown]

        unknown = sorted(cited - {h["article"] for h in hits})
        if unknown:  # LLM nhắc tới Điều không có trong ngữ cảnh: dấu hiệu bịa
            logger.warning("Trích dẫn ngoài ngữ cảnh: %s", unknown)
            lines.append(f"\n⚠️ Câu trả lời có nhắc Điều {', '.join(unknown)} nhưng các điều này "
                         "không nằm trong dữ liệu đã tra cứu, hãy đối chiếu văn bản gốc.")
        return f"\n\n---\n**{title}:**\n" + "\n".join(lines)

    def stream_answer(self, question: str) -> Iterator[str]:
        q = (question or "").strip()
        if not q:
            yield "Bạn hãy nhập câu hỏi."
            return
        if len(q) > settings.max_question_chars:
            yield f"Câu hỏi quá dài (tối đa {settings.max_question_chars} ký tự)."
            return

        t0 = time.perf_counter()
        hits = self.retriever.search(q)
        t_retrieve = time.perf_counter() - t0

        answer, ttft = "", None
        try:
            for delta in self.llm.stream(build_messages(q, hits)):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                answer += delta
                yield answer
        except LLMUnavailable:
            logger.exception("LLM không khả dụng")
            yield (answer + "\n\n" if answer else "") + \
                "Hệ thống đang quá tải hoặc đã hết lượt dùng miễn phí trong ngày. Vui lòng thử lại sau."
            return

        if not answer.strip():
            yield "Không nhận được câu trả lời từ mô hình, vui lòng thử lại."
            return
        if REFUSAL not in answer:
            yield answer + self._sources_block(answer, hits)

        logger.info("retrieve=%.0fms ttft=%.2fs total=%.2fs refused=%s top=%s",
                    t_retrieve * 1000, ttft or -1, time.perf_counter() - t0,
                    REFUSAL in answer, [h["aid"] for h in hits])
