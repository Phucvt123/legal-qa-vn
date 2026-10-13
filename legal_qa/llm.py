"""Client LLM tương thích OpenAI: streaming, retry khi 429, chuyển model khi hết quota ngày."""
from __future__ import annotations

import logging
import time
from typing import Iterator

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from .config import settings

logger = logging.getLogger(__name__)
DAILY_HINTS = ("per day", "TPD", "RPD")


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    def __init__(self, models: tuple[str, ...] | None = None):
        if not settings.llm_api_key:
            raise RuntimeError("Thiếu GROQ_API_KEY (đặt trong .env hoặc Secrets của Space).")
        self.models = list(models or settings.llm_models)
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url,
                             timeout=60, max_retries=0)

    def stream(self, messages: list[dict], max_attempts: int = 3) -> Iterator[str]:
        for model in self.models:
            for attempt in range(max_attempts):
                started = False
                try:
                    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
                    resp = self.client.chat.completions.create(
                        model=model, messages=messages, temperature=0, stream=True, **extra)
                    for ev in resp:
                        delta = ev.choices[0].delta.content if ev.choices else None
                        if delta:
                            started = True
                            yield delta
                    return
                except RateLimitError as e:
                    if started:
                        raise LLMUnavailable("Bị giới hạn giữa chừng") from e
                    if any(h in str(e) for h in DAILY_HINTS):
                        logger.warning("%s hết quota ngày, chuyển model dự phòng", model)
                        break
                    wait = 5 * (attempt + 1)
                    logger.warning("%s bị 429, đợi %ss", model, wait)
                    time.sleep(wait)
                except (APIConnectionError, APITimeoutError, InternalServerError) as e:
                    if started:
                        raise LLMUnavailable("Mất kết nối giữa chừng") from e
                    logger.warning("%s lỗi kết nối (%s), thử lại", model, type(e).__name__)
                    time.sleep(2 * (attempt + 1))
        raise LLMUnavailable("Tất cả model đều không khả dụng")
