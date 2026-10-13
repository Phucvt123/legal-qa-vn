"""Cấu hình đọc từ biến môi trường. Không hardcode API key."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # chạy local thì đọc .env; trên Spaces dùng Secrets
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

ACTIVE_STATUS = "con_hieu_luc"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    artifacts_dir: Path = ROOT / "artifacts"
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    max_chunk_chars: int = int(os.getenv("MAX_CHUNK_CHARS", "1800"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
    llm_models: tuple[str, ...] = tuple(
        m.strip() for m in os.getenv("LLM_MODELS", "openai/gpt-oss-120b,qwen/qwen3.8-27b").split(",") if m.strip()
    )
    max_question_chars: int = 500


settings = Settings()
