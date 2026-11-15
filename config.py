"""Application settings, model registry and pricing.

Every runtime knob lives here so the CLI and the Streamlit UI share one source of
truth. Values come from `.env` (or real environment variables), which always win
over the defaults below.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent

__version__ = "1.0.0"

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

#: Models this app is known to work with. The live ``/models`` list for a given
#: key is the real source of truth; this list only powers UI dropdowns.
KNOWN_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
)

SUMMARY_STYLES: tuple[str, ...] = ("brief", "detailed", "bullet_points", "executive")


class Price(NamedTuple):
    """USD per one million tokens."""

    input_per_mtok: float
    output_per_mtok: float


#: Published list prices. A model missing from this table still gets exact token
#: accounting - only the dollar estimate is reported as unknown.
PRICING: dict[str, Price] = {
    "gemini-2.5-flash": Price(input_per_mtok=0.30, output_per_mtok=2.50),
    "gemini-2.5-flash-lite": Price(input_per_mtok=0.10, output_per_mtok=0.40),
    "gemini-2.5-pro": Price(input_per_mtok=1.25, output_per_mtok=10.00),
}


class Settings(BaseSettings):
    """Validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    gemini_api_key: SecretStr = Field(description="Google AI Studio API key")
    gemini_model: str = DEFAULT_MODEL
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        validation_alias=AliasChoices("GEMINI_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
        description="Embedding model used to build the chat index",
    )

    # Chunking (token based, measured with tiktoken as a proxy for Gemini tokens).
    chunk_size: int = Field(default=4000, ge=100, le=100_000)
    chunk_overlap: int = Field(default=200, ge=0, le=10_000)

    # Generation.
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=2048, ge=64, le=32_768)
    max_retries: int = Field(default=4, ge=0, le=10)
    retry_base_delay: float = Field(default=2.0, ge=0.1, le=30.0)
    request_timeout: float = Field(default=180.0, ge=5.0, le=600.0)

    # OCR fallback for scanned PDFs.
    ocr_enabled: bool = True
    ocr_max_pages: int = Field(default=20, ge=1, le=200)
    ocr_dpi: int = Field(default=200, ge=72, le=400)

    # Storage.
    data_dir: Path = ROOT / "data" / "uploads"
    chroma_dir: Path = ROOT / "chroma_db"

    log_level: str = "INFO"

    @field_validator("gemini_model")
    @classmethod
    def _normalise_model(cls, value: str) -> str:
        return value.strip().removeprefix("models/")

    @field_validator("embedding_model")
    @classmethod
    def _normalise_embedding_model(cls, value: str) -> str:
        return value.strip().removeprefix("models/")

    @field_validator("log_level")
    @classmethod
    def _normalise_level(cls, value: str) -> str:
        level = value.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return level

    @model_validator(mode="after")
    def _overlap_must_fit(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def api_key(self) -> str:
        """Plaintext key for SDK calls. Never log this."""
        return self.gemini_api_key.get_secret_value()

    def price_for(self, model: str | None = None) -> Price | None:
        return PRICING.get(model or self.gemini_model)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()
