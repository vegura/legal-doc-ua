from __future__ import annotations

from dataclasses import dataclass

from ..config import (
    CRIMINAL_PROMPT,
    CRIMINAL_SCHEMA,
    ExtractionSettings,
    StorageSettings,
)


V2_INFO_VERSION = "info_version_9"


@dataclass(frozen=True)
class V2ChainSettings:
    target_batch_tokens: int = 6_000
    overlap_tokens: int = 256

    def validate(self) -> None:
        if self.target_batch_tokens <= 0:
            raise ValueError("target_batch_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")


DEFAULT_V2_CHAIN_SETTINGS = V2ChainSettings()
DEFAULT_V2_EXTRACTION_SETTINGS = ExtractionSettings(
    prompt=CRIMINAL_PROMPT,
    extraction_schema=CRIMINAL_SCHEMA,
    max_new_tokens=8_192,
    json_retries=0,
)
DEFAULT_V2_STORAGE_SETTINGS = StorageSettings(
    info_version=V2_INFO_VERSION,
    justice_kinds=(2,),
)


def validate_v2_extraction_settings(
    settings: ExtractionSettings,
) -> None:
    settings.validate(production=True)
    if not settings.extraction_schema.equals(CRIMINAL_SCHEMA):
        raise ValueError(
            "V2 handlers require the canonical CRIMINAL_SCHEMA unchanged"
        )
    if settings.max_new_tokens >= settings.model_context_tokens:
        raise ValueError(
            "V2 max_new_tokens must be below model_context_tokens"
        )
