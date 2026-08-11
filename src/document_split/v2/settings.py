from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..config import (
    CRIMINAL_SCHEMA,
    DEFAULT_EXTRACTION_PROMPT,
    ExtractionSettings,
    StorageSettings,
)


V2_INFO_VERSION = "info_version_9"
V2_DISTRIBUTED_INFO_VERSION = "info_version_12"
PART_PROCESSING_PROMPT_PLACEHOLDER = "<DEFINE_PART_PROCESSING_INSTRUCTIONS>"
SAMPLE_PART_PROCESSING_MARKER = "<SAMPLE_ONLY>"

V2PartProcessingMode = Literal["filtered", "sequential"]


@dataclass(frozen=True)
class V2PartProcessingPrompts:
    """User-owned ontology instructions for each routed document part."""

    introductory: str = PART_PROCESSING_PROMPT_PLACEHOLDER
    reasoning: str = PART_PROCESSING_PROMPT_PLACEHOLDER
    operative: str = PART_PROCESSING_PROMPT_PLACEHOLDER

    def validate(self, *, production: bool) -> None:
        for name, prompt in (
            ("introductory", self.introductory),
            ("reasoning", self.reasoning),
            ("operative", self.operative),
        ):
            if not prompt.strip():
                raise ValueError(f"{name} processing prompt cannot be empty")
            if production and (
                PART_PROCESSING_PROMPT_PLACEHOLDER in prompt
                or SAMPLE_PART_PROCESSING_MARKER in prompt
            ):
                raise ValueError(
                    f"Define the {name} ontology-processing instructions "
                    "before starting a production run"
                )


@dataclass(frozen=True)
class V2ChainSettings:
    target_batch_tokens: int = 6_000
    overlap_tokens: int = 256
    part_processing_mode: V2PartProcessingMode = "filtered"

    def validate(self) -> None:
        if self.target_batch_tokens <= 0:
            raise ValueError("target_batch_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if self.part_processing_mode not in {"filtered", "sequential"}:
            raise ValueError(
                "part_processing_mode must be 'filtered' or 'sequential'"
            )


DEFAULT_V2_CHAIN_SETTINGS = V2ChainSettings()
DEFAULT_V2_PART_PROCESSING_PROMPTS = V2PartProcessingPrompts()
SAMPLE_V2_PART_PROCESSING_PROMPTS = V2PartProcessingPrompts(
    introductory=(
        "HANDLER: introductory_part\n<SAMPLE_ONLY>\n"
        "This is a routing demonstration. Return introductory_part as null."
    ),
    reasoning=(
        "HANDLER: court_reasoning_part\n<SAMPLE_ONLY>\n"
        "This is a routing demonstration. Return reasoning_part as null."
    ),
    operative=(
        "HANDLER: result_part\n<SAMPLE_ONLY>\n"
        "This is a routing demonstration. Return operative_part as null."
    ),
)
DEFAULT_V2_EXTRACTION_SETTINGS = ExtractionSettings(
    prompt=DEFAULT_EXTRACTION_PROMPT,
    extraction_schema=CRIMINAL_SCHEMA,
    max_new_tokens=8_192,
    json_retries=0,
)
DEFAULT_V2_STORAGE_SETTINGS = StorageSettings(
    info_version=V2_DISTRIBUTED_INFO_VERSION,
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
