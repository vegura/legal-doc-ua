from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ExtractionSettings
from ..processing import Paragraph
from .settings import V2ChainSettings


@dataclass
class V2DocumentState:
    document_id: str
    justice_kind: int
    raw_rtf: bytes
    work_dir: Path
    parts_parquet_bytes: bytes | None = None
    paragraphs: tuple[Paragraph, ...] = ()
    part_assignments: list[dict[str, Any]] = field(default_factory=list)
    numbered_text: str = ""
    extraction: dict[str, Any] = field(default_factory=dict)
    handler_outputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    model_calls: int = 0
    final_object: dict[str, Any] | None = None
    final_row: dict[str, Any] | None = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs)

    def artifact_path(self, relative_path: str) -> Path:
        path = self.work_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json_artifact(
        self,
        relative_path: str,
        value: Any,
    ) -> Path:
        path = self.artifact_path(relative_path)
        path.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path


@dataclass(frozen=True)
class V2HandlerContext:
    model_pipe: Any
    tokenizer: Any
    extraction_settings: ExtractionSettings
    chain_settings: V2ChainSettings
    include_base_prompt: bool = True
