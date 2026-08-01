from __future__ import annotations

import pyarrow as pa
from huggingface_hub import model_info

from ...config import ExtractionSettings
from ...runtime import load_extraction_model
from .settings import DocumentClassificationSettings


def load_classification_model(
    settings: DocumentClassificationSettings,
    *,
    hf_token: str | None = None,
):
    revision = settings.model_revision
    if revision is None:
        revision = model_info(settings.model_id, token=hf_token).sha
    if not revision:
        raise RuntimeError("Could not resolve an immutable model revision")
    extraction_settings = ExtractionSettings(
        prompt=settings.prompt,
        extraction_schema=pa.schema([]),
        model_id=settings.model_id,
        model_revision=revision,
        model_context_tokens=settings.model_context_tokens,
        max_new_tokens=settings.max_new_tokens,
        parquet_compression=settings.parquet_compression,
    )
    model_pipe, tokenizer = load_extraction_model(
        extraction_settings,
        revision,
        hf_token,
    )
    return revision, model_pipe, tokenizer
