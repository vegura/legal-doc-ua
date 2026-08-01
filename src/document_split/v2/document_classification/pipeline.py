from __future__ import annotations

from typing import Any

from google.cloud import bigquery, storage
from tqdm.auto import tqdm

from ..document_text_parsing.cloud import ParagraphRunLogger
from ..document_text_parsing.runtime import create_google_cloud_clients
from .bigquery import (
    ClassificationSource,
    iter_documents_for_classification,
    mark_documents_classified,
)
from .cloud import (
    classification_result_exists,
    classification_result_object_path,
    classification_source_object_path,
    download_paragraph_parquet,
    upload_classification_parquet_atomically,
)
from .processing import classify_paragraph_parquet
from .runtime import load_classification_model
from .settings import (
    DEFAULT_DOCUMENT_CLASSIFICATION_SETTINGS,
    DocumentClassificationSettings,
)


def run_document_classification_pipeline(
    settings: DocumentClassificationSettings = (
        DEFAULT_DOCUMENT_CLASSIFICATION_SETTINGS
    ),
    *,
    storage_client: storage.Client | None = None,
    bigquery_client: bigquery.Client | None = None,
    model_pipe: Any | None = None,
    tokenizer: Any | None = None,
    hf_token: str | None = None,
) -> dict[str, int]:
    settings.validate(production=True)
    if (model_pipe is None) != (tokenizer is None):
        raise ValueError("model_pipe and tokenizer must be provided together")

    if storage_client is None or bigquery_client is None:
        clients = create_google_cloud_clients(
            project_id=settings.project_id,
            auth_mode=settings.auth_mode,
            colab_service_account_secret=(
                settings.colab_service_account_secret
            ),
        )
        active_storage = storage_client or clients.storage
        active_bigquery = bigquery_client or clients.bigquery
    else:
        active_storage = storage_client
        active_bigquery = bigquery_client

    if model_pipe is None or tokenizer is None:
        _revision, active_model, active_tokenizer = (
            load_classification_model(settings, hf_token=hf_token)
        )
    else:
        active_model = model_pipe
        active_tokenizer = tokenizer

    logger = ParagraphRunLogger(
        storage_client=active_storage,
        bucket_name=settings.bucket,
        destination_prefix=(
            f"{settings.normalized_document_prefix}/_classification"
        ),
        flush_size=settings.run_log_flush_size,
    )
    counters = {
        "processed": 0,
        "skipped_existing": 0,
        "failed": 0,
        "paragraphs": 0,
        "model_calls": 0,
        "bigquery_updated": 0,
    }
    completed: list[ClassificationSource] = []

    def flush_progress() -> None:
        if not completed:
            return
        counters["bigquery_updated"] += mark_documents_classified(
            active_bigquery,
            settings,
            tuple(completed),
        )
        completed.clear()

    try:
        sources = iter_documents_for_classification(
            active_bigquery,
            settings,
        )
        for source in tqdm(
            sources,
            desc="Classifying document paragraphs",
            unit="doc",
            disable=not settings.show_progress,
        ):
            result_object = classification_result_object_path(
                settings, source
            )
            try:
                if settings.skip_existing and classification_result_exists(
                    active_storage, settings, source
                ):
                    counters["skipped_existing"] += 1
                    completed.append(source)
                else:
                    source_bytes = download_paragraph_parquet(
                        active_storage, settings, source
                    )
                    result = classify_paragraph_parquet(
                        document_id=source.document_id,
                        parquet_bytes=source_bytes,
                        model_pipe=active_model,
                        tokenizer=active_tokenizer,
                        settings=settings,
                    )
                    created = upload_classification_parquet_atomically(
                        active_storage,
                        settings,
                        source,
                        result.parquet_bytes,
                    )
                    if created:
                        counters["processed"] += 1
                        counters["paragraphs"] += result.paragraph_count
                        counters["model_calls"] += result.model_calls
                    else:
                        counters["skipped_existing"] += 1
                    completed.append(source)
            except Exception as exc:
                counters["failed"] += 1
                logger.record(
                    {
                        "status": "failed",
                        "document_id": source.document_id,
                        "justice_kind": source.justice_kind,
                        "source_object": classification_source_object_path(
                            settings, source
                        ),
                        "destination_object": result_object,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            if len(completed) >= settings.progress_update_batch_size:
                flush_progress()
        flush_progress()
    finally:
        logger.close(counters)
    return counters
