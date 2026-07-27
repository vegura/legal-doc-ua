from __future__ import annotations

import itertools
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from google.cloud import bigquery, storage
from tqdm.auto import tqdm

from .bigquery import (
    iter_unparsed_documents,
    mark_documents_parsed,
)
from .cloud import (
    ParagraphRunLogger,
    SourceDocument,
    download_source_document,
    list_completed_document_ids,
    paragraph_object_path,
    prepare_manifest,
    upload_document_artifacts,
)
from .processing import parse_document_to_artifacts
from .runtime import create_bigquery_client, create_storage_client
from .settings import (
    DEFAULT_DOCUMENT_TEXT_PARSING_SETTINGS,
    DocumentTextParsingSettings,
)


@dataclass(frozen=True)
class _DocumentOutcome:
    source: SourceDocument
    status: str
    paragraph_count: int = 0
    error_type: str | None = None
    error: str | None = None


def _iter_batches(
    values: Iterable[SourceDocument],
    batch_size: int,
) -> Iterator[tuple[SourceDocument, ...]]:
    iterator = iter(values)
    while True:
        batch = tuple(itertools.islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def _process_document(
    storage_client: storage.Client,
    settings: DocumentTextParsingSettings,
    source: SourceDocument,
) -> _DocumentOutcome:
    try:
        raw_rtf = download_source_document(
            storage_client,
            settings,
            source,
        )
        with tempfile.TemporaryDirectory(
            prefix=f"document-paragraphs-{source.document_id}-"
        ) as temp_dir:
            state = parse_document_to_artifacts(
                document_id=source.document_id,
                justice_kind=source.justice_kind,
                raw_rtf=raw_rtf,
                output_dir=Path(temp_dir),
            )
            created = upload_document_artifacts(
                storage_client,
                settings,
                source,
                state.work_dir,
            )
        if not created:
            return _DocumentOutcome(
                source=source,
                status="skipped_existing",
            )
        return _DocumentOutcome(
            source=source,
            status="processed",
            paragraph_count=len(state.paragraphs),
        )
    except Exception as exc:
        return _DocumentOutcome(
            source=source,
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )


def run_document_text_parsing_pipeline(
    settings: DocumentTextParsingSettings = (
        DEFAULT_DOCUMENT_TEXT_PARSING_SETTINGS
    ),
    storage_client: storage.Client | None = None,
    bigquery_client: bigquery.Client | None = None,
) -> dict[str, int]:
    settings.validate()
    with tqdm(
        total=3,
        desc="Preparing pipeline",
        unit="step",
        disable=not settings.show_progress,
    ) as preparation:
        preparation.set_postfix(stage="cloud clients")
        active_client = storage_client or create_storage_client(
            project_id=settings.project_id,
            auth_mode="adc",
        )
        active_bigquery_client = (
            bigquery_client
            or create_bigquery_client(
                project_id=settings.project_id,
                auth_mode="adc",
            )
        )
        preparation.update(1)

        preparation.set_postfix(stage="manifest")
        prepare_manifest(active_client, settings)
        preparation.update(1)

        preparation.set_postfix(stage="existing outputs")
        completed_by_kind: dict[int, set[str]] = {}
        if not settings.overwrite_existing:
            completed_by_kind = {
                justice_kind: list_completed_document_ids(
                    active_client,
                    settings,
                    justice_kind,
                )
                for justice_kind in settings.justice_kinds
            }
        preparation.update(1)

    sources: Iterable[SourceDocument] = iter_unparsed_documents(
        active_bigquery_client,
        settings,
    )

    logger = ParagraphRunLogger(
        storage_client=active_client,
        bucket_name=settings.destination_bucket,
        destination_prefix=(
            settings.normalized_destination_prefix
        ),
        flush_size=settings.run_log_flush_size,
    )
    counters = {
        "processed": 0,
        "skipped_existing": 0,
        "failed": 0,
        "paragraphs": 0,
    }

    try:
        with (
            ThreadPoolExecutor(
                max_workers=settings.max_workers
            ) as executor,
            tqdm(
                desc="Splitting documents",
                unit="doc",
                disable=not settings.show_progress,
            ) as progress,
        ):
            for batch in _iter_batches(sources, settings.batch_size):
                pending: list[SourceDocument] = []
                completed_sources: list[SourceDocument] = []
                for source in batch:
                    if (
                        not settings.overwrite_existing
                        and source.document_id
                        in completed_by_kind[source.justice_kind]
                    ):
                        counters["skipped_existing"] += 1
                        completed_sources.append(source)
                        progress.update(1)
                    else:
                        pending.append(source)
                outcomes = executor.map(
                    lambda source: _process_document(
                        active_client,
                        settings,
                        source,
                    ),
                    pending,
                )
                for outcome in outcomes:
                    counters[outcome.status] += 1
                    if outcome.status == "processed":
                        counters["paragraphs"] += (
                            outcome.paragraph_count
                        )
                        completed_sources.append(outcome.source)
                    elif outcome.status == "skipped_existing":
                        completed_sources.append(outcome.source)
                    elif outcome.status == "failed":
                        logger.record(
                            {
                                "status": "failed",
                                "document_id": (
                                    outcome.source.document_id
                                ),
                                "justice_kind": (
                                    outcome.source.justice_kind
                                ),
                                "source_object": (
                                    outcome.source.object_name
                                ),
                                "destination_object": (
                                    paragraph_object_path(
                                        settings.normalized_destination_prefix,
                                        outcome.source.justice_kind,
                                        outcome.source.document_id,
                                    )
                                ),
                                "error_type": outcome.error_type,
                                "error": outcome.error,
                            }
                        )
                    progress.update(1)
                mark_documents_parsed(
                    active_bigquery_client,
                    settings,
                    completed_sources,
                )
                progress.set_postfix(
                    processed=counters["processed"],
                    skipped=counters["skipped_existing"],
                    failed=counters["failed"],
                    paragraphs=counters["paragraphs"],
                )
    finally:
        logger.close(counters)
    return counters
