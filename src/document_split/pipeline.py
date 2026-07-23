from __future__ import annotations

from tqdm.auto import tqdm

from .cloud import (
    RunLogger,
    destination_object_path,
    download_source_rtf,
    iter_document_ids,
    list_completed_document_ids,
    prepare_version_manifest,
    source_object_path,
    upload_parquet_atomically,
)
from .config import (
    DEFAULT_EXTRACTION_SETTINGS,
    DEFAULT_STORAGE_SETTINGS,
    ExtractionSettings,
    StorageSettings,
)
from .processing import parse_document_to_parquet
from .runtime import load_colab_clients, load_extraction_model


def run_pipeline(
    extraction: ExtractionSettings = DEFAULT_EXTRACTION_SETTINGS,
    storage_settings: StorageSettings = DEFAULT_STORAGE_SETTINGS,
) -> dict[str, int]:
    extraction.validate(production=True)
    storage_settings.validate()

    clients = load_colab_clients(storage_settings)
    revision, manifest = prepare_version_manifest(
        clients.storage,
        extraction,
        storage_settings,
        clients.hf_token,
    )
    print(
        f"Using {extraction.model_id}@{revision} for "
        f"{manifest['identity']['info_version']}"
    )
    model_pipe, tokenizer = load_extraction_model(
        extraction,
        revision,
        clients.hf_token,
    )
    logger = RunLogger(
        storage_client=clients.storage,
        bucket_name=storage_settings.destination_bucket,
        version_prefix=storage_settings.version_prefix,
        flush_size=storage_settings.run_log_flush_size,
    )
    counters = {
        "processed": 0,
        "skipped_existing": 0,
        "failed": 0,
        "paragraphs": 0,
    }
    remaining = storage_settings.limit

    try:
        for justice_kind in storage_settings.justice_kinds:
            if remaining is not None and remaining <= 0:
                break

            completed = (
                list_completed_document_ids(
                    clients.storage,
                    storage_settings.destination_bucket,
                    storage_settings.version_prefix,
                    justice_kind,
                )
                if storage_settings.skip_existing
                else set()
            )
            document_ids = iter_document_ids(
                clients.bigquery,
                storage_settings.bigquery_table,
                justice_kind,
                storage_settings.bigquery_page_size,
                remaining,
            )
            for document_id in tqdm(
                document_ids,
                desc=f"justice_kind={justice_kind}",
                unit="doc",
            ):
                if remaining is not None and remaining <= 0:
                    break
                if document_id in completed:
                    counters["skipped_existing"] += 1
                    if remaining is not None:
                        remaining -= 1
                    continue

                output_path = destination_object_path(
                    storage_settings.version_prefix,
                    justice_kind,
                    document_id,
                )
                try:
                    raw_rtf = download_source_rtf(
                        clients.storage,
                        storage_settings.source_bucket,
                        justice_kind,
                        document_id,
                    )
                    parquet_bytes, paragraph_count = (
                        parse_document_to_parquet(
                            document_id,
                            raw_rtf,
                            model_pipe,
                            tokenizer,
                            extraction,
                        )
                    )
                    created = upload_parquet_atomically(
                        clients.storage,
                        storage_settings.destination_bucket,
                        output_path,
                        parquet_bytes,
                    )
                    if created:
                        counters["processed"] += 1
                        counters["paragraphs"] += paragraph_count
                    else:
                        counters["skipped_existing"] += 1
                except Exception as exc:
                    counters["failed"] += 1
                    logger.record(
                        {
                            "status": "failed",
                            "document_id": document_id,
                            "justice_kind": justice_kind,
                            "source_object": source_object_path(
                                justice_kind,
                                document_id,
                            ),
                            "destination_object": output_path,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                finally:
                    if remaining is not None:
                        remaining -= 1
    finally:
        logger.close(counters)
    return counters

