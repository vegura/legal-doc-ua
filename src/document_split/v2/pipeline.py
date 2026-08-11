from __future__ import annotations

import tempfile
from pathlib import Path

from tqdm.auto import tqdm

from ..cloud import (
    RunLogger,
    download_part_document_parquet,
    iter_part_document_ids,
    part_document_object_path,
    upload_parquet_atomically,
)
from ..config import ExtractionSettings, StorageSettings
from ..runtime import load_colab_clients, load_extraction_model
from .cloud import (
    create_v2_document_folder,
    list_completed_v2_document_ids,
    prepare_v2_manifest,
    upload_v2_artifacts,
    v2_result_object_path,
)
from .document import finalize_v2_document
from .handlers import build_parts_v2_chain
from .settings import (
    DEFAULT_V2_CHAIN_SETTINGS,
    DEFAULT_V2_EXTRACTION_SETTINGS,
    DEFAULT_V2_PART_PROCESSING_PROMPTS,
    DEFAULT_V2_STORAGE_SETTINGS,
    V2ChainSettings,
    V2PartProcessingPrompts,
    validate_v2_extraction_settings,
)
from .state import V2DocumentState, V2HandlerContext


def run_v2_pipeline(
    extraction_settings: ExtractionSettings = (
        DEFAULT_V2_EXTRACTION_SETTINGS
    ),
    storage_settings: StorageSettings = DEFAULT_V2_STORAGE_SETTINGS,
    chain_settings: V2ChainSettings = DEFAULT_V2_CHAIN_SETTINGS,
    part_prompts: V2PartProcessingPrompts = (
        DEFAULT_V2_PART_PROCESSING_PROMPTS
    ),
) -> dict[str, int]:
    validate_v2_extraction_settings(extraction_settings)
    storage_settings.validate()
    chain_settings.validate()
    part_prompts.validate(production=True)
    clients = load_colab_clients(storage_settings)
    revision, manifest = prepare_v2_manifest(
        clients.storage,
        extraction_settings,
        storage_settings,
        chain_settings,
        part_prompts,
        clients.hf_token,
    )
    print(
        f"Using V2 chain {extraction_settings.model_id}@{revision} for "
        f"{manifest['identity']['info_version']}"
    )
    model_pipe, tokenizer = load_extraction_model(
        extraction_settings,
        revision,
        clients.hf_token,
    )
    context = V2HandlerContext(
        model_pipe=model_pipe,
        tokenizer=tokenizer,
        extraction_settings=extraction_settings,
        chain_settings=chain_settings,
        include_base_prompt=False,
    )
    chain = build_parts_v2_chain(part_prompts)
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
        "model_calls": 0,
    }
    remaining = storage_settings.limit

    try:
        for justice_kind in storage_settings.justice_kinds:
            if remaining is not None and remaining <= 0:
                break
            completed = (
                list_completed_v2_document_ids(
                    clients.storage,
                    storage_settings.destination_bucket,
                    storage_settings.version_prefix,
                    justice_kind,
                )
                if storage_settings.skip_existing
                else set()
            )
            document_ids = iter_part_document_ids(
                clients.bigquery,
                storage_settings.bigquery_table,
                justice_kind,
                storage_settings.bigquery_page_size,
                remaining,
            )
            for document_id in tqdm(
                document_ids,
                desc=f"v2 justice_kind={justice_kind}",
                unit="doc",
            ):
                if remaining is not None and remaining <= 0:
                    break
                if document_id in completed:
                    counters["skipped_existing"] += 1
                    if remaining is not None:
                        remaining -= 1
                    continue

                document_prefix = create_v2_document_folder(
                    clients.storage,
                    storage_settings.destination_bucket,
                    storage_settings.version_prefix,
                    justice_kind,
                    document_id,
                    source_object=part_document_object_path(
                        storage_settings.parts_prefix,
                        justice_kind,
                        document_id,
                    ),
                )
                result_path = v2_result_object_path(
                    storage_settings.version_prefix,
                    justice_kind,
                    document_id,
                )
                state: V2DocumentState | None = None
                try:
                    parts_parquet_bytes = (
                        download_part_document_parquet(
                            clients.storage,
                            storage_settings.parts_bucket,
                            storage_settings.parts_prefix,
                            justice_kind,
                            document_id,
                        )
                    )
                    with tempfile.TemporaryDirectory(
                        prefix=f"document-v2-{document_id}-"
                    ) as temp_dir:
                        state = V2DocumentState(
                            document_id=str(document_id),
                            justice_kind=int(justice_kind),
                            raw_rtf=b"",
                            work_dir=Path(temp_dir),
                            parts_parquet_bytes=parts_parquet_bytes,
                        )
                        try:
                            chain.handle(state, context)
                        except Exception as chain_error:
                            state.write_json_artifact(
                                "chain_failure.json",
                                {
                                    "document_id": state.document_id,
                                    "error_type": type(
                                        chain_error
                                    ).__name__,
                                    "error": str(chain_error),
                                    "completed_handlers": list(
                                        state.handler_outputs
                                    ),
                                    "model_calls": state.model_calls,
                                },
                            )
                            try:
                                upload_v2_artifacts(
                                    clients.storage,
                                    storage_settings.destination_bucket,
                                    document_prefix,
                                    state.work_dir,
                                )
                            except Exception as artifact_error:
                                state.warnings.append(
                                    "Failed to upload partial artifacts: "
                                    f"{artifact_error}"
                                )
                            raise
                        finalize_v2_document(
                            state,
                            extraction_settings,
                        )
                        upload_v2_artifacts(
                            clients.storage,
                            storage_settings.destination_bucket,
                            document_prefix,
                            state.work_dir,
                        )
                        result_bytes = state.artifact_path(
                            "result.parquet"
                        ).read_bytes()
                        created = upload_parquet_atomically(
                            clients.storage,
                            storage_settings.destination_bucket,
                            result_path,
                            result_bytes,
                        )
                    if created:
                        counters["processed"] += 1
                        counters["paragraphs"] += len(state.paragraphs)
                        counters["model_calls"] += state.model_calls
                    else:
                        counters["skipped_existing"] += 1
                except Exception as exc:
                    counters["failed"] += 1
                    logger.record(
                        {
                            "status": "failed",
                            "document_id": document_id,
                            "justice_kind": justice_kind,
                            "source_object": part_document_object_path(
                                storage_settings.parts_prefix,
                                justice_kind,
                                document_id,
                            ),
                            "destination_object": result_path,
                            "document_prefix": document_prefix,
                            "handler_outputs": (
                                list(state.handler_outputs)
                                if state is not None
                                else []
                            ),
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
