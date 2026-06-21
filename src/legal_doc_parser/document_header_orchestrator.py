from __future__ import annotations

import pandas as pd

try:
    from .document_header_batches import extract_document_headers_in_batches
    from .document_pipeline_bigquery import (
        fetch_existing_bigquery_ids,
        upload_dataframe_to_bigquery,
    )
except ImportError:  # pragma: no cover - supports notebook-style local imports
    from document_header_batches import extract_document_headers_in_batches
    from document_pipeline_bigquery import (
        fetch_existing_bigquery_ids,
        upload_dataframe_to_bigquery,
    )


def filter_header_dataset_missing_in_bigquery(
    dataset: pd.DataFrame,
    bigquery_table_id: str,
    id_column: str = "doc_id",
) -> tuple[pd.DataFrame, int]:
    if id_column not in dataset.columns:
        raise KeyError(f"Missing id column in dataset: {id_column}")

    existing_ids = fetch_existing_bigquery_ids(bigquery_table_id, id_column=id_column)
    id_values = dataset[id_column]
    id_strings = id_values.astype("string")
    missing_mask = id_values.notna() & ~id_strings.isin(existing_ids)
    missing_count = int(missing_mask.sum())
    return dataset.loc[missing_mask].copy(), missing_count


def extract_document_headers_with_bigquery_tracking(
    dataset: pd.DataFrame,
    batch_size: int,
    bigquery_table_id: str,
    url_column: str | None = None,
    id_column: str = "doc_id",
    limit: int | None = None,
    max_workers: int = 1,
    timeout: int = 30,
    request_headers: dict[str, str] | None = None,
    sleep_seconds: float = 0,
    write_disposition: str = "WRITE_APPEND",
) -> pd.DataFrame:
    if write_disposition != "WRITE_APPEND" and max_workers > 1:
        raise ValueError(
            "Non-append BigQuery writes must run with max_workers=1 so the first "
            "batch can apply the requested write disposition before later batches append."
        )

    is_first_batch = True

    def _on_batch_complete(batch: pd.DataFrame) -> None:
        nonlocal is_first_batch
        batch_write_disposition = (
            write_disposition if is_first_batch else "WRITE_APPEND"
        )
        upload_dataframe_to_bigquery(
            batch,
            table_id=bigquery_table_id,
            write_disposition=batch_write_disposition,
        )
        is_first_batch = False

    return extract_document_headers_in_batches(
        dataset=dataset,
        batch_size=batch_size,
        url_column=url_column,
        id_column=id_column,
        limit=limit,
        max_workers=max_workers,
        timeout=timeout,
        request_headers=request_headers,
        sleep_seconds=sleep_seconds,
        on_batch_complete=_on_batch_complete,
    )


def extract_missing_document_headers_with_bigquery_tracking(
    dataset: pd.DataFrame,
    batch_size: int,
    bigquery_table_id: str,
    url_column: str | None = None,
    id_column: str = "doc_id",
    limit: int | None = None,
    max_workers: int = 1,
    timeout: int = 30,
    request_headers: dict[str, str] | None = None,
    sleep_seconds: float = 0,
    write_disposition: str = "WRITE_APPEND",
) -> pd.DataFrame:
    filtered_dataset, missing_count = filter_header_dataset_missing_in_bigquery(
        dataset=dataset,
        bigquery_table_id=bigquery_table_id,
        id_column=id_column,
    )
    if missing_count == 0:
        return filtered_dataset

    return extract_document_headers_with_bigquery_tracking(
        dataset=filtered_dataset,
        batch_size=batch_size,
        bigquery_table_id=bigquery_table_id,
        url_column=url_column,
        id_column=id_column,
        limit=limit,
        max_workers=max_workers,
        timeout=timeout,
        request_headers=request_headers,
        sleep_seconds=sleep_seconds,
        write_disposition=write_disposition,
    )
