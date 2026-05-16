from __future__ import annotations

from pathlib import Path

import pandas as pd

from document_pipeline_batches import (
    upload_dataset_documents_to_gcs_in_batches,
)
from document_pipeline_bigquery import (
    fetch_existing_bigquery_ids,
    upload_dataframe_to_bigquery,
)


def filter_dataset_missing_in_bigquery(
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


def upload_documents_with_bigquery_tracking(
    dataset: pd.DataFrame,
    batch_size: int,
    bucket_name: str,
    bigquery_table_id: str,
    temp_dir: Path | None = None,
    url_column: str = "doc_url",
    id_column: str = "doc_id",
    prefix: str = "",
    limit: int | None = None,
    max_workers: int = 1,
    catalog_column: str | None = None,
    filename_id_column: str | None = None,
) -> None:
    def _on_batch_complete(batch: pd.DataFrame) -> None:
        upload_dataframe_to_bigquery(batch, table_id=bigquery_table_id)

    upload_dataset_documents_to_gcs_in_batches(
        dataset=dataset,
        batch_size=batch_size,
        bucket_name=bucket_name,
        temp_dir=temp_dir,
        url_column=url_column,
        id_column=id_column,
        prefix=prefix,
        limit=limit,
        max_workers=max_workers,
        catalog_column=catalog_column,
        filename_id_column=filename_id_column,
        on_batch_complete=_on_batch_complete,
    )


def upload_missing_documents_with_bigquery_tracking(
    dataset: pd.DataFrame,
    batch_size: int,
    bucket_name: str,
    bigquery_table_id: str,
    temp_dir: Path | None = None,
    url_column: str = "doc_url",
    id_column: str = "doc_id",
    prefix: str = "",
    limit: int | None = None,
    max_workers: int = 1,
    catalog_column: str | None = None,
    filename_id_column: str | None = None,
) -> int:
    filtered_dataset, missing_count = filter_dataset_missing_in_bigquery(
        dataset=dataset,
        bigquery_table_id=bigquery_table_id,
        id_column=id_column,
    )
    if missing_count == 0:
        return 0

    upload_documents_with_bigquery_tracking(
        dataset=filtered_dataset,
        batch_size=batch_size,
        bucket_name=bucket_name,
        bigquery_table_id=bigquery_table_id,
        temp_dir=temp_dir,
        url_column=url_column,
        id_column=id_column,
        prefix=prefix,
        limit=limit,
        max_workers=max_workers,
        catalog_column=catalog_column,
        filename_id_column=filename_id_column,
    )
    return missing_count
