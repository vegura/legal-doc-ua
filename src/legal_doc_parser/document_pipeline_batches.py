from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from document_pipeline import upload_dataset_documents_to_gcs


def iter_dataframe_batches(dataset: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield dataset.iloc[start:end].copy()


def upload_dataset_documents_to_gcs_in_batches(
    dataset: pd.DataFrame,
    batch_size: int,
    bucket_name: str,
    temp_dir: Path | None = None,
    url_column: str = "doc_url",
    id_column: str = "doc_id",
    prefix: str = "",
    limit: int | None = None,
    max_workers: int = 1,
    catalog_column: str | None = None,
    filename_id_column: str | None = None,
    on_batch_complete: Callable[[pd.DataFrame], None] | None = None,
) -> None:
    if limit is not None:
        dataset = dataset.iloc[:limit]

    batches = list(iter_dataframe_batches(dataset, batch_size))

    def _process_batch(batch: pd.DataFrame) -> None:
        upload_dataset_documents_to_gcs(
            dataset=batch,
            bucket_name=bucket_name,
            temp_dir=temp_dir,
            url_column=url_column,
            id_column=id_column,
            prefix=prefix,
            catalog_column=catalog_column,
            filename_id_column=filename_id_column,
            on_batch_complete=on_batch_complete,
        )

    if max_workers <= 1:
        for batch in batches:
            _process_batch(batch)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_process_batch, batches))
