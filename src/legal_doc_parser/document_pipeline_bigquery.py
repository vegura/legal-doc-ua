from __future__ import annotations

import re
from typing import Optional, Set

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery


def upload_dataframe_to_bigquery(
    dataset: pd.DataFrame,
    table_id: str,
    client: Optional[bigquery.Client] = None,
    write_disposition: str = "WRITE_APPEND",
) -> int:
    bq_client = client or bigquery.Client()
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
    )
    load_job = bq_client.load_table_from_dataframe(
        dataset, table_id, job_config=job_config
    )
    load_job.result()
    return load_job.output_rows or 0


def fetch_existing_bigquery_ids(
    table_id: str,
    id_column: str,
    client: Optional[bigquery.Client] = None,
) -> Set[str]:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", id_column):
        raise ValueError(f"Invalid id_column name: {id_column}")

    bq_client = client or bigquery.Client()
    query = (
        f"SELECT DISTINCT `{id_column}` AS id "
        f"FROM `{table_id}` "
        f"WHERE `{id_column}` IS NOT NULL"
    )

    try:
        query_job = bq_client.query(query)
        return {str(row.id) for row in query_job.result()}
    except NotFound:
        return set()
