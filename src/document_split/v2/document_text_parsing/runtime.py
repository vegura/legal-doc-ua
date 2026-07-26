from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from google.cloud import bigquery, storage


StorageAuthMode = Literal[
    "colab_secret",
    "colab_user",
    "adc",
]


@dataclass(frozen=True)
class DocumentTextParsingClients:
    storage: storage.Client
    bigquery: bigquery.Client


def _resolve_google_cloud_auth(
    *,
    project_id: str,
    auth_mode: StorageAuthMode,
    colab_service_account_secret: str,
) -> tuple[str | None, Any | None]:
    if auth_mode == "colab_secret":
        try:
            from google.colab import userdata
        except ImportError as exc:
            raise RuntimeError(
                "auth_mode='colab_secret' requires Google Colab"
            ) from exc

        raw_secret = userdata.get(colab_service_account_secret)
        if not raw_secret:
            raise RuntimeError(
                f"Missing Colab secret: "
                f"{colab_service_account_secret}"
            )

        from google.oauth2 import service_account

        try:
            service_account_info = json.loads(raw_secret)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Colab secret {colab_service_account_secret!r} "
                "must contain service-account JSON"
            ) from exc
        credentials = (
            service_account.Credentials.from_service_account_info(
                service_account_info
            )
        )
        resolved_project_id = (
            project_id or service_account_info.get("project_id")
        )
        return resolved_project_id or None, credentials

    if auth_mode == "colab_user":
        try:
            from google.colab import auth
        except ImportError as exc:
            raise RuntimeError(
                "auth_mode='colab_user' requires Google Colab"
            ) from exc
        auth.authenticate_user()
        return project_id or None, None

    if auth_mode == "adc":
        return project_id or None, None

    raise ValueError(
        "auth_mode must be 'colab_secret', 'colab_user', or 'adc'"
    )


def create_storage_client(
    *,
    project_id: str,
    auth_mode: StorageAuthMode = "adc",
    colab_service_account_secret: str = "cloud_access",
) -> storage.Client:
    resolved_project_id, credentials = _resolve_google_cloud_auth(
        project_id=project_id,
        auth_mode=auth_mode,
        colab_service_account_secret=(
            colab_service_account_secret
        ),
    )
    try:
        kwargs = {"project": resolved_project_id}
        if credentials is not None:
            kwargs["credentials"] = credentials
        return storage.Client(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Google Cloud Storage credentials are unavailable"
        ) from exc


def create_bigquery_client(
    *,
    project_id: str,
    auth_mode: StorageAuthMode = "adc",
    colab_service_account_secret: str = "cloud_access",
) -> bigquery.Client:
    resolved_project_id, credentials = _resolve_google_cloud_auth(
        project_id=project_id,
        auth_mode=auth_mode,
        colab_service_account_secret=(
            colab_service_account_secret
        ),
    )
    try:
        kwargs = {"project": resolved_project_id}
        if credentials is not None:
            kwargs["credentials"] = credentials
        return bigquery.Client(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "BigQuery credentials are unavailable"
        ) from exc


def create_google_cloud_clients(
    *,
    project_id: str,
    auth_mode: StorageAuthMode = "adc",
    colab_service_account_secret: str = "cloud_access",
) -> DocumentTextParsingClients:
    resolved_project_id, credentials = _resolve_google_cloud_auth(
        project_id=project_id,
        auth_mode=auth_mode,
        colab_service_account_secret=(
            colab_service_account_secret
        ),
    )
    kwargs = {"project": resolved_project_id}
    if credentials is not None:
        kwargs["credentials"] = credentials
    try:
        return DocumentTextParsingClients(
            storage=storage.Client(**kwargs),
            bigquery=bigquery.Client(**kwargs),
        )
    except Exception as exc:
        raise RuntimeError(
            "Google Cloud credentials are unavailable"
        ) from exc
