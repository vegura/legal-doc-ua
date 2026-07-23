from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from google.cloud import bigquery, storage
from huggingface_hub import login
from transformers import pipeline

from .config import ExtractionSettings, StorageSettings


@dataclass(frozen=True)
class RuntimeClients:
    bigquery: bigquery.Client
    storage: storage.Client
    hf_token: str | None


def load_colab_clients(storage_settings: StorageSettings) -> RuntimeClients:
    from google.colab import userdata
    from google.oauth2 import service_account

    raw_secret = userdata.get("cloud_access")
    if not raw_secret:
        raise RuntimeError("Missing Colab secret: cloud_access")
    service_account_info = json.loads(raw_secret)
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )

    try:
        hf_token = userdata.get("HF_TOKEN")
    except Exception:
        hf_token = None
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    return RuntimeClients(
        bigquery=bigquery.Client(
            project=storage_settings.project_id,
            credentials=credentials,
        ),
        storage=storage.Client(
            project=storage_settings.project_id,
            credentials=credentials,
        ),
        hf_token=hf_token,
    )


def load_extraction_model(
    settings: ExtractionSettings,
    revision: str,
    hf_token: str | None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled Colab runtime is required")

    model_pipe = pipeline(
        "image-text-to-text",
        model=settings.model_id,
        revision=revision,
        token=hf_token,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    tokenizer = getattr(model_pipe, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The loaded pipeline did not expose a tokenizer")
    return model_pipe, tokenizer

