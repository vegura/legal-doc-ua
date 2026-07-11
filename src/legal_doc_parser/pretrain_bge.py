from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, Sequence

import pyarrow.parquet as pq
import torch
from google.cloud import storage
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import IterableDataset, get_worker_info
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .pretraining_data import (
    DEFAULT_DATASET_PREFIX,
    MODEL_NAME,
    validate_distinct_buckets,
)


_CHECKPOINT_MARKER_RE = re.compile(r"checkpoint-(\d+)/_SUCCESS$")
LOGGER = logging.getLogger("legal_doc_parser.pretraining")


@dataclass(frozen=True)
class PretrainingConfig:
    project_id: str
    source_bucket: str
    dataset_bucket: str
    checkpoint_bucket: str
    run_id: str
    dataset_prefix: str = DEFAULT_DATASET_PREFIX
    checkpoint_prefix: str = "legal-bge-pretraining"
    model_name: str = MODEL_NAME
    local_root: str = "/content/legal-pretraining"
    new_vocabulary_tokens: int = 16_000
    tokenizer_min_frequency: int = 2
    tokenizer_max_documents: int = 200_000
    max_sequence_length: int = 512
    min_chunk_tokens: int = 32
    mlm_probability: float = 0.15
    max_steps: int = 100_000
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-4
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    max_eval_sequences: int = 2_048
    cache_max_gb: float = 4.0
    dataloader_workers: int = 0
    remote_checkpoint_limit: int = 5
    seed: int = 42
    resume: bool = True

    def validate(self) -> None:
        validate_distinct_buckets(
            self.source_bucket, self.dataset_bucket, self.checkpoint_bucket
        )
        if not self.run_id.strip():
            raise ValueError("run_id is required so Colab sessions can resume the same run")
        if self.new_vocabulary_tokens < 0:
            raise ValueError("new_vocabulary_tokens cannot be negative")
        if self.tokenizer_max_documents <= 0:
            raise ValueError("tokenizer_max_documents must be positive")
        if not 0 < self.max_sequence_length <= 512:
            raise ValueError("max_sequence_length must be in the BGE range 1..512")
        if not 0 < self.min_chunk_tokens <= self.max_sequence_length - 2:
            raise ValueError("min_chunk_tokens must fit inside max_sequence_length")
        if not 0 < self.mlm_probability < 1:
            raise ValueError("mlm_probability must be between 0 and 1")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive for a streaming dataset")
        if self.per_device_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch size and gradient accumulation must be positive")
        if self.eval_steps <= 0 or self.save_steps <= 0 or self.logging_steps <= 0:
            raise ValueError("logging, evaluation, and save intervals must be positive")
        if self.eval_steps != self.save_steps:
            raise ValueError("eval_steps and save_steps must match for best-model loading")
        if self.cache_max_gb <= 0:
            raise ValueError("cache_max_gb must be positive")
        if self.dataloader_workers < 0:
            raise ValueError("dataloader_workers cannot be negative")
        if self.remote_checkpoint_limit <= 0:
            raise ValueError("remote_checkpoint_limit must be positive")

    @property
    def normalized_dataset_prefix(self) -> str:
        return self.dataset_prefix.strip("/")

    @property
    def normalized_checkpoint_prefix(self) -> str:
        return self.checkpoint_prefix.strip("/")

    @property
    def local_run_dir(self) -> Path:
        return Path(self.local_root) / self.run_id


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got: {uri}")
    bucket_and_object = uri[5:]
    bucket, separator, object_name = bucket_and_object.partition("/")
    if not bucket or not separator or not object_name:
        raise ValueError(f"GCS URI must include a bucket and object: {uri}")
    return bucket, object_name


def merge_vocabularies(
    base_vocabulary: dict[str, int],
    candidate_tokens: Sequence[str],
    max_new_tokens: int,
) -> dict[str, int]:
    ordered_base = [
        token for token, _ in sorted(base_vocabulary.items(), key=lambda item: item[1])
    ]
    if [base_vocabulary[token] for token in ordered_base] != list(
        range(len(ordered_base))
    ):
        raise ValueError("base tokenizer vocabulary IDs must be contiguous")
    merged = dict(base_vocabulary)
    for token in candidate_tokens:
        if token in merged:
            continue
        merged[token] = len(merged)
        if len(merged) >= len(ordered_base) + max_new_tokens:
            break
    return merged


class ShardMaterializer(Protocol):
    def materialize(self, uri: str) -> Path: ...


class GCSShardCache:
    def __init__(
        self,
        cache_dir: Path,
        max_bytes: int,
        client: storage.Client | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.max_bytes = max_bytes
        self.client = client or storage.Client()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _local_path(self, uri: str) -> Path:
        digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
        name = Path(parse_gcs_uri(uri)[1]).name
        return self.cache_dir / f"{digest}-{name}"

    def _evict(self, protected: Path) -> None:
        files = [path for path in self.cache_dir.iterdir() if path.is_file()]
        total = sum(path.stat().st_size for path in files)
        for path in sorted(files, key=lambda item: item.stat().st_mtime):
            if total <= self.max_bytes:
                return
            if path == protected:
                continue
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size

    def materialize(self, uri: str) -> Path:
        local_path = self._local_path(uri)
        if local_path.exists():
            os.utime(local_path, None)
            return local_path
        bucket_name, object_name = parse_gcs_uri(uri)
        temporary_path = local_path.with_suffix(local_path.suffix + ".partial")
        self.client.bucket(bucket_name).blob(object_name).download_to_filename(
            str(temporary_path), checksum="auto"
        )
        temporary_path.replace(local_path)
        self._evict(local_path)
        return local_path


class LocalShardMaterializer:
    """Test/local implementation with the same interface as the GCS cache."""

    def materialize(self, uri: str) -> Path:
        return Path(uri)


def _prefetched_paths(
    shard_uris: Sequence[str], materializer: ShardMaterializer
) -> Iterator[Path]:
    if not shard_uris:
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        future: Future[Path] = executor.submit(materializer.materialize, shard_uris[0])
        for index in range(len(shard_uris)):
            current = future.result()
            if index + 1 < len(shard_uris):
                future = executor.submit(
                    materializer.materialize, shard_uris[index + 1]
                )
            yield current


class GCSParquetMLMDataset(IterableDataset):
    def __init__(
        self,
        shard_uris: Sequence[str],
        tokenizer: Any,
        materializer: ShardMaterializer,
        *,
        max_sequence_length: int = 512,
        min_chunk_tokens: int = 32,
        seed: int = 42,
        repeat: bool = True,
        max_sequences: int | None = None,
    ) -> None:
        super().__init__()
        if not shard_uris:
            raise ValueError("At least one Parquet shard is required")
        self.shard_uris = list(shard_uris)
        self.tokenizer = tokenizer
        self.materializer = materializer
        self.max_sequence_length = max_sequence_length
        self.min_chunk_tokens = min_chunk_tokens
        self.seed = seed
        self.repeat = repeat
        self.max_sequences = max_sequences

    def _sequences_from_text(self, text: str) -> Iterator[dict[str, list[int]]]:
        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        content_length = self.max_sequence_length - 2
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        if cls_token_id is None or sep_token_id is None:
            raise ValueError("The tokenizer must define BERT CLS and SEP token IDs")
        for start in range(0, len(token_ids), content_length):
            content = token_ids[start : start + content_length]
            if len(content) < self.min_chunk_tokens:
                continue
            input_ids = [cls_token_id, *content, sep_token_id]
            yield {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        worker_shards = self.shard_uris[worker_id::worker_count]
        if not worker_shards:
            return

        emitted = 0
        epoch = 0
        while True:
            rng = random.Random(self.seed + epoch)
            ordered_shards = list(worker_shards)
            rng.shuffle(ordered_shards)
            for local_path in _prefetched_paths(ordered_shards, self.materializer):
                table = pq.read_table(local_path, columns=["text"])
                texts = table.column("text").to_pylist()
                rng.shuffle(texts)
                for text in texts:
                    if not text:
                        continue
                    for sequence in self._sequences_from_text(str(text)):
                        yield sequence
                        emitted += 1
                        if self.max_sequences is not None and emitted >= self.max_sequences:
                            return
            if not self.repeat:
                return
            epoch += 1


def list_parquet_shards(
    client: storage.Client, bucket_name: str, prefix: str, split: str
) -> list[str]:
    object_prefix = f"{prefix.strip('/')}/clean/{split}/"
    names = sorted(
        blob.name
        for blob in client.list_blobs(bucket_name, prefix=object_prefix)
        if blob.name.endswith(".parquet") and (blob.size is None or blob.size > 0)
    )
    if not names:
        raise FileNotFoundError(
            f"No Parquet shards found under gs://{bucket_name}/{object_prefix}"
        )
    return [f"gs://{bucket_name}/{name}" for name in names]


def iter_clean_texts(
    shard_uris: Sequence[str],
    materializer: ShardMaterializer,
    max_documents: int | None = None,
) -> Iterator[str]:
    emitted = 0
    for local_path in _prefetched_paths(shard_uris, materializer):
        parquet = pq.ParquetFile(local_path)
        for batch in parquet.iter_batches(columns=["text"], batch_size=256):
            for text in batch.column(0).to_pylist():
                if text:
                    yield str(text)
                    emitted += 1
                    if max_documents is not None and emitted >= max_documents:
                        return


def _upload_directory(
    client: storage.Client,
    bucket_name: str,
    object_prefix: str,
    local_dir: Path,
    *,
    marker: bool = True,
) -> None:
    bucket = client.bucket(bucket_name)
    prefix = object_prefix.strip("/")
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_dir).as_posix()
        bucket.blob(f"{prefix}/{relative}").upload_from_filename(
            str(path), checksum="auto"
        )
    if marker:
        bucket.blob(f"{prefix}/_SUCCESS").upload_from_string(
            "ok\n", content_type="text/plain"
        )


def _download_directory(
    client: storage.Client,
    bucket_name: str,
    object_prefix: str,
    local_dir: Path,
) -> Path:
    prefix = object_prefix.strip("/") + "/"
    local_dir.mkdir(parents=True, exist_ok=True)
    found = False
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        relative = blob.name[len(prefix) :]
        if not relative or relative == "_SUCCESS":
            continue
        destination = local_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination), checksum="auto")
        found = True
    if not found:
        raise FileNotFoundError(f"No artifacts found at gs://{bucket_name}/{prefix}")
    return local_dir


def build_or_load_tokenizer(
    config: PretrainingConfig,
    train_shards: Sequence[str],
    materializer: ShardMaterializer,
    client: storage.Client,
) -> Any:
    tokenizer_prefix = (
        f"{config.normalized_dataset_prefix}/tokenizer-"
        f"plus-{config.new_vocabulary_tokens}"
    )
    marker = client.bucket(config.dataset_bucket).blob(f"{tokenizer_prefix}/_SUCCESS")
    local_dir = config.local_run_dir / "tokenizer"
    if marker.exists():
        _download_directory(
            client, config.dataset_bucket, tokenizer_prefix, local_dir
        )
        return AutoTokenizer.from_pretrained(local_dir, use_fast=True)

    base_tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if config.new_vocabulary_tokens == 0:
        local_dir.mkdir(parents=True, exist_ok=True)
        base_tokenizer.save_pretrained(local_dir)
        _upload_directory(
            client, config.dataset_bucket, tokenizer_prefix, local_dir
        )
        return base_tokenizer

    candidate_tokenizer = BertWordPieceTokenizer(
        lowercase=bool(getattr(base_tokenizer, "do_lower_case", True)),
        strip_accents=False,
    )
    candidate_tokenizer.train_from_iterator(
        iter_clean_texts(
            train_shards,
            materializer,
            max_documents=config.tokenizer_max_documents,
        ),
        vocab_size=len(base_tokenizer) + config.new_vocabulary_tokens,
        min_frequency=config.tokenizer_min_frequency,
        special_tokens=list(base_tokenizer.all_special_tokens),
        show_progress=True,
    )
    candidate_vocabulary = candidate_tokenizer.get_vocab()
    candidate_tokens = [
        token
        for token, _ in sorted(
            candidate_vocabulary.items(), key=lambda item: item[1]
        )
    ]
    merged_vocabulary = merge_vocabularies(
        base_tokenizer.get_vocab(),
        candidate_tokens,
        config.new_vocabulary_tokens,
    )
    tokenizer = BertTokenizerFast(
        vocab=merged_vocabulary,
        do_lower_case=bool(getattr(base_tokenizer, "do_lower_case", True)),
        strip_accents=False,
        model_max_length=config.max_sequence_length,
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(local_dir)
    coverage = {
        "base_vocabulary_size": len(base_tokenizer),
        "final_vocabulary_size": len(tokenizer),
        "added_tokens": len(tokenizer) - len(base_tokenizer),
        "model_specification": config.model_name,
    }
    (local_dir / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8"
    )
    _upload_directory(client, config.dataset_bucket, tokenizer_prefix, local_dir)
    return tokenizer


def completed_checkpoint_steps(object_names: Iterable[str]) -> list[int]:
    steps: set[int] = set()
    for name in object_names:
        match = _CHECKPOINT_MARKER_RE.search(name)
        if match:
            steps.add(int(match.group(1)))
    return sorted(steps)


def configure_pretraining_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = LOGGER
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


class GCSCheckpointStore:
    def __init__(
        self,
        client: storage.Client,
        bucket_name: str,
        prefix: str,
        run_id: str,
        remote_limit: int = 5,
    ) -> None:
        self.client = client
        self.bucket_name = bucket_name
        self.base_prefix = f"{prefix.strip('/')}/runs/{run_id}"
        self.remote_limit = remote_limit

    @property
    def checkpoints_prefix(self) -> str:
        return f"{self.base_prefix}/checkpoints"

    def completed_steps(self) -> list[int]:
        names = (
            blob.name
            for blob in self.client.list_blobs(
                self.bucket_name, prefix=f"{self.checkpoints_prefix}/"
            )
        )
        return completed_checkpoint_steps(names)

    def upload_checkpoint(self, local_checkpoint: Path, step: int) -> None:
        prefix = f"{self.checkpoints_prefix}/checkpoint-{step}"
        marker = self.client.bucket(self.bucket_name).blob(f"{prefix}/_SUCCESS")
        if marker.exists():
            return
        _upload_directory(
            self.client,
            self.bucket_name,
            prefix,
            local_checkpoint,
            marker=True,
        )

    def download_latest(self, local_root: Path) -> Path | None:
        steps = self.completed_steps()
        if not steps:
            return None
        latest = steps[-1]
        destination = local_root / f"checkpoint-{latest}"
        if destination.exists():
            return destination
        return _download_directory(
            self.client,
            self.bucket_name,
            f"{self.checkpoints_prefix}/checkpoint-{latest}",
            destination,
        )

    def prune(self, protected_step: int | None = None) -> None:
        completed = self.completed_steps()
        keep = set(completed[-self.remote_limit :])
        if protected_step is not None:
            keep.add(protected_step)
        bucket = self.client.bucket(self.bucket_name)
        for step in completed:
            if step in keep:
                continue
            prefix = f"{self.checkpoints_prefix}/checkpoint-{step}/"
            for blob in self.client.list_blobs(self.bucket_name, prefix=prefix):
                bucket.blob(blob.name).delete()

    def upload_final(self, local_dir: Path) -> None:
        _upload_directory(
            self.client,
            self.bucket_name,
            f"{self.base_prefix}/final",
            local_dir,
            marker=True,
        )

    def restore_log(self, name: str, local_path: Path) -> bool:
        blob = self.client.bucket(self.bucket_name).blob(
            f"{self.base_prefix}/logs/{name}"
        )
        if not blob.exists():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path), checksum="auto")
        return True

    def upload_log(self, name: str, local_path: Path) -> None:
        if not local_path.exists():
            return
        self.client.bucket(self.bucket_name).blob(
            f"{self.base_prefix}/logs/{name}"
        ).upload_from_filename(str(local_path), checksum="auto")


class CheckpointUploadCallback(TrainerCallback):
    def __init__(self, store: GCSCheckpointStore) -> None:
        self.store = store

    def on_save(self, args: TrainingArguments, state: Any, control: Any, **_: Any) -> Any:
        step = int(state.global_step)
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        if checkpoint_dir.exists():
            self.store.upload_checkpoint(checkpoint_dir, step)
            protected_step = None
            if state.best_model_checkpoint:
                match = re.search(r"checkpoint-(\d+)$", state.best_model_checkpoint)
                if match:
                    protected_step = int(match.group(1))
            self.store.prune(protected_step)
        return control


class PretrainingLoggingCallback(TrainerCallback):
    def __init__(
        self,
        store: GCSCheckpointStore,
        runtime_log_path: Path,
        metrics_log_path: Path,
    ) -> None:
        self.store = store
        self.runtime_log_path = runtime_log_path
        self.metrics_log_path = metrics_log_path
        self.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)

    def sync_logs(self) -> None:
        for handler in LOGGER.handlers:
            handler.flush()
        try:
            self.store.upload_log("pretraining.log", self.runtime_log_path)
            self.store.upload_log("trainer_metrics.jsonl", self.metrics_log_path)
        except Exception:
            LOGGER.exception("Failed to synchronize pre-training logs to GCS")

    def on_log(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        values = dict(logs or {})
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": int(state.global_step),
            **values,
        }
        with self.metrics_log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        interesting = [
            f"{key}={values[key]}"
            for key in ("loss", "eval_loss", "learning_rate", "epoch")
            if key in values
        ]
        if interesting:
            LOGGER.info("Trainer step %s: %s", state.global_step, ", ".join(interesting))
        return control

    def on_save(self, args: TrainingArguments, state: Any, control: Any, **_: Any) -> Any:
        self.sync_logs()
        return control

    def on_train_end(
        self, args: TrainingArguments, state: Any, control: Any, **_: Any
    ) -> Any:
        self.sync_logs()
        return control


def initialize_scratch_mlm(config: PretrainingConfig, tokenizer: Any) -> Any:
    model_config = AutoConfig.from_pretrained(config.model_name)
    model_config.vocab_size = len(tokenizer)
    model_config.max_position_embeddings = config.max_sequence_length
    model_config.architectures = ["BertForMaskedLM"]
    return AutoModelForMaskedLM.from_config(model_config)


def _precision_flags() -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    bf16 = bool(torch.cuda.is_bf16_supported())
    return bf16, not bf16


def run_pretraining(config: PretrainingConfig) -> dict[str, float]:
    config.validate()
    config.local_run_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client(project=config.project_id)
    checkpoint_store = GCSCheckpointStore(
        client,
        config.checkpoint_bucket,
        config.normalized_checkpoint_prefix,
        config.run_id,
        config.remote_checkpoint_limit,
    )
    logs_dir = config.local_run_dir / "logs"
    runtime_log_path = logs_dir / "pretraining.log"
    metrics_log_path = logs_dir / "trainer_metrics.jsonl"
    checkpoint_store.restore_log("pretraining.log", runtime_log_path)
    checkpoint_store.restore_log("trainer_metrics.jsonl", metrics_log_path)
    logger = configure_pretraining_logging(runtime_log_path)
    logger.info("Starting pre-training run %s", config.run_id)
    logger.info(
        "Dataset: gs://%s/%s; checkpoints: gs://%s/%s/runs/%s",
        config.dataset_bucket,
        config.normalized_dataset_prefix,
        config.checkpoint_bucket,
        config.normalized_checkpoint_prefix,
        config.run_id,
    )
    logger.info(
        "Training parameters: max_steps=%s, batch=%s, accumulation=%s, sequence_length=%s",
        config.max_steps,
        config.per_device_batch_size,
        config.gradient_accumulation_steps,
        config.max_sequence_length,
    )
    cache = GCSShardCache(
        config.local_run_dir / "shard-cache",
        max_bytes=int(config.cache_max_gb * 1024**3),
        client=client,
    )
    train_shards = list_parquet_shards(
        client,
        config.dataset_bucket,
        config.normalized_dataset_prefix,
        "train",
    )
    validation_shards = list_parquet_shards(
        client,
        config.dataset_bucket,
        config.normalized_dataset_prefix,
        "validation",
    )
    logger.info(
        "Discovered %s training shards and %s validation shards",
        len(train_shards),
        len(validation_shards),
    )
    tokenizer = build_or_load_tokenizer(config, train_shards, cache, client)
    logger.info("Tokenizer ready with %s tokens", len(tokenizer))
    model = initialize_scratch_mlm(config, tokenizer)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Initialized scratch MLM with %s parameters", parameter_count)

    train_dataset = GCSParquetMLMDataset(
        train_shards,
        tokenizer,
        cache,
        max_sequence_length=config.max_sequence_length,
        min_chunk_tokens=config.min_chunk_tokens,
        seed=config.seed,
        repeat=True,
    )
    validation_dataset = GCSParquetMLMDataset(
        validation_shards,
        tokenizer,
        cache,
        max_sequence_length=config.max_sequence_length,
        min_chunk_tokens=config.min_chunk_tokens,
        seed=config.seed,
        repeat=False,
        max_sequences=config.max_eval_sequences,
    )

    output_dir = config.local_run_dir / "trainer"
    resume_checkpoint = (
        checkpoint_store.download_latest(output_dir) if config.resume else None
    )
    if resume_checkpoint:
        logger.info("Resuming from checkpoint %s", resume_checkpoint)
    else:
        logger.info("No completed checkpoint selected; starting from step 0")
    bf16, fp16 = _precision_flags()
    logger.info("Precision selection: bf16=%s, fp16=%s", bf16, fp16)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type="linear",
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=fp16,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        dataloader_num_workers=config.dataloader_workers,
        dataloader_persistent_workers=config.dataloader_workers > 0,
        remove_unused_columns=False,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        optim="adamw_torch",
        restore_callback_states_from_checkpoint=True,
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability,
        seed=config.seed,
    )
    logging_callback = PretrainingLoggingCallback(
        checkpoint_store, runtime_log_path, metrics_log_path
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[CheckpointUploadCallback(checkpoint_store), logging_callback],
    )
    try:
        trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
        )
    except Exception:
        logger.exception("Pre-training failed")
        logging_callback.sync_logs()
        raise
    evaluation = trainer.evaluate()
    eval_loss = float(evaluation["eval_loss"])
    perplexity = math.exp(eval_loss) if eval_loss < 50 else float("inf")

    final_dir = config.local_run_dir / "final"
    mlm_dir = final_dir / "mlm"
    encoder_dir = final_dir / "encoder"
    tokenizer_dir = final_dir / "tokenizer"
    trainer.save_model(str(mlm_dir))
    trainer.model.base_model.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    metrics = {"eval_loss": eval_loss, "perplexity": perplexity}
    logger.info(
        "Training completed: eval_loss=%s, perplexity=%s", eval_loss, perplexity
    )
    (final_dir / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    (final_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    checkpoint_store.upload_final(final_dir)
    logging_callback.sync_logs()
    logger.info("Final model and logs uploaded to GCS")
    logging_callback.sync_logs()
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-train a BGE-small-shaped masked language model in Colab."
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--source-bucket", required=True)
    parser.add_argument("--dataset-bucket", required=True)
    parser.add_argument("--checkpoint-bucket", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-prefix", default=DEFAULT_DATASET_PREFIX)
    parser.add_argument("--checkpoint-prefix", default="legal-bge-pretraining")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--local-root", default="/content/legal-pretraining")
    parser.add_argument("--new-vocabulary-tokens", type=int, default=16_000)
    parser.add_argument("--tokenizer-max-documents", type=int, default=200_000)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--cache-max-gb", type=float, default=4.0)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = PretrainingConfig(
        project_id=args.project_id,
        source_bucket=args.source_bucket,
        dataset_bucket=args.dataset_bucket,
        checkpoint_bucket=args.checkpoint_bucket,
        run_id=args.run_id,
        dataset_prefix=args.dataset_prefix,
        checkpoint_prefix=args.checkpoint_prefix,
        model_name=args.model_name,
        local_root=args.local_root,
        new_vocabulary_tokens=args.new_vocabulary_tokens,
        tokenizer_max_documents=args.tokenizer_max_documents,
        max_steps=args.max_steps,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cache_max_gb=args.cache_max_gb,
        dataloader_workers=args.dataloader_workers,
        seed=args.seed,
        resume=not args.no_resume,
    )
    print(json.dumps(run_pretraining(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
