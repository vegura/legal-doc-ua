## Poetry GPU Profiles

Use one hardware profile per environment:

```bash
# macOS / Apple MPS
poetry sync --with mac --without cuda

# Linux / NVIDIA CUDA 13.2
poetry sync --with cuda --without mac
```

This project uses a local `poetry.toml` with `installer.re-resolve = false`, which is required by Poetry 2.1.x for mutually exclusive PyTorch builds in the same lock file.

## Ukrainian Legal Model Pre-training

The pre-training pipeline has two independent stages:

1. `legal_doc_parser.pretraining_data` cleans the 2024 RTF decisions and writes
   Parquet shards.
2. `legal_doc_parser.pretrain_bge` streams those shards and trains a masked
   language model from random weights using the
   `BAAI/bge-small-en-v1.5` architecture.

The corpus is law-only. It contains 90% criminal decisions and 2.5% from each
of the civil, commercial, administrative, and administrative-offense groups.
This is masked-language-model pre-training; it does not create supervised
query/document pairs and does not run contrastive embedding fine-tuning.

### Storage layout

Use three different GCS buckets. Both commands reject reused bucket names:

```text
gs://<raw-bucket>/                         # existing RTF source; read-only
gs://<dataset-bucket>/legal-bge-pretraining/v1/
  manifests/                              # deterministic source selection
  clean/train/                            # dynamically loaded Parquet shards
  clean/validation/
  errors/
  tokenizer-plus-16000/
gs://<checkpoint-bucket>/legal-bge-pretraining/runs/<run-id>/
  checkpoints/checkpoint-<step>/          # resumable Trainer state
  final/                                  # MLM, encoder, tokenizer, metrics
  logs/pretraining.log                    # human-readable Colab/runtime log
  logs/trainer_metrics.jsonl              # loss, LR and evaluation history
```

Buckets must already exist and the Colab identity needs read access to the raw
bucket and object write/delete access to the two artifact buckets. Checkpoints
are uploaded only after the local checkpoint is complete; `_SUCCESS` marks a
checkpoint that is safe to resume.

### Google Colab setup

Clone the repository, select a GPU runtime, and authenticate before calling the
modules:

```python
from google.colab import auth
auth.authenticate_user()
```

Install the package in the runtime:

```bash
pip install -e .
pip uninstall -y torchaudio torchcodec librosa soundfile
```

Colab may ship an optional audio stack compiled for a different CUDA build than
the PyTorch version selected by this project. The text pre-training pipeline
does not use audio, so remove TorchAudio and packages that can import it
indirectly. Restart the Colab runtime after installation to clear Transformers'
cached package-availability state, then repeat the authentication and
project-directory cells. Do not reinstall TorchAudio unless its PyTorch and
CUDA versions exactly match the active runtime.

First run a small data smoke test. Replace the two artifact bucket names with
separate, globally unique buckets:

```bash
python -m legal_doc_parser.pretraining_data \
  --project-id lab-test-project-1-305710 \
  --table-id lab-test-project-1-305710.court_data_2024.document_data \
  --source-bucket court_data_2024 \
  --dataset-bucket YOUR_DATASET_BUCKET \
  --checkpoint-bucket YOUR_CHECKPOINT_BUCKET \
  --dataset-prefix legal-bge-pretraining/smoke \
  --limit 2000 \
  --shard-rows 250
```

Remove `--limit` for the full preparation run. Completed shards are detected
in GCS and skipped after a Colab restart. Use the default `v1` prefix for that
full run; a prefix records its preparation settings and rejects incompatible
options so smoke and production shards cannot be mixed accidentally.

Parquet files use Zstandard compression by default. If minimizing transferred
bytes is more important than decompression speed, select gzip when creating a
new dataset prefix:

```bash
--shard-rows 250 --parquet-compression gzip
```

The training loader reads only 32 documents from the active shard at a time;
configure this with `parquet_read_batch_rows` or
`--parquet-read-batch-rows`. Gzip and Zstandard shards are detected and decoded
transparently by PyArrow.

Then run a short training smoke test:

```bash
python -m legal_doc_parser.pretrain_bge \
  --project-id lab-test-project-1-305710 \
  --source-bucket court_data_2024 \
  --dataset-bucket YOUR_DATASET_BUCKET \
  --checkpoint-bucket YOUR_CHECKPOINT_BUCKET \
  --run-id bge-small-uk-law-v1 \
  --dataset-prefix legal-bge-pretraining/smoke \
  --max-steps 10 \
  --checkpoint-every-steps 5 \
  --dataloader-workers 0
```

For the full run, omit the smoke overrides. The default is 100,000 optimizer
steps with an effective batch of at least 64 sequences. Batch size, accumulation
and gradient checkpointing are selected from available GPU memory. An 80 GiB
GPU defaults to a physical batch of 64, one accumulation pass, BF16, TF32, and
no gradient checkpointing. It saves every 2,000 optimizer
steps, keeps one local checkpoint and the two latest complete cloud
checkpoints. Override this without editing code using, for example,
`--checkpoint-every-steps 5000`. Reusing the same `--run-id` automatically
downloads the latest complete cloud checkpoint. Each worker downloads one
Parquet shard at a time into a bounded `/content` cache and prefetches the next
shard rather than materializing the full corpus locally.

Manual performance overrides are available through
`--per-device-batch-size`, `--gradient-accumulation-steps`, and
`--gradient-checkpointing auto|on|off`. Start with batch 64 on an 80 GiB GPU;
try batch 96 or 128 if monitoring still shows substantial free GPU memory.

During training, progress is printed in Colab and written locally under
`/content/legal-pretraining/<run-id>/logs/`. The runtime log and structured
Trainer metrics are synchronized to the checkpoint bucket whenever a checkpoint
is saved and at the end of training. Reusing the same run ID restores and
continues the existing cloud logs.

The final encoder is an MLM-pretrained language encoder. Because the requested
pipeline intentionally omits contrastive fine-tuning, it should not be treated
as a drop-in retrieval-optimized BGE embedding model.
