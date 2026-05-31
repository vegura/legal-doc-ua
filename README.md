## Poetry GPU Profiles

Use one hardware profile per environment:

```bash
# macOS / Apple MPS
poetry sync --with mac --without cuda

# Linux / NVIDIA CUDA 13.2
poetry sync --with cuda --without mac
```

This project uses a local `poetry.toml` with `installer.re-resolve = false`, which is required by Poetry 2.1.x for mutually exclusive PyTorch builds in the same lock file.
