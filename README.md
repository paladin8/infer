# infer

Educational LLM inference runtime.

## Environment setup (uv)

1. Install uv.
2. Sync runtime + dev dependencies:

```bash
uv sync --dev
```

3. Optional: install advanced Triton kernel dependencies:

```bash
uv sync --dev --group kernels
```

4. Optional: install benchmarking dependencies:

```bash
uv sync --dev --group bench
```

5. Verify PyTorch + CUDA visibility:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Common commands

Run tests:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check .
```

Format:

```bash
uv run ruff format .
```
