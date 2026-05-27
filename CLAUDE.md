# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Authoritative documents

Two committed docs govern this repo and outrank this file on their topics:

- **`BENCHMARK_SPEC.md`** — the source of truth for *what must be true* (requirements `R0.1`–`R11.4`, acceptance criteria `A1`–`A10`). When code and spec disagree, the spec wins.
- **`AGENTS.md`** — *how to build*: repository layout, the adapter contract, and the **Golden Rules** (methodology invariants). Read its Golden Rules before touching timing, power, equivalence, or any adapter; breaking one produces invalid benchmark numbers even when the code runs. Preserve `(Rx.y)` references in code comments — they map implementation back to spec clauses.

## Commands

```bash
# Install the harness environment (pinned via uv.lock)
uv sync

# Generate + persist the shared input once (latent, timestep, text embedding)
uv run sdbench prepare-input --config config/matrix.yaml

# Full matrix with power/thermal/no-sleep wrapper (needs sudo for powermetrics)
sudo ./scripts/run.sh --config config/matrix.yaml

# One cell by id, or by explicit tuple
uv run sdbench run-cell --cell mlx-gpu-fp16
uv run sdbench run-cell --backend diffusers_mps --compute-unit MPS --attention NATIVE --precision fp16 --resolution 512

# Regenerate tables from existing JSONL without re-running
uv run sdbench tables --input results/data/results.jsonl --output-dir results/tables

# Tests
uv run pytest                                   # full suite
uv run pytest tests/test_orchestrator.py        # one file
uv run pytest tests/test_config.py::test_name   # one test
```

`scripts/run.sh` is a thin wrapper: it blocks sleep with `caffeinate -dimsu`, launches `powermetrics` as a separate privileged process writing a plist to `results/raw/`, and runs the harness via `uv run sdbench`. The harness and the power sampler are independent processes joined only by a shared monotonic clock; `power.py` aligns samples to the timed-iteration window post-hoc. Pass `--dry-run` to print the commands, `--no-power` to skip the sampler (and the sudo requirement).

The `SD15_CHECKPOINT` env var (or `${VAR}` expansion in `config/matrix.yaml`, resolved against `.env` files and the environment) points at the local SD 1.5 `.safetensors` checkpoint. The same weights must be used across all backends.

## Three isolated environments

`coremltools` 8.x and 9.x cannot coexist in one interpreter, so conversion toolchains are split into separate uv projects. **This is the central structural constraint of the repo.**

- **Root (`pyproject.toml`)** — the benchmark harness. Pins `coremltools==8.3.0`, `torch`, `diffusers`, `numpy`, etc. Does **not** depend on `mlx` or `coreml-diffusion`. This is what `uv sync` and all `uv run sdbench` commands use.
- **`envs/apple-ct8/`** — Apple `ml-stable-diffusion` + coremltools 8 conversion toolchain (Python 3.11). Synced with `uv sync --project envs/apple-ct8`.
- **`envs/team-ct9/`** — `coreml-diffusion` + coremltools 9 conversion toolchain (Python 3.12). Synced with `uv sync --project envs/team-ct9`.

`scripts/convert/apple_ct8.py` and `scripts/convert/team_ct9.py` are the one-time conversion drivers run inside those isolated envs; they emit CoreML artifacts under `artifacts/` plus per-build conversion-timing JSON (graph-capture / convert / first-load-compile). The toolchain-version contrast (ct8 vs ct9 on otherwise-matching cells) is the primary purpose of the study.

## Architecture

The harness is backend-agnostic. Flow: `cli.py` loads `config/matrix.yaml` → builds the adapter registry → `orchestrator.run_matrix` iterates cells → each cell runs through one `BackendAdapter` → records are written as JSONL plus generated tables.

- **Adapter contract (`adapter.py`)** — every backend implements `prepare(cfg) -> RealizedConfig`, `step(latent, timestep, text_embedding) -> np.ndarray`, `teardown()`. `prepare()` absorbs *all* one-time cost (load, compile, first graph build); `step()` runs exactly one UNet forward pass and must return a fully host-materialized numpy array with no deferred work. Do not widen this interface to leak backend internals into the harness.
- **Registry (`backends/registry.py`)** — maps backend name → adapter instance. When no `checkpoint_path` is given, or for not-yet-implemented backends, it returns an `UnavailableBackendAdapter` that raises a descriptive error on `prepare()`. **The `mlx` backend is intentionally an `UnavailableBackendAdapter`** — SD 1.5 MLX support is deliberately implemented last, after the other backends establish the numerical cross-check.
- **Reference backend** — `diffusers_mps` is both a measured cell *and* the numerical ground truth. It exposes an extra `reference_step()` (fp32 on CPU) that the orchestrator calls via `hasattr` duck-typing to produce the comparison baseline for MSE/cosine equivalence. Optional adapter capabilities (`reference_step`, `model_size`) are detected the same way, not declared in the Protocol.
- **Resilient orchestration (`orchestrator.py`)** — `_run_cell` wraps each cell in try/except/finally; a failing cell becomes a `failed` record (with `failure_reason`) and `teardown()` still runs, so one failure never aborts the suite. Equivalence failures are recorded as `numerically_divergent=True` but the cell's results are kept, never dropped.
- **Config (`config.py`)** — frozen dataclasses parsed from YAML with `${VAR}` env expansion. Enforces `iterations >= 10`. `CellConfig` carries `enabled` (skipped if false) and `requires` (capability gate → recorded N/A, not failed). Adding/removing a matrix cell is a YAML edit only — no code change.
- **Single-concern modules** — `timing.py` (monotonic loop, cold-run discard, median+IQR), `power.py` (plist parse, baseline-subtract, time-align), `thermal.py` (throttle gate), `equivalence.py` (MSE+cosine), `sizing.py` (4-column size + quant efficiency), `conversion.py` (capture/convert/compile timing), `results.py` (JSONL records + table generation), `env.py` (per-run manifest). Each maps to a spec section.

### Testing pattern

Adapters take their heavy dependencies via constructor injection (e.g. `DiffusersMpsAdapter(checkpoint, torch_module=..., model_cls=...)`). Tests pass fakes (`FakeTorch`, `FakeTensor`) so adapter logic is exercised without torch/coremltools/MPS hardware. When adding a backend, follow this injection pattern and lazy-import the real dependency only inside the `_load_*` helpers — keep import-time side effects out so the test suite runs in the harness env on any machine.

## Conventions

- English only for all code, comments, configs, and docs.
- Conversion-timing fields in records (`graph_capture_s`, `convert_s`, `first_load_compile_s`) are currently `None` because CoreML/MLX cells are gated on converted artifacts; they are wired through `results.py` and populated once conversion drivers feed them in.
