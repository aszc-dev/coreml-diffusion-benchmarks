# AGENTS.md

Operational guide for the code-generation agent working on this repository.

**`BENCHMARK_SPEC.md` is the source of truth.** This file tells you *how to build*; the spec tells you *what must be true*. If anything here appears to conflict with the spec, the spec wins — stop and reconcile. Requirement references like `(R5.2)` point at spec clauses; preserve them in code comments where a non-obvious rule is implemented.

Project in one line: a self-contained, public-GitHub-ready benchmark of the **Stable Diffusion 1.5 UNet only** across four Apple-Silicon backends, measuring inference time, GPU-vs-ANE power, model size, quantization efficiency, and conversion time.

---

## Golden rules (invariants — never violate these)

These encode the methodology traps. A build that breaks any of them produces invalid numbers even if it runs.

1. **UNet only.** No VAE, no CLIP/text-encoder in any timed or power-measured path. (R0.1)
2. **One checkpoint, all backends.** Same SD 1.5 weights everywhere. (R0.2)
3. **Shared input, loaded not regenerated.** The latent, timestep, and text embedding are generated once, persisted, and loaded by every backend, each casting to its native type. No backend makes its own input. (R4.1)
4. **Discard the cold run.** First iteration of every cell is thrown away; then ≥10 timed iterations; report **median + dispersion**, never mean. (R5.2–R5.4)
5. **No lazy work in the timed window.** MLX `step()` MUST force evaluation before returning; CoreML models MUST be pre-compiled before timing. Timing graph construction or compilation = invalid. (R2.2, R3.1.4, R3.2.5, R3.4.2, R5.5)
6. **Power is relative-only.** Per-engine power is the **median** sample over each window (not the mean — the wall-clock-aligned window bounds drift by up to a sample interval, so a mean inherits ramp/leak contamination; see `power.summarize_power`), baseline-subtracted, time-aligned to the timed iterations, and reported as relative for one machine in one sitting — never as device-to-device absolute. Always record both GPU and ANE channels. (R6.2–R6.7)
7. **Numerical equivalence flags, never drops.** Every cell's UNet output is compared (MSE + cosine) against the PyTorch reference on the shared input; failing a threshold marks the cell divergent but keeps its results. (R7.1–R7.4)
8. **One failing cell never aborts the suite.** Record and continue. (R1.3, R10/A10)
9. **No vendoring.** Apple `ml-stable-diffusion`, `coreml-diffusion`, `diffusers`, `mlx` are installed dependencies, not copied into the repo. (R0.6, R11.4)
10. **Pin everything.** All framework versions pinned, **all three `uv.lock` files content-hashed (SHA-256)**, the harness git SHA + dirty flag, and the salted host id hash all written to the per-run manifest; the comparison is version-sensitive AND host-sensitive. (R10.4, R11.3, R11.6, R11.8, R11.9)
11. **English** for all code, comments, configs, and docs.
12. **Reports carry their host.** Every emitted artifact — the JSONL row, the generated table, the environment manifest, the contributor bundle — MUST be self-describing about the machine and the toolchain that produced it. A row missing `host_id_hash` + `provenance_digest` is invalid. Tables MUST carry a caption with `sdbench <ver> · <chip> · macOS <build> · provenance <digest[:12]>`. (R11.6-R11.13)
13. **Schema bumps are breaking.** Renaming or removing any captured manifest / record field MUST bump `TELEMETRY_SCHEMA_VERSION`. Additive changes do not. The `sdbench validate-report` gate refuses bundles newer than the maintainer's supported version; the maintainer's harness MUST be upgraded before such reports can be accepted. (R11.5, R11.14)

---

## Repository layout

Create and respect this structure. One module per spec concern so requirements map to files.

```
.
├── AGENTS.md                       # this file
├── BENCHMARK_SPEC.md               # authoritative requirements — do not contradict
├── README.md                       # human methodology + generated result tables (R10.6)
├── pyproject.toml                  # uv-managed deps; single documented install (R1.5, R11.1)
├── uv.lock                         # exact pins (committed) (R11.3)
├── .python-version                 # pinned Python for uv
├── config/
│   └── matrix.yaml                 # declarative run matrix; adding a cell = no code change (R1.2)
├── assets/
│   └── shared_input/               # persisted latent / timestep / text-embedding (R4.1)
├── src/sdbench/
│   ├── cli.py                      # single entrypoint: full matrix OR one cell (R1.1)
│   ├── orchestrator.py             # iterates cells, resilient, records failures (R1.3)
│   ├── adapter.py                  # BackendAdapter contract (see below) (R2)
│   ├── inputs.py                   # shared-input generation + load (R4)
│   ├── timing.py                   # monotonic loop, warmup/discard, median+IQR (R5)
│   ├── power.py                    # parse powermetrics plist, baseline-sub, align (R6)
│   ├── thermal.py                  # pre-run throttle gate (R5.6)
│   ├── equivalence.py              # MSE + cosine vs reference (R7)
│   ├── sizing.py                   # 4-column size table + quant efficiency (R8)
│   ├── conversion.py               # capture / convert / first-load compile timing (R9)
│   ├── results.py                  # machine-readable records + table generation (R10.1–R10.2)
│   ├── env.py                      # environment manifest (R10.4, R11.2–R11.3)
│   └── backends/
│       ├── registry.py             # maps backend name -> adapter class
│       ├── apple_coreml.py         # Backend 1: Apple ml-stable-diffusion + coremltools 8
│       ├── coreml_diffusion.py     # Backend 2: coreml-diffusion + coremltools 9
│       ├── diffusers_mps.py        # Backend 3: diffusers on MPS (also the reference, R3.3.2)
│       └── mlx_backend.py          # Backend 4: MLX
├── scripts/
│   ├── run.sh                      # wraps caffeinate + sudo powermetrics + cli (R6.1, R6.5)
│   └── convert/                    # one-time per-backend conversion drivers (R9)
└── results/
    ├── raw/                        # retained powermetrics plists (R10.3)
    ├── data/                       # machine-readable results, one record per cell (R10.1)
    └── tables/                     # generated publication-ready tables (R10.2)
```

---

## The adapter contract

Every backend is wrapped behind one interface so the harness stays backend-agnostic (R2). Implement this as the stub below; fill bodies per backend. Do **not** widen the interface to leak backend specifics into the harness.

```python
# src/sdbench/adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class CellConfig:
    backend: str            # "apple_coreml" | "coreml_diffusion" | "diffusers_mps" | "mlx"
    compute_unit: str       # "CPU_AND_NE" | "CPU_AND_GPU" | "MPS" | "GPU"
    attention: str          # "SPLIT_EINSUM_V2" | "ORIGINAL" | "NATIVE"
    precision: str          # "fp16" | "w4" | "w8a8" | ...
    resolution: int         # e.g. 512


@dataclass(frozen=True)
class RealizedConfig:
    # What actually happened, for requested-vs-realized mismatch detection (R4.4)
    compute_unit: str
    attention: str
    precision: str
    artifact_paths: list[str]


class BackendAdapter(Protocol):
    name: str

    def prepare(self, cfg: CellConfig) -> RealizedConfig:
        """Load weights and FULLY warm/compile the model for `cfg`.
        All one-time cost (disk load, ANE/Metal compile, MLX first-build) happens
        here, never in `step`. Returns the realized configuration. (R2.1, R3.*.4)"""
        ...

    def step(self, latent: np.ndarray, timestep: int, text_emb: np.ndarray) -> np.ndarray:
        """Run exactly ONE UNet forward pass and return the output latent,
        FULLY materialized on the host as a numpy array. No lazy/deferred compute
        may survive this call. Contains only what legitimately belongs to one
        denoising step. (R2.2)"""
        ...

    def teardown(self) -> None:
        """Release resources. (R2.1)"""
        ...
```

Implementation notes per backend (full requirements in spec §3):

- **`apple_coreml`** — Apple converter pinned to **coremltools 8.x**; produce `SPLIT_EINSUM_V2` (ANE) and `ORIGINAL` (GPU) builds; benchmark under both `CPU_AND_NE` and `CPU_AND_GPU`; pre-compile to the Apple-compiled on-disk form and load from it. (R3.1)
- **`coreml_diffusion`** — the team's pipeline pinned to **coremltools 9.x** + the team's torch/numpy. **Single conversion path** mirroring Apple's method (same custom diffusers pipeline + trace API); the only intended difference vs Backend 1 is the toolchain version. Benchmark under both compute units. At least one weight-quantized variant in addition to fp16; any W8A8 variant is hardware-gated and recorded N/A elsewhere. (R3.2)
- **`diffusers_mps`** — same checkpoint, fp16, `mps`, UNet only. **This backend is the numerical reference**: its fp32-on-CPU output is ground truth for §7; its fp16-on-MPS output is one compared point. Any `torch.compile` variant is a separate labeled cell. (R3.3)
- **`mlx`** — SD 1.5 UNet, fp16, UNet only. Upstream `mlx-examples` ships SD 2.1-base / SDXL / FLUX but **not** SD 1.5, so supply a verified SD 1.5 config on the canonical architecture loading the same checkpoint (a version-pinned community port is an acceptable fallback). `step()` MUST force evaluation before returning. (R3.4)

---

## Run matrix (config-driven)

A matrix cell is a superset of `CellConfig`. **`config/matrix.yaml` is the canonical, committed matrix** — adding or removing a cell there requires no code change (R1.2). Per-cell fields:

- required (map to `CellConfig`): `backend`, `compute_unit`, `attention`, `precision`, `resolution`
- operational: `id` (stable key, used by `run-cell` and as table row key), `label` (human/table caption), `enabled` (bool; disabled cells are skipped), `requires` (optional capability gate; if the host lacks it the cell is recorded **N/A**, not failed — R8.4), `notes`

Top-level keys: `checkpoint` (same weights for all backends — R0.2), `seed` (R1.4), `warmup`/`iterations` (R5.2–R5.3), an `equivalence` block whose `reference` defines the ground-truth cell (diffusers UNet, fp32, CPU — R3.3.2) plus `mse_max`/`cosine_min` flag thresholds (R7.2), a `power` block (`interval_ms`, `baseline_seconds`, `samplers` — R6.1–R6.2), and a `thermal` gate (R5.6). See `config/matrix.yaml` for the full populated matrix.

---

## Commands

Provide a single documented entrypoint plus a measurement wrapper.

```bash
# install (one command, pinned) — R1.5, R11.1
uv sync

# generate + persist the shared input once — R4.1
uv run python -m sdbench.cli prepare-input --config config/matrix.yaml

# full matrix, with power + thermal + no-sleep wrapper (needs sudo for powermetrics) — R6
sudo ./scripts/run.sh --config config/matrix.yaml

# a single cell in isolation — R1.1
uv run python -m sdbench.cli run-cell --cell mlx-gpu-fp16

# regenerate tables from existing data without re-running — R10.2
uv run python -m sdbench.cli tables
```

`scripts/run.sh` responsibilities: `caffeinate -dimsu` to block sleep (R6.5); launch `powermetrics --samplers cpu_power,gpu_power,ane_power -i <interval> -f plist -o results/raw/<run>.plist` as a separate privileged process (R6.1); run the harness via `uv run sdbench`, which records each cell's timed-window bounds as **wall-clock (epoch) timestamps** in the results record (latency itself stays monotonic-derived); after the run completes and the sampler is stopped, `run.sh` invokes `sdbench power`, which parses the plist and baseline-subtracts each cell's power aligned to its wall-clock window post-hoc (R6.2-R6.3). The harness and the power sampler are separate processes joined only by the shared **wall clock** (monotonic is process-relative and cannot align across the two processes). (Note: `powermetrics` requires `sudo`; keep `uv run` for the harness so the pinned environment is used.)

---

## Outputs

- `results/data/` — one machine-readable record per cell containing: realized config (R4.4), latency median + dispersion (R5.4), per-engine GPU/ANE average power and energy-per-image (R6.4), equivalence MSE/cosine + flag (R7), size 4-tuple (R8.1), and conversion timings for CoreML cells (R9). (R10.1)
- `results/tables/` — generated latency / power+energy / size+quantization / conversion-time tables, publication-ready. (R10.2)
- `results/raw/` — retained powermetrics plists. (R10.3)
- environment manifest per run: chip/model, OS version, all framework versions, seed, run conditions. (R10.4)

---

## Definition of done

The build is complete when acceptance criteria **A1–A10** in `BENCHMARK_SPEC.md §12` all hold. Before declaring done, self-check against the Golden Rules above and confirm each spec section §4–§11 has a corresponding implemented module that satisfies its `R` clauses.

---

## Do NOT

- Do not put VAE/CLIP, lazy MLX evaluation, or model compilation inside the timed window.
- Do not let any backend generate its own input or use its own RNG for the shared input.
- Do not report mean latency or present power as absolute/cross-device.
- Do not drop numerically-divergent cells; flag them.
- Do not vendor backend source or leave any dependency version unpinned.
- Do not add SDXL / LCM / Flux cells to v1; they are documented future work only. (R0.5)
- Do not widen the adapter interface to expose backend internals to the harness.
