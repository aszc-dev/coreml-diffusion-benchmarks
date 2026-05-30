# SD 1.5 UNet Cross-Backend Benchmark

Benchmark harness for comparing Stable Diffusion 1.5 UNet execution across Apple-Silicon backends.

The benchmark measures only the UNet. VAE and CLIP/text-encoder execution are outside the timed and power-measured path.

## Quick start

The tool ships a guided terminal flow. Run it with no arguments and it walks you through getting the checkpoint, converting artifacts, configuring a run, running it, and cleaning up:

```bash
uvx coreml-diffusion-benchmarks                  # one-off, no install
# or
uv tool install coreml-diffusion-benchmarks
cdbench                                           # guided menu (sdbench is a legacy alias)
```

A single install provisions the full benchmark stack (torch, coremltools, diffusers, mlx, …). There is no light-only mode — every meaningful flow (`convert`, `run`, `prepare-input`, `download`) needs the full stack, so splitting the install only confused contributors who saw a menu but couldn't actually run anything.

## From a repository checkout

```bash
uv sync                                           # full dev stack
```

CoreML conversion toolchains are isolated because `coremltools` 8.x and 9.x cannot be imported into the same Python environment. The conversion driver runs each in its own pinned uv project:

```bash
uv sync --project envs/apple-ct8
uv sync --project envs/team-ct9
```

## Checkpoint

The same SD 1.5 weights must back every backend, so the checkpoint is pinned by SHA-256 and verified whether you point at a local file or have it downloaded:

```bash
cdbench download --checkpoint /path/to/v1-5-pruned-emaonly.safetensors   # verify a local file
cdbench download --download                                              # fetch + verify from the official HF repo
```

Alternatively set `SD15_CHECKPOINT` to the local `.safetensors` path; it is recorded (with its hash) in the per-run environment manifest. A SHA mismatch is fatal — the harness refuses to benchmark non-identical weights.

## Commands

| Command | Purpose |
| --- | --- |
| `cdbench` | Guided menu (no arguments; `sdbench` is a legacy alias of the same app). |
| `cdbench download` | Resolve and SHA-verify the checkpoint. |
| `cdbench convert` | Convert CoreML artifacts in the isolated ct8/ct9 envs (cached by checkpoint SHA). |
| `cdbench measure-disk` | Measure converted-artifact sizes into `config/disk_footprint.yaml`. |
| `cdbench config` | Interactively select cells, power, and verbosity; save a run plan. |
| `cdbench run` | Run with live progress and minimal-root power; results are upserted. |
| `cdbench verify` | Check that all results share one provenance fingerprint matching this environment. |
| `cdbench cleanup` | Reclaim generated state (models, captures, results) with measured sizes. |
| `cdbench tables` | Regenerate publication tables from an existing JSONL file. |
| `cdbench run-matrix` | Headless engine run used by `scripts/run.sh` and CI. |
| `cdbench run-cell` | Run a single cell by id or explicit tuple. |

Full-suite runs are never the default: cell selection is an explicit checkbox, and "FULL SUITE" is a separate, deliberate menu choice.

## Power measurement, sudo, and your safety

Per-engine power requires Apple's `powermetrics`, which needs root. Root is kept to the minimum: **only the sampler runs under `sudo`** — the benchmark harness itself stays unprivileged. You can audit exactly what is elevated in `scripts/run.sh` and `src/sdbench/tui/power_session.py` before granting your password.

If you would rather not grant sudo, decline at the prompt (or pass `--no-power`): power metering is disabled and every other metric — latency, equivalence, model size, conversion time — still runs. The guided flow also reminds you to close other heavy apps before a run, since background load skews latency and pollutes the per-engine power baseline.

### Power numbers require AC and a quiet host

Before the sampler starts, `cdbench run --power` checks three preconditions:

- the laptop is plugged into AC (battery routes thermal and clock budgets differently, so the W and J figures are not comparable to AC numbers);
- low-power mode is off (it caps CPU/GPU clocks);
- `loadavg_1m ≤ 2.0` (a noisy background contaminates the baseline window captured before each cell).

A failed check refuses to record power and leaves latency/equivalence/size as the trustworthy outputs. Pass `--force-power` to override — the resulting numbers will still be flagged in the manifest and should not be quoted next to clean runs.

## Reproducibility and provenance

Every result is stamped with a provenance fingerprint over the checkpoint hash, the tool version, the host chip, and the pinned dependency sets of all three uv environments. If the checkpoint or a dependency changes, dependent results are invalidated before new ones are written, and `cdbench verify` flags any mix of datapoints from different provenances.

## Methodology

Every cell uses the same persisted latent, timestep, and text embedding. The first UNet step is discarded as a cold run, then at least 10 timed iterations are measured. Latency is reported as median plus IQR, not mean.

Power metrics are collected as CPU/GPU/ANE engine channels and are intended only for relative comparison on the same machine in one sitting. Reported energy is `energy_per_unet_step_j`; `estimated_energy_per_50_step_image_j` is a labeled extrapolation, not a directly measured image-generation result. A pre-run thermal check (`pmset -g therm`) gates timing when the CPU is throttled.

Numerical equivalence is computed against the PyTorch reference output using MSE and cosine similarity. Divergent cells are flagged but retained in the results.

## Current Status

The harness, config contract, CLI, guided front-end, summary-table generation, isolated CoreML env definitions, and the `diffusers_mps` UNet adapter (with CPU fp32 reference comparison) are in place and tested. CoreML and MLX adapters are gated on converted artifacts and backend-specific validation. MLX SD 1.5 support is intentionally last because it needs the strongest equivalence cross-check.
