# SD 1.5 UNet Cross-Backend Benchmark

Benchmark harness for comparing Stable Diffusion 1.5 UNet execution across Apple-Silicon backends.

The benchmark measures only the UNet. VAE and CLIP/text-encoder execution are outside the timed and power-measured path.

## Install

```bash
uv sync
```

CoreML conversion toolchains are isolated because `coremltools` 8.x and 9.x cannot be imported into the same Python environment:

```bash
uv sync --project envs/apple-ct8
uv sync --project envs/team-ct9
```

## Checkpoint

Set `SD15_CHECKPOINT` to a local vanilla Stable Diffusion 1.5 `.safetensors` checkpoint. The intended source is the Stable Diffusion 1.5 model family on Hugging Face; the harness records the resolved path in config-derived outputs, and production runs should record the file hash in the environment manifest.

```bash
export SD15_CHECKPOINT=/absolute/path/to/sd-v1-5.safetensors
```

## Commands

Generate the shared backend-neutral input:

```bash
uv run sdbench prepare-input --config config/matrix.yaml
```

Run the full configured matrix:

```bash
sudo ./scripts/run.sh --config config/matrix.yaml
```

Run one matrix cell:

```bash
uv run sdbench run-cell \
  --backend mlx \
  --compute-unit GPU \
  --attention NATIVE \
  --precision fp16 \
  --resolution 512
```

Regenerate publication tables from an existing JSONL result file:

```bash
uv run sdbench tables --input results/data/results.jsonl --output-dir results/tables
```

## Methodology

Every cell uses the same persisted latent, timestep, and text embedding. The first UNet step is discarded as a cold run, then at least 10 timed iterations are measured. Latency is reported as median plus IQR, not mean.

Power metrics are collected as CPU/GPU/ANE engine channels and are intended only for relative comparison on the same machine in one sitting. Reported energy is `energy_per_unet_step_j`; `estimated_energy_per_50_step_image_j` is a labeled extrapolation, not a directly measured image-generation result.

Numerical equivalence is computed against the PyTorch reference output using MSE and cosine similarity. Divergent cells are flagged but retained in the results.

## Current Status

The repository currently contains the tested harness foundation, config contract, CLI, summary-table generation, isolated CoreML env definitions, and a working `diffusers_mps` UNet adapter with CPU fp32 reference comparison. CoreML and MLX adapters are still gated on converted artifacts and backend-specific validation. MLX SD 1.5 support is intentionally left for the final backend because it needs the strongest equivalence cross-check.
