# Handoff: VAE + CLIP conversion and end-to-end Core ML inference

**Audience:** maintainer of `coreml-diffusion-benchmarks`.
**Upstream:** `coreml-diffusion` (the `team-ct9` toolchain in `envs/team-ct9/`).
**Date:** 2026-06-04.
**Status of upstream work:** merged to `coreml-diffusion` `main` (commits `dc1f85b`
conversion, `ca08b16` inference). Verified by Tier 0 (75 tests) + Tier 1 smoke (12
tests, real coremltools conversion on Apple Silicon). Full-image golden on a real
checkpoint is still the upstream Tier 2 anchor (run before release).

This document has two parts: **what changed upstream** (the new capabilities this
repo can now consume), and **recommendations** for turning them into a full-image
comparison track here — the "compare generated images" metric that is more
meaningful than the per-forward MSE/cosine deviation the current matrix measures.

---

## Part A — What changed in `coreml-diffusion`

Until now `coreml-diffusion` converted **only the UNet**. It now converts the rest
of the checkpoint and can run the **whole pipeline on Core ML**, not just the UNet.

### A.1 New conversion components

`convert()` gained a `component` keyword (default `"unet"`, so every existing call
is unchanged). New components, each emitted as its own `.mlpackage`:

| component        | input → output                         | notes                                  |
|------------------|----------------------------------------|----------------------------------------|
| `unet`           | (unchanged)                            | historical path                        |
| `vae_decoder`    | latent `(B,4,h,w)` → image `(B,3,8h,8w)` | needed for any image output            |
| `vae_encoder`    | image → latent moments `(B,8,h,w)`     | img2img / round-trip only              |
| `text_encoder`   | token ids `(B,77)` → embeddings        | SD1.5 final state; SDXL enc1 penultimate |
| `text_encoder_2` | token ids `(B,77)` → embeddings + pooled | SDXL only (`CLIPTextModelWithProjection`) |

Wrappers are **1:1** — they call the reference submodules directly, no custom
networks. Scaling and sampling stay in the pipeline so artifacts match the
reference models bit-for-bit (modulo fp16). The VAE mid-block self-attention is
routed through the existing `ORIGINAL` (full, fp32-score) attention processor.

### A.2 Public API the benchmark will call

**Conversion** (run inside `envs/team-ct9`, exactly like the UNet today):

```python
import coreml_diffusion.convert as team
from coreml_diffusion.model_version import ModelVersion

# VAE decoder — resolution-dependent (latent/image spatial baked at trace time)
team.convert(
    str(checkpoint), ModelVersion.SD15, str(vae_dec_mlpackage),
    component="vae_decoder",
    batch_size=1,                                  # 1 image decoded per call
    sample_size=(resolution // 8, resolution // 8),
    quantize_nbits="none",                         # or "8"/"6"/"4" (k-means palettize)
)

# CLIP text encoder — resolution-INDEPENDENT (only batch + the fixed 77 tokens)
team.convert(
    str(checkpoint), ModelVersion.SD15, str(te_mlpackage),
    component="text_encoder",
    batch_size=1,
)
# SDXL also needs: component="text_encoder_2"
```

**End-to-end inference** — `build_pipeline` now optionally swaps the VAE and the
text encoder(s) too. The UNet path is unchanged; pass only the extra packages you
want served from Core ML:

```python
from coreml_diffusion import build_pipeline
import torch

pipe = build_pipeline(
    ckpt_path, unet_mlpackage, ModelVersion.SD15,
    vae_decoder_mlpackage=vae_dec_mlpackage,
    text_encoder_mlpackage=te_mlpackage,
    # optional per-component placement (VAE is often faster on the GPU):
    compute_unit="CPU_AND_NE",
    vae_compute_unit="CPU_AND_GPU",
    text_encoder_compute_unit="CPU_AND_NE",
    torch_device="cpu",
)
image = pipe(
    "a photograph of an astronaut riding a horse",
    num_inference_steps=20,
    guidance_scale=7.5,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
```

The components are exposed as adapters (`CoreMLVAE`, `CoreMLTextEncoder`) that
stand in for `AutoencoderKL` / `CLIPTextModel`; you can also wire them by hand if
you need finer control than `build_pipeline` gives.

**Constraints carried over from the UNet:**
- Shapes are baked at conversion. The UNet must be converted at `batch_size=2`
  for classifier-free guidance (already this repo's default). The VAE decoder is
  `batch_size=1`; text encoder `batch_size=1`.
- Resolution must match across UNet and VAE decoder (`sample_size` ↔ image size).

### A.3 Discovery / naming contract (consumed for cache keys)

- `list_convertible_components()` → `["unet","vae_decoder","vae_encoder","text_encoder","text_encoder_2"]`.
  `CONTRACT_VERSION` bumped `1.0 → 1.1` (additive).
- `compose_component_name(...)` is the cache-key filename for non-UNet components,
  the sibling of `compose_out_name`. VAE names carry resolution; text-encoder names
  do not (the artifact is resolution-independent). If this repo composes artifact
  filenames itself, use these so cache hits line up.

### A.4 Two coremltools-9 conversion gotchas (FYI — why this was non-trivial)

These are fixed upstream; noted so the same traps don't bite a parallel ct8 or
re-implementation here:
- **VAE self-attention is 4-D.** `height * width` in the attention reshape emits a
  symbolic `aten::Int` that ct9 cannot fold → conversion fails. Fixed by
  `flatten(2)` instead of `view(B, C, h*w)`. (The UNet attention is 3-D and never
  hit this.)
- **CLIP causal mask is dynamic.** transformers builds it from
  `query_length + past_key_values_length` (another unfoldable `aten::Int`). Fixed
  by a trace-only patch that substitutes a constant mask (sequence length is fixed
  at convert time). `sdpa` attention converts fine once the mask is constant — no
  need to force eager.

---

## Part B — Why this matters for the benchmark

The current matrix is, by design (`R0.1`), a **UNet-only, single-`step()`** study:
`step()` runs exactly one UNet forward and `equivalence.py` scores it (MSE + cosine)
against a diffusers fp32-CPU reference forward. That is the right tool for the
ct8-vs-ct9 toolchain contrast — it isolates one operator graph and avoids scheduler
/ sampler confounds.

But a single-forward MSE says little about the thing a user actually sees. fp16 and
quantization errors **accumulate over the sampling loop and pass through the VAE**;
two backends can both clear `cosine_min=0.999` on one step yet produce visibly
different final images. Now that VAE + CLIP convert, this repo can measure the
**end-to-end generated image** — the metric that answers "does the converted model
still make the right picture, and how much energy did the whole generation cost?"

This is **additive**, not a replacement. Keep the UNet study exactly as is.

---

## Part C — Recommendations

Ordered by dependency. Each notes the spec/Golden-Rule tension it touches so you
can decide deliberately rather than discover it mid-refactor.

### C.1 Treat e2e image as a SEPARATE track, not a mutation of the UNet path

- **Do not widen `step()`** (`AGENTS.md` adapter contract: `step()` = one forward,
  host-materialized numpy). A full generation is a different unit of work.
- Add a parallel capability, detected the same duck-typed way the repo already
  detects `reference_step` / `model_size`: e.g. `generate(cfg) -> ImageResult`
  on adapters that support it, with its own timing/power window. Backends without
  it are simply absent from the e2e track (recorded N/A per `R8.4`), exactly like
  `mlx` is absent from cells today.
- Give it its own spec section (suggest `R12.*`) and its own matrix
  (`config/matrix_e2e.yaml`) so the clean UNet matrix stays uncontaminated. The
  `enabled`/`requires` machinery and YAML-only cell editing (`R1.2`) carry over.

### C.2 Extend the ct9 conversion driver to emit VAE + text encoder

`scripts/convert/team_ct9.py` currently calls `team.convert(...)` once for the
UNet. For e2e it must also produce the VAE decoder and the text encoder (see A.2).
Recommended:
- Add `--component {unet,vae_decoder,text_encoder,all}` (default `unet`, preserving
  today's behaviour and the conversion-timing sidecars).
- For `all`, convert the three packages into the same `artifacts/coreml_diffusion/`
  build dir and record per-component conversion timing
  (`graph_capture_s`/`convert_s`/`first_load_compile_s`) — the VAE/CLIP convert
  cost is itself a comparison point vs Apple's ct8 toolchain.
- The Apple ct8 driver (`scripts/convert/apple_ct8.py`) already has VAE + text
  encoder available from `ml-stable-diffusion` — wire those through too so the e2e
  track is a fair **ct8-vs-ct9 full-pipeline** contrast, the same head-to-head the
  UNet matrix runs. (`ml-stable-diffusion` is checked out at `~/dev/ml-stable-diffusion`.)

### C.3 New image-equivalence module (the headline metric)

Add `image_equivalence.py` alongside `equivalence.py` (don't overload the per-forward
one). For a generated image vs a reference image:
- **PSNR** and **SSIM** — cheap, no extra weights, deterministic. Good default gate.
- **LPIPS** — perceptual; closest to "looks the same to a human". Needs a small
  pretrained net (AlexNet/VGG) — pin it and treat it as an optional capability
  (`requires: { lpips: true }`) so the harness env stays light.
- Report all three; gate on a documented threshold, not exact match.

**Reference image:** compute once from the diffusers fp32 path (consistent with the
existing reference definition `R3.3.2`: diffusers, fp32, CPU). It is slow at
512×N-steps but runs once per (prompt, seed, steps) tuple and is cached. The MPS
fp16 image is a useful *second* reference for "fp16 baseline" comparisons.

**Determinism caveat (important):** the full pipeline is multi-step and the ANE is
nondeterministic; per-step UNet drift compounds across the schedule (upstream's
golden anchor uses PSNR ≥ 20 dB same-scene, ~23 dB observed). Cross-backend
**exact** image match is impossible — pin seed + scheduler + step count + guidance
scale, and compare with a perceptual + PSNR threshold. Document the threshold and
its provenance, the way the UNet `mse_max`/`cosine_min` are documented.

### C.4 Energy/latency for a whole generation

The current power pipeline (`power.py`: per-engine, baseline-subtracted, time-aligned,
`R6`) already does what e2e needs — just point its window at the whole `generate()`
call instead of one `step()`:
- Headline e2e metric: **Joules per image** (mean active power × generation wall
  time) and **seconds per image**, per compute unit. This is the real-world number
  the README's energy claim implies but the per-step study only approximates.
- Per-engine attribution still works and now spans CPU (text encode) + ANE (UNet) +
  GPU/ANE (VAE) — surfacing where the non-UNet cost actually lands, which the
  UNet-only study can't see.
- Note the larger sampling window may leave the cold-thermal regime; keep the
  throttle gate (`R5.6`) and consider fewer steps or explicit cool-down between
  cells.

### C.5 Visual gallery (deliverable the maintainer asked for)

The e2e track naturally produces the actual images per (backend, compute unit,
precision) cell. Emit them to `results/images/<cell-id>.png` and generate a
contact-sheet table (reference vs each backend, with the PSNR/SSIM/LPIPS numbers
under each). That is the "compare images" artifact for the GitHub Pages knowledge
base and the log.aszc.dev writeup — far more convincing than a deviation column.

### C.6 Matrix shape for e2e

Mirror a subset of the UNet matrix (don't run the full cross-product first):
- `ours-ane-fp16`, `ours-gpu-fp16`, `ours-ane-w4` (quant-quality is the most
  interesting e2e story — quantization error is exactly what survives to the image).
- `apple-ane-fp16` as the ct8 counterpart.
- `diffusers-mps-fp16` as the on-device fp16 baseline; diffusers-fp32-CPU as the
  reference (not a timed cell).
- A few fixed prompts × one seed first; widen prompts once the gate thresholds are
  trusted.

---

## Open decisions for the maintainer

1. **Scope boundary.** Confirm e2e is a separate track (`R12`, `matrix_e2e.yaml`)
   and the UNet study (`R0.1`) stays untouched. (Recommended.)
2. **Quant precision parity.** The UNet matrix uses `w4`/`w8a8`. For e2e, is the VAE
   also quantized, or kept fp16 while only the UNet is palettized? (VAE quant tends
   to hurt image quality disproportionately — recommend fp16 VAE first.)
3. **LPIPS dependency.** In-harness (heavier env) or a separate scoring step run
   post-hoc on the saved PNGs? (Recommend post-hoc to keep the harness env light.)
4. **Reference cost.** fp32-CPU full-image reference is slow; acceptable as a cached
   one-shot per (prompt, seed, steps)?
5. **SDXL.** First e2e pass SD1.5-only (matches the current matrix), or include SDXL
   (needs `text_encoder_2` + the SDXL pipeline, both supported upstream)?

---

## Pointers

- Upstream API surface: `coreml_diffusion/__init__.py` (discovery + lazy exports),
  `convert.py` (`convert(component=...)`, `convert_vae_*`, `convert_text_encoder`,
  `load_vae`, `load_text_encoders`), `inference.py` (`build_pipeline`, `CoreMLVAE`,
  `CoreMLTextEncoder`), `naming.py` (`compose_component_name`).
- Conversion gotchas: `coreml_diffusion/conversion/vae.py`,
  `conversion/text_encoder.py` (`static_causal_mask`), `conversion/attention.py`
  (4-D `flatten(2)` branch).
- Upstream tests to mirror for confidence: `tests/smoke/test_coreml_adapters.py`
  (round-trips VAE/CLIP through the adapters on CPU — the closest existing analogue
  to an e2e check).
- This repo's touch points: `scripts/convert/team_ct9.py`,
  `scripts/convert/apple_ct8.py`, `src/sdbench/backends/coreml_diffusion.py`,
  `src/sdbench/equivalence.py`, `src/sdbench/power.py`, `config/matrix.yaml`,
  `BENCHMARK_SPEC.md`, `AGENTS.md`.
