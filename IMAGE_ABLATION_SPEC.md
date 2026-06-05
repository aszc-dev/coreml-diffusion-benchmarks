# Image-Space Conversion-Fidelity Ablation — Build Specification

**Audience:** code-generation agent / maintainer.
**Nature:** requirements only. *What* the experiment must do and the contract it satisfies.
**Relationship to `BENCHMARK_SPEC.md`:** this is a **sibling** experiment, not an extension of
the sdbench matrix. sdbench measures a single UNet `step()` in *latent* space (R0.1). This
experiment measures *end-to-end image output* of the full SD 1.5 pipeline, and exists for one
purpose: **attribute image-space divergence of the converted Core ML pipeline to individual
components (UNet / VAE / text-encoder).** "All-diffusers vs all-CoreML" lumps three error
sources together and cannot attribute; this experiment isolates them.

---

## A0. Scope & non-goals

- **A0.1** Model under test MUST be SD 1.5 (the VERIFIED version). SDXL/LCM are out of scope for v1.
- **A0.2** The deliverable is a **per-component attribution** of image divergence, plus an
  optional full-pipeline endpoint. Raw latency/power are NOT in scope here (that is sdbench).
- **A0.3** The single isolated variable across configs MUST be *which component is served from
  Core ML*. Everything else (weights, scheduler, steps, guidance, seed, initial noise,
  resolution, torch device) MUST be held identical.
- **A0.4** Output MUST be reusable in a blog writeup: machine-readable JSONL + generated image
  grids + a markdown summary table.

---

## A1. Configurations (the ablation ladder)

Each config is the tuple `{unet, vae, text_encoder}` where each flag = "served from Core ML".
The harness MUST run at least these five, one component swapped at a time (OAT) plus the endpoint:

| id              | UNet     | VAE      | text-encoder | isolates                          |
|-----------------|----------|----------|--------------|-----------------------------------|
| `reference`     | diffusers| diffusers| diffusers    | ground truth (fp32, CPU)          |
| `coreml-unet`   | **CoreML**| diffusers| diffusers   | UNet conversion (loop-accumulated)|
| `coreml-vae`    | diffusers| **CoreML**| diffusers   | VAE decode (single application)   |
| `coreml-clip`   | diffusers| diffusers| **CoreML**   | text-encoder (loop-propagated)    |
| `coreml-full`   | **CoreML**| **CoreML**| **CoreML**  | full pipeline + interaction       |

- **A1.1** `reference` MUST be fully torch, fp32, on CPU — the numerical ground truth, consistent
  with sdbench's reference choice (`BENCHMARK_SPEC` R3.3.2).
- **A1.2** Attribution rule: each OAT config's divergence vs `reference` is that component's
  isolated contribution. If `coreml-full` divergence ≉ a simple combination of the three OAT
  divergences, the components interact and that MUST be reported, not hidden.
- **A1.3** For `coreml-vae` the post-loop latent is identical to `reference` by construction;
  the harness MAY assert this (latent metric ≈ 0) as a self-check that only the decode differs.

---

## A2. Controlled variables (the fairness traps, made explicit)

- **A2.1 Shared initial noise.** The initial latent MUST be identical across all configs for a
  given `(prompt, seed)`. Achieved by keeping every torch component on **CPU** and drawing noise
  from a CPU `torch.Generator(seed)`; Core ML stand-ins report `device == cpu`, so the pipeline's
  noise sampling stays on CPU and is bit-identical across configs. (RNG on different devices
  yields different noise — this is why everything stays CPU-side.)
- **A2.2 Scheduler.** One deterministic scheduler, fixed config, set identically on every pipeline
  (default: DDIM, `eta=0`). Steps and `guidance_scale` are fixed and recorded.
- **A2.3 `scaling_factor`.** The VAE wrapper deliberately does NOT bake the scale
  (`conversion/vae.py`); the pipeline owns it. The harness MUST let the diffusers pipeline apply
  scaling on the standard decode path (do NOT decode latents manually) so scaling is applied
  once, identically, for every config. A manual-decode path is the most likely place to introduce
  a systematic brightness/contrast offset that masquerades as a quality difference.
- **A2.4 VAE precision.** `CoreMLVAE` runs fp16 internally; the `reference` VAE is fp32. So
  `coreml-vae` divergence includes the fp16 effect — this is honest (it is what the artifact
  delivers). The harness MAY add an OPTIONAL `diffusers-vae-fp16` control to separate conversion
  error from precision error; if added it MUST be a separate labeled config.
- **A2.5 Batch baked at conversion.** Core ML packages have FIXED traced input shapes. CFG feeds
  batch=2, so the UNet `.mlpackage` MUST be converted with `batch_size=2` (per
  `inference.py` `CoreMLUNet` docstring). The harness MUST fail loudly with a clear message if the
  UNet package's expected batch does not match the guided batch, never silently fall back.
- **A2.6 Safety checker.** MUST be disabled — a blanked NSFW frame would corrupt every metric.

---

## A3. Inputs

- **A3.1** A fixed, committed prompt set (default: the 10 prompts in the harness) crossed with a
  fixed seed list (default `[0, 1, 2]`). Adding prompts/seeds MUST NOT require code changes
  beyond the committed lists.
- **A3.2** One single-file SD 1.5 checkpoint, identical across all configs.
- **A3.3** Resolution fixed at 512×512 (matching the converted `sample_size`).

---

## A4. Metrics (image space)

Computed per `(config, prompt, seed)` against the `reference` image, then aggregated:

- **A4.1 LPIPS** (headline; perceptual distance, lower = closer).
- **A4.2 SSIM** and **PSNR** (structural / pixel fidelity).
- **A4.3 CLIP-score** of the test image against its prompt — catches *semantic* drift (an image
  can be perceptually distant yet still on-prompt, or vice-versa). Report the test image's
  CLIP-score and the reference's, so a drop is visible.
- **A4.4** Reported per-config value MUST be the **median** with a dispersion measure (IQR or
  p10–p90) over the prompt×seed set — same robustness stance as sdbench R5.4.
- **A4.5 (optional)** Final pre-decode latent MSE/cosine vs `reference`, to bridge to sdbench's
  latent-space numbers. Version-sensitive (callback capture); MAY degrade gracefully if absent.

---

## A5. Output

- **A5.1** `results.jsonl`: one row per `(config, prompt, seed)` with all metrics + provenance
  (checkpoint sha, package paths, scheduler, steps, guidance, seed, git sha).
- **A5.2** Per-config images on disk + a per-prompt side-by-side **grid** (reference | each config)
  for visual inspection — the "eyeball" artifact, paired with A4 numbers (numbers-first; the grid
  illustrates, the table claims).
- **A5.3** `summary.md`: median ± IQR per config per metric.

---

## A6. Extension — quantization ladder + "GPU original" anchor (Thread B)

This experiment is the scaffold for the extended quantization comparison; it slots in without
new infrastructure:

- **A6.1** Add UNet `.mlpackage`s at `w8` / `w6` / `w4` (palettized, each converted at
  `batch_size=2`) as additional `coreml-unet`-shaped configs (`coreml-unet-w8`, …). Each is
  measured against the **same fp32 `reference`** — giving *absolute* perceptual degradation per
  quant level, not merely ct9-w4-vs-ct8-w4.
- **A6.2** Run the same quant ladder for **both** ct8 (Apple) and ct9 (ours) UNet packages. The
  ct8-vs-ct9 delta at matched quant is the toolchain contrast; the vs-reference distance is the
  absolute cost.
- **A6.3** Add a `diffusers-mps-fp16` config as the practical "GPU original" baseline — the
  un-converted PyTorch model a user would actually run on GPU — so the story references the real
  origin, not only another conversion. (Note: MPS-vs-CPU RNG differs, so its initial noise will
  not be bit-identical; treat it as a quality anchor, not a paired-noise OAT config, and say so.)
