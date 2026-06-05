"""Image-space conversion-fidelity ablation for SD 1.5.

Implements IMAGE_ABLATION_SPEC.md: swap one Core ML component at a time into an
otherwise-identical fp32 diffusers pipeline and attribute image divergence to the
UNet, VAE, and text-encoder conversions individually.

Reuses the drop-in adapters from `coreml_diffusion.inference`
(`CoreMLUNet` / `CoreMLVAE` / `CoreMLTextEncoder`), which already implement the
diffusers component contracts. Unlike `build_pipeline`, this assembles configs by
hand so the UNet can stay on torch (needed for the VAE-only / CLIP-only configs).

Run on-device (Apple Silicon, CPU_AND_NE): see `runs/ablation-ct9-full/` for the
full 10-prompt x 3-seed output (120 rows, summary, grids) and `provenance.json`
for the pinned toolchain. Requires: torch, diffusers, coreml_diffusion, and
`torchmetrics[image]` + transformers for the metrics (uv add torchmetrics
torchvision transformers). Run inside `envs/team-ct9` (coremltools 9 toolchain).

Usage:
    uv run python ablation_e2e.py \
        --ckpt /path/v1-5-pruned-emaonly.safetensors \
        --unet-mlpackage artifacts/ct9/unet_b2.mlpackage \
        --vae-decoder-mlpackage artifacts/ct9/vae_decoder.mlpackage \
        --text-encoder-mlpackage artifacts/ct9/text_encoder.mlpackage \
        --out runs/ablation-ct9 \
        --compute-unit CPU_AND_NE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

# ---- Fixed, committed inputs (A3.1). Edit the lists, not the code. -----------

DEFAULT_PROMPTS = [
    "a photograph of an astronaut riding a horse on the moon",
    "a bowl of ramen, steam rising, cinematic lighting",
    "a red fox sitting in a snowy forest at dawn",
    "an oil painting of a lighthouse in a storm",
    "a cute corgi wearing sunglasses, studio portrait",
    "a futuristic city skyline at night, neon reflections",
    "a still life of lemons and a glass jug on a wooden table",
    "a watercolor of cherry blossoms over a quiet river",
    "a vintage motorcycle parked on a cobblestone street",
    "a macro shot of a dragonfly on a green leaf",
]
DEFAULT_SEEDS = [0, 1, 2]


# ---- Config types ------------------------------------------------------------


@dataclass(frozen=True)
class SwapConfig:
    """Which components are served from Core ML (True) vs diffusers (False)."""

    id: str
    unet: bool
    vae: bool
    text_encoder: bool


# A1: the OAT ladder + endpoint. reference MUST be first (it is the comparison target).
LADDER = [
    SwapConfig("reference", unet=False, vae=False, text_encoder=False),
    SwapConfig("coreml-unet", unet=True, vae=False, text_encoder=False),
    SwapConfig("coreml-vae", unet=False, vae=True, text_encoder=False),
    SwapConfig("coreml-clip", unet=False, vae=False, text_encoder=True),
    SwapConfig("coreml-full", unet=True, vae=True, text_encoder=True),
]


@dataclass(frozen=True)
class GenParams:
    steps: int = 30
    guidance: float = 7.5
    height: int = 512
    width: int = 512


@dataclass(frozen=True)
class Artifacts:
    ckpt: str
    unet_mlpackage: str
    vae_decoder_mlpackage: str | None = None
    text_encoder_mlpackage: str | None = None


# ---- Provenance --------------------------------------------------------------


def _sha256(path: str | Path, limit_mb: int | None = None) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


# Packages whose version determines whether this run reproduces: the converter
# toolchain (coreml-diffusion + coremltools), the inference stack (diffusers /
# torch), and the metric backbones (LPIPS net, CLIP via transformers). A bump in
# any of these can move the numbers, so they are pinned in provenance.
_PROVENANCE_PACKAGES = (
    "coreml-diffusion",
    "coremltools",
    "diffusers",
    "torch",
    "torchmetrics",
    "transformers",
    "numpy",
    "pillow",
)


def _pkg_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in _PROVENANCE_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _platform_info() -> dict[str, str]:
    import platform

    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


# ---- Pipeline assembly (A1, A2) ---------------------------------------------


def build_config_pipeline(cfg: SwapConfig, artifacts: Artifacts, compute_unit: str):
    """Assemble an SD 1.5 pipeline with the requested components swapped to Core ML.

    Everything torch stays fp32 on CPU (A2.1) and the scheduler is pinned identically
    (A2.2). Scaling is left to the pipeline's standard decode path (A2.3).
    """
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    from coreml_diffusion.inference import CoreMLTextEncoder, CoreMLUNet, CoreMLVAE
    from coreml_diffusion.model_version import ModelVersion

    pipe = StableDiffusionPipeline.from_single_file(
        artifacts.ckpt, torch_dtype=torch.float32
    )
    pipe.to("cpu")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)  # A2.2
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None  # A2.6

    if cfg.unet:
        # NOTE: package must be converted at batch_size=2 for guided CFG (A2.5).
        pipe.unet = CoreMLUNet(
            artifacts.unet_mlpackage, pipe.unet, ModelVersion.SD15, compute_unit
        )
    if cfg.vae:
        if artifacts.vae_decoder_mlpackage is None:
            raise ValueError(f"{cfg.id} needs --vae-decoder-mlpackage")
        pipe.vae = CoreMLVAE(
            pipe.vae,
            decoder_mlpackage=artifacts.vae_decoder_mlpackage,
            compute_unit=compute_unit,
        )
    if cfg.text_encoder:
        if artifacts.text_encoder_mlpackage is None:
            raise ValueError(f"{cfg.id} needs --text-encoder-mlpackage")
        # The adapter emits fp16 embeddings (its full-CoreML default), but the OAT
        # `coreml-clip` config keeps the UNet on torch fp32. diffusers propagates
        # prompt_embeds.dtype to the latents and time embedding, so fp16 embeds clash
        # with the fp32 UNet ("mat1 and mat2 must have the same dtype"). Ask for fp32
        # output to bridge the boundary; the fp16 rounding (the real conversion effect
        # we attribute to CLIP) is already baked into the values. coreml-diffusion
        # >=0.1.4 takes `output_dtype` directly; older builds fall back to a wrapper.
        import inspect

        if "output_dtype" in inspect.signature(CoreMLTextEncoder.__init__).parameters:
            pipe.text_encoder = CoreMLTextEncoder(
                artifacts.text_encoder_mlpackage, pipe.text_encoder,
                compute_unit=compute_unit, output_dtype=torch.float32,
            )
        else:
            te = CoreMLTextEncoder(
                artifacts.text_encoder_mlpackage, pipe.text_encoder,
                compute_unit=compute_unit,
            )
            pipe.text_encoder = _Fp32TextEncoder(te)
    return pipe


class _Fp32TextEncoder:
    """Wrap a Core ML text encoder so its embeddings are cast to fp32 (A2.x boundary).

    Delegates everything to the wrapped adapter; only the forward output dtype changes.
    """

    def __init__(self, inner):
        import torch

        self._inner = inner
        self.dtype = torch.float32

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, *args, **kwargs):
        import torch

        out = self._inner(*args, **kwargs)
        # SD1.5 reads out[0] (-> out._first); SDXL reads .text_embeds / hidden_states.
        # Cast every tensor attribute so no fp16 reference survives to the fp32 UNet.
        embeds = out.last_hidden_state.float()
        pooled = out.pooler_output.float() if out.pooler_output is not None else None
        out.last_hidden_state = embeds
        out.text_embeds = pooled
        out.pooler_output = pooled
        out.hidden_states = (embeds, embeds)
        out._first = embeds if pooled is None else pooled
        return out


def generate(pipe, prompt: str, seed: int, params: GenParams) -> np.ndarray:
    """One deterministic generation. Returns HWC float image in [0, 1]."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)  # A2.1
    out = pipe(
        prompt,
        num_inference_steps=params.steps,
        guidance_scale=params.guidance,
        height=params.height,
        width=params.width,
        generator=generator,
        output_type="np",
    )
    return np.asarray(out.images[0], dtype=np.float32)


# ---- Metrics (A4) ------------------------------------------------------------


class MetricBank:
    """Lazily-built torchmetrics modules, reused across calls (weights load once)."""

    def __init__(self):
        import torch
        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        from torchmetrics.multimodal.clip_score import CLIPScore

        self.torch = torch
        self._lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self._psnr = PeakSignalNoiseRatio(data_range=1.0)
        self._clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

    def _nchw(self, img: np.ndarray):
        # HWC [0,1] -> 1CHW float
        return self.torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    def _clip_score(self, img: np.ndarray, prompt: str) -> float:
        imgs = (self._nchw(img) * 255).to(self.torch.uint8)
        val = self._clip(imgs, [prompt]).item()
        self._clip.reset()
        return val

    def score(self, ref: np.ndarray, test: np.ndarray, prompt: str) -> dict:
        t_ref, t_test = self._nchw(ref), self._nchw(test)
        with self.torch.no_grad():
            lpips = self._lpips(t_test, t_ref).item();  self._lpips.reset()
            ssim = self._ssim(t_test, t_ref).item();    self._ssim.reset()
            psnr = self._psnr(t_test, t_ref).item();    self._psnr.reset()
        return {
            "lpips": lpips,
            "ssim": ssim,
            "psnr": psnr,
            "clip_score_test": self._clip_score(test, prompt),
            "clip_score_ref": self._clip_score(ref, prompt),
        }


# ---- Aggregation + reporting (A4.4, A5) -------------------------------------


def _spread(values: list[float]) -> tuple[float, float]:
    """(p10, p90) via linear interpolation; falls back to min/max for tiny n."""
    if len(values) < 3:
        return (min(values), max(values))
    p = np.percentile(values, [10, 90])
    return (float(p[0]), float(p[1]))


def summarize(rows: list[dict]) -> str:
    metrics = ["lpips", "ssim", "psnr", "clip_score_test"]
    by_cfg: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], {m: [] for m in metrics})
        for m in metrics:
            by_cfg[r["config"]][m].append(r[m])

    lines = ["# Conversion-fidelity ablation — summary", "",
             "median [p10–p90] over the prompt×seed set; vs the `reference` image.",
             "lpips: lower=closer · ssim/psnr: higher=closer · clip: higher=on-prompt", ""]
    header = "| config | LPIPS↓ | SSIM↑ | PSNR↑ | CLIP↑ |"
    lines += [header, "|" + "---|" * 5]
    for cfg in LADDER:
        if cfg.id not in by_cfg:
            continue
        cells = [cfg.id]
        for m in metrics:
            vals = by_cfg[cfg.id][m]
            lo, hi = _spread(vals)
            cells.append(f"{median(vals):.3f} [{lo:.3f}–{hi:.3f}]")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_grid(images_by_cfg: dict[str, np.ndarray], out_path: Path, prompt: str):
    """Side-by-side reference | each config (A5.2). Best-effort; needs Pillow."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    order = [c.id for c in LADDER if c.id in images_by_cfg]
    tiles = [(np.clip(images_by_cfg[c], 0, 1) * 255).astype(np.uint8) for c in order]
    h, w = tiles[0].shape[:2]
    strip = Image.new("RGB", (w * len(tiles), h + 20), "white")
    for i, (cid, tile) in enumerate(zip(order, tiles)):
        strip.paste(Image.fromarray(tile), (i * w, 20))
        ImageDraw.Draw(strip).text((i * w + 4, 4), cid, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_path)


# ---- Runner ------------------------------------------------------------------


def run(artifacts: Artifacts, out_dir: Path, compute_unit: str,
        prompts: list[str], seeds: list[int], params: GenParams):
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = MetricBank()

    provenance = {
        "checkpoint_sha256": _sha256(artifacts.ckpt),
        "unet_mlpackage": artifacts.unet_mlpackage,
        "vae_decoder_mlpackage": artifacts.vae_decoder_mlpackage,
        "text_encoder_mlpackage": artifacts.text_encoder_mlpackage,
        "compute_unit": compute_unit,
        "git_sha": _git_sha(),
        "package_versions": _pkg_versions(),
        "platform": _platform_info(),
        "gen_params": asdict(params),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    # Build each pipeline once; reuse across all prompts/seeds.
    pipes = {c.id: build_config_pipeline(c, artifacts, compute_unit) for c in LADDER}

    rows: list[dict] = []
    results_path = out_dir / "results.jsonl"
    with results_path.open("w") as fh:
        for pi, prompt in enumerate(prompts):
            for seed in seeds:
                ref_img = generate(pipes["reference"], prompt, seed, params)
                grid = {"reference": ref_img}
                for cfg in LADDER:
                    if cfg.id == "reference":
                        continue
                    img = generate(pipes[cfg.id], prompt, seed, params)
                    grid[cfg.id] = img
                    m = bank.score(ref_img, img, prompt)
                    row = {"config": cfg.id, "prompt_idx": pi, "prompt": prompt,
                           "seed": seed, **m}
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    img_path = out_dir / "images" / cfg.id / f"{pi:02d}_s{seed}.png"
                    save_grid({cfg.id: img}, img_path, prompt)  # single tile per cfg
                save_grid(grid, out_dir / "grids" / f"{pi:02d}_s{seed}.png", prompt)

    (out_dir / "summary.md").write_text(summarize(rows))
    print(f"Wrote {len(rows)} rows -> {results_path}")
    print(f"Summary -> {out_dir / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--unet-mlpackage", required=True)
    ap.add_argument("--vae-decoder-mlpackage")
    ap.add_argument("--text-encoder-mlpackage")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--compute-unit", default="CPU_AND_NE")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--limit-prompts", type=int, default=None,
                    help="Use only the first N committed prompts (smoke runs).")
    args = ap.parse_args()

    artifacts = Artifacts(
        ckpt=args.ckpt,
        unet_mlpackage=args.unet_mlpackage,
        vae_decoder_mlpackage=args.vae_decoder_mlpackage,
        text_encoder_mlpackage=args.text_encoder_mlpackage,
    )
    prompts = DEFAULT_PROMPTS[: args.limit_prompts] if args.limit_prompts else DEFAULT_PROMPTS
    run(artifacts, args.out, args.compute_unit, prompts, args.seeds,
        GenParams(steps=args.steps, guidance=args.guidance))


if __name__ == "__main__":
    main()
