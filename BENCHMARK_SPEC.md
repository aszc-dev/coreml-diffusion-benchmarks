# SD 1.5 UNet Cross-Backend Benchmark — Build Specification

**Audience:** code-generation agent (Codex).
**Nature of this document:** requirements only. Describe *what* each component must do and the contract it must satisfy. Do not invent product behavior beyond these requirements. Justifications appear only where omitting them would lead to an incorrect implementation.

---

## 0. Scope & non-goals

- **R0.1** The benchmark MUST measure the **UNet only** of Stable Diffusion 1.5. VAE and CLIP/text-encoder are out of scope and MUST NOT be in any timed or power-measured path.
- **R0.2** The model under test MUST be one single vanilla SD 1.5 checkpoint, identical (same weights) across all backends.
- **R0.3** Four backends ("sides") are in scope:
  1. **Apple `ml-stable-diffusion`** built with **coremltools 8** — CoreML, measured under both ANE and GPU compute units.
  2. **`coreml-diffusion`** (the team's own pipeline) built with **coremltools 9** — CoreML, measured under both ANE and GPU compute units.
  3. **`diffusers`** loading `.safetensors` on **MPS** (GPU).
  4. **MLX** (GPU; no ANE path exists).
- **R0.4** Five metric families are in scope: **inference time, power draw (GPU vs ANE separated), model size, quantization efficiency, conversion time.**
- **R0.5** Non-goals for v1: SDXL, LCM, and Flux MUST NOT block the suite. They MAY appear only as a documented "future work / not-applicable" section. Default target resolution is 512×512; the harness MUST allow other resolutions via config but only 512×512 is required to pass.
- **R0.6** The experiment MUST be a **self-contained, public-GitHub-ready repository**: only this repo's own scripts, configs, and results are committed; every backend (Apple repo, `coreml-diffusion`, `diffusers`, `mlx`) is consumed as an installed dependency, not vendored.
- **R0.7** Results MUST be emitted in a form directly reusable in a written blog post (see §10): machine-readable data plus generated tables.

---

## 1. Repository & orchestration

- **R1.1** A single CLI entrypoint MUST run the full matrix end-to-end and MUST also support running any single (backend × compute-unit × variant) cell in isolation.
- **R1.2** The run matrix MUST be **config-driven** (declarative file). A matrix cell is the tuple: `{backend, compute_unit, attention_variant, precision/quantization, resolution}`. Adding a cell MUST NOT require code changes.
- **R1.3** The orchestrator MUST be resilient: failure of one cell MUST be recorded and MUST NOT abort the remaining cells.
- **R1.4** All randomness MUST be seeded from a single configured seed, recorded in the output manifest.
- **R1.5** Dependencies MUST be managed with **uv**: a committed **`uv.lock`** pins exact versions and the Python version is pinned (`requires-python` / `.python-version`). Install MUST be a single documented command (`uv sync`).

---

## 2. Backend adapter contract (uniform interface)

- **R2.1** Every backend MUST be wrapped behind one common adapter interface so the harness is backend-agnostic. The interface MUST expose, at minimum, three methods with these contracts:
  - **`prepare()`** — load weights, compile/warm the model for the requested compute unit, and bring the model to a state where the first `step()` is representative. All one-time cost (disk load, compile) MUST happen here, never inside the timed loop.
  - **`step(latent, timestep, text_embedding) -> latent_out`** — execute exactly one UNet forward pass and return the output latent in a host-comparable form (see §7). MUST contain only compute that legitimately belongs to one denoising step.
  - **`teardown()`** — release resources.
- **R2.2** `step()` MUST be free of any hidden lazy/deferred computation: the returned latent MUST be fully materialized on the host side before `step()` returns (rationale: lazy frameworks otherwise let the harness time graph construction instead of compute — see R3.4/R5.5).
- **R2.3** Each adapter MUST report static metadata: backend name, compute unit actually used, attention variant, precision/quantization, and the resolved model artifact path(s).

---

## 3. The four backends — per-backend requirements

### 3.1 Backend 1 — Apple `ml-stable-diffusion` + coremltools 8
- **R3.1.1** MUST convert the SD 1.5 UNet using Apple's official converter pinned to a coremltools 8.x release; the exact version MUST be recorded.
- **R3.1.2** MUST produce a **`SPLIT_EINSUM_V2`** (ANE-oriented) build and MUST also support an **`ORIGINAL`** (GPU-oriented) build; both MUST be benchmarkable.
- **R3.1.3** MUST be measured under both `CPU_AND_NE` and `CPU_AND_GPU` compute units.
- **R3.1.4** MUST be pre-compiled to the Apple-compiled on-disk form and loaded from that form for timing (rationale: per-load recompilation otherwise contaminates timing and the ANE compile pass alone can take minutes).

### 3.2 Backend 2 — `coreml-diffusion` + coremltools 9
- **R3.2.1** MUST be wired in as a dependency providing its own conversion + inference path, pinned to coremltools 9.x with the team's torch/numpy versions; exact versions MUST be recorded.
- **R3.2.2** MUST use a **single conversion path** that mirrors Apple's method (same custom diffusers pipeline and trace-based API as Backend 1), differing only in the toolchain (coremltools 9 + the team's torch/numpy versions). This isolates **toolchain version** as the single variable between Backend 1 and Backend 2 (rationale: that toolchain-vs-toolchain comparison is the primary purpose of the study). No separate `torch.export` build is required for v1.
- **R3.2.3** MUST be measured under both `CPU_AND_NE` and `CPU_AND_GPU`.
- **R3.2.4** MUST support at least one weight-quantized variant (e.g. low-bit palettized) in addition to fp16, for the quantization-efficiency metric (§8). Any activation-quantized (W8A8) variant is OPTIONAL and, if included, MUST be gated to hardware that supports it and skipped (recorded as N/A) elsewhere.
- **R3.2.5** Same pre-compile-before-timing rule as R3.1.4.

### 3.3 Backend 3 — `diffusers` on MPS
- **R3.3.1** MUST load the same checkpoint via `diffusers` in fp16 on the `mps` device, exposing only the UNet through the adapter.
- **R3.3.2** This backend serves as the **PyTorch numerical reference** for §7 (its fp32-on-CPU output is the ground truth; its fp16-on-MPS output is one of the compared points).
- **R3.3.3** MUST record whether any compile/graph-optimization (e.g. `torch.compile`) is enabled; if offered as a variant it MUST be a separate, labeled matrix cell.

### 3.4 Backend 4 — MLX
- **R3.4.1** MUST run an SD 1.5 UNet in MLX in fp16, exposing only the UNet through the adapter. The upstream `ml-explore/mlx-examples` `stable_diffusion` module does **not** ship an SD 1.5 configuration (it provides SD 2.1-base, SDXL, FLUX). The adapter MUST therefore supply a verified SD 1.5 UNet configuration on the canonical mlx-examples architecture (SD 1.5 differs from the shipped SD 2.1-base UNet essentially by cross-attention dim 768 and eps- rather than v-prediction) and load the **same checkpoint** as the other backends. A version-pinned community SD 1.5 port MAY be used as a fallback/cross-check, but the same-checkpoint guarantee (R4.1) MUST hold. A cell is valid only once SD 1.5 is confirmed running and passes the numerical-equivalence check (§7).
- **R3.4.2** Inside `step()`, the implementation MUST force evaluation of the output before returning (rationale: MLX is lazy; without forced evaluation the harness measures graph construction, producing implausibly fast and invalid timings).
- **R3.4.3** All weight loading and first-call graph build MUST occur in `prepare()`, not in the timed loop.
- **R3.4.4** An OPTIONAL MLX-native quantized variant MAY be added for §8; if added it MUST be a separate labeled cell.

---

## 4. Shared inputs & fairness controls

- **R4.1** A single fixed input set MUST be generated once, persisted to disk, and loaded by **every** backend: the input latent, the integer timestep, and the text-embedding tensor. Each backend casts this shared input to its native dtype/array type — no backend generates its own inputs (rationale: the only variable across cells must be the implementation, not the data).
- **R4.2** The shared input MUST be generated from the recorded seed in a backend-neutral way (host-side, not via any one backend's RNG).
- **R4.3** All backends in a comparison MUST use the same resolution, the same timestep, and the same number of timed iterations.
- **R4.4** The harness MUST record, per cell, the exact compute unit, attention variant, and precision actually realized (not merely requested), and MUST flag any mismatch between requested and realized configuration.

---

## 5. Timing methodology

- **R5.1** Inference time MUST be measured around `step()` only, using a monotonic clock.
- **R5.2** The first iteration of every cell MUST be discarded as a cold run (rationale: first-call compile/shader/graph warmup is not representative).
- **R5.3** At least **10 timed iterations** MUST be executed back-to-back after the discarded cold run.
- **R5.4** Reported per-cell latency MUST be the **median** with a dispersion measure (IQR or min/max), not the mean (rationale: thermal drift skews the tail; median is the robust statistic).
- **R5.5** No lazy/deferred work may leak out of the timed window (enforced via R2.2 / R3.4.2).
- **R5.6** The harness MUST verify thermal state is unthrottled before/around a run and MUST flag or abort cells executed while the system is thermally throttled.

---

## 6. Power methodology (GPU vs ANE separation)

- **R6.1** Per-engine power MUST be collected via the OS power sampler producing **separate CPU, GPU, and ANE power channels**, streamed to a file in a machine-parseable format. The sampling interval MUST be fine enough to resolve a single UNet step and is configurable; a default around 100 ms MUST be used.
- **R6.2** Each measured run MUST capture an **idle baseline window** before and after the active window; reported active power MUST be **baseline-subtracted** per engine.
- **R6.3** Power samples MUST be **time-aligned** to the timed-iteration window using shared monotonic timestamps, so reported power corresponds to the same iterations as reported latency.
- **R6.4** The harness MUST report, per cell: average GPU power and average ANE power over the active window, and an **energy-per-image** figure (power × wall time across the timed iterations).
- **R6.5** The suite MUST prevent display/system sleep during measurement.
- **R6.6** Output and documentation MUST state that these power figures are **estimates valid only for relative comparison on the same machine in one sitting**, and MUST NOT present them as device-to-device absolute measurements (rationale: the OS sampler explicitly disclaims cross-device accuracy).
- **R6.7** For GPU-only backends the ANE channel is expected near zero and vice-versa; the harness MUST record both channels for every cell regardless, with no per-process attribution assumed.
- **R6.8** Background ANE/GPU contention MUST be minimized and the run conditions (what else was running) recorded; the privileged-access requirement of the sampler MUST be documented as a prerequisite.

---

## 7. Numerical equivalence (fairness gate)

- **R7.1** For the shared input (R4.1), each backend's output latent MUST be compared against the PyTorch reference (R3.3.2) by computing **MSE and cosine similarity** in host-side fp32.
- **R7.2** Default acceptance thresholds MUST be configurable; defaults: MSE below 1e-3 and cosine above 0.999 versus reference. Actual measured values MUST be recorded for every cell regardless of pass/fail.
- **R7.3** A cell that fails the threshold MUST be **flagged, not dropped**: its timing/power/size results remain in the output, marked as numerically divergent (rationale: fp16 accumulation-order differences are expected and still produce visually identical images; the reader needs the number, not a silent exclusion).
- **R7.4** The equivalence check MUST run on the same fixed input used for timing, not a fresh random one.

---

## 8. Model size & quantization reporting

- **R8.1** For every model artifact (per backend and per variant) the harness MUST report four columns: **on-disk size**, **weight-only size**, **effective bits per parameter**, and **compute precision** (e.g. fp16 / weight-4-bit / W8A8).
- **R8.2** On-disk size MUST measure the full artifact as actually shipped (for directory-form artifacts, the whole directory including metadata).
- **R8.3** Quantization efficiency MUST be reported as the relationship between three deltas relative to the fp16 baseline of the same backend: **size reduction**, **latency change**, and **quality change**. Quality change MUST be sourced from the §7 metrics of the quantized variant versus that backend's fp16 variant.
- **R8.4** Where a quantization mode is unavailable on the test hardware, the cell MUST be recorded as N/A with the reason, never silently omitted.

---

## 9. Conversion-time measurement

- **R9.1** For each CoreML backend (Backends 1 and 2) the harness MUST capture and report three separate one-time wall-clock costs: **graph capture (trace/export) time**, **convert time**, and **first-load compile time** for the target compute unit.
- **R9.2** Conversion time MUST be reported per build variant and side-by-side for Backend 1 (coremltools 8) versus Backend 2 (coremltools 9) (rationale: the toolchain-version comparison is a primary purpose of the study).
- **R9.3** Conversion-time figures MUST be clearly separated from per-step inference timing and never mixed into latency results.

---

## 10. Outputs & artifacts

- **R10.1** Every run MUST emit a **machine-readable results file** (one row/record per matrix cell) containing all metrics from §5–§9 plus the realized configuration from R4.4.
- **R10.2** The harness MUST generate **publication-ready summary tables** derived from the machine-readable results (for direct inclusion in the README and the planned blog post), at minimum: a latency table, a power/energy table, a size+quantization table, and a conversion-time table.
- **R10.3** Raw power-sampler output MUST be retained per run for auditability.
- **R10.4** An **environment manifest** MUST be written per run: chip/model, OS version, and the versions of all relevant frameworks (torch, coremltools per build, mlx, numpy, diffusers), plus the seed and run conditions.
- **R10.5** Results and generated tables MUST live under a results directory in the repo, suitable for committing alongside the scripts.
- **R10.6** The README MUST contain a methodology section that is auto-consistent with the harness behavior (warmup/discard policy, statistic reported, power-as-relative caveat, fairness controls).

---

## 11. Environment & reproducibility

- **R11.1** A single documented command MUST reproduce the full run on a correctly provisioned machine.
- **R11.2** Hardware/OS prerequisites and the privileged-access requirement for power sampling MUST be documented as preconditions.
- **R11.3** All framework versions MUST be pinned and recorded in the manifest (R10.4); the comparison is version-sensitive and unpinned runs are not acceptable.
- **R11.4** The repo MUST run with each backend installed as an external dependency; no backend source may be copied into the repo.

---

## 12. Acceptance criteria

The build is accepted when all of the following hold:

- **A1** The single entrypoint runs the full configured matrix and also runs any single cell in isolation.
- **A2** All four backends execute SD 1.5 UNet through the common adapter; CoreML backends run under both `CPU_AND_NE` and `CPU_AND_GPU`; MLX forces evaluation inside the timed step.
- **A3** Every cell uses the shared persisted input; realized configuration is recorded and any requested-vs-realized mismatch is flagged.
- **A4** For each cell: cold run discarded, ≥10 timed iterations, median+dispersion reported; thermal throttling detected and flagged/aborted.
- **A5** Per-engine GPU and ANE power are baseline-subtracted, time-aligned to the timed iterations, and energy-per-image is reported; outputs carry the relative-only caveat.
- **A6** Numerical equivalence (MSE + cosine vs PyTorch reference) is computed for every cell on the shared input, with thresholds applied as flags, not as exclusions.
- **A7** Model-size table (4 columns) and quantization-efficiency deltas are produced; unavailable modes are N/A with reason.
- **A8** Conversion time (capture / convert / first-load compile) is reported per CoreML build, with coremltools 8 vs 9 side by side.
- **A9** Machine-readable results, raw power output, environment manifest, and publication-ready tables are emitted; a single command reproduces the run; all versions pinned and recorded.
- **A10** One cell failing does not abort the suite; its failure is recorded.
