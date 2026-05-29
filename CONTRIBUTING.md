# Contributing a benchmark report

This document is the contract between an external contributor running the
benchmark on their Apple Silicon machine and the maintainer who will compare,
verify, and publish the results.

> **Easiest path (recommended)** — `uv tool install coreml-diffusion-benchmarks`
> → `cdbench` → guided full-screen menu walks you through download, convert,
> run, and bundle. The post-run step prompts to build and validate the
> submission `.zip`. Attach the zip to a GitHub Discussion in this repo.
>
> **Scripted path** — install with `uv sync`, prepare the checkpoint, run
> `sudo ./scripts/run.sh`, then `cdbench report --zip --anonymize --salt
> <whatever>` and attach the resulting `.zip`.
>
> `sdbench` is kept as a legacy alias of `cdbench`; both run the same app.

The harness captures everything needed to reproduce a run on an identical
machine: chip variant + core counts, macOS build, kernel, repo SHA + dirty flag,
`uv.lock` hashes, full resolved package set, RNG seed, latent/text-embedding
hashes, runtime power/load/thermal at start and end, and the powermetrics
sampler metadata. Those land in `results/data/environment.json` and are mirrored
into every JSONL record. The submission bundle just zips them.

---

## Hardware prerequisites

- Apple Silicon Mac (M1 or newer; the harness records the chip variant so any
  generation is acceptable as long as we know which one ran).
- macOS that exposes `powermetrics` (any recent release).
- ~30 GB of free disk space for the converted CoreML artifacts of the full
  matrix.
- Either the SD 1.5 `.safetensors` checkpoint on disk, or network access to
  fetch it from the official HF repo on first run.

The benchmark MUST NOT run under Rosetta — the harness records the Rosetta flag
and reports refuse to validate when it's true.

---

## One-time setup

### Guided (recommended)

```bash
uv tool install coreml-diffusion-benchmarks
cdbench   # full-screen menu: download → convert → configure → run → report
```

The guided menu walks you through every step (download checkpoint, convert
CoreML artifacts, configure cells, run with live progress, build submission
bundle). The post-run step prompts to build and validate a report bundle
with the same reproducibility guarantees as the scripted path.

### Scripted

```bash
# 1. install (pins everything — R1.5, R11.3)
uv sync
uv sync --project envs/apple-ct8
uv sync --project envs/team-ct9

# 2. resolve and SHA-verify the SD 1.5 checkpoint (point at a local file
#    or auto-download from the official HF repo)
uv run cdbench download --download

# 3. convert the CoreML artifacts (cached by checkpoint SHA, so you only
#    pay this cost once per machine + checkpoint)
uv run cdbench convert
```

---

## Running the benchmark

```bash
# full matrix with power + thermal + no-sleep wrapper.
# sudo is needed only for powermetrics; the harness itself runs unprivileged.
sudo ./scripts/run.sh --config config/matrix.yaml
```

Useful flags:

- `--no-power` — skip the powermetrics sampler (drops the sudo requirement,
  fills `gpu_power_w` / `ane_power_w` with `null`).
- `--dry-run` — print the command lines without executing.
- `--cell <id>` — run a single cell (matches `config/matrix.yaml :: cells.id`).

### Running without sudo

You don't need to type the prefix `sudo`. If you launch `./scripts/run.sh` as a
normal user the script will prompt:

```
[sdbench] powermetrics needs sudo for per-engine power sampling (R6.1).
[sdbench] You can:
          (a) re-run with 'sudo ./scripts/run.sh ...' for power figures, or
          (b) continue WITHOUT power (latency / size / equivalence still measured).
[sdbench] Authorize sudo for powermetrics now? [y/N]
```

Answering **n** (or pressing Enter) continues without power measurement —
`gpu_power_w` and `ane_power_w` are `null` in every record, everything else
(latency, equivalence, size, conversion time) is still measured. The manifest
records `power_sampler=null` so the maintainer sees power was opted out.

If you prefer to skip the prompt entirely, pass `--no-power`.

The wrapper exports `SDBENCH_RUN_ID` so the plist filename, the
`environment.json` manifest, and the per-cell records all share one identifier.

---

## Submitting the report

After the run completes, the guided flow prompts to build the bundle inline
(with anonymization + salt prompts). If you skipped that step, or want to
re-bundle later, run:

```bash
uv run cdbench report --zip --anonymize --salt <your-private-salt>
```

That produces a directory **and** a zip under `results/reports/<run_id>/`. Attach
the `.zip` to a GitHub Discussion in this repo.

### What goes in the bundle

| File | What it carries |
| --- | --- |
| `manifest.json` | Full environment manifest (host, OS, toolchain, repo, determinism, runtime conditions). |
| `results.jsonl` | One record per matrix cell, including per-cell thermal/load snapshots. |
| `tables/` | Generated latency / power_energy / size_quantization / conversion_time / environment Markdown tables. |
| `raw/` | Retained `*-powermetrics.plist` (R10.3) for auditability. |
| `matrix.yaml` | The matrix config the run actually consumed. |
| `README.md` | Bundle header — chip, build, schema version, anonymization status. |
| `SCHEMA.md` | Schema version notes + supported compatibility window. |

### What `--anonymize` strips

The flag is opt-in and **preserves every field needed for reproducibility**. Only
filesystem-local PII is removed:

- `determinism.checkpoint_path` (leaks `$HOME`).
- `repo.upstream_url` and `repo.dirty_files`.
- `hardware.host_id_hash` is **re-hashed** with your `--salt`, so the same
  contributor's runs still dedup against each other if they reuse the same salt.

Everything else — the chip variant, core counts, macOS build, kernel, RAM, repo
SHA, `uv.lock` hashes, package versions, RNG seed, latent SHA, checkpoint SHA,
power sampler metadata, thermal snapshots — is retained. Without these the
report cannot be reproduced.

---

## What the maintainer does

```bash
uv run cdbench validate-report path/to/<run_id>/
```

The validator gates on:

1. `schema_version` is supported by the maintainer's harness.
2. All records share one `provenance_digest`, matching the manifest.
3. `latent_input_sha256` / `text_embedding_input_sha256` are consistent across
   all records and match `manifest.determinism`.
4. Records flagged with `numerically_divergent=true` are surfaced but not
   rejected (Golden Rule 7).

A bundle that fails any of (1)-(3) is rejected; the contributor is asked to
re-run after fixing the noted drift. (4) is surfaced for discussion.

---

## Pitfalls

- **Dirty repo**: the harness records `repo.dirty=true` and the dirty file list.
  Reports run with a dirty harness are not rejected, but the maintainer sees the
  flag in every table caption and will ask what changed.
- **Background load**: avoid running other heavyweights. The harness records
  `loadavg` and the top-5 CPU consumers at start and end so unusual load is
  visible.
- **AC vs battery**: macOS clocks down on battery. The manifest records the AC
  state at start and end; expect a maintainer to ask for an AC-only re-run if
  the numbers are surprisingly slow.
- **Thermal throttling**: `pmset -g therm` is sampled at run start, after each
  cell, and at run end. Throttled runs are flagged but not aborted by default;
  the `thermal.abort_on_throttle` config knob lets you change that.
- **Re-running invalidates `provenance.json`**: bumping the toolchain or moving
  the checkpoint changes the fingerprint, which wipes the previous run's
  results and tables on next run. That is by design — mixed-fingerprint
  datapoints are the failure mode the provenance ledger exists to prevent.
