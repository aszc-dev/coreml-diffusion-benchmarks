"""Interactive terminal front-end for the benchmark harness.

Modules here own the human-facing experience (cleanup, config, run views) and
must stay light: no torch/coremltools/mlx imports at module load, so the CLI
front-end starts instantly and the heavy backend dependencies are only paid for
when an actual benchmark runs.
"""
