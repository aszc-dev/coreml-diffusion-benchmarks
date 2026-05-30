"""Pure-logic tests for the matrix picker helpers.

The interactive ``_matrix_picker`` itself needs a tty so it isn't exercised
here; the abbreviation tables and per-column header tokens are the only
parts that have to stay stable as new backends, compute units, or precisions
are added to the matrix.
"""

from types import SimpleNamespace

from sdbench.tui.prompts import (
    _abbrev_attention,
    _abbrev_backend,
    _abbrev_compute_unit,
    _column_header_tokens,
)


def _row(**cell_fields):
    return SimpleNamespace(cell=SimpleNamespace(**cell_fields))


def test_abbreviations_keep_columns_within_six_chars():
    # The matrix renderer reserves 6 chars per column token; anything longer
    # bleeds into the neighbouring column and breaks alignment.
    for backend in ("apple_coreml", "coreml_diffusion", "diffusers_mps", "mlx"):
        assert len(_abbrev_backend(backend)) <= 6
    for cu in ("CPU_AND_NE", "CPU_AND_GPU", "MPS", "GPU"):
        assert len(_abbrev_compute_unit(cu)) <= 6
    for attn in ("SPLIT_EINSUM_V2", "ORIGINAL", "NATIVE"):
        assert len(_abbrev_attention(attn)) <= 6


def test_column_header_tokens_match_publication_layout():
    row = _row(
        id="apple-ane-fp16",
        label="apple ANE",
        backend="apple_coreml",
        compute_unit="CPU_AND_NE",
        attention="SPLIT_EINSUM_V2",
        precision="fp16",
    )
    assert _column_header_tokens(row) == ["apple", "ANE", "SE2", "fp16"]


def test_compile_variant_tags_precision_token():
    """The two diffusers MPS cells differ only in torch.compile; the header
    must surface that or the user cannot tell the columns apart."""
    row = _row(
        id="diffusers-mps-fp16-compile",
        label="diffusers MPS +compile",
        backend="diffusers_mps",
        compute_unit="MPS",
        attention="NATIVE",
        precision="fp16",
    )
    tokens = _column_header_tokens(row)
    assert tokens[0] == "diff"
    assert tokens[3] == "fp16+c"
