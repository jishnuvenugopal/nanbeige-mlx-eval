"""Convert Nanbeige4.2-3B HF weights into MLX quants.

Uses mlx-lm's ``model_file`` hook: the MLX port (``model.py``) is dropped into a
*staged copy* of the HF repo and referenced from that copy's ``config.json``, so
the standard ``mlx_lm.convert`` pipeline (load -> quantize -> save) works without
a registry entry. ``auto_map`` is stripped from the *output* config because the
MLX side loads via ``model_file``; it is left intact in the source so the HF
parity reference can still load the remote modeling code.

Two correctness properties (P3.3, P3.4 in docs/investigation-log.md):

* **Never mutate the source.** ``prepare_source`` stages a conversion-ready copy
  in a temp dir, symlinks the (large) safetensors shards instead of duplicating
  them, and writes the ``model_file`` config into the *copy*. Pointing ``--src``
  at an HF cache snapshot is therefore safe — the snapshot's hashes are
  untouched. ``to_mlx`` calls ``prepare_source`` itself and cleans up the stage,
  so ``convert`` is one step and idempotent from a pristine HF download.
* **Preserve tokenizer files and verify them.** mlx-lm's convert writes a
  trimmed ``tokenizer_config.json`` and drops ``added_tokens.json`` /
  ``special_tokens_map.json`` / ``tokenizer.model``. We copy those through,
  drop transient init keys, restore ``model_max_length``, and assert the
  tokenizer round-trips the EOS token before declaring success.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from mlx_lm.convert import convert

# The single source of truth for the port. Copied verbatim into each staged
# (and therefore each converted) weight repo as the shipped ``model_file``.
PORT_FILE = Path(__file__).resolve().parent / "model.py"

# Tokenizer / template files mlx-lm's convert trims that we must carry through.
CARRY = [
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "chat_template.jinja",
]

# Transient keys mlx-lm leaks into tokenizer_config.json from
# tokenizer.init_kwargs; they don't belong in a published config.
_TRANSIENT_TOKENIZER_KEYS = ("is_local", "local_files_only", "backend")

# HF ``trust_remote_code`` files that the source repo carries for the PyTorch
# reference path. They must NOT ship in an MLX repo (which loads via
# ``model_file``), so remove them from the output after convert.
_STRIP_FROM_OUTPUT = {
    "configuration_nanbeige.py",
    "modeling_nanbeige.py",
}

# Nanbeige4.2-3B supports 256K context; mlx-lm's convert writes the int64
# sentinel instead of this concrete value. Restore it.
MODEL_MAX_LENGTH = 262144

# EOS token the model must be able to emit to ever stop generating.
EOS_TOKEN = "<|im_end|>"
EOS_TOKEN_ID = 166101


def prepare_source(
    src_dir: str | Path,
    model_file: str | Path = PORT_FILE,
    workdir: str | Path | None = None,
) -> Path:
    """Stage a conversion-ready copy of ``src_dir``. **Never writes into ``src_dir``**.

    Non-safetensor files are copied; ``*.safetensors`` shards are *symlinked*
    (not duplicated — the BF16 checkpoint is ~8 GB). The port is copied in as
    ``nanbeige.py`` and ``config.json["model_file"]`` is set on the *copy*.

    Returns the stage directory. The caller is responsible for cleanup if they
    don't go through :func:`to_mlx` (which handles it).
    """
    src = Path(src_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"source repo not found: {src}")

    stage = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="nanbeige-stage-")
    )
    stage.mkdir(parents=True, exist_ok=True)

    # Copy small files verbatim.
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, stage / f.name)

    # Symlink the weight shards — don't duplicate ~8 GB.
    for f in src.glob("*.safetensors"):
        (stage / f.name).symlink_to(f.resolve())

    # Drop in the port and wire up model_file on the staged config copy.
    shutil.copy2(model_file, stage / "nanbeige.py")
    cfg_path = stage / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["model_file"] = "nanbeige.py"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    return stage


def _finalize(src: Path, out: Path) -> None:
    """Carry tokenizer files through, scrub transient keys, restore max length.

    Idempotent: only copies a CARRY file when the output is missing it.
    """
    for name in CARRY:
        if (src / name).exists() and not (out / name).exists():
            shutil.copy2(src / name, out / name)

    tc = out / "tokenizer_config.json"
    if tc.exists():
        cfg = json.loads(tc.read_text(encoding="utf-8"))
        for k in _TRANSIENT_TOKENIZER_KEYS:
            cfg.pop(k, None)
        cfg["model_max_length"] = MODEL_MAX_LENGTH
        tc.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    # Strip HF trust_remote_code files the convert step may have copied through;
    # the MLX repo loads via model_file and shouldn't ship the PyTorch reference.
    for name in _STRIP_FROM_OUTPUT:
        p = out / name
        if p.exists():
            p.unlink()


def verify(out_dir: str | Path) -> None:
    """Assert the converted repo's tokenizer round-trips the EOS token.

    A conversion that silently loses ``<|im_end|>`` produces a model that never
    stops generating — cheap to assert, expensive to debug.
    """
    from transformers import AutoTokenizer  # type: ignore

    out = Path(out_dir)
    tok = AutoTokenizer.from_pretrained(str(out), trust_remote_code=False)
    got = tok.convert_tokens_to_ids(EOS_TOKEN)
    assert got == EOS_TOKEN_ID, (
        f"EOS token lost in conversion: {EOS_TOKEN!r} -> {got} (expected {EOS_TOKEN_ID})"
    )
    # ``eos_token_id`` may be stored as a string in some tokenizer configs.
    assert int(tok.eos_token_id) == EOS_TOKEN_ID, (
        f"tokenizer.eos_token_id={tok.eos_token_id} != {EOS_TOKEN_ID}"
    )
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<|im_start|>assistant" in rendered, "chat template missing assistant marker"
    assert "<think>" in rendered, "chat template missing the reasoning (<think>) block"


def to_mlx(
    src_dir: str | Path,
    out_dir: str | Path,
    bits: int,
    group_size: int = 64,
    *,
    skip_verify: bool = False,
) -> Path:
    """Convert ``src_dir`` to a quantized MLX repo at ``out_dir`` (``bits`` per weight).

    Stages a non-mutating copy (P3.3), runs mlx-lm's convert, strips
    ``auto_map`` from the output config, carries tokenizer files through and
    verifies the round-trip (P3.4). The stage dir is always cleaned up.
    """
    src = Path(src_dir)
    out = Path(out_dir)

    stage = prepare_source(src)
    try:
        convert(
            hf_path=str(stage),
            mlx_path=str(out),
            quantize=True,
            q_group_size=group_size,
            q_bits=bits,
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    # Strip auto_map from the output config: MLX loads via model_file, and
    # leaving auto_map triggers an irrelevant trust_remote_code prompt on load.
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.pop("auto_map", None)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    _finalize(src, out)

    if not skip_verify:
        try:
            verify(out)
            print(
                f"PASS: tokenizer round-trips {EOS_TOKEN!r} -> {EOS_TOKEN_ID}; "
                f"chat template intact."
            )
        except Exception as exc:  # pragma: no cover - surfaces as a hard error
            print(f"VERIFY FAILED: {exc}")
            raise

    return out


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(
        prog="mlx-nanbeige-convert",
        description="Convert Nanbeige4.2-3B HF weights to a quantized MLX repo.",
    )
    ap.add_argument("--src", required=True, help="local Nanbeige HF repo (left untouched)")
    ap.add_argument("--out", required=True, help="output MLX repo path")
    ap.add_argument("--bits", type=int, required=True, help="quantization bits (4, 6, 8)")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the post-convert tokenizer round-trip assertion",
    )
    a = ap.parse_args(argv)
    to_mlx(a.src, a.out, a.bits, a.group_size, skip_verify=a.skip_verify)
    print(a.out)


if __name__ == "__main__":  # pragma: no cover
    main()
