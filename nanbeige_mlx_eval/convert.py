"""Convert Nanbeige4.2-3B HF weights into MLX quants.

Uses mlx-lm's ``model_file`` hook: the MLX port (``nanbeige.py``) is dropped into
a local copy of the HF repo and referenced from its ``config.json``, so the
standard ``mlx_lm.convert`` pipeline (load -> quantize -> save) works without a
registry entry. ``auto_map`` is stripped from the *output* config because the MLX
side loads via ``model_file``; it is left intact in the source so the HF parity
reference can still load the remote modeling code.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from mlx_lm.convert import convert

PORT_FILE = Path(__file__).resolve().parent / "models" / "nanbeige.py"


def prepare_source(src_dir: str | Path, port_file: str | Path = PORT_FILE) -> None:
    """Inject the port and ``model_file`` into a local HF repo copy."""
    src = Path(src_dir)
    shutil.copy(port_file, src / "nanbeige.py")
    cfg_path = src / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["model_file"] = "nanbeige.py"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def to_mlx(
    src_dir: str | Path,
    out_dir: str | Path,
    bits: int,
    group_size: int = 64,
) -> Path:
    """Convert ``src_dir`` to a quantized MLX repo at ``out_dir`` (``bits`` per weight)."""
    out = Path(out_dir)
    convert(
        hf_path=str(src_dir),
        mlx_path=str(out),
        quantize=True,
        q_group_size=group_size,
        q_bits=bits,
    )
    # Strip auto_map from the output config: MLX loads via model_file, and
    # leaving auto_map triggers an irrelevant trust_remote_code prompt on load.
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.pop("auto_map", None)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Convert Nanbeige HF -> MLX quant")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group-size", type=int, default=64)
    a = ap.parse_args()
    print(to_mlx(a.src, a.out, a.bits, a.group_size))
